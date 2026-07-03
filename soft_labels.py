"""
Dictionary soft-label cross-lingual alignment for causal LM training.

Given single-token dictionary pairs, we nudge the model so that when it predicts
a word, a little probability mass also lands on that word's translations. This
requires the source and target words to be single tokens (guaranteed by the
forced-token tokenizer).
"""

from collections import defaultdict

import torch


def _read_pairs(path):
    """Yield (source, target) word pairs from a MUSE dict (tab- or space-separated)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t") if "\t" in line else line.rsplit(" ", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                yield parts[0].strip().lower(), parts[1].strip().lower()


def build_soft_map(tokenizer, dict_en_nl=None, dict_en_zh=None, use_t2s=True):
    """Build {src_id: set(tgt_ids)} for pairs that are single tokens, both directions."""
    convert = (lambda w: w)
    if use_t2s:
        from opencc import OpenCC
        convert = OpenCC("t2s").convert   # traditional -> simplified Chinese

    def single_id(word):
        ids = tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    soft_map = defaultdict(set)

    def add(a, b):
        ia, ib = single_id(a), single_id(b)
        if ia is not None and ib is not None and ia != ib:
            soft_map[ia].add(ib)   # a -> b
            soft_map[ib].add(ia)   # b -> a  (reverse direction)

    if dict_en_nl:
        for en, nl in _read_pairs(dict_en_nl):
            add(en, nl)
    if dict_en_zh:
        for en, zh in _read_pairs(dict_en_zh):
            add(en, convert(zh))

    return soft_map


def compile_tables(soft_map, vocab_size, eps=0.15, K=4, device="cuda:0"):
    """Turn the map into lookup tables indexed by token id.

    Defaults mean 'train normally': true_weight=1, trans_wts=0. Only mapped
    tokens get true_weight=1-eps and eps spread over their translations.
    """
    true_weight = torch.ones(vocab_size)
    trans_ids = torch.zeros(vocab_size, K, dtype=torch.long)   # pad id 0
    trans_wts = torch.zeros(vocab_size, K)

    for src_id, targets in soft_map.items():
        targets = list(targets)[:K]
        if not targets:
            continue
        true_weight[src_id] = 1.0 - eps
        for j, t in enumerate(targets):
            trans_ids[src_id, j] = t
            trans_wts[src_id, j] = eps / len(targets)

    return true_weight.to(device), trans_ids.to(device), trans_wts.to(device)


def soft_label_loss(logits, labels, true_weight, trans_ids, trans_wts):
    """Soft-label cross-entropy for causal LM (does the one-position shift)."""
    V = logits.size(-1)

    # shift: logits at position t predict token t+1
    shift_logits = logits[:, :-1, :].reshape(-1, V)   # [N, V]
    y = labels[:, 1:].reshape(-1)                      # [N]

    # drop padding positions
    keep = y != -100
    if keep.sum() == 0:
        return logits.sum() * 0.0                      # nothing to train on
    shift_logits = shift_logits[keep]                  # [M, V]
    y = y[keep]                                         # [M]

    # log-normalizer per position (avoids building a full [M, V] softmax)
    lse = torch.logsumexp(shift_logits, dim=-1)        # [M]

    # main term: weighted -log p(true token)
    logit_true = shift_logits.gather(1, y[:, None]).squeeze(1)   # [M]
    w = true_weight[y]                                            # [M]
    main = w * (lse - logit_true)                                # [M]

    # translation terms: sum_k weight_k * -log p(translation_k)
    tid = trans_ids[y]                                 # [M, K]
    tw = trans_wts[y]                                  # [M, K]
    logit_trans = shift_logits.gather(1, tid)          # [M, K]
    extra = (tw * (lse[:, None] - logit_trans)).sum(1) # [M]

    return (main + extra).mean()
