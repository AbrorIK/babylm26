"""
Triplet soft-label cross-lingual alignment for causal LM training.

Given (english, dutch, chinese) triplets, we nudge the model so that when it
predicts an English word, a little probability mass also lands on the FIRST
token of each translation. Only the first token is modelled: the rest of the
translation is ignored, which keeps the target well defined under left-to-right
decoding.

Keys are English words that are a single token, so one token id means one word.
"""

import torch


def _readers(tokenizer):
    """Two ways to get a word's first token id.

    initial(): the word as it appears after whitespace, i.e. metaspace-prefixed.
    Correct for English and Dutch.

    mid(): the word as it appears with no preceding space. Correct for Chinese,
    where the corpus is unsegmented and the prefixed form is a different id.
    """
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)

    def initial(word):
        ids = encode(word)
        return ids[0] if ids else None

    def mid(word):
        ids = encode("x" + word)
        return ids[1] if len(ids) > 1 else None

    return encode, initial, mid


def build_triplet_map(tokenizer, triplets_path):
    """Build {en_id: [nl_id, zh_id]} from a tab-separated triplet file."""
    encode, initial, mid = _readers(tokenizer)

    triplet_map = {}
    with open(triplets_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            en, nl, zh = (p.strip() for p in parts)

            ids = encode(en)
            if len(ids) != 1:
                continue
            en_id = ids[0]
            targets = [t for t in (initial(nl), mid(zh)) if t is not None and t != en_id]
            if targets:
                triplet_map[en_id] = targets

    return triplet_map


def compile_tables(soft_map, vocab_size, eps=0.10, K=2, device="cuda:0"):
    """Turn the map into lookup tables indexed by token id.

    eps is the mass given to EACH translation, so a token with both gets
    80/10/10 and one with only a usable Chinese translation gets 90/10.
    Unmapped tokens keep true_weight=1 and trans_wts=0, i.e. plain CE.
    """
    true_weight = torch.ones(vocab_size)
    trans_ids = torch.zeros(vocab_size, K, dtype=torch.long)   # pad id 0
    trans_wts = torch.zeros(vocab_size, K)

    for src_id, targets in soft_map.items():
        targets = list(targets)[:K]
        if not targets:
            continue
        true_weight[src_id] = 1.0 - eps * len(targets)
        for j, t in enumerate(targets):
            trans_ids[src_id, j] = t
            trans_wts[src_id, j] = eps

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
