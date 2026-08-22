"""
PreAlign: before training the language model, briefly train the model so that
translations of the same word (e.g. "cat", "kat", "猫") end up with similar
internal representations. This shapes the shared trunk before the LM objective
pushes languages apart.

Only the trunk (embeddings + transformer layers) is aligned.
The language heads are left alone.
"""

from __future__ import annotations

import argparse
import contextlib
import random
from typing import TYPE_CHECKING, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

# A group of translations of the same concept, e.g. ["cat", "kat", "猫"].
WordGroup = list[str]
# A standard HuggingFace batch: {"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor}.
LMBatch = dict[str, Tensor]

# Which device to run on. Overridden to "cpu" for testing.
DEVICE: str = "cuda:0"


def _autocast() -> contextlib.AbstractContextManager[None]:
    """Use half-precision (bfloat16) on GPU for speed. Do nothing on CPU."""
    if DEVICE.startswith("cuda"):
        return torch.autocast(dtype=torch.bfloat16, device_type="cuda:0")
    return contextlib.nullcontext()


def load_word_groups(path: str) -> list[WordGroup]:
    """Read a TSV file of translations. Each line is "eng\\tnld\\tzho".

    Returns e.g. [["cat", "kat", "猫"], ["dog", "hond", "狗"], ...].
    """
    groups: list[WordGroup] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split("\t")]
            if len(parts) == 3 and all(parts):
                groups.append(parts)
    return groups


def flatten_groups(groups: list[WordGroup]) -> tuple[list[str], Tensor]:
    """Unnest translation groups into a flat word list + a group label per word.

    Example:
        Input:  [["cat", "kat", "猫"], ["dog", "hond", "狗"]]
        Output: words    = ["cat", "kat", "猫", "dog", "hond", "狗"]
                group_ids = tensor([0, 0, 0, 1, 1, 1])

    Returns (words, group_ids).
        words:     list of N strings
        group_ids: [N] — integer saying which group each word belongs to
    """
    words: list[str] = []
    group_ids: list[int] = []
    for gid, group in enumerate(groups):       # gid = 0, 1, 2, ...
        for word in group:                     # 3 words per group
            words.append(word)
            group_ids.append(gid)
    return words, torch.tensor(group_ids)      # group_ids shape: [N]


def encode_words(
    words: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_len: int = 8,
) -> tuple[Tensor, Tensor]:
    """Convert words to token IDs and pad them to equal length.

    A word like "unbelievable" may become multiple subword tokens [348, 12, 9921].
    All words are padded to the length of the longest one.

    Returns (input_ids, attention_mask).
        input_ids:      [N, L]  — token numbers, 0 = padding
        attention_mask: [N, L]  — 1 = real token, 0 = padding
        where N = number of words, L = length of longest word in subwords
    """
    sequences: list[list[int]] = []
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)[:max_len]
        sequences.append(ids or [tokenizer.unk_token_id])

    longest = max(len(s) for s in sequences)
    # input_ids:      [N, longest] filled with 0s
    input_ids = torch.zeros(len(sequences), longest, dtype=torch.long)
    # attention_mask: [N, longest] filled with 0s
    attention_mask = torch.zeros(len(sequences), longest, dtype=torch.long)
    for i, ids in enumerate(sequences):
        input_ids[i, :len(ids)] = torch.tensor(ids)       # fill real tokens
        attention_mask[i, :len(ids)] = 1                   # mark them as real
    return input_ids, attention_mask


def word_reps(
    model: PreTrainedModel,
    input_ids: Tensor,        # [N, L]
    attention_mask: Tensor,   # [N, L]
) -> list[Tensor]:
    """Run words through the transformer trunk and get one vector per word per layer.

    Words that span multiple subwords are mean-pooled into a single vector.
    Uses model.base_model — the shared trunk — so the output head(s) are
    skipped entirely and no architecture-specific attributes are needed.

    Returns a list of (num_layers + 1) tensors, each of shape [N, D].
        Index 0 = embedding layer output
        Index 1..12 = transformer layer outputs
        D = hidden dimension (e.g. 768)
    """
    outputs = model.base_model(
        input_ids=input_ids,              # [N, L]
        attention_mask=attention_mask,     # [N, L]
        output_hidden_states=True,
    )
    # outputs.hidden_states is a tuple of (num_layers+1) tensors, each [N, L, D]

    # mask: [N, L] -> [N, L, 1]  (add a dimension so it can multiply with [N, L, D])
    mask = attention_mask.unsqueeze(-1)    # [N, L, 1]

    reps: list[Tensor] = []
    for hidden in outputs.hidden_states:   # each hidden: [N, L, D]
        m = mask.to(hidden.dtype)          # [N, L, 1] — same dtype as hidden

        # Mean-pool: zero out padding positions, sum across subwords, divide by count
        # hidden * m:          [N, L, D] * [N, L, 1] -> [N, L, D]  (broadcast on D)
        #   padding positions become 0
        # .sum(dim=1):         [N, L, D] -> [N, D]   (sum over the L subword positions)
        # m.sum(dim=1):        [N, L, 1] -> [N, 1]   (count of real tokens per word)
        # .clamp(min=1e-6):    avoid division by zero
        # division:            [N, D] / [N, 1] -> [N, D]  (broadcast on D)
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)  # [N, D]
        reps.append(pooled)

    return reps  # list of (num_layers+1) tensors, each [N, D]


def contrastive_loss(reps: Tensor, group_ids: Tensor, temperature: float = 0.1) -> Tensor:
    """Contrastive loss: push translations together, push unrelated words apart.

    For each word (the "anchor"), its translations are "positives" and everything
    else is "negatives". The loss is low when translations are more similar to
    each other than to unrelated words.

    Args:
        reps:        [N, D] — one vector per word
        group_ids:   [N]    — which translation group each word belongs to
        temperature: scalar — sharpness. 0.1 means similarities in [-1,1] get
                     stretched to [-10,10] before softmax, making the loss very
                     sensitive to small differences.

    Returns a scalar loss tensor.
    """
    # ---- Step 1: normalize vectors to unit length ----
    # After this, dot product = cosine similarity (range [-1, 1])
    z = F.normalize(reps.float(), dim=-1)           # [N, D]

    # ---- Step 2: compute all pairwise similarities, scaled by temperature ----
    # z @ z.t():  [N, D] × [D, N] -> [N, N]
    # Each cell (i,j) = cosine similarity between word i and word j
    # Dividing by 0.1 stretches the range from [-1,1] to [-10,10]
    sim = (z @ z.t()) / temperature                 # [N, N]

    # ---- Step 3: mask out self-similarity (the diagonal) ----
    # A word compared to itself always scores 1.0 / 0.1 = 10.0
    # That would dominate the softmax, so we set it to -infinity
    n = z.size(0)                                   # N
    eye = torch.eye(n, dtype=torch.bool, device=z.device)  # [N, N] diagonal = True
    sim = sim.masked_fill(eye, float("-inf"))       # [N, N] diagonal = -inf

    # ---- Step 4: find which pairs are translations (positives) ----
    # group_ids[:, None]: [N] -> [N, 1]  (column vector)
    # group_ids[None, :]: [N] -> [1, N]  (row vector)
    # comparison:         [N, 1] == [1, N] -> [N, N]  (broadcast: True where same group)
    # & ~eye: exclude self-pairs
    positives = (group_ids[:, None] == group_ids[None, :]) & ~eye  # [N, N] bool

    n_positives = positives.sum()                   # scalar: total number of positive pairs
    if n_positives == 0:
        return reps.sum() * 0.0                     # no positives -> zero loss (with grad)

    # ---- Step 5: compute log-softmax for each row ----
    # For each anchor word i, this computes:
    #   log_prob[i,j] = sim[i,j] - log(sum_over_k(exp(sim[i,k])))
    # This is the log probability that word i "picks" word j out of all others
    # logsumexp(sim, dim=1): [N, N] -> [N, 1]  (one normalizer per row)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # [N, N]

    # ---- Step 6: average the log-prob of the positive pairs ----
    # log_prob[positives]: select only the positive entries -> [n_positives]
    # Negate because we want to MAXIMIZE log_prob (minimizing negative log_prob)
    return -log_prob[positives].sum() / n_positives  # scalar


def alignment_loss(
    model: PreTrainedModel,
    input_ids: Tensor,        # [N, L]
    attention_mask: Tensor,   # [N, L]
    group_ids: Tensor,        # [N]
    temperature: float = 0.1,
) -> Tensor:
    """Compute contrastive loss at every layer, then average across layers.

    Averaging (instead of summing) makes the loss scale independent of model
    depth — so alpha means the same thing for a 6-layer and 12-layer model.

    Returns a scalar loss tensor.
    """
    reps = word_reps(model, input_ids, attention_mask)  # list of (L+1) × [N, D]
    losses = [contrastive_loss(r, group_ids, temperature) for r in reps]  # list of (L+1) scalars
    return torch.stack(losses).mean()                   # scalar: average across layers


def negative_similarity(reps: Tensor, group_ids: Tensor) -> float:
    """How similar are unrelated words? (0 for different words => good, 1 for identical words => bad)

    Computes average cosine similarity between all word pairs from different groups.

    Args:
        reps:      [N, D] — word vectors from one layer
        group_ids: [N]    — group labels

    Returns a float between 0 and 1.
    """
    with torch.no_grad():
        z = F.normalize(reps.float(), dim=-1)          # [N, D]
        sim = z @ z.t()                                # [N, N] cosine similarities
        # different: [N, 1] != [1, N] -> [N, N] (True where words are from different groups)
        different = group_ids[:, None] != group_ids[None, :]  # [N, N] bool
        if not different.any():
            return 0.0
        return float(sim[different].mean().item())     # average of off-group similarities


def sample_alignment_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    groups: list[WordGroup],
    n_groups: int,
    max_len: int,
    temperature: float,
    device: str,
) -> Tensor:
    """Pick random word groups and compute their alignment loss. One-liner
    used by both PreAlign and (optionally) the main training loop.

    Returns a scalar loss tensor.
    """
    sample = random.sample(groups, min(n_groups, len(groups)))
    words, group_ids = flatten_groups(sample)
    input_ids, attention_mask = encode_words(words, tokenizer, max_len)
    return alignment_loss(
        model,
        input_ids.to(device),
        attention_mask.to(device),
        group_ids.to(device),
        temperature,
    )


def prealign(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    groups: list[WordGroup],
    dataloader: DataLoader[LMBatch],
    args: argparse.Namespace,
) -> PreTrainedModel:
    """Run 500 steps of alignment + language modeling, then return the model.

    Each step:
      1. Sample 64 word groups (192 words) and compute contrastive alignment loss
      2. Take one batch of normal text and compute next-word-prediction loss
      3. Combine: loss = alpha * align + lm
      4. Update the model

    After all steps, copy the updated embedding table into the language heads.

    Returns the model with aligned embeddings.
    """
    device: str = DEVICE
    model.train()
    # The ignores are a stub artefact: transformers decorates
    # PreTrainedModel.to(), and mypy reads the wrapper as an unbound method
    # wanting an explicit `self`. The call is ordinary nn.Module.to().
    if device.startswith("cuda"):
        model = model.to(dtype=torch.bfloat16, device=device)  # type: ignore[call-arg]
    else:
        model = model.to(device=device)  # type: ignore[call-arg]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.prealign_lr)
    n_groups: int = min(args.prealign_groups, len(groups))

    # ---- Create an iterator over the dataloader ----
    # We only need ~500 batches out of millions, so we just grab an iterator
    # and pull from it. If the dataloader runs out (unlikely), we make a new one.
    lm_iter: Iterator[LMBatch] = iter(dataloader)

    print(f"PreAlign: {args.prealign_steps} steps, "
          f"{len(groups):,} word groups, "
          f"{n_groups} per step, "
          f"alpha={args.prealign_alpha}, "
          f"tau={args.prealign_tau}", flush=True)

    last_align: float = float("nan")
    last_lm: float = float("nan")
    last_neg_sim: float = float("nan")

    for step in range(args.prealign_steps):

        # ==== ALIGNMENT BATCH ====
        sample = random.sample(groups, n_groups) # Sample 64 random word groups -> 192 words (64 groups × 3 languages)
        words, group_ids = flatten_groups(sample)
        # words:     list of 192 strings
        # group_ids: [192]

        input_ids, attention_mask = encode_words(words, tokenizer, args.prealign_max_len)
        # input_ids:      [192, L]  where L = longest word in subwords
        # attention_mask: [192, L]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        group_ids = group_ids.to(device)

        # ==== LANGUAGE MODELING BATCH ====
        minibatch: int = max(1, getattr(args, "batch_size", 32) //
                                max(1, getattr(args, "grad_acc", 1)))

        try:
            raw_batch = next(lm_iter)
        except StopIteration:
            lm_iter = iter(dataloader)
            raw_batch = next(lm_iter)

        lm_batch: LMBatch = {k: v[:minibatch].to(device) for k, v in raw_batch.items()}
        # lm_batch["input_ids"]:      [B, S]  where B = minibatch, S = sequence length
        # lm_batch["attention_mask"]: [B, S]
        # lm_batch["labels"]:         [B, S]

        # ==== FORWARD PASS ====
        with _autocast():
            reps = word_reps(model, input_ids, attention_mask)
            # reps: list of 13 tensors, each [192, D] where D = hidden dim (e.g. 768)
            #   reps[0]  = embedding layer output
            #   reps[1]  = transformer layer 1 output
            #   ...
            #   reps[12] = transformer layer 12 output

            align = torch.stack(
                [contrastive_loss(r, group_ids, args.prealign_tau) for r in reps]
            ).mean()
            # supcon_loss returns a scalar for each of the 13 layers
            # torch.stack makes them into a [13] tensor
            # .mean() averages them into one scalar

            lm = model(**lm_batch).loss

            loss = args.prealign_alpha * align + lm

        # ==== BACKWARD PASS + UPDATE ====
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # ==== LOGGING (every 100 steps + last step) ====
        if step % 100 == 0 or step == args.prealign_steps - 1:
            neg_sim = negative_similarity(reps[-1], group_ids) # float in [0, 1] — how similar unrelated words are

            last_align, last_lm, last_neg_sim = align.item(), lm.item(), neg_sim

            print(f"  prealign {step:>4}/{args.prealign_steps}  "
                  f"align {align.item():.4f}  lm {lm.item():.4f}  "
                  f"neg_sim {neg_sim:.3f}", flush=True)

            if getattr(args, "wandb", False) and wandb_available:
                wandb.log({
                    "prealign/align_loss": align.item(),
                    "prealign/lm_loss": lm.item(),
                    "prealign/neg_similarity": neg_sim,
                    "prealign/step": step,
                })
    embeddings = model.get_input_embeddings().weight
    
    with torch.no_grad():
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            print("no output layer found, nothing to refresh", flush=True)
        if lm_head.weight is embeddings:            # tied: literally the same tensor
            print("tied to embeddings, already aligned", flush=True)
        lm_head.weight.copy_(embeddings)
        print("refreshed untied lm_head from embeddings", flush=True)

    print(f"PREALIGN SUMMARY  "
          f"alpha={args.prealign_alpha}  "
          f"tau={args.prealign_tau}  "
          f"final_align={last_align:.4f}  "
          f"final_lm={last_lm:.4f}  "
          f"final_neg_sim={last_neg_sim:.3f}", flush=True)

    return model