# align_loss.py
"""
ALIGN-MLM Steps 3 & 5: Embedding-alignment loss and intrinsic evaluation.

The alignment loss operates directly on the embedding table (no forward pass
needed).  For multi-token words, subword embeddings are mean-pooled into a
single vector before computing cosine similarity.

Usage:
    from align_loss import align_loss_batch, compute_alignment_accuracy
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def word_embedding(emb_weight: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool the embedding rows for a set of subword token IDs.

    Args:
        emb_weight: The full embedding table, shape (V, d).
        token_ids:  1-D tensor of subword IDs for one word.

    Returns:
        A single (d,) vector — the mean of the selected rows.
    """
    return emb_weight[token_ids].mean(dim=0)


def align_loss_batch(
    model,
    dict_batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    Compute the embedding-alignment loss for a batch of dictionary pairs.

    For each (src_ids, tgt_ids) pair, mean-pools subword embeddings on each
    side and computes cosine similarity.  The loss is the *negative* mean
    cosine similarity (so minimizing the loss maximizes alignment).

    This is efficient because it only touches the embedding table — no
    transformer forward pass is needed.

    Args:
        model:      A HuggingFace model with ``get_input_embeddings()``.
        dict_batch: List of (src_token_ids, tgt_token_ids) tuples.
                    Each element is a 1-D LongTensor of subword IDs.

    Returns:
        Scalar loss tensor (negative mean cosine similarity).
    """
    emb = model.get_input_embeddings().weight  # (V, d)

    # Pad sequences for batched gathering
    # Find max lengths
    src_lens = [s.size(0) for s, _ in dict_batch]
    tgt_lens = [t.size(0) for _, t in dict_batch]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    batch_size = len(dict_batch)

    # Build padded index tensors (pad with 0, any valid token id)
    src_padded = torch.zeros(batch_size, max_src, dtype=torch.long, device=emb.device)
    tgt_padded = torch.zeros(batch_size, max_tgt, dtype=torch.long, device=emb.device)
    src_mask = torch.zeros(batch_size, max_src, dtype=torch.float, device=emb.device)
    tgt_mask = torch.zeros(batch_size, max_tgt, dtype=torch.float, device=emb.device)

    for i, (s, t) in enumerate(dict_batch):
        sl, tl = s.size(0), t.size(0)
        src_padded[i, :sl] = s
        tgt_padded[i, :tl] = t
        src_mask[i, :sl] = 1.0
        tgt_mask[i, :tl] = 1.0

    # Batched embedding lookup: (batch_size, max_len, d)
    src_embs = emb[src_padded]
    tgt_embs = emb[tgt_padded]

    # Masked mean pooling → (batch_size, d)
    src_vecs = (src_embs * src_mask.unsqueeze(-1)).sum(dim=1) / src_mask.sum(dim=1, keepdim=True)
    tgt_vecs = (tgt_embs * tgt_mask.unsqueeze(-1)).sum(dim=1) / tgt_mask.sum(dim=1, keepdim=True)

    # Negative mean cosine similarity
    return -F.cosine_similarity(src_vecs, tgt_vecs, dim=-1).mean()


@torch.no_grad()
def compute_alignment_accuracy(
    model,
    dict_pairs: list,
    k: int = 5000,
) -> dict:
    """
    Intrinsic alignment evaluation (Step 5).

    For a sample of dictionary pairs, check whether the source word's
    nearest neighbour in the embedding space (by cosine) is the correct
    target translation.

    This is a cheap sanity signal: alignment accuracy should climb during
    training.  Note: for multi-token words we use the mean-pooled embedding.

    Args:
        model:      A HuggingFace model with ``get_input_embeddings()``.
        dict_pairs: Full list of (src_ids, tgt_ids, lang_pair) entries.
        k:          Number of pairs to evaluate (random sample if > k).

    Returns:
        Dict with keys: "align_acc" (fraction correct, 0-100),
                        "avg_cosine" (average cosine similarity).
    """
    import random

    emb = model.get_input_embeddings().weight  # (V, d)

    # Sample a subset
    if k < len(dict_pairs):
        sampled = random.sample(dict_pairs, k)
    else:
        sampled = dict_pairs

    # Build unique target vocabulary for nearest-neighbour search
    # Each target is a mean-pooled embedding; we compare against all targets
    # in the sampled set.
    src_vecs = []
    tgt_vecs = []
    for src_ids, tgt_ids, _ in sampled:
        src_ids_dev = src_ids.to(emb.device)
        tgt_ids_dev = tgt_ids.to(emb.device)
        src_vecs.append(word_embedding(emb, src_ids_dev))
        tgt_vecs.append(word_embedding(emb, tgt_ids_dev))

    src_matrix = torch.stack(src_vecs)  # (N, d)
    tgt_matrix = torch.stack(tgt_vecs)  # (N, d)

    # Normalize for cosine similarity
    src_norm = F.normalize(src_matrix, dim=-1)
    tgt_norm = F.normalize(tgt_matrix, dim=-1)

    # Cosine similarity matrix: (N, N)
    sim_matrix = src_norm @ tgt_norm.T

    # For each source, check if nearest target is the diagonal (correct pair)
    nearest = sim_matrix.argmax(dim=-1)  # (N,)
    correct = (nearest == torch.arange(len(sampled), device=emb.device)).sum().item()

    # Average cosine of correct pairs (diagonal)
    avg_cosine = sim_matrix.diag().mean().item()

    return {
        "align_acc": 100.0 * correct / len(sampled),
        "avg_cosine": avg_cosine,
    }
