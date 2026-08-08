"""
PreAlign: cross-lingual word alignment before language pretraining.

A randomly initialised model is trained briefly so that translations land close
together in representation space, then normal CLM training proceeds. The point
is to shape the shared trunk before the LM objective carves out separate
per-language regions.

Only the trunk is aligned — the embedding layer and the transformer layers.
The language heads are left alone: they are meant to diverge, and at PreAlign
time all three are still identical copies of the embedding table anyway, so
aligning them would duplicate the embedding-layer term.
"""

import torch
import torch.nn.functional as F

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

# Overridable so the loop can be dry-run on CPU; training always uses the GPU.
DEVICE = "cuda:0"


def _autocast():
    if DEVICE.startswith("cuda"):
        return torch.autocast(dtype=torch.bfloat16, device_type="cuda:0")
    import contextlib
    return contextlib.nullcontext()


def load_word_groups(path):
    """Read ``eng \t nld \t zho`` triplets into groups of translations."""
    groups = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split("\t")]
            if len(parts) == 3 and all(parts):
                groups.append(parts)
    return groups


def flatten_groups(groups):
    """[["cat","kat","猫"], ...] -> (words, group_ids) with one entry per word."""
    words, group_ids = [], []
    for gid, group in enumerate(groups):
        for word in group:
            words.append(word)
            group_ids.append(gid)
    return words, torch.tensor(group_ids)


def encode_words(words, tokenizer, max_len=8):
    """Tokenize words into one padded batch.

    Words may be multiple subwords — they get mean-pooled later — so there is
    no single-token requirement and no dependence on tokenizer segmentation.
    """
    sequences = []
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)[:max_len]
        sequences.append(ids or [tokenizer.unk_token_id])

    longest = max(len(s) for s in sequences)
    input_ids = torch.zeros(len(sequences), longest, dtype=torch.long)
    attention_mask = torch.zeros(len(sequences), longest, dtype=torch.long)
    for i, ids in enumerate(sequences):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attention_mask[i, :len(ids)] = 1
    return input_ids, attention_mask


def word_reps(model, input_ids, attention_mask):
    """One forward pass -> a mean-pooled vector per word, per layer.

    Returns a list of L+1 tensors of shape [num_words, d]: the embedding layer
    followed by each transformer layer. Going through model.transformer skips
    the language heads entirely, so no lang_ids are needed.
    """
    outputs = model.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    mask = attention_mask.unsqueeze(-1)
    reps = []
    for hidden in outputs.hidden_states:
        m = mask.to(hidden.dtype)
        reps.append((hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6))
    return reps


def supcon_loss(reps, group_ids, temperature=0.1):
    """Supervised contrastive loss: words in the same group are positives.

    Every word is an anchor; its translations are the positives and all other
    words in the batch are negatives. Self-similarity is masked out — it is
    always 1.0 and, at temperature 0.1, exp(1/0.1) would dominate the
    denominator and flatten the loss.
    """
    # float32 for the similarity maths: normalize and logsumexp lose too much
    # in bf16, and these tensors are small ([words, d]) so it costs nothing.
    z = F.normalize(reps.float(), dim=-1)
    sim = (z @ z.t()) / temperature

    n = z.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))

    positives = (group_ids[:, None] == group_ids[None, :]) & ~eye
    n_positives = positives.sum()
    if n_positives == 0:
        return reps.sum() * 0.0

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # Index rather than multiply by the mask: the masked diagonal is -inf, and
    # 0 * -inf is NaN.
    return -log_prob[positives].sum() / n_positives


def alignment_loss(model, input_ids, attention_mask, group_ids, temperature=0.1):
    """Contrastive loss averaged over the embedding and transformer layers.

    Averaging rather than summing keeps the scale independent of depth, so the
    alignment weight means the same thing for a 6- and a 12-layer model.
    """
    reps = word_reps(model, input_ids, attention_mask)
    losses = [supcon_loss(r, group_ids, temperature) for r in reps]
    return torch.stack(losses).mean()


def negative_similarity(reps, group_ids):
    """Mean cosine similarity between words in DIFFERENT groups.

    The collapse detector. Contrastive loss has a trivial solution where every
    word maps to the same vector; if this climbs toward 1.0 the model is taking
    it and the alignment weight is too high.
    """
    with torch.no_grad():
        z = F.normalize(reps.float(), dim=-1)
        sim = z @ z.t()
        different = group_ids[:, None] != group_ids[None, :]
        if not different.any():
            return 0.0
        return sim[different].mean().item()


def sample_alignment_loss(model, tokenizer, groups, n_groups, max_len,
                          temperature, device):
    """Draw a batch of word groups and return their alignment loss.

    Used both by the PreAlign phase and, when enabled, by the main training
    loop — the paper keeps sampling word pairs for the alignment loss during
    language pretraining, otherwise the alignment established up front is
    forgotten over the much longer pretraining run.
    """
    import random

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


def _infinite(dataloader):
    """Yield LM batches forever; PreAlign needs far fewer than one epoch."""
    while True:
        for batch in dataloader:
            yield batch


def prealign(model, tokenizer, groups, dataloader, args):
    """Align translations in the trunk before language pretraining starts.

    Each step draws a batch of word groups for the contrastive loss and one
    ordinary LM batch. The LM term is what stops the contrastive loss taking
    its trivial solution of mapping every word to the same vector.
    """
    import random

    device = DEVICE
    model.train()
    if device.startswith("cuda"):
        model = model.to(dtype=torch.bfloat16, device=device)
    else:
        model = model.to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.prealign_lr)
    lm_batches = _infinite(dataloader)
    n_groups = min(args.prealign_groups, len(groups))

    print(f"PreAlign: {args.prealign_steps} steps, {len(groups):,} word groups, "
          f"{n_groups} per step, alpha={args.prealign_alpha}, "
          f"tau={args.prealign_tau}", flush=True)

    peak_neg_sim = 0.0
    last_align = last_lm = last_neg_sim = float("nan")

    for step in range(args.prealign_steps):
        sample = random.sample(groups, n_groups)
        words, group_ids = flatten_groups(sample)
        input_ids, attention_mask = encode_words(words, tokenizer, args.prealign_max_len)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        group_ids = group_ids.to(device)

        minibatch = max(1, getattr(args, "batch_size", 32) // max(1, getattr(args, "grad_acc", 1)))
        lm_batch = {k: v[:minibatch].to(device) for k, v in next(lm_batches).items()}

        with _autocast():
            # one forward pass serves both the loss and the collapse diagnostic
            reps = word_reps(model, input_ids, attention_mask)
            align = torch.stack([supcon_loss(r, group_ids, args.prealign_tau) for r in reps]).mean()
            lm = model(**lm_batch).loss
            loss = args.prealign_alpha * align + lm

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % args.prealign_log_every == 0 or step == args.prealign_steps - 1:
            neg_sim = negative_similarity(reps[-1], group_ids)
            peak_neg_sim = max(peak_neg_sim, neg_sim)
            last_align, last_lm, last_neg_sim = align.item(), lm.item(), neg_sim
            print(f"  prealign {step:>4}/{args.prealign_steps}  "
                  f"align {align.item():.4f}  lm {lm.item():.4f}  "
                  f"neg_sim {neg_sim:.3f}", flush=True)
            if neg_sim > 0.9:
                print("    WARNING: representations are collapsing "
                      "(all words alike) — lower --prealign_alpha", flush=True)

            if getattr(args, "wandb", False) and wandb_available:
                wandb.log({
                    "prealign/align_loss": align.item(),
                    "prealign/lm_loss": lm.item(),
                    "prealign/neg_similarity": neg_sim,
                    "prealign/step": step,
                })

    # The heads were copied from wte at init, and PreAlign has since moved wte.
    # Refresh them so main training starts from the aligned embeddings.
    with torch.no_grad():
        for head in model.heads:
            head.weight.copy_(model.transformer.wte.weight)
    print("PreAlign done; heads refreshed from aligned embeddings", flush=True)
    print(f"PREALIGN SUMMARY  alpha={args.prealign_alpha}  tau={args.prealign_tau}  "
          f"final_align={last_align:.4f}  final_lm={last_lm:.4f}  "
          f"final_neg_sim={last_neg_sim:.3f}  peak_neg_sim={peak_neg_sim:.3f}",
          flush=True)
    if peak_neg_sim > 0.7:
        print("  -> peak_neg_sim above 0.7: representations collapsed at some "
              "point; try a lower --prealign_alpha", flush=True)

    return model
