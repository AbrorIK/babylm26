"""
Multi-head multilingual causal LM (BabyLM 2026 multilingual track).

One shared GPT-2 trunk with one output head per language (eng/nld/zho). Each
position is routed to the head of its own language for the standard CLM loss.
The shared trunk is forced to encode a language-neutral "what comes next"
representation; each head renders it as tokens in its own language.

lang_ids convention: 0 = pad/unknown, 1 = eng, 2 = nld, 3 = zho.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

LANG2ID = {"eng": 1, "nld": 2, "zho": 3}
ID2LANG = {v: k for k, v in LANG2ID.items()}
NUM_LANGS = 3


class MultiHeadGPT2Config(GPT2Config):
    model_type = "multihead-gpt2"

    def __init__(self, num_lang_heads=NUM_LANGS, **kwargs):
        super().__init__(**kwargs)
        self.num_lang_heads = num_lang_heads


class MultiHeadGPT2LMHeadModel(GPT2PreTrainedModel):
    config_class = MultiHeadGPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.heads = nn.ModuleList(
            nn.Linear(config.n_embd, config.vocab_size, bias=False)
            for _ in range(config.num_lang_heads)
        )
        self.post_init()
        # Start every head at the input embedding table. A single-head GPT-2
        # ties lm_head to the embeddings; we lose that with three heads, so we
        # at least initialize them there — it keeps the heads mutually aligned
        # and gives each one a sensible starting point.
        with torch.no_grad():
            for head in self.heads:
                head.weight.copy_(self.transformer.wte.weight)

    def route_logits(self, hidden, lang_ids):
        B, T, d = hidden.shape
        flat_h = hidden.reshape(-1, d)                       # [B*T, d]
        head_idx = (lang_ids.reshape(-1) - 1).clamp(min=0)   # lang 0 -> head 0
        logits = flat_h.new_zeros(B * T, self.config.vocab_size)
        for l, head in enumerate(self.heads):
            mask = head_idx == l
            if mask.any():
                # under autocast the head runs in bf16; cast back to the
                # destination dtype so the scatter-assign dtypes match
                logits[mask] = head(flat_h[mask]).to(logits.dtype)
        return logits.view(B, T, -1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, lang_ids=None, **kwargs):
        hidden = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                                   # [B,T,d]

        if lang_ids is None:
            lang_ids = torch.ones_like(input_ids)             # default: eng head
        logits = self.route_logits(hidden, lang_ids)

        loss = None
        if labels is not None:
            # position t predicts token t+1
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return CausalLMOutput(loss=loss, logits=logits, hidden_states=(hidden,))


def head_divergence(model):
    """Mean pairwise cosine similarity between the three heads' weights.

    All heads start as identical copies of wte, so this begins at 1.0. If it
    stays at 1.0 the heads never specialise and routing is a no-op; falling
    values mean each head is adapting to its own language.
    """
    sims = []
    heads = list(model.heads)
    with torch.no_grad():
        for i in range(len(heads)):
            for j in range(i + 1, len(heads)):
                a = heads[i].weight.flatten().float()
                b = heads[j].weight.flatten().float()
                sims.append(F.cosine_similarity(a, b, dim=0).item())
    return sum(sims) / len(sims)
