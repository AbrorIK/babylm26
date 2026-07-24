"""Export per-language models from a multi-head checkpoint.

For each language (eng/nld/zho), builds a plain GPT2LMHeadModel with:
  - the shared trunk weights
  - that language's head baked in as lm_head
  - tie_word_embeddings=False (so loading lm_head doesn't overwrite wte)

The BabyLM eval pipeline can then load each export as a standard GPT-2.

Usage:
    python export_multihead.py --checkpoint output/gpt2-multihead/checkpoint-XXXX
"""

import argparse
import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, DebertaV2Tokenizer

from multihead_model import MultiHeadGPT2LMHeadModel, LANG2ID

ID2LANG = {v: k for k, v in LANG2ID.items()}


def export(checkpoint_path, output_dir=None, tokenizer_path=None):
    if output_dir is None:
        output_dir = checkpoint_path + "-export"

    print(f"Loading multi-head checkpoint from: {checkpoint_path}")
    mh_model = MultiHeadGPT2LMHeadModel.from_pretrained(checkpoint_path)
    mh_model.eval()
    mh_config = mh_model.config

    if tokenizer_path:
        tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = DebertaV2Tokenizer.from_pretrained(checkpoint_path)

    trunk_state = mh_model.transformer.state_dict()

    for lang_id in (1, 2, 3):
        lang = ID2LANG[lang_id]
        head = mh_model.heads[lang_id - 1]

        config = GPT2Config(
            vocab_size=mh_config.vocab_size,
            n_positions=mh_config.n_positions,
            n_embd=mh_config.n_embd,
            n_layer=mh_config.n_layer,
            n_head=mh_config.n_head,
            n_inner=mh_config.n_inner,
            resid_pdrop=mh_config.resid_pdrop,
            embd_pdrop=mh_config.embd_pdrop,
            attn_pdrop=mh_config.attn_pdrop,
            pad_token_id=mh_config.pad_token_id,
            bos_token_id=mh_config.bos_token_id,
            eos_token_id=mh_config.eos_token_id,
            tie_word_embeddings=False,
        )

        gpt2 = GPT2LMHeadModel(config)
        gpt2.transformer.load_state_dict(trunk_state)

        with torch.no_grad():
            gpt2.lm_head.weight.copy_(head.weight)

        lang_dir = os.path.join(output_dir, lang)
        gpt2.save_pretrained(lang_dir)
        tokenizer.save_pretrained(lang_dir)
        print(f"  {lang} -> {lang_dir}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir, args.tokenizer)
