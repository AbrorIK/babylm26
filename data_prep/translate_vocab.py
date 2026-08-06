"""Translate the English word list into Dutch and Chinese with Qwen3-8B.

Produces [eng, nld, zho] triplets for PreAlign. One prompt asks for both
languages at once, and prompts are generated in batches — one word at a time
would take most of a day for 12k words.

Output: data/prealign_triplets.tsv   ``eng \t nld \t zho``
Progress is flushed after every batch, and an existing output file is resumed
rather than restarted.

    python data_prep/translate_vocab.py --model ./Qwen3-8B
"""

import argparse
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VOCAB = "data/prealign_vocab.txt"
OUTPUT = "data/prealign_triplets.tsv"

PROMPT = (
    "Translate the English word into Dutch and Simplified Chinese.\n"
    "Answer with exactly: dutch | chinese\n"
    "No explanation, no pinyin, no extra words.\n\n"
    "Word: {word}"
)

CJK = re.compile(r"[一-鿿]")


def load_words(path, done):
    with open(path, encoding="utf-8") as f:
        return [w.strip() for w in f if w.strip() and w.strip() not in done]


def load_done(path):
    """Words already translated, so a re-run resumes instead of redoing work."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.split("\t")[0] for line in f if line.strip()}


def build_prompts(words, tokenizer):
    prompts = []
    for word in words:
        messages = [{"role": "user", "content": PROMPT.format(word=word)}]
        prompts.append(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,      # Qwen3 emits <think> blocks otherwise
        ))
    return prompts


def parse(reply):
    """'weer | 天气' -> ('weer', '天气'), or None if the reply is unusable."""
    reply = reply.strip().split("\n")[0]
    if "|" not in reply:
        return None
    dutch, _, chinese = reply.partition("|")
    dutch, chinese = dutch.strip(), chinese.strip()

    # Dutch must be Latin script; Chinese must actually contain CJK.
    if not dutch or CJK.search(dutch):
        return None
    if not chinese or not CJK.search(chinese):
        return None
    # A single word, not a sentence or an explanation.
    if len(dutch.split()) > 3 or len(chinese) > 8:
        return None
    return dutch, chinese


@torch.no_grad()
def translate_batch(model, tokenizer, prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=24,          # translating single words
        do_sample=False,            # deterministic
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./Qwen3-8B")
    parser.add_argument("--vocab", default=VOCAB)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    done = load_done(args.output)
    words = load_words(args.vocab, done)
    print(f"{len(done):,} already done, {len(words):,} to translate")
    if not words:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Decoder-only models must be left-padded for batched generation, or the
    # generated text continues from padding instead of from the prompt.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="cuda")
    model.eval()

    kept = failed = 0
    with open(args.output, "a", encoding="utf-8") as out:
        for i in range(0, len(words), args.batch_size):
            batch = words[i:i + args.batch_size]
            replies = translate_batch(model, tokenizer, build_prompts(batch, tokenizer))

            for word, reply in zip(batch, replies):
                parsed = parse(reply)
                if parsed is None:
                    failed += 1
                    continue
                dutch, chinese = parsed
                out.write(f"{word}\t{dutch}\t{chinese}\n")
                kept += 1
            out.flush()

            print(f"{i + len(batch):>6,}/{len(words):,}  kept {kept:,}  failed {failed:,}",
                  flush=True)

    print(f"\nwrote {kept:,} triplets to {args.output} ({failed:,} rejected)")


if __name__ == "__main__":
    main()
