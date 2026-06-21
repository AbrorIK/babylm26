"""
ALIGN-MLM Step 1: Preprocess MUSE bilingual dictionaries into token-ID pairs.

For every (src_word, tgt_word) in a MUSE dictionary, tokenize both sides with
the shared BPE tokenizer and store the two token-ID lists.  Keeps *all* pairs
including multi-token ones.  Each pair is tagged with its language pair so they
can be balanced during training.
"""

import random
from typing import List, Tuple

import torch


# Type alias: each dictionary entry is (src_token_ids, tgt_token_ids, lang_pair)
DictEntry = Tuple[torch.Tensor, torch.Tensor, str]


def load_align_dictionary(
    filepath: str,
    tokenizer,
    lang_pair: str,
) -> List[DictEntry]:
    """
    Load a MUSE-format bilingual dictionary and tokenize both sides.

    Supports both tab-separated (``word<TAB>translation``) and
    space-separated (``word translation``) formats.

    Args:
        filepath:   Path to the MUSE dictionary file.
        tokenizer:  A HuggingFace tokenizer (e.g. DebertaV2Tokenizer) with
                     the shared BPE vocabulary.
        lang_pair:  Tag string, e.g. "en-nl" or "en-zh".

    Returns:
        List of (src_token_ids, tgt_token_ids, lang_pair) tuples.
        Token IDs are 1-D int64 tensors (no special tokens).
    """
    entries: List[DictEntry] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            # Parse: tab-separated first, then space-separated
            if "\t" in line:
                parts = line.split("\t", maxsplit=1)
            else:
                parts = line.split(" ", maxsplit=1)

            if len(parts) != 2:
                continue

            src_word, tgt_word = parts[0].strip(), parts[1].strip()
            if not src_word or not tgt_word:
                continue

            # Tokenize without special tokens ([CLS], [SEP], etc.)
            src_ids = tokenizer.encode(src_word, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt_word, add_special_tokens=False)

            # Filter out empty tokenizations
            if len(src_ids) == 0 or len(tgt_ids) == 0:
                continue

            entries.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
                lang_pair,
            ))

    return entries


def load_all_dictionaries(
    dict_paths: dict,
    tokenizer,
) -> List[DictEntry]:
    """
    Load multiple MUSE dictionaries and combine them.

    Args:
        dict_paths: Mapping of lang_pair -> filepath,
                    e.g. {"en-nl": "data/en-nl.txt", "en-zh": "data/en-zh.txt"}
        tokenizer:  Shared BPE tokenizer.

    Returns:
        Combined list of all dictionary entries.
    """
    all_entries: List[DictEntry] = []
    for lang_pair, filepath in dict_paths.items():
        entries = load_align_dictionary(filepath, tokenizer, lang_pair)
        print(f"  Loaded {len(entries):,} {lang_pair} dictionary pairs from {filepath}")
        all_entries.extend(entries)
    print(f"  Total alignment dictionary pairs: {len(all_entries):,}")
    return all_entries


def sample_dict_batch(
    dict_pairs: List[DictEntry],
    k: int = 256,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Randomly sample k (src_ids, tgt_ids) pairs for the alignment loss.

    Samples uniformly from the combined dictionary (EN-NL + EN-ZH mixed).
    Returns only the token-ID tensors (drops the lang_pair tag).

    Args:
        dict_pairs: Full list of dictionary entries.
        k:          Number of pairs to sample per training step.

    Returns:
        List of (src_ids, tgt_ids) tuples, each a 1-D LongTensor.
    """
    if k >= len(dict_pairs):
        sampled = dict_pairs
    else:
        sampled = random.sample(dict_pairs, k)
    return [(src, tgt) for src, tgt, _ in sampled]
