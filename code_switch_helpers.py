# code_switch_preprocessing.py
"""
Offline Code-Switching (Cross-Lingual Word Replacement) for BabyLM 2026.

This module provides utilities to probabilistically replace English
**nouns and verbs** with their Dutch or Chinese translations using MUSE
bilingual dictionaries.  It is designed to run as an offline preprocessing
step so that the downstream MLM training loop remains unchanged.

Usage:
    from code_switch_preprocessing import (
        load_dictionary, code_switch_sentence,
        DICT_EN_NL, DICT_EN_ZH,
    )
"""

import random
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk

# Ensure required NLTK data is available (silent no-op if already present)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ------------------------------------------------------------------ #
# Configuration                                                      #
# ------------------------------------------------------------------ #
DICT_EN_NL = './data/en-nl.txt'
DICT_EN_ZH = './data/en-zh.txt'
CS_SWAP_PROB = 1.0          # per-word swap probability
DICT_MAX_ENTRIES = 40_000    # keep only top-N most frequent translations

# POS tags for content words (nouns + verbs) that we allow code-switching on.
# Determiners, prepositions, pronouns, conjunctions, etc. are never swapped.
_CONTENT_POS_TAGS = frozenset({
    'NN', 'NNS',                              # nouns
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # verbs
})


# ------------------------------------------------------------------ #
# Dictionary loading                                                   #
# ------------------------------------------------------------------ #

def load_dictionary(filepath: str, max_entries: int = DICT_MAX_ENTRIES) -> dict[str, str]:
    """
    Load a MUSE-format bilingual dictionary.

    Supports both tab-separated (``word<TAB>translation``) and
    space-separated (``word translation``) formats.  When the line is
    space-separated the first token is taken as the source word and the
    last token as the translation (handles multi-word Chinese entries).

    MUSE dictionaries are ordered by descending frequency, so keeping only
    the first *max_entries* lines filters out obscure translations.

    Returns a ``dict`` mapping **lowercased** English words to their
    target-language translations.
    """
    dictionary: dict[str, str] = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue

            # Try tab-separated first (en-nl.txt format)
            if '\t' in line:
                parts = line.split('\t', maxsplit=1)
            else:
                # Space-separated (en-zh.txt format): first token = src,
                # everything after the first space = target
                parts = line.split(' ', maxsplit=1)

            if len(parts) != 2:
                continue

            src, tgt = parts
            src_lower = src.strip().lower()
            tgt = tgt.strip()
            if not src_lower or not tgt:
                continue

            # Keep only the first (most frequent) translation per word
            if src_lower not in dictionary:
                dictionary[src_lower] = tgt
            if len(dictionary) >= max_entries:
                break
    return dictionary


# ------------------------------------------------------------------ #
# Code-switching                                                       #
# ------------------------------------------------------------------ #

def code_switch_sentence(
    sentence: str,
    dict_nl: dict[str, str],
    dict_zh: dict[str, str],
    swap_prob: float = CS_SWAP_PROB,
) -> str:
    """
    Probabilistically replace English **nouns and verbs** with Dutch or
    Chinese translations.

    Uses NLTK POS tagging to identify content words (nouns/verbs).  Only
    those tokens are candidates for replacement; function words ("the",
    "on", "and", …) are never swapped.

    For every eligible word, with probability *swap_prob* attempt to swap
    it for a translation from a **randomly chosen** dictionary (Dutch or
    Chinese — the same language is used for the whole sentence).

    If at least one swap succeeds, the appropriate control token
    (``[PREDICT_NL]`` or ``[PREDICT_ZH]``) is prepended to the sentence.
    If no swap occurs (all lookups miss), the sentence is returned unchanged.
    """
    # Choose target language for this sentence
    if random.random() < 0.5:
        chosen_dict, control_token = dict_nl, '[PREDICT_NL]'
    else:
        chosen_dict, control_token = dict_zh, '[PREDICT_ZH]'

    # Tokenize and POS-tag
    words = word_tokenize(sentence)
    tagged = pos_tag(words)

    result: list[str] = []

    for word, tag in tagged:
        if tag in _CONTENT_POS_TAGS and random.random() < swap_prob:
            replacement = chosen_dict.get(word.lower())
            if replacement is not None:
                if word[0].isupper() and not word.isupper():
                    replacement = replacement.capitalize()
                elif word.isupper():
                    replacement = replacement.upper()
                result.append(control_token)
                result.append(replacement)
                continue

        result.append(word)

    out = _detokenize(result)
    return out


def _detokenize(tokens: list[str]) -> str:
    """
    Rejoin NLTK word_tokenize tokens into a readable string.

    Handles common punctuation attachment (periods, commas, quotes, etc.)
    without requiring the heavy TreebankWordDetokenizer.
    """
    if not tokens:
        return ''
    # Punctuation that attaches to the *preceding* word (no space before)
    _ATTACH_LEFT = frozenset({'.', ',', '!', '?', ';', ':', "'", "'s",
                              "'t", "'re", "'ve", "'ll", "'d", "'m",
                              "n't", ')', ']', '}', '%', '...', '--'})
    # Punctuation that attaches to the *following* word (no space after)
    _ATTACH_RIGHT = frozenset({'(', '[', '{', '$', '#'})

    parts: list[str] = [tokens[0]]
    for tok in tokens[1:]:
        if tok in _ATTACH_LEFT or (len(tok) == 1 and not tok.isalnum()):
            parts.append(tok)
        elif parts and parts[-1] in _ATTACH_RIGHT:
            parts.append(tok)
        else:
            parts.append(' ')
            parts.append(tok)
    return ''.join(parts)


