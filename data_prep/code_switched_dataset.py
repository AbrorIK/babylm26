import os
import random

from code_switch_helpers import (
    load_dictionary, code_switch_sentence,
    DICT_EN_NL, DICT_EN_ZH,
)
from dataset_common import (
    BYTE_PREMIUM_ENGLISH,
    BYTE_PREMIUM_DUTCH,
    BYTE_PREMIUM_CHINESE,
    adjusted_budget,
    write_train_val_split,
)

# ================================================================== #
# Configuration                                                        #
# ================================================================== #

# ---- Token budget & ratios ----
TOKEN_TARGET = 10_000_000

RATIO_ENG = 1 / 3
RATIO_NLD = 1 / 3
RATIO_ZHO = 1 / 3

# Fraction of each language's budget kept pure (phase 1); the rest (phase 2) is
# eligible for code-switching.
PHASE_1_RATIO = 0.5

BUDGET_ENG = adjusted_budget(TOKEN_TARGET, RATIO_ENG, BYTE_PREMIUM_ENGLISH)
BUDGET_NLD = adjusted_budget(TOKEN_TARGET, RATIO_NLD, BYTE_PREMIUM_DUTCH)
BUDGET_ZHO = adjusted_budget(TOKEN_TARGET, RATIO_ZHO, BYTE_PREMIUM_CHINESE)

LANG_BUDGETS = {
    "eng": BUDGET_ENG,
    "nld": BUDGET_NLD,
    "zho": BUDGET_ZHO,
}

# ---- Source files (relative to project root) ----
INPUT_FILES = {
    "eng": [
        "data/en-nl/OpenSubtitles.en-nl.en",
        "data/en-zh/OpenSubtitles.en-zh.en",
    ],
    "nld": ["data/en-nl/OpenSubtitles.en-nl.nl"],
    "zho": ["data/en-zh/OpenSubtitles.en-zh.zh"],
}

# ---- Output paths ----
OUTPUT_TRAIN      = "./data/bb26_cs_train.txt"
OUTPUT_VALIDATION = "./data/bb26_cs_validation.txt"

# ---- Misc ----
VALIDATION_RATIO = 0.1
SEED             = 42


def load_code_switch_dictionaries():
    """Load EN->NL and EN->ZH dictionaries if present.

    Returns ``(dict_nl, dict_zh, do_code_switch)``; falls back to pure text when
    the MUSE dictionaries are missing.
    """
    if os.path.isfile(DICT_EN_NL) and os.path.isfile(DICT_EN_ZH):
        print(f"Loading EN→NL dictionary from {DICT_EN_NL} …")
        dict_nl = load_dictionary(DICT_EN_NL)
        print(f"  → {len(dict_nl):,} entries")

        print(f"Loading EN→ZH dictionary from {DICT_EN_ZH} …")
        dict_zh = load_dictionary(DICT_EN_ZH)
        print(f"  → {len(dict_zh):,} entries")
        return dict_nl, dict_zh, True

    print("WARNING: MUSE dictionaries not found — skipping code-switching.")
    print(f"  Expected: {DICT_EN_NL} and {DICT_EN_ZH}")
    return {}, {}, False


def collect_phase_lines(filepaths, limit_total):
    """Read ``filepaths`` into phase-1 (pure) and phase-2 lines by word budget."""
    limit_phase1 = int(limit_total * PHASE_1_RATIO)

    current_words = 0
    lines_phase1: list[str] = []
    lines_phase2: list[str] = []

    for filepath in filepaths:
        if current_words >= limit_total:
            break
        print(f"  Reading {filepath} …")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                if not words:
                    continue

                num_words = len(words)
                if current_words < limit_phase1:
                    lines_phase1.append(line.strip())
                    current_words += num_words
                elif current_words < limit_total:
                    lines_phase2.append(line.strip())
                    current_words += num_words
                else:
                    break

    return lines_phase1, lines_phase2


def code_switch_lines(lines, dict_nl, dict_zh):
    """Code-switch English ``lines`` in place; return the count actually changed."""
    cs_count = 0
    for i in range(len(lines)):
        switched = code_switch_sentence(lines[i], dict_nl, dict_zh)
        if switched != lines[i]:
            cs_count += 1
        lines[i] = switched
    return cs_count


def create_code_switched_dataset():
    dict_nl, dict_zh, do_code_switch = load_code_switch_dictionaries()

    phase1_lines: list[str] = []  # pure
    phase2_lines: list[str] = []  # code-switched (English only)

    for lang, filepaths in INPUT_FILES.items():
        print(f"Processing {lang}...")

        lang_lines_phase1, lang_lines_phase2 = collect_phase_lines(
            filepaths, LANG_BUDGETS[lang]
        )

        # Code-switch phase 2 (English only).
        if lang == "eng" and do_code_switch:
            cs_count = code_switch_lines(lang_lines_phase2, dict_nl, dict_zh)
            print(f"  → Code-switched {cs_count:,} lines in Phase 2")

        phase1_lines.extend(lang_lines_phase1)
        phase2_lines.extend(lang_lines_phase2)

    # Shuffle phases independently, then concatenate (pure before code-switched).
    print("\nShuffling Phase 1 and Phase 2 internally...")
    random.seed(SEED)
    random.shuffle(phase1_lines)
    random.shuffle(phase2_lines)

    all_sampled_lines = phase1_lines + phase2_lines

    # Order is meaningful, so split without shuffling.
    print(f"Writing datasets to {OUTPUT_TRAIN} and {OUTPUT_VALIDATION}...")
    write_train_val_split(
        all_sampled_lines, OUTPUT_TRAIN, OUTPUT_VALIDATION,
        val_ratio=VALIDATION_RATIO, shuffle=False,
    )


if __name__ == "__main__":
    create_code_switched_dataset()
