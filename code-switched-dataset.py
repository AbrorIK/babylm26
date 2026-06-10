import random
import os
from code_switch_helpers import (
    load_dictionary, code_switch_sentence,
    DICT_EN_NL, DICT_EN_ZH,
)

# ================================================================== #
# Configuration                                                        #
# ================================================================== #

# ---- Byte premium factors (information capacity per token) ----
BYTE_PREMIUM_ENGLISH = 1.000000
BYTE_PREMIUM_DUTCH   = 1.051606
BYTE_PREMIUM_CHINESE = 0.935966

# ---- Token budget & Ratios ----
TOKEN_TARGET = 10_000_000

RATIO_ENG = 1/3
RATIO_NLD = 1/3
RATIO_ZHO = 1/3

# Determine how much of the budget is pure vs. code-switched
PHASE_1_RATIO = 0.5  

# ---- Per-language adjusted budgets ----
BUDGET_ENG = round((TOKEN_TARGET * RATIO_ENG) / BYTE_PREMIUM_ENGLISH)
BUDGET_NLD = round((TOKEN_TARGET * RATIO_NLD) / BYTE_PREMIUM_DUTCH)
BUDGET_ZHO = round((TOKEN_TARGET * RATIO_ZHO) / BYTE_PREMIUM_CHINESE)

LANG_BUDGETS = {
    "eng": BUDGET_ENG,
    "nld": BUDGET_NLD,
    "zho": BUDGET_ZHO,
}

# ---- Source files ----
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


# ================================================================== #
# Main                                                                 #
# ================================================================== #

def create_code_switched_dataset():
    # ------------------------------------------------------------------ #
    # Load bilingual dictionaries for code-switching (English only)        #
    # ------------------------------------------------------------------ #
    if os.path.isfile(DICT_EN_NL) and os.path.isfile(DICT_EN_ZH):
        print(f"Loading EN→NL dictionary from {DICT_EN_NL} …")
        dict_nl = load_dictionary(DICT_EN_NL)
        print(f"  → {len(dict_nl):,} entries")

        print(f"Loading EN→ZH dictionary from {DICT_EN_ZH} …")
        dict_zh = load_dictionary(DICT_EN_ZH)
        print(f"  → {len(dict_zh):,} entries")
        do_code_switch = True
    else:
        print("WARNING: MUSE dictionaries not found — skipping code-switching.")
        print(f"  Expected: {DICT_EN_NL} and {DICT_EN_ZH}")
        do_code_switch = False
        dict_nl, dict_zh = {}, {}

    phase1_lines = [] # First 20M tokens (Pure)
    phase2_lines = [] # Last 10M tokens (Code-Switched)

    for lang, filepaths in INPUT_FILES.items():
        print(f"Processing {lang}...")
        
        # Calculate 1/2 for Phase 1, 1/2 for Phase 2
        limit_total = LANG_BUDGETS[lang]
        limit_phase1 = int(limit_total * (1/2))
        limit_phase2 = limit_total - limit_phase1
        
        current_words = 0
        lang_lines_phase1 = []
        lang_lines_phase2 = []
        
        for filepath in filepaths:
            if current_words >= limit_total:
                break
            print(f"  Reading {filepath} …")
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    words = line.split()
                    if not words:
                        continue
                    
                    num_words = len(words)
                    # Fill Phase 1 first
                    if current_words < limit_phase1:
                        lang_lines_phase1.append(line.strip())
                        current_words += num_words
                    # Then fill Phase 2
                    elif current_words < limit_total:
                        lang_lines_phase2.append(line.strip())
                        current_words += num_words
                    else:
                        break
        
        # ----- Code-switch Phase 2 (English only) ----- #
        if lang == 'eng' and do_code_switch:
            cs_count = 0
            for i in range(len(lang_lines_phase2)):
                switched = code_switch_sentence(lang_lines_phase2[i], dict_nl, dict_zh)
                if switched != lang_lines_phase2[i]:
                    cs_count += 1
                lang_lines_phase2[i] = switched
            print(f"  → Code-switched {cs_count:,} lines in Phase 2")

        phase1_lines.extend(lang_lines_phase1)
        phase2_lines.extend(lang_lines_phase2)

    # Shuffle Phase 1 and Phase 2 INDEPENDENTLY
    print("\nShuffling Phase 1 and Phase 2 internally...")
    random.seed(42)
    random.shuffle(phase1_lines)
    random.shuffle(phase2_lines)

    # Combine them sequentially: Phase 1 followed by Phase 2
    all_sampled_lines = phase1_lines + phase2_lines

    train_lines = all_sampled_lines[:int(0.9 * len(all_sampled_lines))]
    validation_lines = all_sampled_lines[int(0.9 * len(all_sampled_lines)):]

    print(f"Writing training dataset to {OUTPUT_TRAIN}...")
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')

    print(f"Writing validation dataset to {OUTPUT_VALIDATION}...")
    with open(OUTPUT_VALIDATION, 'w', encoding='utf-8') as f:
        for line in validation_lines:
            f.write(line + '\n')

if __name__ == "__main__":
    create_code_switched_dataset()