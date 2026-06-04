import random
import os
from dotenv import load_dotenv
from datasets import load_dataset
from code_switch_preprocessing import (
    load_dictionary, code_switch_sentence,
    DICT_EN_NL, DICT_EN_ZH,
)

BYTE_PREMIUM_FACTOR_ENGLISH = 1.000000
BYTE_PREMIUM_FACTOR_DUTCH = 1.051606
BYTE_PREMIUM_FACTOR_CHINESE = 0.935966

targets = {
    "eng": round(10_000_000 / BYTE_PREMIUM_FACTOR_ENGLISH),
    "nld": round(10_000_000 / BYTE_PREMIUM_FACTOR_DUTCH),
    "zho": round(10_000_000 / BYTE_PREMIUM_FACTOR_CHINESE)
}

INPUT_FILES = {
    'eng': './data/babylm-eng.txt',
    'nld': './data/babylm-nld.txt',
    'zho': './data/babylm-zho.txt'
}

OUTPUT_TRAIN = './data/bb26_30m_train.txt'
OUTPUT_VALIDATION = './data/bb26_30m_validation.txt'

OUTPUT_CODE_SWITCHING_TRAIN = './data/bb26_cs_train.txt'
OUTPUT_CODE_SWITCHING_VALIDATION = './data/bb26_cs_validation.txt'

# TLM dataset outputs
TLM_OUTPUT_TRAIN = './data/tlm_30m_train.txt'
TLM_OUTPUT_VALIDATION = './data/tlm_30m_validation.txt'

# OpenSubtitles baseline (plain monolingual) dataset outputs
OPENSUBS_OUTPUT_TRAIN = './data/opensubs_30m_train.txt'
OPENSUBS_OUTPUT_VALIDATION = './data/opensubs_30m_validation.txt'

# Target total token count to match bb26_30m (~30 million tokens)
TLM_TOKEN_TARGET = 30_000_000

# Per-language token targets for OpenSubtitles baseline (adjusted by byte premium)
OPENSUBS_TARGETS = {
    "eng": round(10_000_000 / BYTE_PREMIUM_FACTOR_ENGLISH),
    "nld": round(10_000_000 / BYTE_PREMIUM_FACTOR_DUTCH),
    "zho": round(10_000_000 / BYTE_PREMIUM_FACTOR_CHINESE),
}

# OpenSubtitles source files – plain monolingual sentences
OPENSUBS_INPUT_FILES = {
    "eng": [
        "data/en-nl/OpenSubtitles.en-nl.en",  # English side of EN-NL corpus
        "data/en-zh/OpenSubtitles.en-zh.en",   # English side of EN-ZH corpus
    ],
    "nld": [
        "data/en-nl/OpenSubtitles.en-nl.nl",
    ],
    "zho": [
        "data/en-zh/OpenSubtitles.en-zh.zh",
    ],
}


def load_and_save_datasets():
    load_dotenv()
    my_token = os.getenv("AUTH_TOKEN")

    langs = ["eng", "nld", "zho"]

    os.makedirs("data", exist_ok=True)

    for l in langs:
        dataset = load_dataset(f"BabyLM-community/babylm-{l}", split="train", token=my_token)
        
        file_path = f"data/babylm-{l}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            for row in dataset:
                f.write(row['text'] + '\n')
        
        print(f"Finished! Saved to {file_path}")

def mix_and_sample():
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

    for lang, filepaths in OPENSUBS_INPUT_FILES.items():
        print(f"Processing {lang}...")
        
        # Calculate 1/2 for Phase 1, 1/2 for Phase 2
        limit_total = targets[lang]
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

    print(f"Writing training dataset to {OUTPUT_CODE_SWITCHING_TRAIN}...")
    with open(OUTPUT_CODE_SWITCHING_TRAIN, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')

    print(f"Writing validation dataset to {OUTPUT_CODE_SWITCHING_VALIDATION}...")
    with open(OUTPUT_CODE_SWITCHING_VALIDATION, 'w', encoding='utf-8') as f:
        for line in validation_lines:
            f.write(line + '\n')

def mix_and_sample_opensubs(
    train_output: str = OPENSUBS_OUTPUT_TRAIN,
    val_output: str = OPENSUBS_OUTPUT_VALIDATION,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Build a 30M-token baseline dataset of plain monolingual sentences
    sourced from the OpenSubtitles parallel corpora.

    This is the fair comparison counterpart to the TLM dataset:
    same source domain (OpenSubtitles), same total size (~30M tokens),
    same per-language budget (~10M tokens adjusted by byte premium factors),
    but sentences are kept as independent monolingual lines (no [SEP] pairing).

    English tokens are drawn from both the EN-NL and EN-ZH English sides
    (reading EN-NL first, then EN-ZH if more are needed).
    """
    all_sampled_lines: list[str] = []

    for lang, filepaths in OPENSUBS_INPUT_FILES.items():
        limit = OPENSUBS_TARGETS[lang]
        current_words = 0
        lang_lines: list[str] = []

        print(f"Processing {lang} (target: {limit:,} tokens) …")

        for filepath in filepaths:
            if current_words >= limit:
                break
            print(f"  Reading {filepath} …")
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    words = line.split()
                    if not words:
                        continue

                    num_words = len(words)
                    if current_words + num_words <= limit:
                        lang_lines.append(line.strip())
                        current_words += num_words
                    else:
                        needed = limit - current_words
                        if needed > 0:
                            lang_lines.append(" ".join(words[:needed]))
                            current_words += needed
                        break

        print(f"  → Extracted {current_words:,} tokens for {lang}.")
        all_sampled_lines.extend(lang_lines)

    print(f"\nTotal lines collected: {len(all_sampled_lines):,}")
    print("Shuffling …")
    random.seed(seed)
    random.shuffle(all_sampled_lines)

    # Train / validation split
    split_idx = int((1.0 - val_ratio) * len(all_sampled_lines))
    train_lines = all_sampled_lines[:split_idx]
    val_lines = all_sampled_lines[split_idx:]

    os.makedirs(os.path.dirname(train_output) or ".", exist_ok=True)

    print(f"Writing {len(train_lines):,} training lines to {train_output} …")
    with open(train_output, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    print(f"Writing {len(val_lines):,} validation lines to {val_output} …")
    with open(val_output, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    print("Done! OpenSubtitles baseline dataset creation complete.")
    print(f"  Train:      {train_output}")
    print(f"  Validation: {val_output}")

def convert_moses_to_tlm(en_filepath, target_filepath, output_filepath, sep_token="[SEP]"):
    print(f"Reading from {en_filepath} and {target_filepath}...")
    
    # Open all three files simultaneously
    with open(en_filepath, 'r', encoding='utf-8') as f_en, \
         open(target_filepath, 'r', encoding='utf-8') as f_target, \
         open(output_filepath, 'w', encoding='utf-8') as f_out:
        
        count = 0
        # zip() pairs Line 1 of English with Line 1 of the Target Language
        for en_line, target_line in zip(f_en, f_target):
            # .strip() removes any hidden newline characters or trailing spaces
            en_clean = en_line.strip()
            target_clean = target_line.strip()
            
            # Skip empty lines to keep data clean
            if not en_clean or not target_clean:
                continue
                
            # Concatenate them for Translation Language Modeling
            combined_line = f"{en_clean} {sep_token} {target_clean}\n"
            f_out.write(combined_line)
            count += 1
            
    print(f"Success! Processed {count} parallel sentences and saved to {output_filepath}")


def create_tlm_dataset(
    sep_token: str = "[SEP]",
    token_target: int = TLM_TOKEN_TARGET,
    train_output: str = TLM_OUTPUT_TRAIN,
    val_output: str = TLM_OUTPUT_VALIDATION,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Build a combined TLM dataset from EN-NL and EN-ZH parallel corpora.

    Each line has the form:  English sentence [SEP] Target sentence
    Lines from both language pairs are pooled, shuffled, then token-counted
    until ``token_target`` whitespace-split tokens are collected.  The result
    is split into train / validation sets and written to disk.

    Args:
        sep_token:    Separator token inserted between the two sentences.
        token_target: Approximate total whitespace-token budget (default 30M).
        train_output: Path for the training split.
        val_output:   Path for the validation split.
        val_ratio:    Fraction of lines to reserve for validation.
        seed:         Random seed for reproducible shuffling.
    """
    corpora = [
        {
            "name": "EN-NL",
            "en":     "data/en-nl/OpenSubtitles.en-nl.en",
            "target": "data/en-nl/OpenSubtitles.en-nl.nl",
        },
        {
            "name": "EN-ZH",
            "en":     "data/en-zh/OpenSubtitles.en-zh.en",
            "target": "data/en-zh/OpenSubtitles.en-zh.zh",
        },
    ]

    # ------------------------------------------------------------------ #
    # Step 1 – read all valid sentence pairs from every corpus             #
    # ------------------------------------------------------------------ #
    all_pairs: list[str] = []

    for corpus in corpora:
        print(f"Reading {corpus['name']} corpus …")
        pair_count = 0
        with open(corpus["en"], "r", encoding="utf-8") as f_en, \
             open(corpus["target"], "r", encoding="utf-8") as f_tgt:
            for en_line, tgt_line in zip(f_en, f_tgt):
                en_clean  = en_line.strip()
                tgt_clean = tgt_line.strip()
                if not en_clean or not tgt_clean:
                    continue
                all_pairs.append(f"{en_clean} {sep_token} {tgt_clean}")
                pair_count += 1
        print(f"  → {pair_count:,} sentence pairs loaded from {corpus['name']}")

    print(f"\nTotal sentence pairs across all corpora: {len(all_pairs):,}")

    # ------------------------------------------------------------------ #
    # Step 2 – shuffle                                                     #
    # ------------------------------------------------------------------ #
    print("Shuffling …")
    random.seed(seed)
    random.shuffle(all_pairs)

    # ------------------------------------------------------------------ #
    # Step 3 – sample until we reach the token budget                     #
    # ------------------------------------------------------------------ #
    print(f"Sampling up to {token_target:,} whitespace tokens …")
    sampled: list[str] = []
    total_tokens = 0

    for line in all_pairs:
        tokens = line.split()
        n = len(tokens)
        if total_tokens + n <= token_target:
            sampled.append(line)
            total_tokens += n
        else:
            # Fit the remaining budget with a partial line
            needed = token_target - total_tokens
            if needed > 0:
                sampled.append(" ".join(tokens[:needed]))
                total_tokens += needed
            break

    print(f"Collected {total_tokens:,} tokens across {len(sampled):,} lines.")

    # ------------------------------------------------------------------ #
    # Step 4 – train / validation split                                    #
    # ------------------------------------------------------------------ #
    split_idx   = int((1.0 - val_ratio) * len(sampled))
    train_lines = sampled[:split_idx]
    val_lines   = sampled[split_idx:]

    os.makedirs(os.path.dirname(train_output) or ".", exist_ok=True)

    print(f"Writing {len(train_lines):,} training lines to {train_output} …")
    with open(train_output, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    print(f"Writing {len(val_lines):,} validation lines to {val_output} …")
    with open(val_output, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    print("Done! TLM dataset creation complete.")
    print(f"  Train:      {train_output}")
    print(f"  Validation: {val_output}")


if __name__ == "__main__":
    mix_and_sample()