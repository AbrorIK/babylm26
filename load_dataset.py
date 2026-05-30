import random
import os
from dotenv import load_dotenv
from datasets import load_dataset

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

# TLM dataset outputs
TLM_OUTPUT_TRAIN = './data/tlm_30m_train.txt'
TLM_OUTPUT_VALIDATION = './data/tlm_30m_validation.txt'

# Target total token count to match bb26_30m (~30 million tokens)
TLM_TOKEN_TARGET = 30_000_000

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
    all_sampled_lines = []

    for lang, filepath in INPUT_FILES.items():
        print(f"Processing {lang}...")
        limit = targets[lang]
        current_words = 0
        lang_lines = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
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
                    
        print(f"Extracted {current_words} words for {lang}.")
        all_sampled_lines.extend(lang_lines)

    print("\nShuffling the combined lines...")
    random.seed(42)
    random.shuffle(all_sampled_lines)

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
    create_tlm_dataset()