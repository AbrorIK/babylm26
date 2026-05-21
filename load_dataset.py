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

if __name__ == "__main__":
    mix_and_sample()