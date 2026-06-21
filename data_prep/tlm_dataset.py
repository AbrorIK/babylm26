import random

from dataset_common import (
    BYTE_PREMIUM_DUTCH,
    BYTE_PREMIUM_CHINESE,
    adjusted_budget,
    sample_to_budget,
    write_train_val_split,
)

# ================================================================== #
# Configuration                                                        #
# ================================================================== #

# ---- Token budget ----
TOKEN_TARGET = 30_000_000  # total whitespace-token budget

# ---- Per-corpus ratio (must sum to 1.0), applied BEFORE byte-premium adjust ----
RATIO_EN_NL = 0.5
RATIO_EN_ZH = 0.5

BUDGET_EN_NL = adjusted_budget(TOKEN_TARGET, RATIO_EN_NL, BYTE_PREMIUM_DUTCH)
BUDGET_EN_ZH = adjusted_budget(TOKEN_TARGET, RATIO_EN_ZH, BYTE_PREMIUM_CHINESE)

# ---- Parallel corpus file paths (relative to project root) ----
CORPUS_EN_NL = {
    "en":     "data/en-nl/OpenSubtitles.en-nl.en",
    "target": "data/en-nl/OpenSubtitles.en-nl.nl",
}
CORPUS_EN_ZH = {
    "en":     "data/en-zh/OpenSubtitles.en-zh.en",
    "target": "data/en-zh/OpenSubtitles.en-zh.zh",
}

# ---- Output paths ----
OUTPUT_TRAIN      = "./data/tlm_30m_train.txt"
OUTPUT_VALIDATION = "./data/tlm_30m_validation.txt"

# ---- Misc ----
SEP_TOKEN        = "[SEP]"
VALIDATION_RATIO = 0.1
SEED             = 42


def load_parallel_pairs(en_path: str, tgt_path: str, sep_token: str) -> list[str]:
    """Read a parallel corpus and return TLM-formatted lines: 'en [SEP] tgt'."""
    pairs: list[str] = []
    with open(en_path, encoding="utf-8") as f_en, \
         open(tgt_path, encoding="utf-8") as f_tgt:
        for en_line, tgt_line in zip(f_en, f_tgt):
            en_clean  = en_line.strip()
            tgt_clean = tgt_line.strip()
            if en_clean and tgt_clean:
                pairs.append(f"{en_clean} {sep_token} {tgt_clean}")
    return pairs


def create_tlm_dataset():
    print(f"Token budget  EN-NL: {BUDGET_EN_NL:,} tokens (adjusted by Dutch  BPF={BYTE_PREMIUM_DUTCH})")
    print(f"Token budget  EN-ZH: {BUDGET_EN_ZH:,} tokens (adjusted by Chinese BPF={BYTE_PREMIUM_CHINESE})")
    print(f"Token budget  Total: {BUDGET_EN_NL + BUDGET_EN_ZH:,} tokens")

    # --- Load & shuffle each corpus independently ---
    random.seed(SEED)

    print(f"\nLoading EN-NL corpus …")
    pairs_nl = load_parallel_pairs(CORPUS_EN_NL["en"], CORPUS_EN_NL["target"], SEP_TOKEN)
    random.shuffle(pairs_nl)
    print(f"  → {len(pairs_nl):,} sentence pairs")

    print(f"Loading EN-ZH corpus …")
    pairs_zh = load_parallel_pairs(CORPUS_EN_ZH["en"], CORPUS_EN_ZH["target"], SEP_TOKEN)
    random.shuffle(pairs_zh)
    print(f"  → {len(pairs_zh):,} sentence pairs")

    # --- Sample each corpus to its adjusted budget ---
    sampled_nl = sample_to_budget(pairs_nl, BUDGET_EN_NL)
    sampled_zh = sample_to_budget(pairs_zh, BUDGET_EN_ZH)
    print(f"\nSampled EN-NL: {len(sampled_nl):,} lines (~{BUDGET_EN_NL:,} tokens)")
    print(f"Sampled EN-ZH: {len(sampled_zh):,} lines (~{BUDGET_EN_ZH:,} tokens)")

    # --- Pool, shuffle, split, write (SEED already set above; do not reseed) ---
    all_lines = sampled_nl + sampled_zh
    train_lines, val_lines = write_train_val_split(
        all_lines, OUTPUT_TRAIN, OUTPUT_VALIDATION,
        val_ratio=VALIDATION_RATIO, shuffle=True,
    )

    print(f"\nDone!")
    print(f"  Train:      {OUTPUT_TRAIN}  ({len(train_lines):,} lines)")
    print(f"  Validation: {OUTPUT_VALIDATION}  ({len(val_lines):,} lines)")


if __name__ == '__main__':
    create_tlm_dataset()
