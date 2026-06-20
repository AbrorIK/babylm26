import random
import os

# ---- Byte premium factors (information capacity per token) ----
# Dutch tokens carry ~5% less info → need more tokens for same content.
# Chinese tokens carry ~6% more info → need fewer tokens.
BYTE_PREMIUM_ENGLISH = 1.000000
BYTE_PREMIUM_DUTCH   = 1.051606
BYTE_PREMIUM_CHINESE = 0.935966

# ---- Token budget ----
TOKEN_TARGET = 30_000_000        # total whitespace-token budget

# ---- Per-corpus ratio (must sum to 1.0) ----
# Controls how the base budget is split between EN-NL and EN-ZH pairs
# BEFORE applying byte premium adjustments.
RATIO_EN_NL = 0.5
RATIO_EN_ZH = 0.5

# ---- Per-corpus adjusted budgets ----
# Divide by target-language byte premium so that both corpora contribute
# the same amount of *information*, not the same raw token count.
BUDGET_EN_NL = round((TOKEN_TARGET * RATIO_EN_NL) / BYTE_PREMIUM_DUTCH)
BUDGET_EN_ZH = round((TOKEN_TARGET * RATIO_EN_ZH) / BYTE_PREMIUM_CHINESE)

# ---- Parallel corpus file paths ----
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


def sample_to_budget(lines: list[str], token_budget: int) -> list[str]:
    """Keep lines from the list until the whitespace-token budget is exhausted."""
    sampled: list[str] = []
    total = 0
    for line in lines:
        n = len(line.split())
        if total + n <= token_budget:
            sampled.append(line)
            total += n
        else:
            break
    return sampled


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

    # --- Pool, shuffle, and split ---
    all_lines = sampled_nl + sampled_zh
    random.shuffle(all_lines)

    split_idx   = int((1.0 - VALIDATION_RATIO) * len(all_lines))
    train_lines = all_lines[:split_idx]
    val_lines   = all_lines[split_idx:]

    # --- Write to disk ---
    os.makedirs(os.path.dirname(OUTPUT_TRAIN) or ".", exist_ok=True)

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    with open(OUTPUT_VALIDATION, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    print(f"\nDone!")
    print(f"  Train:      {OUTPUT_TRAIN}  ({len(train_lines):,} lines)")
    print(f"  Validation: {OUTPUT_VALIDATION}  ({len(val_lines):,} lines)")

if __name__ == '__main__':
    create_tlm_dataset()