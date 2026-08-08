import argparse
import random

BP = {"nld": 1.0516, "zho": 0.9894}

parser = argparse.ArgumentParser(description="Build a byte-premium-adjusted multilingual mixture for BabyLM 2026.")
parser.add_argument("--eng", required=True, help="path to English corpus (one sentence per line)")
parser.add_argument("--nld", required=True, help="path to Dutch corpus")
parser.add_argument("--zho", required=True, help="path to Chinese corpus")
parser.add_argument("--budget", type=int, default=100_000_000, help="total word budget in English-equivalent words (default: 100M)")
parser.add_argument("--out-train", default="data/bb26_train.tsv", help="output training file")
parser.add_argument("--out-valid", default="data/bb26_validation.tsv", help="output validation file")
parser.add_argument("--val-ratio", type=float, default=0.1, help="fraction held out for validation (default: 0.1)")
parser.add_argument("--seed", type=int, default=42, help="random seed for shuffling (default: 42)")
parser.add_argument("--tagged", action="store_true", help="prefix each line with its language tag (eng/nld/zho)")
parser.add_argument("--ratio", default="0.333,0.333,0.334", help="eng,nld,zho content ratio, must sum to 1.0 (default: equal thirds)")
args = parser.parse_args()

ratios = dict(zip(["eng", "nld", "zho"], [float(r) for r in args.ratio.split(",")]))
assert sum(ratios.values()) <= 1.0, f"ratios must sum to ≤ 1.0, got {sum(ratios.values())}"

CORPORA = {"eng": args.eng, "nld": args.nld, "zho": args.zho}
WORDS_PER_LANG = int(args.budget * ratios["eng"])
rng = random.Random(args.seed)
bytes_used = {"eng": 0, "nld": 0, "zho": 0}
sampled = []

# --- step 1: sample exactly budget/3 words from English --------------------

lines = open(CORPORA["eng"], encoding="utf-8").read().splitlines()
rng.shuffle(lines)

eng_words = 0
for line in lines:
    text = line.strip()
    if not text:
        continue
    w = len(text.split())
    if eng_words + w > WORDS_PER_LANG:
        break
    sampled.append(("eng", text))
    eng_words += w
    bytes_used["eng"] += len(text.encode("utf-8"))

print(f"eng: {eng_words:,} words, {bytes_used['eng']:,} bytes")

# --- step 2: sample Dutch and Chinese by file size -------------------------

for lang in ["nld", "zho"]:
    allowance = bytes_used["eng"] * (ratios[lang] / ratios["eng"]) * BP[lang]
    lines = open(CORPORA[lang], encoding="utf-8").read().splitlines()
    rng.shuffle(lines)

    kept = 0
    for line in lines:
        text = line.strip()
        if not text:
            continue
        n = len(text.encode("utf-8"))
        if bytes_used[lang] + n > allowance:
            break
        sampled.append((lang, text))
        bytes_used[lang] += n
        kept += 1

    print(f"{lang}: {kept:,} lines, {bytes_used[lang]:,} bytes (limit {allowance:,.0f})")

# --- shuffle and split into train / validation ------------------------------

rng.shuffle(sampled)
split = int(len(sampled) * (1 - args.val_ratio))

with open(args.out_train, "w", encoding="utf-8") as f:
    for lang, text in sampled[:split]:
        f.write(f"{lang}\t{text}\n" if args.tagged else f"{text}\n")

with open(args.out_valid, "w", encoding="utf-8") as f:
    for lang, text in sampled[split:]:
        f.write(f"{lang}\t{text}\n" if args.tagged else f"{text}\n")

print(f"\nwrote {split:,} train / {len(sampled) - split:,} validation lines")

# --- verify -----------------------------------------------------------------

bytes_per_word = bytes_used["eng"] / eng_words
nld_words = bytes_used["nld"] / BP["nld"] / bytes_per_word
zho_words = bytes_used["zho"] / BP["zho"] / bytes_per_word
total_words = eng_words + nld_words + zho_words

print(f"\nVerification:")
print(f"  bytes per English word: {bytes_per_word:.2f}")
print(f"  eng: {eng_words:,} words (exact)")
print(f"  nld: {nld_words:,.0f} eng-equivalent words")
print(f"  zho: {zho_words:,.0f} eng-equivalent words")
print(f"  total: {total_words:,.0f} eng-equivalent words (limit {args.budget:,})")