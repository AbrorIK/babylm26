import os
import re
from collections import Counter

from opencc import OpenCC
import ahocorasick

# ---- config ----
DICT_ENNL = "data/dictionaries/en-nl.txt"   # tab-separated:  english <TAB> dutch
DICT_ENZH = "data/dictionaries/en-zh.txt"   # space-separated: english <space> chinese
CORPUS_EN = "data/babylm-eng.txt"
CORPUS_NL = "data/babylm-nld.txt"
CORPUS_ZH = "data/babylm-zho.txt"
N_PER_LANG = 3000
OUT_FILE = "tokenizers/forced_tokens.txt"

t2s = OpenCC("t2s")               # traditional -> simplified Chinese
WORD = re.compile(r"[^\W\d_]+")   # runs of letters (Latin), Unicode-aware


# ---- 1. collect candidate words from the dictionaries ----
def read_pairs(path):
    """Yield (source, target) word pairs from a MUSE dictionary."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t") if "\t" in line else line.rsplit(" ", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                yield parts[0].strip().lower(), parts[1].strip().lower()

en_words, nl_words, zh_words = set(), set(), set()
for en, nl in read_pairs(DICT_ENNL):
    en_words.add(en)
    nl_words.add(nl)
for en, zh in read_pairs(DICT_ENZH):
    en_words.add(en)
    zh_words.add(t2s.convert(zh))


# ---- 2. count how often each candidate appears in the corpus ----
def whitespace_freq(words, corpus_path):
    """Frequencies for space-segmented text (English, Dutch)."""
    words = set(words)
    counts = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            for w in WORD.findall(line.lower()):
                if w in words:
                    counts[w] += 1
    return counts

def substring_freq(words, corpus_path):
    """Frequencies for non-segmented text (Chinese), via substring search."""
    automaton = ahocorasick.Automaton()
    for w in words:
        automaton.add_word(w, w)
    automaton.make_automaton()
    counts = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            for _, w in automaton.iter(line.replace(" ", "")):
                counts[w] += 1
    return counts


# ---- 3. keep the top-N most frequent per language ----
def top_n(counts, n):
    return [w for w, _ in counts.most_common(n)]

en_top = top_n(whitespace_freq(en_words, CORPUS_EN), N_PER_LANG)
nl_top = top_n(whitespace_freq(nl_words, CORPUS_NL), N_PER_LANG)
zh_top = top_n(substring_freq(zh_words, CORPUS_ZH), N_PER_LANG)


# ---- 4. write symbols (Latin words get the word-start marker, Chinese doesn't) ----
symbols = ["▁" + w for w in en_top] + ["▁" + w for w in nl_top] + zh_top
symbols = list(dict.fromkeys(symbols))   # dedupe, keep order

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for s in symbols:
        f.write(s + "\n")

print(f"en={len(en_top)}  nl={len(nl_top)}  zh={len(zh_top)}  total={len(symbols)}")
print(f"wrote {OUT_FILE}")
