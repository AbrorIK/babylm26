import string
from collections import Counter

DATA = "data"


def corpus_counts(path):
    """word -> frequency for space-segmented text (English, Dutch)."""
    punct = string.punctuation + "“”‘’—…«»"
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            for w in line.lower().split():
                w = w.strip(punct)
                if w:
                    counts[w] += 1
    return counts


def dict_words(path, side):
    """Unique words on one side of a MUSE dict. side=0 = source, side=1 = target."""
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t") if "\t" in line else line.rsplit(" ", 1)
            if len(parts) == 2:
                out.add(parts[side].strip().lower())
    return {w for w in out if w}


def report(name, words, counts):
    total = sum(counts.values())
    present = [w for w in words if w in counts]
    hits = sum(counts[w] for w in present)
    print(f"{name}")
    print(f"   words found in data : {len(present)}/{len(words)} ({100*len(present)/len(words):.1f}%)")
    print(f"   share of text covered: {100*hits/total:.1f}%\n")


def report_chinese(name, words, path):
    import ahocorasick
    words = list(words)
    A = ahocorasick.Automaton()
    for i, w in enumerate(words):
        A.add_word(w, i)
    A.make_automaton()
    found, hit_chars, total_chars = set(), 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.replace(" ", "")
            total_chars += len(line)
            for _, idx in A.iter(line):
                found.add(idx)
                hit_chars += len(words[idx])
    print(f"{name}")
    print(f"   words found in data : {len(found)}/{len(words)} ({100*len(found)/len(words):.1f}%)")
    print(f"   share of text covered: ~{100*hit_chars/total_chars:.1f}%\n")


print("Reading corpora...\n")
eng = corpus_counts(f"{DATA}/babylm-eng.txt")
nl = corpus_counts(f"{DATA}/babylm-nld.txt")

report("en-nl  English", dict_words(f"{DATA}/dictionaries/en-nl.txt", 0), eng)
report("en-nl  Dutch",   dict_words(f"{DATA}/dictionaries/en-nl.txt", 1), nl)
report("en-zh  English", dict_words(f"{DATA}/dictionaries/en-zh.txt", 0), eng)
report_chinese("en-zh  Chinese", dict_words(f"{DATA}/dictionaries/en-zh.txt", 1), f"{DATA}/babylm-zho.txt")
