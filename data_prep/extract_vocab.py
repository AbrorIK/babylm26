"""Pick the English words worth aligning: frequent content words.

Words are POS-tagged in sentence context with NLTK, and only content words are
kept: nouns, verbs, adjectives, adverbs, pronouns, numerals. That drops
articles, prepositions and conjunctions (no clean cross-lingual counterpart —
"the" maps to de/het by Dutch gender, and Chinese has no articles), and drops
proper nouns (NNP), which only transliterate.

Tagging must happen in context: an isolated word list gives the tagger nothing
to work with, and capitalisation — its main cue for proper nouns — is lost.

The Penn tagset has no AUX category, so "is"/"have"/"do" carry the same VB*
tags as "run"/"eat". No tagger can separate them, hence the small AUXILIARIES
list below.

Size matters more than the frequency threshold: PreAlign samples
steps x groups_per_step word groups in total, so a list much larger than that
contains words the training loop never draws.

    python data_prep/extract_vocab.py                  # top 12000
    python data_prep/extract_vocab.py --top_n 20000
"""

import argparse
import collections

CORPUS = "data/babylm-eng.txt"
OUTPUT = "data/prealign_vocab.txt"
TOP_N = 12_000
MIN_LEN = 2
TAG_EVERY = 5            # POS-tag every Nth line; majority tags converge fast

# Nouns, verbs, adjectives, adverbs, pronouns, numerals.
CONTENT_TAGS = ("NN", "VB", "JJ", "RB", "PRP", "CD")
# NNP/NNPS also start with "NN", so proper nouns need excluding explicitly.
PROPER_TAGS = ("NNP", "NNPS")

# The tagset cannot express these: auxiliaries are tagged as ordinary verbs,
# and a handful of function words are tagged as adverbs or pronouns.
AUXILIARIES = set("""
is am are was were be been being get got
do does did doing done have has had having
will would shall should can could may might must
not no nor very too just only even still also there here
""".split())


def count_and_tag(path, tag_every):
    """One pass: word frequencies over the whole corpus, POS tags over a sample.

    Returns (frequency counter, {word: majority tag}).
    """
    import nltk
    nltk.data.path.insert(0, "data/nltk_data")
    from nltk import pos_tag

    counts = collections.Counter()
    tag_votes = collections.defaultdict(collections.Counter)

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            words = [w for w in line.split() if w.isalpha()]
            if not words:
                continue
            counts.update(w.lower() for w in words)
            if i % tag_every == 0:
                for word, tag in pos_tag(words):
                    tag_votes[word.lower()][tag] += 1

    majority = {w: votes.most_common(1)[0][0] for w, votes in tag_votes.items()}
    return counts, majority


def is_content_word(word, tag):
    if word in AUXILIARIES or len(word) < MIN_LEN:
        return False
    if tag is None or tag in PROPER_TAGS:
        return False
    return tag.startswith(CONTENT_TAGS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--top_n", type=int, default=TOP_N)
    parser.add_argument("--tag_every", type=int, default=TAG_EVERY)
    args = parser.parse_args()

    counts, majority = count_and_tag(args.corpus, args.tag_every)
    print(f"{sum(counts.values()):,} tokens, {len(counts):,} types, "
          f"{len(majority):,} tagged")

    kept = [w for w, _ in counts.most_common()
            if is_content_word(w, majority.get(w))][:args.top_n]

    with open(args.output, "w", encoding="utf-8") as f:
        for word in kept:
            f.write(f"{word}\n")

    print(f"kept {len(kept):,} words (freq {counts[kept[-1]]:,}..{counts[kept[0]]:,})")
    print(f"wrote {args.output}")
    print("sample:", ", ".join(kept[:20]))


if __name__ == "__main__":
    main()
