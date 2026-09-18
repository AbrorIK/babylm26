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


# Penn Treebank tag -> WordNet part of speech, for the lemmatizer.
_WORDNET_POS = {"NN": "n", "VB": "v", "JJ": "a", "RB": "r"}


def lemmatize_counts(counts, majority):
    """Collapse inflected forms onto their dictionary form, summing frequencies.

    "takes"/"taking"/"took" all become "take". This matters because the words
    are translated one at a time with no context: an inflected form has no clean
    single-word translation (Dutch inflects differently), and the translator
    tends to copy the English through unchanged rather than guess. Feeding
    dictionary forms removes that failure at the source.

    The POS tag decides how to lemmatize — "looking" is only reduced to "look"
    when tagged as a verb, and "better" only becomes "good" as an adjective.
    """
    import nltk
    nltk.data.path.insert(0, "data/nltk_data")
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    lemma_counts = collections.Counter()
    lemma_tag = {}

    for word, n in counts.items():
        tag = majority.get(word)
        if not is_content_word(word, tag):
            continue
        lemma = lemmatizer.lemmatize(word, _WORDNET_POS.get(tag[:2], "n"))
        # Lemmatising can land on a word we exclude ("was" -> "be"), and can
        # shorten below the minimum length.
        if lemma in AUXILIARIES or len(lemma) < MIN_LEN:
            continue
        lemma_counts[lemma] += n
        lemma_tag.setdefault(lemma, tag)

    return lemma_counts, lemma_tag


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

    lemma_counts, _ = lemmatize_counts(counts, majority)
    print(f"{len(lemma_counts):,} content lemmas after merging inflected forms")

    kept = [w for w, _ in lemma_counts.most_common()][:args.top_n]

    with open(args.output, "w", encoding="utf-8") as f:
        for word in kept:
            f.write(f"{word}\n")

    print(f"kept {len(kept):,} lemmas "
          f"(freq {lemma_counts[kept[-1]]:,}..{lemma_counts[kept[0]]:,})")
    print(f"wrote {args.output}")
    print("sample:", ", ".join(kept[:20]))


if __name__ == "__main__":
    main()
