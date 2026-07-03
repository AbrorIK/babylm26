import sentencepiece as spm
from pathlib import Path


def load_forced_tokens(path="tokenizers/forced_tokens.txt"):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def train_tokenizer():
    print("Training tokenizer")
    forced = load_forced_tokens()
    print(f"Forcing {len(forced)} dictionary words to be single tokens")

    Path("tokenizer").mkdir(exist_ok=True)
    spm.SentencePieceTrainer.train(
        input='data/bb26_train.txt',
        model_prefix='tokenizers/bb26-50k',
        vocab_size=50000,                       # bumped up to leave room for forced words
        model_type='bpe',
        byte_fallback=True,
        character_coverage=0.9995,
        user_defined_symbols=forced,   # these stay whole
        normalization_rule_name="identity",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3
    )
    print("Tokenizer training complete!")


def main():
    train_tokenizer()


if __name__ == "__main__":
    main()
