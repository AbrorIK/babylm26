import sentencepiece as spm
from pathlib import Path

def train_tokenizer():
    print("Training tokenizer")
    Path("tokenizer").mkdir(exist_ok=True)
    spm.SentencePieceTrainer.train(
        input='data/multilingual_mixture.txt',
        model_prefix='tokenizer/babylm_multilingual_bpe_40k',
        vocab_size=40000,
        model_type='bpe',
        byte_fallback=True,
        character_coverage=0.9995,
        user_defined_symbols=["[MASK]"],
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
    