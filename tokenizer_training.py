from tokenizers import (Tokenizer, models, normalizers, pre_tokenizers,
                        decoders, processors, trainers, Regex)
from transformers import PreTrainedTokenizerFast

OUT_DIR = "tokenizers/bb26-40k"

tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(Regex(r"\s+"), " ") , normalizers.Lowercase()])
tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

trainer = trainers.BpeTrainer(
    vocab_size=40000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=2,
)
tok.train(["data/bb26_train.txt"], trainer)

# Must come after train(): the ids only exist once the vocab is built.
tok.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[("[CLS]", tok.token_to_id("[CLS]")),
                    ("[SEP]", tok.token_to_id("[SEP]"))],
)

hf = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
    pad_token="[PAD]", mask_token="[MASK]",
    bos_token="[CLS]", eos_token="[SEP]",
    model_max_length=1024,
)
hf.save_pretrained(OUT_DIR)
print(f"saved {OUT_DIR} (vocab {len(hf)})")
