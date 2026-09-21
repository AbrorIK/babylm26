# BabyLM26

## Building the training mixture

`build_dataset.py` samples a byte-premium-adjusted multilingual dataset from three monolingual corpora.
English is sampled by word count; Dutch and Chinese are sampled by file size, scaled by their
[Byte Premium](https://aclanthology.org/2024.sigul-1.1/) so that all three languages contribute equal content.

### Usage

```bash
python data_prep/build_dataset.py \
    --eng data/babylm-eng.txt \
    --nld data/babylm-nld.txt \
    --zho data/babylm-zho.txt \
    --tagged
```

This produces `data/bb26_train.tsv` and `data/bb26_validation.tsv` with a 90/10 split,
equal language ratios, and language tags (`eng\t<text>`) needed by the multi-head model.

### Options

| Flag          | Default                    | Description                                              |
| ------------- | -------------------------- | -------------------------------------------------------- |
| `--budget`    | `100000000`                | Total word budget (English-equivalent)                   |
| `--ratio`     | `0.333,0.333,0.334`        | eng,nld,zho content shares (must sum to ≤ 1.0)           |
| `--val-ratio` | `0.1`                      | Fraction held out for validation                         |
| `--seed`      | `42`                       | Random seed for reproducibility                          |
| `--tagged`    | off                        | Prefix lines with language tag (required for multi-head) |
| `--out-train` | `data/bb26_train.tsv`      | Output training file                                     |
| `--out-valid` | `data/bb26_validation.tsv` | Output validation file                                   |

## Running the experiments (4 conditions x 3 seeds)

Every condition is a SLURM job array of three tasks, one per seed. The seeds
are **0, 1, 2**, fixed a priori and identical across all four conditions, so a
difference between conditions is a difference in mechanism rather than in which
seeds happened to be drawn. Seed 0 is the training scripts' own default, so the
earlier single-seed results reappear as the seed-0 replicate and can be used to
check that the retrain reproduces them.

The corpus shuffle/split seed is a *different* seed and stays pinned at 42 in
`data_prep/build_dataset.py`; every run therefore sees exactly the same training
and validation data, and the three seeds vary only initialisation, batch order,
and (for the contrastive condition) the word-group sampling in `prealign.py`.

```bash
bash jobs/submit_all.sh                               # 4 arrays x 3 seeds = 12 runs
bash jobs/submit_all.sh baseline softlabel multihead  # or a subset
```

The submit script creates each `logs/<condition>/` directory before submitting,
which matters because SLURM opens its log files before the job body runs and
will not create a missing directory — the job fails instead. Submitting a job by
hand therefore needs the directory first:

```bash
mkdir -p logs/baseline && sbatch jobs/base.sh
```

The same applies to the two jobs `submit_all.sh` does not cover:
`logs/sweep/` for `sweep_alpha.sh`, `logs/translate/` for
`translate.sh`.

To rerun a single seed, select its array index (index i is seed i):

```bash
sbatch --array=1 jobs/contrastive.sh
```

Checkpoints land in `$WORK/output/gpt2-<condition>-seed<N>/`, whose basename is
also the W&B run name; the three seeds of a condition share a `WANDB_RUN_GROUP`
so they group together in the dashboard. Logs stay under the project directory
in `$HOME`, organised per condition as
`logs/<condition>/log_<arrayjobid>_<task>.{out,err}`.

Model outputs go to `$WORK` (10 TB, shared with the group, **not backed up**)
because `$HOME` (100 GB, backed up) is full and holds the code, data and logs.
`$WORK` also limits the number of files, so only few-and-large things belong
there — checkpoints qualify, the virtualenv and caches do not.

Note: `jobs/contrastive.sh` was previously named
`prealign.sh` and wrote to `output/gpt2-multihead-prealign`. That name
was a mislabel — `train_contrastive.py` builds a plain `AutoModelForCausalLM`
and has no multi-head component.

## Training

To train our hard_decay model, for example, run:

```bash
python train_mask.py --train_data data/bb24.train --valid_data data/bb25_small.dev --tokenizer tokenizers/bb24.model --output_path models/test_model/ --mask_update_steps 200 --logging_steps 200 --intermediate_size 1280 --hidden_size 384 --max_seq_len 0:64,5:256 --lamb --all_checkpoints --mlm_prob 0.4 --mask_decay 0.25 --seed 0
```

All flags are explained here:

```
--train_data            Path to the training data file.
--valid_data            Path to the validation data file.
--max_seq_len           Maximum sequence length. Can be a single number (e.g. 64) or depending on epoch (e.g. 0:64,5:128).
--model_path            Model path. Defaults to microsoft/deberta-v3-base.
--output_path           Output directory for model checkpoints and logs.
--tokenizer             Tokenizer path or name. If not specified, uses the model’s default tokenizer.
--batch_size            Batch size. Default: 256
--grad_acc              Gradient accumulation steps. Default: 1
--lr                    Learning rate. Default: 0.007
--epochs                Number of training epochs. Default: 10
--cpus                  Number of CPU workers for data loading. Default: 64
--logging_steps         Log training metrics every N steps. Default: 100
--eval_steps            Run evaluation every N steps. Default: 1000
--save_steps            Save a checkpoint every N steps. Default: 1000
--all_checkpoints       Save and evaluate model at multiple checkpoints (1/10/100M words) according to BabyLM requirements. Overrides eval_steps and save_steps.
--log_mlm_probs         Log masked language model probabilities for analysis.
--mask_update_steps     Steps between dynamic mask updates. Default: 100
--first_mask_update     Do not perform a mask update before this global step. Default: 0
--hidden_size           Hidden size of the model. Default: 768
--intermediate_size     Intermediate size of the feedforward layers. Default: 3072
--dropout               Dropout probability. Default: 0.1
--weight_decay          Weight decay for optimizer. Default: 0.01
--mlm_prob              Probability of masking a token for MLM. Default: 0.15
--mask_replace_prob     Probability of replacing a masked token with [MASK]. Default: 0.8
--random_replace_prob   Probability of replacing a masked token with a random token. Default: 0.1
--seed                  Random seed. Default: 0
--pretrained            Load pretrained model weights.
--eval_only             Run evaluation only, without training.
--debug                 Activate debug mode.
--wandb                 Report metrics to Weights & Biases.
--regular_mlm           Use standard masked language modeling objective.
--custom                Path or identifier for a custom model.
--lamb                  Use LAMB optimizer instead of AdamW.
--lower                 Lowercase all input text.
--soft                  Use soft masking strategy.
--mask_decay            Mask decay rate. For example, 0.1 decays masking probability linearly by 0.1 over training.
```

## Acknowledgements

This project is a fork of the [babylm25](https://github.com/Leukas/babylm25) repository developed by my supervisor, Lukas. I am extending this work as part of my thesis research.

## PreAlign: cross-lingual word alignment

PreAlign needs `[eng, nld, zho]` translation triplets. They are produced in two
steps: extract frequent English content words from the corpus, then translate
them with a local Qwen3-8B.

### External model

|              |                                                                                                                      |
| ------------ | -------------------------------------------------------------------------------------------------------------------- |
| Model        | `Qwen/Qwen3-8B`                                                                                                      |
| Revision     | `b968826d9c46dd6066d109eabc6255188de91218`                                                                           |
| Downloaded   | 2026-08-06 (UTC)                                                                                                     |
| Purpose      | Translating the PreAlign word list into Dutch and Chinese                                                            |
| Size         | ~16 GB in bf16, fits alongside nothing else on a 40 GB A100                                                          |
| Why this one | On the BabyLM 2026 approved external-model list (Qwen 2.5/3 families up to 9B) and covers all three target languages |

Qwen3 needs `transformers>=4.51.0`; earlier versions do not know the
architecture. This is pinned in `requirements.txt`.

**Budget note:** words generated by an external model count toward the 100M
budget. The 12,000 triplets are ~36,000 words (~0.036%), but they must still be
declared.

### Download

Run once on a login node — not inside a job, so the 16 GB transfer does not
consume GPU allocation. Pin the revision so the run is reproducible; models on
the Hub get updated in place:

```bash
huggingface-cli download \
    --local-dir ./Qwen3-8B \
    --revision b968826d9c46dd6066d109eabc6255188de91218 \
    Qwen/Qwen3-8B
```

Omitting `--revision` takes whatever `main` points to today. To recover the
revision of a copy that was downloaded without pinning:

```bash
head -1 Qwen3-8B/.cache/huggingface/download/*.metadata | sort -u
```

Each `.metadata` file holds three lines — commit hash, that file's blob hash,
download timestamp — so the first line is the repo revision and is the same
across every file.

### Building the triplets

```bash
python data_prep/extract_vocab.py          # -> data/prealign_vocab.txt (12k lemmas)
sbatch jobs/job_translate_test.sh          # 10 words, check the output format
sbatch jobs/translate.sh               # -> data/prealign_triplets.tsv (~30 min)
```

`extract_vocab.py` POS-tags in context, keeps content words only (dropping
function words and proper nouns), and lemmatises so inflected forms are merged
onto their dictionary form — single-word translation of `takes` or `looking`
has no clean answer and the translator tends to copy the English through.

`translate_vocab.py` batches 64 prompts at a time, disables Qwen3's thinking
mode, left-pads for batched generation, and resumes from an existing output
file. It reports how often the Dutch came back identical to the English; some
of those are real cognates (`man`, `help`, `water`), but a high rate means the
prompt is not landing.
