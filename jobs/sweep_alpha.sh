#!/bin/bash
#SBATCH -J prealign-sweep
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err

# Sweep the PreAlign alpha with --prealign_only: run the 500-step alignment
# phase and stop, skipping the 11-hour pretraining. The dataset is tokenized
# once and cached by HF datasets, so runs after the first are quick.
#
# Judge each alpha on two numbers from its PREALIGN SUMMARY line:
#   final_align - lower is better (alignment actually learned something)
#   final_lm    - the stopping signal: while this stays flat, alpha is still
#                 safe; when it starts climbing, alignment is beginning to cost
#                 language modelling and you have found the ceiling
#
# Do NOT judge on the early neg_sim spike. It reaches ~0.98 around step 50 at
# every alpha tested, including 0.01, because a randomly initialised LM first
# learns the unigram distribution. A higher alpha lowers it, not raises it.

echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

PROJECT_DIR=$HOME/thesis/babylm26
cd $PROJECT_DIR
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# ---- Storage ----
# Model outputs go to $WORK (10 TB, shared, not backed up); $HOME is full.
if [ ! -d "$WORK/output" ]; then
    mkdir "$WORK/output"
fi

TRAIN_DATA="data/bb26_tagged_train.tsv"
VALID_DATA="data/bb26_tagged_validation.tsv"
TOKENIZER_DIR="tokenizers/bb26-40k"
TRIPLETS="data/prealign_triplets.tsv"

for ALPHA in 0.3 1.0 3.0 10.0; do
    echo ""
    echo "########## alpha = $ALPHA ##########"
    python train_contrastive.py \
        --train_data $TRAIN_DATA \
        --valid_data $VALID_DATA \
        --tokenizer $TOKENIZER_DIR \
        --output_path "$WORK/output/sweep-alpha-$ALPHA" \
        --model_path "gpt2" \
        --max_seq_len "0:64" \
        --batch_size 256 \
        --grad_acc 8 \
        --cpus $SLURM_CPUS_PER_TASK \
        --prealign_steps 500 \
        --prealign_triplets $TRIPLETS \
        --prealign_alpha $ALPHA \
        --prealign_only
done

echo ""
echo "=========================================="
echo "Sweep finished. Collect the results with:"
echo "  grep 'PREALIGN SUMMARY' logs/log_$SLURM_JOB_ID.out"
echo "(grepping inside the job finds nothing — SLURM has not flushed this file yet)"
echo "=========================================="
echo "End time: $(date)"
