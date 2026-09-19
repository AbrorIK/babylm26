#!/bin/bash
#SBATCH -J babylm-clm-baseline
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH -o logs/log_%A_%a.out
#SBATCH -e logs/log_%A_%a.err

# Baseline: plain multilingual causal LM, no alignment mechanism.
# This is the reference point every other condition is compared against, so the
# shared settings below must match the other job scripts exactly — only the
# mechanism flags differ.
#
# Submitted as a 3-task array, one task per seed (see SEEDS below). To rerun a
# single seed, select its array index: sbatch --array=1 jobs/job_train_baseline.sh

# ---- Seeds ----
# Seeds are fixed a priori and identical across all four conditions, so a
# difference between conditions is a difference in mechanism and not in which
# seeds happened to be drawn. The corpus shuffle/split seed is NOT this seed —
# it stays pinned at 42 in data_prep/build_dataset.py so every run sees exactly
# the same training and validation data.
SEEDS=(0 1 2)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}

# ---- Setup ----
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array task:   ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Seed:         $SEED"
echo "Node:         $(hostname)"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "GPUs:         $SLURM_GPUS_ON_NODE"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Start time:   $(date)"
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

# ---- Configuration Variables ----
TRAIN_DATA="data/bb26_train.txt"
VALID_DATA="data/bb26_validation.txt"
TOKENIZER_DIR="tokenizers/bb26-50k.model"      # same tokenizer as every other condition
OUTPUT_DIR="$WORK/output/gpt2-baseline-seed$SEED"

# wandb takes the run name from the output directory basename, so each seed is
# its own run; WANDB_RUN_GROUP collects the three of them under one condition.
export WANDB_RUN_GROUP="baseline"

echo "Starting baseline training (seed $SEED)..."

# ---- Run Training ----
python train_base.py \
    --train_data $TRAIN_DATA \
    --valid_data $VALID_DATA \
    --tokenizer $TOKENIZER_DIR \
    --output_path $OUTPUT_DIR \
    --model_path "gpt2" \
    --max_seq_len "0:64,5:256" \
    --batch_size 256 \
    --grad_acc 8 \
    --epochs 10 \
    --lr 5e-4 \
    --seed $SEED \
    --cpus $SLURM_CPUS_PER_TASK \
    --wandb

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
