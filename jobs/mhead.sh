#!/bin/bash
#SBATCH -J babylm-clm-multihead
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH -o logs/log_%A_%a.out
#SBATCH -e logs/log_%A_%a.err

# ---- Seeds ----
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
TRAIN_DATA="data/bb26_tagged_train.tsv"
VALID_DATA="data/bb26_tagged_validation.tsv"
TOKENIZER_DIR="tokenizers/bb26-40k"
OUTPUT_DIR="$WORK/output/gpt2-multihead-seed$SEED"

export WANDB_RUN_GROUP="multihead"

echo "Starting multi-head training (seed $SEED)..."

# ---- Run Training ----
python train_mhead.py \
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
echo "Exporting per-language models (seed $SEED)..."
echo "=========================================="

# Find the last checkpoint
LAST_CKPT=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -n "$LAST_CKPT" ]; then
    python export_multihead.py \
        --checkpoint $LAST_CKPT \
        --tokenizer $TOKENIZER_DIR
    echo "Export done: $LAST_CKPT-export/{eng,nld,zho}"
else
    echo "No checkpoint found to export."
fi

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
