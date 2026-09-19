#!/bin/bash
#SBATCH -J babylm-clm-contrastive
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
if [ ! -d "$WORK/output" ]; then
    mkdir "$WORK/output"
fi

# ---- Configuration Variables ----
TRAIN_DATA="data/bb26_tagged_train.tsv"
VALID_DATA="data/bb26_tagged_validation.tsv"
TOKENIZER_DIR="tokenizers/bb26-50k.model"
OUTPUT_DIR="$WORK/output/gpt2-contrastive-seed$SEED"
TRIPLETS="data/prealign_triplets.tsv"

export WANDB_RUN_GROUP="contrastive"

# ---- PreAlign settings ----
PREALIGN_STEPS=500
PREALIGN_ALPHA=1.0
PREALIGN_GROUPS=64
PREALIGN_TAU=0.1

ALIGN_LAMBDA=0.03
ALIGN_EVERY=1

if [ ! -f "$TRIPLETS" ]; then
    echo "ERROR: $TRIPLETS not found."
    echo "  1) python data_prep/extract_vocab.py"
    echo "  2) sbatch jobs/job_translate.sh"
    exit 1
fi
echo "word groups available: $(wc -l < $TRIPLETS)"

echo "Starting training (PreAlign + contrastive alignment, seed $SEED)..."

python train_contrastive.py \
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
    --wandb \
    --log_gpu_mem \
    --prealign_steps $PREALIGN_STEPS \
    --prealign_triplets $TRIPLETS \
    --prealign_alpha $PREALIGN_ALPHA \
    --prealign_groups $PREALIGN_GROUPS \
    --prealign_tau $PREALIGN_TAU \
    --align_lambda $ALIGN_LAMBDA \
    --align_every $ALIGN_EVERY

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
