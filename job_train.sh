#!/bin/bash
#SBATCH -J babylm-mask
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err

# ---- Setup ----
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
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

# ---- Configuration Variables ----
TRAIN_DATA="data/tlm_30m_train.txt"
VALID_DATA="data/tlm_30m_validation.txt"
TOKENIZER_DIR="tokenizers/bb26.model"             
OUTPUT_DIR="output/tlm-30m"

echo "Starting training..."

# ---- Run Training ----
python train_mask.py \
    --train_data $TRAIN_DATA \
    --valid_data $VALID_DATA \
    --tokenizer $TOKENIZER_DIR \
    --output_path $OUTPUT_DIR \
    --model_path "microsoft/deberta-v3-base" \
    --max_seq_len "0:64,5:256" \
    --batch_size 256 \
    --grad_acc 8 \
    --epochs 10 \
    --lr 5e-4 \
    --cpus $SLURM_CPUS_PER_TASK \
    --all_checkpoints \
    --wandb

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="