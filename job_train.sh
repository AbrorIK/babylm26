#!/bin/bash
#SBATCH -J babylm-mask                                  # Job name
#SBATCH -p lrz-v100x2                                  # Partition
#SBATCH --gres=gpu:1                                    # Request 1 GPU
#SBATCH --cpus-per-task=10                              # CPUs for data loading (increased to 10 to match your python script)
#SBATCH --mem=32G                                       # Memory
#SBATCH --time=24:00:00                                 # Max wall time
#SBATCH -o logs/log_%j.out                              # Standard output
#SBATCH -e logs/log_%j.err                              # Standard error

# ---- Setup ----
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Partition:    $SLURM_JOB_PARTITION"
echo "GPUs:         $SLURM_GPUS_ON_NODE"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Start time:   $(date)"
echo "=========================================="

# Navigate to project directory
PROJECT_DIR=/dss/dsshome1/09/ge58lix2/babylm26
cd $PROJECT_DIR

# Activate virtual environment
source ../.venv/bin/activate

# Force Python to print logs instantly instead of buffering them
export PYTHONUNBUFFERED=1

# ---- Configuration Variables ----
# Change these paths to match your actual files!
TRAIN_DATA="data/bb26_30m_train.txt"
VALID_DATA="data/bb26_30m_validation.txt"
TOKENIZER_DIR="tokenizers/bb26.model"             # Folder containing spm.model & tokenizer_config.json
OUTPUT_DIR="output/babylm_deberta_run"

echo "Starting training..."

# ---- Run Training ----
srun python train_mask_basic.py \
    --train_data $TRAIN_DATA \
    --valid_data $VALID_DATA \
    --tokenizer $TOKENIZER_DIR \
    --output_path $OUTPUT_DIR \
    --model_path "microsoft/deberta-v3-base" \
    --max_seq_len "0:64,5:256" \
    --batch_size 256 \
    --grad_acc 8 \
    --epochs 10 \
    --lr 0.007 \
    --cpus $SLURM_CPUS_PER_TASK \
    --all_checkpoints \
    --wandb

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="