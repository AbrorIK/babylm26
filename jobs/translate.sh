#!/bin/bash
#SBATCH -J qwen-translate
#SBATCH -p a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err

# ---- Setup ----
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "GPUs:         $SLURM_GPUS_ON_NODE"
echo "Start time:   $(date)"
echo "=========================================="

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

PROJECT_DIR=$HOME/thesis/babylm26
cd $PROJECT_DIR

source $HOME/thesis/.venv/bin/activate

export PYTHONUNBUFFERED=1

# ---- Configuration Variables ----
MODEL="./Qwen3-8B"
VOCAB="data/prealign_vocab.txt"
OUTPUT="data/prealign_triplets.tsv"
BATCH_SIZE=64

# ---- Checks ----
if [ ! -d "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL"
    echo "Download it once on a login node with:"
    echo "  huggingface-cli download --local-dir ./Qwen3-8B Qwen/Qwen3-8B"
    exit 1
fi

if [ ! -f "$VOCAB" ]; then
    echo "ERROR: $VOCAB not found. Run: python data_prep/extract_vocab.py"
    exit 1
fi

# NOTE: $OUTPUT is deliberately NOT deleted. translate_vocab.py skips words
# already present, so re-submitting this job resumes where it stopped instead
# of redoing finished work.
echo "words to translate: $(wc -l < $VOCAB)"
if [ -f "$OUTPUT" ]; then
    echo "already done:       $(wc -l < $OUTPUT)  (resuming)"
fi
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/translate_vocab.py \
    --model $MODEL \
    --vocab $VOCAB \
    --output $OUTPUT \
    --batch_size $BATCH_SIZE

# ---- Summary ----
echo "=========================================="
KEPT=$(wc -l < $OUTPUT)
TOTAL=$(wc -l < $VOCAB)
echo "triplets written: $KEPT / $TOTAL words"
echo ""
echo "first 5:"
head -5 $OUTPUT
echo "last 5:"
tail -5 $OUTPUT
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
