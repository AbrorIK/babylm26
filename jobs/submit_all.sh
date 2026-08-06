#!/bin/bash
# Submit all four conditions × 3 seeds = 12 training runs.
# Each sbatch below queues a 3-task array (seeds 0, 1, 2).
#
#   bash jobs/submit_all_seeds.sh
#
# SLURM writes its logs before the job body runs, so logs/ must already exist.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

for job in baseline softlabel multihead contrastive; do
    sbatch "jobs/job_train_$job.sh"
done

echo
echo "Queued 4 arrays * 3 seeds = 12 runs. Watch them with: squeue -u $USER"
