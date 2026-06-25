#!/bin/bash
# Run the wrench-to-hook density queue manager as a lightweight Slurm job.
#
# This job does not train a model itself. It submits and monitors individual
# train / score jobs, keeping at most MAX_TRAIN_JOBS model-training jobs in
# Slurm at once. Cancel this manager job to stop automatic resubmission.

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --job-name=wth_den_mgr
#SBATCH --output=/iris/u/jasonyan/slurm/%j_wth_den_mgr.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_wth_den_mgr.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

cd "${REPO}"
python scripts/quality/manage_wrench_density_queue.py \
  --max-train-jobs "${MAX_TRAIN_JOBS:-48}" \
  --poll-seconds "${POLL_SECONDS:-300}" \
  --max-attempts "${MAX_ATTEMPTS:-5}" \
  --state-file "${STATE_FILE:-/iris/u/jasonyan/data/wrench_to_hook_density_queue_state.json}"
