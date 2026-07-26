#!/bin/bash
# Run the corrected Wrench 0722 full-frame density queue manager.

#SBATCH --account=iliad
#SBATCH --partition=sc-freecpu
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=w0722_ff_mgr
#SBATCH --output=/iris/u/jasonyan/slurm/%j_w0722_ff_mgr.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_w0722_ff_mgr.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"

cd "${REPO}"
python scripts/quality/manage_wrench_0722_fullframe_pomdp_queue.py \
  --poll-seconds "${POLL_SECONDS:-300}" \
  --max-active-gpu "${MAX_ACTIVE_GPU:-12}" \
  --max-attempts "${MAX_ATTEMPTS:-8}" \
  --pending-migrate-seconds "${PENDING_MIGRATE_SECONDS:-1200}"
