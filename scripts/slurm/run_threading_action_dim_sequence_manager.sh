#!/bin/bash
# Manage Square MH and Threading D1 image+proprio 7D/6D density runs.

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --job-name=pomdp_ip_mgr
#SBATCH --output=/iris/u/jasonyan/slurm/%j_pomdp_ip_mgr.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_pomdp_ip_mgr.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

cd "${REPO}"
python scripts/quality/manage_threading_action_dim_sequence.py \
  --state-file "${STATE_FILE:-/iris/u/jasonyan/data/pomdp_image_proprio_20260723/manager.json}" \
  --max-active-gpu "${MAX_ACTIVE_GPU:-32}" \
  --max-attempts "${MAX_ATTEMPTS:-6}" \
  --pending-migrate-seconds "${PENDING_MIGRATE_SECONDS:-1800}" \
  --poll-seconds "${POLL_SECONDS:-300}" \
  --stages "${STAGES:-square_7d,square_6d,d1_8d,d1_7d}"
