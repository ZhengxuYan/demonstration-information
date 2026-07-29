#!/bin/bash
# Run the resumable Wrench 0722 visual-control queue manager.

#SBATCH --account=iliad
#SBATCH --partition=sc-freecpu
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=w0722_root_mgr
#SBATCH --output=/iris/u/jasonyan/slurm/%j_w0722_root_mgr.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_w0722_root_mgr.err

set -euo pipefail
REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
cd "${REPO}"
python3 scripts/quality/manage_wrench_0722_visual_controls.py "$@"
