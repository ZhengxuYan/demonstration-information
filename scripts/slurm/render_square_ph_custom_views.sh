#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --job-name=ph_cam_views
#SBATCH --output=/iris/u/jasonyan/slurm/%j_ph_cam_views.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_ph_cam_views.err

set -euo pipefail

REPO=/iris/u/jasonyan/repos/demonstration-information
DATASET="${DATASET:-/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image.hdf5}"
DEMO_IDX="${1:-0}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/camera_view_previews}"
OUT_DIR="${OUT_ROOT}/square_ph_demo_${DEMO_IDX}"

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

python scripts/quality/render_square_ph_custom_views.py \
  --dataset "${DATASET}" \
  --demo-idx "${DEMO_IDX}" \
  --output-dir "${OUT_DIR}"

echo "Wrote preview to ${OUT_DIR}/index.html"
