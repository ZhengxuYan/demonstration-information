#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=ph_cam_views
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_ph_cam_views.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_ph_cam_views.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET="${DATASET:-/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image.hdf5}"
ANNOTATIONS_CSV="${ANNOTATIONS_CSV:-${REPO}/observability_annotations.csv}"
ANNOTATION_DATASET="${ANNOTATION_DATASET:-square_ph}"
DEMOS_PER_LABEL="${DEMOS_PER_LABEL:-5}"
NUM_CANDIDATE_VIEWS="${NUM_CANDIDATE_VIEWS:-50}"
MAX_FRAMES="${MAX_FRAMES:-120}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/camera_view_previews}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/square_ph_${NUM_CANDIDATE_VIEWS}_views}"

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-robodiff}"

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"

python scripts/quality/render_square_ph_custom_views.py \
  --dataset "${DATASET}" \
  --annotations-csv "${ANNOTATIONS_CSV}" \
  --annotation-dataset "${ANNOTATION_DATASET}" \
  --demos-per-label "${DEMOS_PER_LABEL}" \
  --output-dir "${OUT_DIR}" \
  --num-candidate-views "${NUM_CANDIDATE_VIEWS}" \
  --max-frames "${MAX_FRAMES}"

echo "Wrote preview to ${OUT_DIR}/index.html"
echo "Wrote poses to ${OUT_DIR}/camera_views.json and ${OUT_DIR}/camera_views.csv"
