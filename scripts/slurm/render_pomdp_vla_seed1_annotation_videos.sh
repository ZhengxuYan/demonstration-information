#!/bin/bash
# Render side-by-side left_close_low + wrist videos for annotating seed1 rollouts.
#
# Usage:
#   sbatch scripts/slurm/render_pomdp_vla_seed1_annotation_videos.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=pvla_annot_vids
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_pvla_annot_vids.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_pvla_annot_vids.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-robodiff}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
INPUT_HDF5="${INPUT_HDF5:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_raw/low_dim_bc_gmm_seed1_success200.hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/iris/u/jasonyan/data/pomdp_vla_seed1_annotation}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
FPS="${FPS:-20}"
VIDEO_SKIP="${VIDEO_SKIP:-1}"

mkdir -p /iris/u/jasonyan/slurm "${OUTPUT_ROOT}"
cd "${REPO}"

python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "hostname=$(hostname)"
echo "input_hdf5=${INPUT_HDF5}"
echo "output_root=${OUTPUT_ROOT}"
echo "height=${HEIGHT}"
echo "width=${WIDTH}"
echo "fps=${FPS}"
echo "video_skip=${VIDEO_SKIP}"

python scripts/quality/render_pomdp_vla_annotation_videos.py \
  --input-hdf5 "${INPUT_HDF5}" \
  --output-root "${OUTPUT_ROOT}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --fps "${FPS}" \
  --video-skip "${VIDEO_SKIP}" \
  --max-demos 200 \
  --overwrite

find "${OUTPUT_ROOT}/videos" -maxdepth 1 -type f -name '*.mp4' | wc -l
du -sh "${OUTPUT_ROOT}"
