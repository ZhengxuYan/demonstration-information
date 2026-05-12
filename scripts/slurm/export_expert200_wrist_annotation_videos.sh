#!/bin/bash
# Export high-resolution expert200 wrist-view MP4s for annotation.
#
# Usage:
#   sbatch scripts/slurm/export_expert200_wrist_annotation_videos.sh
#   RES=336 sbatch scripts/slurm/export_expert200_wrist_annotation_videos.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=expert_wrist_vids
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_expert_wrist_vids.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_expert_wrist_vids.err

set -euo pipefail

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

REPO=/iris/u/jasonyan/repos/demonstration-information
SOURCE="${SOURCE:-/iris/u/jasonyan/data/policy_view_experiments/expert200/expert200_agent_wrist_image_abs.hdf5}"
FALLBACK="${FALLBACK:-/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image_abs.hdf5}"
RES="${RES:-224}"
FPS="${FPS:-20}"
OUT_DIR="${OUT_DIR:-/iris/u/jasonyan/data/observability_annotation_videos/expert200_wrist_${RES}}"

mkdir -p /iris/u/jasonyan/slurm "${OUT_DIR}"
cd "${REPO}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"

python scripts/quality/export_expert200_wrist_videos.py \
  --source "${SOURCE}" \
  --env-meta-fallback "${FALLBACK}" \
  --out-dir "${OUT_DIR}" \
  --height "${RES}" \
  --width "${RES}" \
  --fps "${FPS}"

find "${OUT_DIR}" -maxdepth 1 -type f -name 'demo_*.mp4' | wc -l
du -sh "${OUT_DIR}"
