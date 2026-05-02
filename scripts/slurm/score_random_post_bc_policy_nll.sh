#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=random_post_nll
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_random_post_nll.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_random_post_nll.err

set -euo pipefail

ALGO="${1:-}"
VIEW="${2:-}"
if [[ "${ALGO}" != "gmm" && "${ALGO}" != "discrete" ]]; then
  echo "Usage: sbatch $0 gmm|discrete agent_wrist|left_close_low_wrist"
  exit 2
fi
if [[ "${VIEW}" != "agent_wrist" && "${VIEW}" != "left_close_low_wrist" ]]; then
  echo "Usage: sbatch $0 gmm|discrete agent_wrist|left_close_low_wrist"
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
RUN_NAME="random_post_bc_${ALGO}_${VIEW}_seed1"
DATASET="/iris/u/jasonyan/data/policy_view_experiments/random_post/random_post_${VIEW}_image.hdf5"
RUN_DIR="/iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments/${RUN_NAME}"
SCORE_DIR="/iris/u/jasonyan/data/robomimic_policy_scores/random_post_policy_view_bc/${RUN_NAME}"

mkdir -p /iris/u/jasonyan/slurm "${SCORE_DIR}"
cd "${REPO}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"

mapfile -t CHECKPOINTS < <(find "${RUN_DIR}" -path "*/models/*.pth" -type f | sort)
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No checkpoints found under ${RUN_DIR}"
  exit 1
fi

declare -A SELECTED=()
for ckpt in "${CHECKPOINTS[@]}"; do
  base="$(basename "${ckpt}")"
  if [[ "${base}" == *best*validation* || "${base}" == *valid*best* ]]; then
    SELECTED[best_validation]="${ckpt}"
  fi
done
SELECTED[final]="${CHECKPOINTS[-1]}"

for label in "${!SELECTED[@]}"; do
  ckpt="${SELECTED[$label]}"
  echo "scoring ${label}: ${ckpt}"
  for split in train valid; do
    python scripts/quality/score_robomimic_policy_nll.py \
      --checkpoint "${ckpt}" \
      --dataset "${DATASET}" \
      --output "${SCORE_DIR}" \
      --name "${RUN_NAME}_${label}_${split}" \
      --filter-key "${split}" \
      --batch-size "${BATCH_SIZE:-128}" \
      --num-workers "${NUM_WORKERS:-0}"
  done
done
