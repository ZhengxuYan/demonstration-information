#!/bin/bash
# Score smooth-target discrete BC policies for expert200 random-post policy views.
#
# Usage:
#   sbatch scripts/slurm/score_expert200_random_post_smooth_discrete_nll.sh agent_wrist
#   sbatch scripts/slurm/score_expert200_random_post_smooth_discrete_nll.sh left_close_low_wrist
#   sbatch scripts/slurm/score_expert200_random_post_smooth_discrete_nll.sh both

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=expert200_smooth_nll
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_expert200_smooth_nll.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_expert200_smooth_nll.err

set -euo pipefail

MODE="${1:-both}"
if [[ "${MODE}" != "agent_wrist" && "${MODE}" != "left_close_low_wrist" && "${MODE}" != "both" ]]; then
  echo "Usage: sbatch $0 agent_wrist|left_close_low_wrist|both" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
DATASET_ROOT="${DATASET_ROOT:-/iris/u/jasonyan/data/policy_view_experiments/expert200_random_post_bc}"
RUN_ROOT="${RUN_ROOT:-/iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/robomimic_policy_scores/expert200_random_post_bc}"

mkdir -p /iris/u/jasonyan/slurm "${SCORE_ROOT}"
cd "${REPO}"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"

score_one() {
  local view="$1"
  local dataset="$2"
  local run_name="expert200_random_post_bc_discrete_smooth_${view}_seed1"
  local run_dir="${RUN_ROOT}/${run_name}"
  local score_dir="${SCORE_ROOT}/${run_name}"

  mapfile -t checkpoints < <(find "${run_dir}" -path "*/models/*.pth" -type f | sort)
  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "No checkpoints found under ${run_dir}" >&2
    exit 1
  fi

  declare -A selected=()
  for ckpt in "${checkpoints[@]}"; do
    base="$(basename "${ckpt}")"
    if [[ "${base}" == *best*validation* || "${base}" == *valid*best* ]]; then
      selected[best_validation]="${ckpt}"
    fi
  done
  selected[final]="${checkpoints[-1]}"

  mkdir -p "${score_dir}"
  for label in "${!selected[@]}"; do
    ckpt="${selected[$label]}"
    echo "scoring ${run_name} ${label}: ${ckpt}"
    for split in train valid; do
      python scripts/quality/score_robomimic_policy_nll.py \
        --checkpoint "${ckpt}" \
        --dataset "${dataset}" \
        --output "${score_dir}" \
        --name "${run_name}_${label}_${split}" \
        --filter-key "${split}" \
        --batch-size "${BATCH_SIZE:-128}" \
        --num-workers "${NUM_WORKERS:-0}" \
        --discrete-loss-type soft_ce \
        --soft-sigma-bins "${SOFT_SIGMA_BINS:-1.5}" \
        --soft-truncate-bins "${SOFT_TRUNCATE_BINS:-6}"
    done
  done
}

if [[ "${MODE}" == "agent_wrist" || "${MODE}" == "both" ]]; then
  score_one agent_wrist "${DATASET_ROOT}/expert200_random_post_agent_wrist_image_abs.hdf5"
fi

if [[ "${MODE}" == "left_close_low_wrist" || "${MODE}" == "both" ]]; then
  score_one left_close_low_wrist "${DATASET_ROOT}/expert200_random_post_left_close_low_wrist_image_abs.hdf5"
fi
