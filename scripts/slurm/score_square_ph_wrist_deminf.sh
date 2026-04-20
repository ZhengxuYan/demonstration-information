#!/bin/bash
# Score Square PH demos with wrist-view DemInf checkpoints.
#
# Usage:
#   sbatch scripts/slurm/score_square_ph_wrist_deminf.sh image
#   sbatch scripts/slurm/score_square_ph_wrist_deminf.sh image_proprio
#   sbatch scripts/slurm/score_square_ph_wrist_deminf.sh all
#
# Optional overrides:
#   IMAGE_OBS_CKPT=/path IMAGE_PROPRIO_OBS_CKPT=/path ACTION_CKPT=/path sbatch ...
#   DISTANCE_DIAGNOSTICS=1 sbatch ...

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=square_ph_wrist_score
#SBATCH --output=/iris/u/jasonyan/slurm/%j_square_ph_wrist_score.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_square_ph_wrist_score.err

set -euo pipefail

MODE="${1:-all}"
if [[ "${MODE}" != "image" && "${MODE}" != "image_proprio" && "${MODE}" != "all" ]]; then
  echo "Expected mode to be one of: image, image_proprio, all. Got: ${MODE}" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
PH_RLDS="${PH_RLDS:-/iris/u/jasonyan/data/robomimic_square_ph_rlds/robo_mimic/1.0.0}"
CKPT_ROOT=/iris/u/jasonyan/data/deminf_outputs/square_ph_wrist_image
OUT_ROOT=/iris/u/jasonyan/data/deminf_outputs/square_ph_wrist_scores

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"

cd "${REPO}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

latest_ckpt_dir() {
  local pattern="$1"
  find "${CKPT_ROOT}" -maxdepth 1 -type d -name "${pattern}" | sort | tail -n 1
}

IMAGE_OBS_CKPT="${IMAGE_OBS_CKPT:-$(latest_ckpt_dir 'square_ph_wrist_image_only_obs_vae_seed1_*')}"
IMAGE_PROPRIO_OBS_CKPT="${IMAGE_PROPRIO_OBS_CKPT:-$(latest_ckpt_dir 'square_ph_wrist_image_proprio_obs_vae_seed1_*')}"
ACTION_CKPT="${ACTION_CKPT:-$(latest_ckpt_dir 'square_ph_action_vae_seed1_*')}"
DISTANCE_DIAGNOSTICS="${DISTANCE_DIAGNOSTICS:-0}"

if [[ -z "${ACTION_CKPT}" ]]; then
  echo "Could not find square_ph_action_vae_seed1_* under ${CKPT_ROOT}" >&2
  exit 1
fi

EXTRA_FLAGS=()
if [[ "${DISTANCE_DIAGNOSTICS}" == "1" ]]; then
  EXTRA_FLAGS+=(--distance_diagnostics)
fi

run_score() {
  local obs_ckpt="$1"
  local out_dir="$2"
  local label="$3"

  if [[ -z "${obs_ckpt}" ]]; then
    echo "Could not find ${label} checkpoint under ${CKPT_ROOT}" >&2
    exit 1
  fi

  python scripts/quality/estimate_quality_combined_robomimic.py \
    --obs_ckpt="${obs_ckpt}" \
    --action_ckpt="${ACTION_CKPT}" \
    --square_dataset_name="square_ph" \
    --square_path_override="${PH_RLDS}" \
    --batch_size=1024 \
    --output="${OUT_ROOT}/${out_dir}" \
    "${EXTRA_FLAGS[@]}"
}

if [[ "${MODE}" == "image" || "${MODE}" == "all" ]]; then
  run_score "${IMAGE_OBS_CKPT}" "image_only" "square_ph_wrist_image_only_obs_vae_seed1_*"
fi

if [[ "${MODE}" == "image_proprio" || "${MODE}" == "all" ]]; then
  run_score "${IMAGE_PROPRIO_OBS_CKPT}" "image_proprio" "square_ph_wrist_image_proprio_obs_vae_seed1_*"
fi
