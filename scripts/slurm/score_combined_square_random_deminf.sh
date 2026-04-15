#!/bin/bash
# Score RoboMimic Square MH + collected random-init demos with combined DemInf checkpoints.
#
# Usage:
#   sbatch scripts/slurm/score_combined_square_random_deminf.sh wrist
#   sbatch scripts/slurm/score_combined_square_random_deminf.sh agent
#   sbatch scripts/slurm/score_combined_square_random_deminf.sh both
#   sbatch scripts/slurm/score_combined_square_random_deminf.sh all

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=score_deminf
#SBATCH --output=/iris/u/jasonyan/slurm/%j_score_deminf.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_score_deminf.err

set -euo pipefail

MODE="${1:-all}"
if [[ "${MODE}" != "wrist" && "${MODE}" != "agent" && "${MODE}" != "both" && "${MODE}" != "all" ]]; then
  echo "Expected mode to be one of: wrist, agent, both, all. Got: ${MODE}" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
CKPT_ROOT=/iris/u/jasonyan/data/deminf_outputs/combined_square_random_image
RANDOM_RLDS=/iris/u/jasonyan/data/random_square_post_rlds/robo_mimic/1.0.0
OUT_ROOT=/iris/u/jasonyan/data/deminf_outputs/combined_square_random_scores

ACTION_CKPT="${CKPT_ROOT}/combined_square_random_action_vae_seed1_20260414_193301"
WRIST_OBS_CKPT="${CKPT_ROOT}/combined_square_random_wrist_obs_vae_seed1_20260414_193937"
AGENT_OBS_CKPT="${CKPT_ROOT}/combined_square_random_agent_obs_vae_seed1_20260414_193224"
BOTH_OBS_CKPT="${BOTH_OBS_CKPT:-}"

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"

cd "${REPO}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

run_score() {
  local camera="$1"
  local obs_ckpt="$2"
  local output="${OUT_ROOT}/${camera}"

  python scripts/quality/estimate_quality_combined_robomimic.py \
    --obs_ckpt="${obs_ckpt}" \
    --action_ckpt="${ACTION_CKPT}" \
    --batch_size=1024 \
    --output="${output}" \
    --extra_dataset="random_square_post=${RANDOM_RLDS}"
}

latest_ckpt_dir() {
  local pattern="$1"
  find "${CKPT_ROOT}" -maxdepth 1 -type d -name "${pattern}" | sort | tail -n 1
}

if [[ "${MODE}" == "wrist" || "${MODE}" == "all" ]]; then
  run_score "wrist" "${WRIST_OBS_CKPT}"
fi

if [[ "${MODE}" == "agent" || "${MODE}" == "all" ]]; then
  run_score "agent" "${AGENT_OBS_CKPT}"
fi

if [[ "${MODE}" == "both" || "${MODE}" == "all" ]]; then
  if [[ -z "${BOTH_OBS_CKPT}" ]]; then
    BOTH_OBS_CKPT="$(latest_ckpt_dir 'combined_square_random_both_obs_vae_seed1_*')"
  fi
  if [[ -z "${BOTH_OBS_CKPT}" ]]; then
    echo "Could not find combined_square_random_both_obs_vae_seed1_* under ${CKPT_ROOT}" >&2
    exit 1
  fi
  run_score "both" "${BOTH_OBS_CKPT}"
fi
