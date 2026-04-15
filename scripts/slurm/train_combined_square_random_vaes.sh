#!/bin/bash
# Train combined RoboMimic Square MH + collected random-init observation/action VAEs.
#
# Usage:
#   sbatch scripts/slurm/train_combined_square_random_vaes.sh wrist
#   sbatch scripts/slurm/train_combined_square_random_vaes.sh agent
#   sbatch scripts/slurm/train_combined_square_random_vaes.sh both
#   sbatch scripts/slurm/train_combined_square_random_vaes.sh action
#   sbatch scripts/slurm/train_combined_square_random_vaes.sh all

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=combined_vaes
#SBATCH --output=/iris/u/jasonyan/slurm/%j_combined_vaes.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_combined_vaes.err

set -euo pipefail

MODE="${1:-all}"
if [[ "${MODE}" != "wrist" && "${MODE}" != "agent" && "${MODE}" != "both" && "${MODE}" != "action" && "${MODE}" != "all" ]]; then
  echo "Expected mode to be one of: wrist, agent, both, action, all. Got: ${MODE}" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
OUT=/iris/u/jasonyan/data/deminf_outputs/combined_square_random_image
MH_RLDS=/iris/u/jasonyan/data/robomimic_rlds_v2/robo_mimic/1.0.0
RANDOM_RLDS=/iris/u/jasonyan/data/random_square_post_rlds/robo_mimic/1.0.0

MH_WEIGHT=72707
RANDOM_WEIGHT=8004
DATASETS="square_mh=${MH_RLDS}@${MH_WEIGHT}::random_square_post=${RANDOM_RLDS}@${RANDOM_WEIGHT}"

mkdir -p /iris/u/jasonyan/slurm "${OUT}"

cd "${REPO}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

run_train() {
  local config_type="$1"
  local camera="$2"
  local name="$3"

  python scripts/train.py \
    --config="configs/quality/vae_robomimic_image.py:combined_square_random,${config_type},1,${camera},${DATASETS}" \
    --path "${OUT}" \
    --name "${name}"
}

if [[ "${MODE}" == "wrist" || "${MODE}" == "all" ]]; then
  run_train "s" "wrist" "combined_square_random_wrist_obs_vae_seed1"
fi

if [[ "${MODE}" == "agent" || "${MODE}" == "all" ]]; then
  run_train "s" "agent" "combined_square_random_agent_obs_vae_seed1"
fi

if [[ "${MODE}" == "both" || "${MODE}" == "all" ]]; then
  run_train "s" "both" "combined_square_random_both_obs_vae_seed1"
fi

if [[ "${MODE}" == "action" || "${MODE}" == "all" ]]; then
  # Camera is ignored for action-only training, but the config parser requires a valid value.
  run_train "a" "wrist" "combined_square_random_action_vae_seed1"
fi
