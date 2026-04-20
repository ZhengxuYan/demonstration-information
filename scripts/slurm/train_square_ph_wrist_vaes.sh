#!/bin/bash
# Train Square PH wrist-view VAEs for DemInf.
#
# Usage:
#   sbatch scripts/slurm/train_square_ph_wrist_vaes.sh image
#   sbatch scripts/slurm/train_square_ph_wrist_vaes.sh image_proprio
#   sbatch scripts/slurm/train_square_ph_wrist_vaes.sh action
#   sbatch scripts/slurm/train_square_ph_wrist_vaes.sh all

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=square_ph_wrist_vae
#SBATCH --output=/iris/u/jasonyan/slurm/%j_square_ph_wrist_vae.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_square_ph_wrist_vae.err

set -euo pipefail

MODE="${1:-all}"
if [[ "${MODE}" != "image" && "${MODE}" != "image_proprio" && "${MODE}" != "action" && "${MODE}" != "all" ]]; then
  echo "Expected mode to be one of: image, image_proprio, action, all. Got: ${MODE}" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
PH_RLDS="${PH_RLDS:-/iris/u/jasonyan/data/robomimic_square_ph_rlds/robo_mimic/1.0.0}"
OUT=/iris/u/jasonyan/data/deminf_outputs/square_ph_wrist_image

mkdir -p /iris/u/jasonyan/slurm "${OUT}"

cd "${REPO}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

run_train() {
  local config_type="$1"
  local name="$2"

  python scripts/train.py \
    --config="configs/quality/vae_robomimic_image.py:square/ph,${config_type},1,wrist,square_ph=${PH_RLDS}" \
    --path "${OUT}" \
    --name "${name}"
}

if [[ "${MODE}" == "image" || "${MODE}" == "all" ]]; then
  run_train "i" "square_ph_wrist_image_only_obs_vae_seed1"
fi

if [[ "${MODE}" == "image_proprio" || "${MODE}" == "all" ]]; then
  run_train "s" "square_ph_wrist_image_proprio_obs_vae_seed1"
fi

if [[ "${MODE}" == "action" || "${MODE}" == "all" ]]; then
  run_train "a" "square_ph_action_vae_seed1"
fi
