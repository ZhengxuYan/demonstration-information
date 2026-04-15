#!/bin/bash
# Train Square MH-only VAEs for fused multi-view DemInf.
#
# Usage:
#   sbatch scripts/slurm/train_square_mh_multiview_vaes.sh both
#   sbatch scripts/slurm/train_square_mh_multiview_vaes.sh action
#   sbatch scripts/slurm/train_square_mh_multiview_vaes.sh all

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=square_mh_mv_vae
#SBATCH --output=/iris/u/jasonyan/slurm/%j_square_mh_mv_vae.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_square_mh_mv_vae.err

set -euo pipefail

MODE="${1:-all}"
if [[ "${MODE}" != "both" && "${MODE}" != "action" && "${MODE}" != "all" ]]; then
  echo "Expected mode to be one of: both, action, all. Got: ${MODE}" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
OUT=/iris/u/jasonyan/data/deminf_outputs/square_mh_multiview_image

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
    --config="configs/quality/vae_robomimic_image.py:square/mh,${config_type},1,${camera}" \
    --path "${OUT}" \
    --name "${name}"
}

if [[ "${MODE}" == "both" || "${MODE}" == "all" ]]; then
  run_train "s" "both" "square_mh_both_obs_vae_seed1"
fi

if [[ "${MODE}" == "action" || "${MODE}" == "all" ]]; then
  # Camera is ignored for action-only training, but the config parser requires a valid value.
  run_train "a" "wrist" "square_mh_action_vae_seed1"
fi
