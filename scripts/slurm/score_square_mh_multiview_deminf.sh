#!/bin/bash
# Score Square MH-only demos with fused multi-view DemInf checkpoints.
#
# Usage:
#   sbatch scripts/slurm/score_square_mh_multiview_deminf.sh
#
# Optional overrides:
#   BOTH_OBS_CKPT=/path/to/obs_ckpt ACTION_CKPT=/path/to/action_ckpt sbatch ...

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=square_mh_mv_score
#SBATCH --output=/iris/u/jasonyan/slurm/%j_square_mh_mv_score.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_square_mh_mv_score.err

set -euo pipefail

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
CKPT_ROOT=/iris/u/jasonyan/data/deminf_outputs/square_mh_multiview_image
OUT_ROOT=/iris/u/jasonyan/data/deminf_outputs/square_mh_multiview_scores

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"

cd "${REPO}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

latest_ckpt_dir() {
  local pattern="$1"
  find "${CKPT_ROOT}" -maxdepth 1 -type d -name "${pattern}" | sort | tail -n 1
}

BOTH_OBS_CKPT="${BOTH_OBS_CKPT:-$(latest_ckpt_dir 'square_mh_both_obs_vae_seed1_*')}"
ACTION_CKPT="${ACTION_CKPT:-$(latest_ckpt_dir 'square_mh_action_vae_seed1_*')}"
DISTANCE_DIAGNOSTICS="${DISTANCE_DIAGNOSTICS:-0}"

if [[ -z "${BOTH_OBS_CKPT}" ]]; then
  echo "Could not find square_mh_both_obs_vae_seed1_* under ${CKPT_ROOT}" >&2
  exit 1
fi

if [[ -z "${ACTION_CKPT}" ]]; then
  echo "Could not find square_mh_action_vae_seed1_* under ${CKPT_ROOT}" >&2
  exit 1
fi

EXTRA_FLAGS=()
if [[ "${DISTANCE_DIAGNOSTICS}" == "1" ]]; then
  EXTRA_FLAGS+=(--distance_diagnostics)
fi

python scripts/quality/estimate_quality_combined_robomimic.py \
  --obs_ckpt="${BOTH_OBS_CKPT}" \
  --action_ckpt="${ACTION_CKPT}" \
  --batch_size=1024 \
  --output="${OUT_ROOT}/both" \
  "${EXTRA_FLAGS[@]}"
