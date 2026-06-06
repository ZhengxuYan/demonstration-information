#!/bin/bash
# Copy random-view DemInf CSV scores and VAE checkpoints to Tian-visible /scr.
#
# Usage:
#   bash scripts/slurm/upload_square_randomviews_thirdperson1_deminf_outputs.sh

set -euo pipefail

DATASET_NAME="${DATASET_NAME:-square_1400_rollouts_randomviews_thirdperson1}"
VAE_ROOT="${VAE_ROOT:-/iris/u/jasonyan/data/deminf_outputs/${DATASET_NAME}_original_pipeline}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/deminf_outputs/${DATASET_NAME}_original_pipeline_scores}"
SEED="${SEED:-1}"
DEST_ROOT="${DEST_ROOT:-/scr/tiangao/deminf_outputs/${DATASET_NAME}}"

OBS_CKPT="${VAE_ROOT}/config-vae_robomimic_image_env-${DATASET_NAME}_type-s_seed-${SEED}/100000"
ACTION_CKPT="${VAE_ROOT}/config-vae_robomimic_image_env-${DATASET_NAME}_type-a_seed-${SEED}/50000"
SCORE_DIR="${SCORE_ROOT}/${DATASET_NAME}/ksg/seed-${SEED}"
SCORE_CSV="${SCORE_DIR}/${DATASET_NAME}_scores_sorted_high_to_low.csv"

for path in "${OBS_CKPT}" "${ACTION_CKPT}" "${SCORE_CSV}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required output: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${DEST_ROOT}/checkpoints" "${DEST_ROOT}/scores"
rsync -ah --delete "${OBS_CKPT}/" "${DEST_ROOT}/checkpoints/obs_vae_100000/"
rsync -ah --delete "${ACTION_CKPT}/" "${DEST_ROOT}/checkpoints/action_vae_50000/"
rsync -ah "${SCORE_DIR}/" "${DEST_ROOT}/scores/"

echo "uploaded to ${DEST_ROOT}"
find "${DEST_ROOT}" -maxdepth 3 -type f | sort | head -100
