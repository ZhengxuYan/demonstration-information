#!/bin/bash
# Score all episodes in the random-view Square HDF5 with trained DemInf VAEs.
#
# This intentionally scores the HDF5 directly instead of the RLDS train split,
# because Tian requested scores for all episodes.
#
# Usage:
#   sbatch scripts/slurm/score_square_randomviews_thirdperson1_hdf5_ksg.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=randview_ksg
#SBATCH --output=/iris/u/jasonyan/slurm/%j_randview_ksg.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_randview_ksg.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_NAME="${DATASET_NAME:-square_1400_rollouts_randomviews_thirdperson1}"
DATASET_HDF5="${DATASET_HDF5:-/iris/u/jasonyan/data/${DATASET_NAME}/image.hdf5}"
VAE_ROOT="${VAE_ROOT:-/iris/u/jasonyan/data/deminf_outputs/${DATASET_NAME}_original_pipeline}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/deminf_outputs/${DATASET_NAME}_original_pipeline_scores}"
SEED="${SEED:-1}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-1400}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

OBS_CKPT="${VAE_ROOT}/config-vae_robomimic_image_env-${DATASET_NAME}_type-s_seed-${SEED}/100000"
ACTION_CKPT="${VAE_ROOT}/config-vae_robomimic_image_env-${DATASET_NAME}_type-a_seed-${SEED}/50000"
OUT_DIR="${SCORE_ROOT}/${DATASET_NAME}/ksg/seed-${SEED}"
SCORE_PKL="${OUT_DIR}/${DATASET_NAME}.pkl"
SCORE_CSV="${OUT_DIR}/${DATASET_NAME}_scores_sorted_high_to_low.csv"

if [[ ! -d "${OBS_CKPT}" ]]; then
  echo "missing obs checkpoint: ${OBS_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${ACTION_CKPT}" ]]; then
  echo "missing action checkpoint: ${ACTION_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_HDF5}" ]]; then
  echo "missing dataset hdf5: ${DATASET_HDF5}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${OUT_DIR}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "dataset=${DATASET_NAME}"
echo "dataset_hdf5=${DATASET_HDF5}"
echo "obs_ckpt=${OBS_CKPT}"
echo "action_ckpt=${ACTION_CKPT}"
echo "out_dir=${OUT_DIR}"

python scripts/quality/score_robomimic_hdf5.py \
  --obs_ckpt "${OBS_CKPT}" \
  --action_ckpt "${ACTION_CKPT}" \
  --dataset "${DATASET_NAME}=0=${DATASET_HDF5}" \
  --batch_size "${BATCH_SIZE}" \
  --camera agent \
  --output "${OUT_DIR}"

python scripts/quality/export_deminf_scores_csv.py \
  --score-pkl "${SCORE_PKL}" \
  --output-csv "${SCORE_CSV}" \
  --expected-episodes "${EXPECTED_EPISODES}"

echo "${SCORE_CSV}"
