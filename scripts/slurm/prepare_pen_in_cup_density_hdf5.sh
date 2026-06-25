#!/bin/bash
# Export one pen-in-cup RLDS builder directory to robomimic HDF5 for density sweeps.
#
# Example:
#   DATASET_TAG=0610_89 \
#   RLDS_PATH=/iris/u/jasonyan/data/droid_pen_in_cup_06102026_89_rlds/droid_pen_in_cup/1.0.0 \
#   sbatch scripts/slurm/prepare_pen_in_cup_density_hdf5.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --job-name=pic_density_h5
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_pic_density_h5.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_pic_density_h5.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_TAG="${DATASET_TAG:?Set DATASET_TAG, e.g. 0610_89 or 0612_100}"
RLDS_PATH="${RLDS_PATH:?Set RLDS_PATH to droid_pen_in_cup/.../1.0.0}"
TASK_TAG="${TASK_TAG:-pen_in_cup}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/${TASK_TAG}_density_datasets}"
ACTION_SOURCE="${ACTION_SOURCE:-action}"
ACTION_TARGET="${ACTION_TARGET:-single}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-none}"
ACTION_BOUND_LOW_PERCENTILE="${ACTION_BOUND_LOW_PERCENTILE:-1}"
ACTION_BOUND_HIGH_PERCENTILE="${ACTION_BOUND_HIGH_PERCENTILE:-99}"
VALID_RATIO="${VALID_RATIO:-0.1}"
NUM_FOLDS="${NUM_FOLDS:-0}"
FOLD_VALID_RATIO="${FOLD_VALID_RATIO:-${VALID_RATIO}}"
SEED="${SEED:-1}"
ENV_NAME="${ENV_NAME:-${TASK_TAG}_density}"
OUTPUT="${OUTPUT:-${OUT_ROOT}/${TASK_TAG}_${DATASET_TAG}_${ACTION_TARGET}_${ACTION_SOURCE}_${ACTION_NORMALIZATION}.hdf5}"

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "rlds_path=${RLDS_PATH}"
echo "output=${OUTPUT}"
echo "task_tag=${TASK_TAG}"
echo "action_source=${ACTION_SOURCE}"
echo "action_target=${ACTION_TARGET}"
echo "action_normalization=${ACTION_NORMALIZATION}"
echo "action_bound_low_percentile=${ACTION_BOUND_LOW_PERCENTILE}"
echo "action_bound_high_percentile=${ACTION_BOUND_HIGH_PERCENTILE}"
echo "num_folds=${NUM_FOLDS}"
echo "fold_valid_ratio=${FOLD_VALID_RATIO}"

python scripts/quality/export_droid_rlds_to_robomimic_density_hdf5.py \
  --rlds-path "${RLDS_PATH}" \
  --output "${OUTPUT}" \
  --action-source "${ACTION_SOURCE}" \
  --action-target "${ACTION_TARGET}" \
  --chunk-size "${CHUNK_SIZE}" \
  --action-normalization "${ACTION_NORMALIZATION}" \
  --action-bound-low-percentile "${ACTION_BOUND_LOW_PERCENTILE}" \
  --action-bound-high-percentile "${ACTION_BOUND_HIGH_PERCENTILE}" \
  --env-name "${ENV_NAME}" \
  --valid-ratio "${VALID_RATIO}" \
  --num-folds "${NUM_FOLDS}" \
  --fold-valid-ratio "${FOLD_VALID_RATIO}" \
  --seed "${SEED}" \
  --overwrite

echo "PREP_PEN_IN_CUP_DENSITY_HDF5_OK"
