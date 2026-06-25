#!/bin/bash
# Submit wrench-to-hook 2-fold density HDF5 export + model training jobs.

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
TASK_TAG="${TASK_TAG:-wrench_to_hook}"
RUN_PREFIX="${RUN_PREFIX:-wrench_to_hook}"
ACTION_SOURCE="${ACTION_SOURCE:-cartesian_velocity}"
ACTION_TARGET="${ACTION_TARGET:-single}"
ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-none}"
DATA_ROOT="${DATA_ROOT:-/iris/u/jasonyan/data}"
DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT}/wrench_to_hook_density_datasets}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/robomimic_outputs/wrench_to_hook_density}"
CONFIG_ROOT="${CONFIG_ROOT:-${DATA_ROOT}/wrench_to_hook_density_configs}"
DATASETS="${DATASETS:-0613_98 0615_96}"
FOLDS="${FOLDS:-0 1}"

WRENCH0613_RLDS="${WRENCH0613_RLDS:-${DATA_ROOT}/droid_wrench_to_hook_06132026_98_rlds/droid_pen_in_cup/1.0.0}"
WRENCH0615_RLDS="${WRENCH0615_RLDS:-${DATA_ROOT}/droid_wrench_on_hook_06152026_96_rlds/droid_pen_in_cup/1.0.0}"

submit_one_dataset() {
  local dataset_tag="$1"
  local rlds_path="$2"
  local hdf5_path="${DATASET_ROOT}/${TASK_TAG}_${dataset_tag}_${ACTION_TARGET}_${ACTION_SOURCE}_${ACTION_NORMALIZATION}.hdf5"

  if [[ ! -d "${rlds_path}" ]]; then
    echo "Missing RLDS path for ${dataset_tag}: ${rlds_path}" >&2
    return 2
  fi

  local prep_job
  prep_job="$(
    TASK_TAG="${TASK_TAG}" \
    DATASET_TAG="${dataset_tag}" \
    RLDS_PATH="${rlds_path}" \
    OUT_ROOT="${DATASET_ROOT}" \
    OUTPUT="${hdf5_path}" \
    ACTION_SOURCE="${ACTION_SOURCE}" \
    ACTION_TARGET="${ACTION_TARGET}" \
    ACTION_NORMALIZATION="${ACTION_NORMALIZATION}" \
    NUM_FOLDS=2 \
    sbatch --parsable "${REPO}/scripts/slurm/prepare_pen_in_cup_density_hdf5.sh"
  )"
  echo "${dataset_tag} prep_job=${prep_job}"

  for fold in ${FOLDS}; do
    local fold_tag="fold${fold}"
    local train_job
    train_job="$(
      TASK_TAG="${TASK_TAG}" \
      RUN_PREFIX="${RUN_PREFIX}" \
      DATASET_TAG="${dataset_tag}" \
      DATASET_HDF5="${hdf5_path}" \
      OUT_ROOT="${OUT_ROOT}" \
      CONFIG_ROOT="${CONFIG_ROOT}" \
      ACTION_SOURCE="${ACTION_SOURCE}" \
      ACTION_TARGET="${ACTION_TARGET}" \
      ACTION_NORMALIZATION="${ACTION_NORMALIZATION}" \
      FOLD_TAG="${fold_tag}" \
      TRAIN_FILTER_KEY="${fold_tag}_train" \
      VALID_FILTER_KEY="${fold_tag}_valid" \
      WANDB_PROJECT="wrench-to-hook-density" \
      RESUME=1 \
      sbatch --parsable --dependency="afterok:${prep_job}" --array=1-12%6 "${REPO}/scripts/slurm/train_pen_in_cup_density_models_array.sh"
    )"
    echo "${dataset_tag} ${fold_tag} train_job=${train_job}"
  done
}

for dataset in ${DATASETS}; do
  case "${dataset}" in
    0613_98)
      submit_one_dataset "0613_98" "${WRENCH0613_RLDS}"
      ;;
    0615_96)
      submit_one_dataset "0615_96" "${WRENCH0615_RLDS}"
      ;;
    *)
      echo "Unknown dataset ${dataset}; expected 0613_98 or 0615_96" >&2
      exit 2
      ;;
  esac
done
