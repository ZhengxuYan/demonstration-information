#!/bin/bash
# Submit wrench-to-hook density scoring and score-vs-label evaluation jobs.
#
# Requires trained runs from launch_wrench_to_hook_density_0613_0615.sh.
# Label CSVs should exist on the cluster; override LABELS_0613 / LABELS_0615.

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
TASK_TAG="${TASK_TAG:-wrench_to_hook}"
RUN_PREFIX="${RUN_PREFIX:-wrench_to_hook}"
ACTION_TARGET="${ACTION_TARGET:-single}"
ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-bounded_minmax}"
DATA_ROOT="${DATA_ROOT:-/iris/u/jasonyan/data}"
DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT}/wrench_to_hook_density_datasets}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/robomimic_outputs/wrench_to_hook_density}"
SCORE_ROOT="${SCORE_ROOT:-${DATA_ROOT}/wrench_to_hook_density_scores}"
EVAL_ROOT="${EVAL_ROOT:-${DATA_ROOT}/wrench_to_hook_density_eval}"
CKPT_MODE="${CKPT_MODE:-best_validation}"
LABELS_0613="${LABELS_0613:-${DATA_ROOT}/wrench_to_hook_06132026_annotations_renumbered_98.csv}"
LABELS_0615="${LABELS_0615:-${DATA_ROOT}/wrench_on_hook_06152026_annotations.csv}"
ALGOS_CSV="${ALGOS:-gaussian,gmm,discrete}"
DATASETS="${DATASETS:-0613_98 0615_96}"

submit_one_dataset() {
  local dataset_tag="$1"
  local labels_csv="$2"
  local hdf5_path="${DATASET_ROOT}/${TASK_TAG}_${dataset_tag}_${ACTION_TARGET}_${ACTION_NORMALIZATION}.hdf5"

  if [[ ! -f "${hdf5_path}" ]]; then
    echo "Missing density HDF5 for ${dataset_tag}: ${hdf5_path}" >&2
    return 2
  fi
  if [[ ! -f "${labels_csv}" ]]; then
    echo "Missing labels CSV for ${dataset_tag}: ${labels_csv}" >&2
    return 2
  fi

  local score_job
  score_job="$(
    TASK_TAG="${TASK_TAG}" \
    RUN_PREFIX="${RUN_PREFIX}" \
    DATASET_TAG="${dataset_tag}" \
    DATASET_HDF5="${hdf5_path}" \
    OUT_ROOT="${OUT_ROOT}" \
    SCORE_ROOT="${SCORE_ROOT}" \
    ACTION_TARGET="${ACTION_TARGET}" \
    ACTION_NORMALIZATION="${ACTION_NORMALIZATION}" \
    CKPT_MODE="${CKPT_MODE}" \
    sbatch --parsable --array=1-12%6 "${REPO}/scripts/slurm/score_pen_in_cup_density_models_array.sh"
  )"
  echo "${dataset_tag} score_job=${score_job}"

  IFS=',' read -r -a algos <<< "${ALGOS_CSV}"
  for algo in "${algos[@]}"; do
    for label_column in observability optimality; do
      TASK_TAG="${TASK_TAG}" \
      DATASET_TAG="${dataset_tag}" \
      LABELS_CSV="${labels_csv}" \
      SCORE_ROOT="${SCORE_ROOT}" \
      EVAL_ROOT="${EVAL_ROOT}" \
      ACTION_TARGET="${ACTION_TARGET}" \
      ACTION_NORMALIZATION="${ACTION_NORMALIZATION}" \
      ALGO="${algo}" \
      CKPT_MODE="${CKPT_MODE}" \
      LABEL_COLUMN="${label_column}" \
      HIGHER_IS_BETTER=0 \
      sbatch --dependency="afterok:${score_job}" "${REPO}/scripts/slurm/combine_eval_pen_in_cup_density_scores.sh"
    done
  done
}

for dataset in ${DATASETS}; do
  case "${dataset}" in
    0613_98)
      submit_one_dataset "0613_98" "${LABELS_0613}"
      ;;
    0615_96)
      submit_one_dataset "0615_96" "${LABELS_0615}"
      ;;
    *)
      echo "Unknown dataset ${dataset}; expected 0613_98 or 0615_96" >&2
      exit 2
      ;;
  esac
done
