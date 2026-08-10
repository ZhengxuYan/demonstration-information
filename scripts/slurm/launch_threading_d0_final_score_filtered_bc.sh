#!/bin/bash
# Submit 54 final-density-score Threading filtered-BC runs across safe GPU tiers.

set -euo pipefail

REPO="${REPO:-/iris/u/${USER}/repos/demonstration-information}"
ROOT="${ROOT:-/iris/u/${USER}/data/threading_d0_final_score_filtered_bc_20260808}"
RESULT_ROOT="${RESULT_ROOT:-/iris/u/${USER}/data/threading_d0_final_score_filtered_bc_results_20260808}"
SOURCE_HDF5="${SOURCE_HDF5:-/iris/u/${USER}/data/threading_d0_final200_abs_delta_20260730/hdf5/image_final200_joint_absolute_fixedobs_contiguous.hdf5}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/iris/u/${USER}/slurm}"
WORKER="${REPO}/scripts/slurm/train_threading_d0_final_score_filtered_bc_array.sh"
MANIFEST="${ROOT}/config_manifest.csv"
SELECTIONS="${ROOT}/score_selections.json"

if [[ ! -f "${ROOT}/PREP_COMPLETE.json" || ! -f "${MANIFEST}" || ! -f "${SELECTIONS}" ]]; then
  echo "preparation missing under ${ROOT}" >&2
  exit 1
fi

mkdir -p "${SLURM_LOG_DIR}" "${RESULT_ROOT}"

make_ids() {
  local offset="$1"
  local values=()
  local task
  for ((task=offset; task<=54; task+=3)); do values+=("${task}"); done
  local joined
  IFS=,; joined="${values[*]}"; unset IFS
  printf '%s' "${joined}"
}

COMMON_EXCLUDES="cocoflops-hgx-1,iliad-hgx-1,iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8,pasteur-hgx-1,pasteur-hgx-2,tiger-hgx-1,sphinx9,viscam9,pasteur6"
SAFE_CONSTRAINT="24G|40G|48G|96G|ampere|ada|blackwell|turing"

submit_tier() {
  local name="$1" account="$2" partition="$3" offset="$4" parallel="$5"
  local ids job_id
  ids="$(make_ids "${offset}")"
  job_id="$(
    sbatch --parsable \
      --account="${account}" --partition="${partition}" \
      --array="${ids}%${parallel}" --requeue \
      --time=48:00:00 --cpus-per-task=20 --mem=64G --gres=gpu:1 \
      --constraint="${SAFE_CONSTRAINT}" --exclude="${COMMON_EXCLUDES}" \
      --job-name="d0fbc_${name}" \
      --output="${SLURM_LOG_DIR}/%A_%a_d0fbc.out" \
      --error="${SLURM_LOG_DIR}/%A_%a_d0fbc.err" \
      --export="ALL,REPO=${REPO},ROOT=${ROOT},RESULT_ROOT=${RESULT_ROOT},SOURCE_HDF5=${SOURCE_HDF5},MANIFEST=${MANIFEST},SELECTIONS=${SELECTIONS},SLURM_LOG_DIR=${SLURM_LOG_DIR},EXPECTED_TASKS=54" \
      "${WORKER}"
  )"
  echo "${name}_job=${job_id} tasks=${ids} max_parallel=${parallel}"
}

# Submit in requested priority order. Use the available 6 + 8 non-preemptible
# quota first, then up to 13 low-priority preemptible jobs (27 total).
submit_tier iris_hi iris iris-hi 1 6
submit_tier iliad iliad iliad 2 8
submit_tier sc_loprio iliad sc-loprio 3 13

echo "expected_runs=54 checkpoints_saved=0 rollout_episodes=100 final_epoch=600"
echo "root=${ROOT} result_root=${RESULT_ROOT}"
