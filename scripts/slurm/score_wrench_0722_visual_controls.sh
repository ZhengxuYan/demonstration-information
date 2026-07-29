#!/bin/bash
# Score proprio, single-camera, and dual-camera GMM controls for one regime.

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_HDF5="${DATASET_HDF5:?}"
MANIFEST="${MANIFEST:?}"
LABELS_CSV="${LABELS_CSV:?}"
CONTROL_OUT_ROOT="${CONTROL_OUT_ROOT:?}"
CONTROL_RUN_PREFIX="${CONTROL_RUN_PREFIX:?}"
DATASET_TAG="${DATASET_TAG:?}"
SCORE_ROOT="${SCORE_ROOT:?}"
REGIME="${REGIME:?}"
FOLD_TAG="${FOLD_TAG:-}"
FILTER_KEY="${FILTER_KEY:?}"
SEED="${SEED:-20260725}"
SNAPSHOT="${SNAPSHOT:-/iris/u/jasonyan/data/wrench_on_hook_0722_pomdp/snapshot_20260727_0700_cst/checkpoints.csv}"

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx
set -u

cd "${REPO}"

snapshot_checkpoint() {
  local condition="$1"
  python3 - "${SNAPSHOT}" "${REGIME}" "${condition}" <<'PY'
import pandas as pd
import sys
path, regime, condition = sys.argv[1:]
frame = pd.read_csv(path)
row = frame[
    (frame.experiment == "gmm_1e-2")
    & (frame.algo == "gmm")
    & (frame.regime == regime)
    & (frame.condition == condition)
    & (frame["mode"] == "best_validation")
]
if len(row) != 1:
    raise SystemExit(f"Expected one frozen checkpoint, got {len(row)}")
print(row.iloc[0].checkpoint)
PY
}

run_dir() {
  local condition="$1"
  local middle="single_image_proprio_none"
  if [[ -n "${FOLD_TAG}" ]]; then
    middle+="_${FOLD_TAG}"
  fi
  printf '%s/%s_%s_%s_gmm_%s_seed1' \
    "${CONTROL_OUT_ROOT}" "${CONTROL_RUN_PREFIX}" "${DATASET_TAG}" "${middle}" "${condition}"
}

PRIOR_CKPT="$(snapshot_checkpoint action_prior)"
BOTH_CKPT="$(snapshot_checkpoint image_proprio_euler)"
CONDITIONS=(proprio_euler exterior_proprio_euler wrist_proprio_euler image_proprio_euler)

for condition in "${CONDITIONS[@]}"; do
  if [[ "${condition}" == "image_proprio_euler" ]]; then
    COND_CKPT="${BOTH_CKPT}"
  else
    COND_CKPT="$(python3 scripts/quality/select_robomimic_checkpoint.py \
      --run-dir "$(run_dir "${condition}")" --mode best_validation \
      --max-epoch 100)"
  fi
  condition_root="${SCORE_ROOT}/${condition}"
  REPO="${REPO}" \
  DATASET_TAG="${DATASET_TAG}" \
  DATASET_HDF5="${DATASET_HDF5}" \
  SCORE_ROOT="${condition_root}" \
  ALGO=gmm \
  REGIME="${REGIME}" \
  FOLD_TAG="${FOLD_TAG}" \
  FILTER_KEY="${FILTER_KEY}" \
  ACTION_SOURCE=image_proprio \
  CONDITIONAL_CONDITION="${condition}" \
  COND_CKPT="${COND_CKPT}" \
  PRIOR_CKPT="${PRIOR_CKPT}" \
  M=16 K=512 SEED="${SEED}" \
  bash scripts/slurm/score_threading_pomdp_6.sh
done

counterfactual_root="${SCORE_ROOT}/image_counterfactual/${REGIME}"
if [[ -n "${FOLD_TAG}" ]]; then
  counterfactual_root="${SCORE_ROOT}/image_counterfactual/${FOLD_TAG}"
fi
cd "${REPO}/robomimic"
export USE_FLAX=0
export PYTHONPATH="${PWD}:${REPO}/scripts/quality:${PYTHONPATH:-}"
if [[ ! -s "${counterfactual_root}/episode_image_counterfactuals.csv" ]]; then
  python "${REPO}/scripts/quality/score_wrench_0722_image_counterfactuals.py" \
    --checkpoint "${BOTH_CKPT}" \
    --dataset "${DATASET_HDF5}" \
    --manifest "${MANIFEST}" \
    --filter-key "${FILTER_KEY}" \
    --output "${counterfactual_root}" \
    --seed "${SEED}"
fi

marker="${SCORE_ROOT}/gmm/${REGIME}"
mkdir -p "${marker}"
if [[ -n "${FOLD_TAG}" ]]; then
  marker="${SCORE_ROOT}/gmm/${FOLD_TAG}"
  mkdir -p "${marker}"
fi
cp "${SCORE_ROOT}/image_proprio_euler/gmm/${REGIME}/threading_pomdp_6_scores.csv" \
  "${marker}/threading_pomdp_6_scores.csv"
if [[ -n "${FOLD_TAG}" ]]; then
  cp "${SCORE_ROOT}/image_proprio_euler/gmm/${FOLD_TAG}/threading_pomdp_6_scores.csv" \
    "${marker}/threading_pomdp_6_scores.csv"
fi
