#!/bin/bash
# Merge 2-fold held-out density scores, combine 8 scores, and evaluate against labels.

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --job-name=den_2fold_eval
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_den_2fold_eval.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_den_2fold_eval.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_TAG="${DATASET_TAG:?Set DATASET_TAG}"
LABELS_CSV="${LABELS_CSV:?Set LABELS_CSV}"
TASK_TAG="${TASK_TAG:-wrench_to_hook}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/${TASK_TAG}_density_scores}"
EVAL_ROOT="${EVAL_ROOT:-/iris/u/jasonyan/data/${TASK_TAG}_density_eval}"
ACTION_SOURCE="${ACTION_SOURCE:-cartesian_velocity}"
ACTION_TARGET="${ACTION_TARGET:-single}"
ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-none}"
ALGO="${ALGO:-gmm}"
CKPT_MODE="${CKPT_MODE:-best_validation}"
LABEL_COLUMN="${LABEL_COLUMN:-observability}"
HIGHER_IS_BETTER="${HIGHER_IS_BETTER:-0}"
FOLDS_CSV="${FOLDS_CSV:-fold0,fold1}"

BASE_RECIPE="${DATASET_TAG}/${ACTION_TARGET}_${ACTION_SOURCE}_${ACTION_NORMALIZATION}"
MERGED_RECIPE="${BASE_RECIPE}/2fold/${ALGO}/${CKPT_MODE}"
MERGED_SCORE_DIR="${SCORE_ROOT}/${MERGED_RECIPE}"
OUT_DIR="${EVAL_ROOT}/${MERGED_RECIPE}"
CURVE_CSV="${OUT_DIR}/retained_${LABEL_COLUMN}_curves.csv"
CURVE_PNG="${OUT_DIR}/retained_${LABEL_COLUMN}_curves.png"

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

mkdir -p /iris/u/jasonyan/slurm "${MERGED_SCORE_DIR}" "${OUT_DIR}"
cd "${REPO}"

IFS=',' read -r -a folds <<< "${FOLDS_CSV}"
for condition in image_state image state action_prior; do
  merge_args=()
  for fold in "${folds[@]}"; do
    merge_args+=(--input "${SCORE_ROOT}/${BASE_RECIPE}/${fold}/${ALGO}/${CKPT_MODE}/${condition}.pkl")
  done
  python scripts/quality/merge_density_fold_score_pkls.py \
    "${merge_args[@]}" \
    --output "${MERGED_SCORE_DIR}/${condition}.pkl" \
    --csv-output "${MERGED_SCORE_DIR}/${condition}_trajectory_scores.csv"
done

python scripts/quality/combine_pen_in_cup_density_scores.py \
  --score-root "${MERGED_SCORE_DIR}" \
  --output-pkl "${OUT_DIR}/combined_8_scores.pkl" \
  --output-csv "${OUT_DIR}/combined_8_scores.csv"

python scripts/quality/evaluate_density_scores_against_labels.py \
  --scores-csv "${OUT_DIR}/combined_8_scores.csv" \
  --labels-csv "${LABELS_CSV}" \
  --output-csv "${CURVE_CSV}" \
  --output-png "${CURVE_PNG}" \
  --label-column "${LABEL_COLUMN}" \
  --ylabel "Average human ${LABEL_COLUMN} among retained episodes" \
  $([[ "${HIGHER_IS_BETTER}" == "1" ]] && echo "--higher-is-better" || echo "--no-higher-is-better")

echo "MERGE_COMBINE_EVAL_DENSITY_2FOLD_OK"
echo "out_dir=${OUT_DIR}"
