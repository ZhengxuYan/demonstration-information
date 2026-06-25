#!/bin/bash
# Combine condition-level density scores into 8 scores and evaluate against labels.
#
# Example:
#   DATASET_TAG=0612_100 LABELS_CSV=/iris/u/.../pen_in_cup_06122026_100_observability_annotations.csv \
#   ALGO=gmm sbatch scripts/slurm/combine_eval_pen_in_cup_density_scores.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --job-name=pic_den_eval
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_pic_den_eval.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_pic_den_eval.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_TAG="${DATASET_TAG:?Set DATASET_TAG}"
LABELS_CSV="${LABELS_CSV:?Set LABELS_CSV}"
TASK_TAG="${TASK_TAG:-pen_in_cup}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/${TASK_TAG}_density_scores}"
EVAL_ROOT="${EVAL_ROOT:-/iris/u/jasonyan/data/${TASK_TAG}_density_eval}"
ACTION_SOURCE="${ACTION_SOURCE:-action}"
ACTION_TARGET="${ACTION_TARGET:-single}"
ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-none}"
ALGO="${ALGO:-gmm}"
CKPT_MODE="${CKPT_MODE:-best_validation}"
LABEL_COLUMN="${LABEL_COLUMN:-observability}"
HIGHER_IS_BETTER="${HIGHER_IS_BETTER:-0}"
RECIPE="${DATASET_TAG}/${ACTION_TARGET}_${ACTION_SOURCE}_${ACTION_NORMALIZATION}/${ALGO}/${CKPT_MODE}"
SCORE_DIR="${SCORE_ROOT}/${RECIPE}"
OUT_DIR="${EVAL_ROOT}/${RECIPE}"
CURVE_CSV="${OUT_DIR}/retained_${LABEL_COLUMN}_curves.csv"
CURVE_PNG="${OUT_DIR}/retained_${LABEL_COLUMN}_curves.png"

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

mkdir -p /iris/u/jasonyan/slurm "${OUT_DIR}"
cd "${REPO}"

python scripts/quality/combine_pen_in_cup_density_scores.py \
  --score-root "${SCORE_DIR}" \
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

echo "COMBINE_EVAL_PEN_IN_CUP_DENSITY_OK"
echo "out_dir=${OUT_DIR}"
