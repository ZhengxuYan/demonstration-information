#!/bin/bash
# Score trained pen-in-cup robomimic density models.
#
# Defaults match train_pen_in_cup_density_models_array.sh. This writes one
# <condition>.pkl per algo / recipe directory, suitable for
# combine_pen_in_cup_density_scores.py.

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=pic_den_score
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8,iliad1,iliad2,iliad3,iliad4
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_pic_den_score.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_pic_den_score.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_TAG="${DATASET_TAG:?Set DATASET_TAG, e.g. 0610_89 or 0612_100}"
DATASET_HDF5="${DATASET_HDF5:?Set DATASET_HDF5}"
TASK_TAG="${TASK_TAG:-pen_in_cup}"
RUN_PREFIX="${RUN_PREFIX:-${TASK_TAG}}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/robomimic_outputs/${TASK_TAG}_density}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/${TASK_TAG}_density_scores}"
ACTION_SOURCE="${ACTION_SOURCE:-action}"
ACTION_TARGET="${ACTION_TARGET:-single}"
ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-none}"
FOLD_TAG="${FOLD_TAG:-}"
ALGOS_CSV="${ALGOS:-gaussian,gmm,discrete}"
CONDITIONS_CSV="${CONDITIONS:-image_state,image,state,action_prior}"
CKPT_MODE="${CKPT_MODE:-best_validation}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GMM_ENTROPY_SAMPLES="${GMM_ENTROPY_SAMPLES:-128}"
SCORE_FILTER_KEY="${SCORE_FILTER_KEY:-}"

IFS=',' read -r -a ALGOS_ARR <<< "${ALGOS_CSV}"
IFS=',' read -r -a CONDITIONS_ARR <<< "${CONDITIONS_CSV}"
TOTAL=$(( ${#ALGOS_ARR[@]} * ${#CONDITIONS_ARR[@]} ))
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK_ID < 1 || TASK_ID > TOTAL )); then
  echo "Task ${TASK_ID} outside range 1-${TOTAL}; ALGOS=${ALGOS_CSV} CONDITIONS=${CONDITIONS_CSV}" >&2
  exit 2
fi

ZERO=$((TASK_ID - 1))
ALGO_INDEX=$((ZERO / ${#CONDITIONS_ARR[@]}))
COND_INDEX=$((ZERO % ${#CONDITIONS_ARR[@]}))
ALGO="${ALGOS_ARR[$ALGO_INDEX]}"
CONDITION="${CONDITIONS_ARR[$COND_INDEX]}"

RUN_MIDDLE="${ACTION_TARGET}_${ACTION_SOURCE}_${ACTION_NORMALIZATION}"
RECIPE_BASE="${ACTION_TARGET}_${ACTION_SOURCE}_${ACTION_NORMALIZATION}"
if [[ -n "${FOLD_TAG}" ]]; then
  RUN_MIDDLE="${RUN_MIDDLE}_${FOLD_TAG}"
  RECIPE_BASE="${RECIPE_BASE}/${FOLD_TAG}"
fi
RUN_NAME="${RUN_PREFIX}_${DATASET_TAG}_${RUN_MIDDLE}_${ALGO}_${CONDITION}_seed1"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
RECIPE="${DATASET_TAG}/${RECIPE_BASE}/${ALGO}/${CKPT_MODE}"
OUTPUT="${SCORE_ROOT}/${RECIPE}"

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

mkdir -p /iris/u/jasonyan/slurm "${OUTPUT}"
cd "${REPO}"
python scripts/setup/patch_robomimic_optional_diffusion.py
if [[ "${ALGO}" == "discrete" ]]; then
  python scripts/setup/patch_robomimic_discrete_action.py
fi

CKPT="$(python scripts/quality/select_robomimic_checkpoint.py --run-dir "${RUN_DIR}" --mode "${CKPT_MODE}")"

cd "${REPO}/robomimic"
export USE_FLAX=0
export PYTHONPATH="${PWD}:${REPO}/scripts/quality:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "hostname=$(hostname)"
echo "dataset_tag=${DATASET_TAG}"
echo "task_tag=${TASK_TAG}"
echo "action_source=${ACTION_SOURCE}"
echo "fold_tag=${FOLD_TAG}"
echo "algo=${ALGO}"
echo "condition=${CONDITION}"
echo "run_dir=${RUN_DIR}"
echo "checkpoint=${CKPT}"
echo "output=${OUTPUT}/${CONDITION}.pkl"

if [[ -s "${OUTPUT}/${CONDITION}.pkl" ]]; then
  echo "score already exists; skipping ${OUTPUT}/${CONDITION}.pkl"
  exit 0
fi

FILTER_ARGS=()
if [[ -n "${SCORE_FILTER_KEY}" ]]; then
  FILTER_ARGS=(--filter-key "${SCORE_FILTER_KEY}")
fi

python "${REPO}/scripts/quality/score_robomimic_policy_nll.py" \
  --checkpoint "${CKPT}" \
  --dataset "${DATASET_HDF5}" \
  --output "${OUTPUT}" \
  --name "${CONDITION}" \
  --batch-size "${BATCH_SIZE}" \
  --gmm-entropy-samples "${GMM_ENTROPY_SAMPLES}" \
  --entropy-seed 1 \
  "${FILTER_ARGS[@]}"
