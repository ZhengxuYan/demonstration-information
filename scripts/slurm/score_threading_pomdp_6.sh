#!/bin/bash
# Score one Threading POMDP density pair with the six PDF-defined scores.

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=thread_pomdp_score
#SBATCH --output=/iris/u/jasonyan/slurm/%j_thread_pomdp_score.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_thread_pomdp_score.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATA_ROOT="${DATA_ROOT:-/iris/u/jasonyan/data}"
DATASET_TAG="${DATASET_TAG:-curated100}"
DATASET_HDF5="${DATASET_HDF5:-${DATA_ROOT}/threading_pomdp_density/threading_curated100_state_action.hdf5}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/robomimic_outputs/threading_pomdp_density}"
SCORE_ROOT="${SCORE_ROOT:-${DATA_ROOT}/threading_pomdp_scores}"
RUN_PREFIX="${RUN_PREFIX:-threading_pomdp}"
ACTION_SOURCE="${ACTION_SOURCE:-state}"
CONDITIONAL_CONDITION="${CONDITIONAL_CONDITION:-state}"
ALGO="${ALGO:?Set ALGO to gaussian or gmm}"
REGIME="${REGIME:-normal}"
FOLD_TAG="${FOLD_TAG:-}"
VARIANT_TAG="${VARIANT_TAG:-}"
FILTER_KEY="${FILTER_KEY:-score_all}"
CKPT_MODE="${CKPT_MODE:-best_validation}"
M="${M:-16}"
K="${K:-512}"
SEED="${SEED:-20260704}"
ACTION_DIMS="${ACTION_DIMS:-}"

RUN_MIDDLE="single_${ACTION_SOURCE}_none"
if [[ -n "${FOLD_TAG}" ]]; then
  RUN_MIDDLE="${RUN_MIDDLE}_${FOLD_TAG}"
fi
if [[ -n "${VARIANT_TAG}" ]]; then
  RUN_MIDDLE="${RUN_MIDDLE}_${VARIANT_TAG}"
fi
COND_RUN="${OUT_ROOT}/${RUN_PREFIX}_${DATASET_TAG}_${RUN_MIDDLE}_${ALGO}_${CONDITIONAL_CONDITION}_seed1"
PRIOR_RUN="${OUT_ROOT}/${RUN_PREFIX}_${DATASET_TAG}_${RUN_MIDDLE}_${ALGO}_action_prior_seed1"
OUTPUT="${SCORE_ROOT}/${ALGO}/${REGIME}"
if [[ -n "${FOLD_TAG}" ]]; then
  OUTPUT="${SCORE_ROOT}/${ALGO}/${FOLD_TAG}"
fi

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

cd "${REPO}"
COND_CKPT="$(python scripts/quality/select_robomimic_checkpoint.py --run-dir "${COND_RUN}" --mode "${CKPT_MODE}")"
PRIOR_CKPT="$(python scripts/quality/select_robomimic_checkpoint.py --run-dir "${PRIOR_RUN}" --mode "${CKPT_MODE}")"

cd "${REPO}/robomimic"
export USE_FLAX=0
export PYTHONPATH="${PWD}:${REPO}/scripts/quality:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

score_args=(
  python "${REPO}/scripts/quality/score_threading_pomdp_6.py"
  --conditional-checkpoint "${COND_CKPT}"
  --prior-checkpoint "${PRIOR_CKPT}"
  --dataset "${DATASET_HDF5}"
  --output "${OUTPUT}"
  --filter-key "${FILTER_KEY}"
  --mc-action-samples "${M}"
  --mc-marginal-states "${K}"
  --seed "${SEED}"
)
if [[ -n "${ACTION_DIMS}" ]]; then
  score_args+=(--action-dims "${ACTION_DIMS}")
fi
"${score_args[@]}"
