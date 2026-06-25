#!/bin/bash
# Train robomimic BC action-density models for pen-in-cup NLL / entropy scores.
#
# Default array is 12 jobs: gaussian,gmm,discrete x image_state,image,state,action_prior.
# Use ALGOS or CONDITIONS to restrict, for example:
#   ALGOS=gmm CONDITIONS=image_state,image,state,action_prior sbatch --array=1-4%4 ...

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=pic_density
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8,iliad1,iliad2,iliad3,iliad4
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_pic_density.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_pic_density.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_TAG="${DATASET_TAG:?Set DATASET_TAG, e.g. 0610_89 or 0612_100}"
DATASET_HDF5="${DATASET_HDF5:?Set DATASET_HDF5 to prepared robomimic density HDF5}"
TASK_TAG="${TASK_TAG:-pen_in_cup}"
RUN_PREFIX="${RUN_PREFIX:-${TASK_TAG}}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/robomimic_outputs/${TASK_TAG}_density}"
CONFIG_ROOT="${CONFIG_ROOT:-/iris/u/jasonyan/data/${TASK_TAG}_density_configs}"
ACTION_SOURCE="${ACTION_SOURCE:-action}"
ACTION_TARGET="${ACTION_TARGET:-single}"
ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-none}"
ALGOS_CSV="${ALGOS:-gaussian,gmm,discrete}"
CONDITIONS_CSV="${CONDITIONS:-image_state,image,state,action_prior}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCH_STEPS="${EPOCH_STEPS:-100}"
VALIDATION_STEPS="${VALIDATION_STEPS:-25}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
L2_REGULARIZATION="${L2_REGULARIZATION:-0.0}"
ACTOR_LAYER_DIMS="${ACTOR_LAYER_DIMS:-1024,1024}"
GMM_MODES="${GMM_MODES:-5}"
DISCRETE_BINS="${DISCRETE_BINS:-256}"
DISCRETE_LOSS_TYPE="${DISCRETE_LOSS_TYPE:-hard_ce}"
WANDB_PROJECT="${WANDB_PROJECT:-${TASK_TAG}-density}"

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

RUN_NAME="${RUN_PREFIX}_${DATASET_TAG}_${ACTION_TARGET}_${ACTION_SOURCE}_${ACTION_NORMALIZATION}_${ALGO}_${CONDITION}_seed1"
CONFIG="${CONFIG_ROOT}/${DATASET_TAG}/${ACTION_TARGET}_${ACTION_SOURCE}_${ACTION_NORMALIZATION}/${RUN_NAME}.json"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
if [[ -d "${RUN_DIR}" ]] && ! find "${RUN_DIR}" -path '*/models/*.pth' -print -quit 2>/dev/null | grep -q .; then
  echo "removing failed empty run directory without checkpoints: ${RUN_DIR}"
  rm -rf "${RUN_DIR}"
fi
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]] && find "${RUN_DIR}" -path '*/models/last.pth' -print -quit 2>/dev/null | grep -q .; then
  RESUME_FLAG=(--resume)
fi

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

mkdir -p /iris/u/jasonyan/slurm "$(dirname "${CONFIG}")" "${OUT_ROOT}"
cd "${REPO}"
python scripts/setup/patch_robomimic_optional_diffusion.py
if [[ "${ALGO}" == "discrete" ]]; then
  python scripts/setup/patch_robomimic_discrete_action.py
fi

python scripts/quality/write_pen_in_cup_density_bc_config.py \
  --algo "${ALGO}" \
  --condition "${CONDITION}" \
  --dataset "${DATASET_HDF5}" \
  --output "${CONFIG}" \
  --output-dir "${OUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --num-epochs "${NUM_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --epoch-steps "${EPOCH_STEPS}" \
  --validation-steps "${VALIDATION_STEPS}" \
  --save-every-n-epochs "${SAVE_EVERY_N_EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --l2-regularization "${L2_REGULARIZATION}" \
  --actor-layer-dims "${ACTOR_LAYER_DIMS}" \
  --gmm-modes "${GMM_MODES}" \
  --discrete-bins "${DISCRETE_BINS}" \
  --discrete-loss-type "${DISCRETE_LOSS_TYPE}" \
  --wandb-project "${WANDB_PROJECT}" \
  ${LOG_WANDB:+--log-wandb}

cd "${REPO}/robomimic"
export MUJOCO_GL=egl
export USE_FLAX=0
export PYTHONPATH="${PWD}:${REPO}/scripts/quality:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "hostname=$(hostname)"
echo "task=${TASK_ID}/${TOTAL}"
echo "dataset_tag=${DATASET_TAG}"
echo "task_tag=${TASK_TAG}"
echo "dataset_hdf5=${DATASET_HDF5}"
echo "action_source=${ACTION_SOURCE}"
echo "algo=${ALGO}"
echo "condition=${CONDITION}"
echo "run_name=${RUN_NAME}"
echo "run_dir=${RUN_DIR}"
echo "resume_requested=${RESUME:-0}"
echo "resume_enabled=$([[ ${#RESUME_FLAG[@]} -gt 0 ]] && echo 1 || echo 0)"

python robomimic/scripts/train.py \
  --config "${CONFIG}" \
  --dataset "${DATASET_HDF5}" \
  --name "${RUN_NAME}" \
  "${RESUME_FLAG[@]}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "training did not create run directory: ${RUN_DIR}" >&2
  exit 1
fi
if ! find "${RUN_DIR}" -path '*/models/*.pth' -print -quit 2>/dev/null | grep -q .; then
  echo "training finished without producing a checkpoint under: ${RUN_DIR}" >&2
  exit 1
fi
