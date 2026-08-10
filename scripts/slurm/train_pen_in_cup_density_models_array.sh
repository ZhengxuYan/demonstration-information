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
SOURCE_DATASET_HDF5="${DATASET_HDF5}"
STAGE_DATASET_TO_TMP="${STAGE_DATASET_TO_TMP:-0}"
TASK_TAG="${TASK_TAG:-pen_in_cup}"
RUN_PREFIX="${RUN_PREFIX:-${TASK_TAG}}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/robomimic_outputs/${TASK_TAG}_density}"
CONFIG_ROOT="${CONFIG_ROOT:-/iris/u/jasonyan/data/${TASK_TAG}_density_configs}"
ACTION_SOURCE="${ACTION_SOURCE:-action}"
ACTION_TARGET="${ACTION_TARGET:-single}"
ACTION_NORMALIZATION="${ACTION_NORMALIZATION:-none}"
FOLD_TAG="${FOLD_TAG:-}"
TRAIN_FILTER_KEY="${TRAIN_FILTER_KEY:-train}"
VALID_FILTER_KEY="${VALID_FILTER_KEY:-valid}"
ALGOS_CSV="${ALGOS:-gaussian,gmm,discrete}"
CONDITIONS_CSV="${CONDITIONS:-image_state,image,state,action_prior}"
ALGOS_CSV="${ALGOS_CSV//:/,}"
CONDITIONS_CSV="${CONDITIONS_CSV//:/,}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCH_STEPS="${EPOCH_STEPS:-100}"
VALIDATION_STEPS="${VALIDATION_STEPS:-25}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-50}"
SAVE_BEST_VALIDATION="${SAVE_BEST_VALIDATION:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
EARLY_STOPPING_MIN_EPOCH="${EARLY_STOPPING_MIN_EPOCH:-0}"
LOCAL_BEST_SYNC_INTERVAL="${LOCAL_BEST_SYNC_INTERVAL:-0}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
L2_REGULARIZATION="${L2_REGULARIZATION:-0.0}"
ACTOR_LAYER_DIMS="${ACTOR_LAYER_DIMS:-1024,1024}"
GMM_MODES="${GMM_MODES:-5}"
GAUSSIAN_MIN_STD="${GAUSSIAN_MIN_STD:-0.0001}"
ACTION_TRANSFORM="${ACTION_TRANSFORM:-identity}"
MEAN_SQUASH="${MEAN_SQUASH:-tanh}"
COVARIANCE_TYPE="${COVARIANCE_TYPE:-diag}"
TRAIN_SEED="${TRAIN_SEED:-1}"
RUN_REGIME="${RUN_REGIME:-normal}"
VARIANT_TAG="${VARIANT_TAG:-}"
DISABLE_RGB_RANDOMIZER="${DISABLE_RGB_RANDOMIZER:-0}"
HDF5_NORMALIZE_OBS="${HDF5_NORMALIZE_OBS:-0}"
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

RECIPE_BASE="${ACTION_TARGET}_${ACTION_SOURCE}_${ACTION_NORMALIZATION}"
if [[ -n "${FOLD_TAG}" ]]; then
  RECIPE_BASE="${RECIPE_BASE}/${FOLD_TAG}"
fi
RUN_MIDDLE="${ACTION_TARGET}_${ACTION_SOURCE}_${ACTION_NORMALIZATION}"
if [[ -n "${FOLD_TAG}" ]]; then
  RUN_MIDDLE="${RUN_MIDDLE}_${FOLD_TAG}"
fi
if [[ -n "${VARIANT_TAG}" ]]; then
  RECIPE_BASE="${RECIPE_BASE}/${VARIANT_TAG}"
  RUN_MIDDLE="${RUN_MIDDLE}_${VARIANT_TAG}"
fi
RUN_NAME="${RUN_PREFIX}_${DATASET_TAG}_${RUN_MIDDLE}_${ALGO}_${CONDITION}_seed${TRAIN_SEED}"
CONFIG="${CONFIG_ROOT}/${DATASET_TAG}/${RECIPE_BASE}/${RUN_NAME}.json"
RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
if [[ -d "${RUN_DIR}" ]] && ! find "${RUN_DIR}" \( -path '*/models/*.pth' -o -path '*/last.pth' -o -path '*/last_bak.pth' \) -print -quit 2>/dev/null | grep -q .; then
  echo "removing failed empty run directory without checkpoints: ${RUN_DIR}"
  rm -rf "${RUN_DIR}"
fi
RESUME_REQUESTED="${RESUME:-0}"
AUTO_RESUME_PARTITION=0
case "${SLURM_JOB_PARTITION:-}" in
  iris|iliad-lo|sc-loprio) AUTO_RESUME_PARTITION=1 ;;
esac
RESUME_FLAG=()
if [[ "${RESUME_REQUESTED}" == "1" || "${AUTO_RESUME_PARTITION}" == "1" ]] && \
   find "${RUN_DIR}" \( -path '*/last.pth' -o -path '*/models/last.pth' \) -print -quit 2>/dev/null | grep -q .; then
  RESUME_FLAG=(--resume)
elif [[ -d "${RUN_DIR}" ]]; then
  BACKUP_DIR="${RUN_DIR}.failed_resume_backup.$(date -u +%Y%m%dT%H%M%SZ).${SLURM_JOB_ID:-manual}"
  echo "existing run directory cannot be resumed because no last.pth was found; moving it to ${BACKUP_DIR}"
  mv "${RUN_DIR}" "${BACKUP_DIR}"
fi

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

if [[ "${STAGE_DATASET_TO_TMP}" == "1" ]]; then
  DATASET_CACHE_DIR="/tmp/density_dataset_cache_${USER}"
  DATASET_CACHE_FILE="${DATASET_CACHE_DIR}/$(basename "${SOURCE_DATASET_HDF5}")"
  DATASET_CACHE_LOCK="${DATASET_CACHE_FILE}.lock"
  mkdir -p "${DATASET_CACHE_DIR}"
  (
    flock 9
    source_size="$(stat -c '%s' "${SOURCE_DATASET_HDF5}")"
    cached_size="$(stat -c '%s' "${DATASET_CACHE_FILE}" 2>/dev/null || echo 0)"
    if [[ "${cached_size}" != "${source_size}" ]]; then
      staged_tmp="${DATASET_CACHE_FILE}.tmp.${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
      rm -f -- "${staged_tmp}"
      cp --reflink=auto "${SOURCE_DATASET_HDF5}" "${staged_tmp}"
      mv -f -- "${staged_tmp}" "${DATASET_CACHE_FILE}"
    fi
  ) 9>"${DATASET_CACHE_LOCK}"
  DATASET_HDF5="${DATASET_CACHE_FILE}"
fi

mkdir -p /iris/u/jasonyan/slurm "$(dirname "${CONFIG}")" "${OUT_ROOT}"
cd "${REPO}"
python scripts/setup/patch_robomimic_optional_diffusion.py
if [[ "${ALGO}" == "discrete" ]]; then
  python scripts/setup/patch_robomimic_discrete_action.py
fi
CODE_VERSION="${CODE_VERSION:-$(git rev-parse --verify HEAD 2>/dev/null || echo unknown)}"
if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
  CODE_VERSION="${CODE_VERSION}-dirty"
fi

RGB_RANDOMIZER_FLAG=()
if [[ "${DISABLE_RGB_RANDOMIZER}" == "1" ]]; then
  RGB_RANDOMIZER_FLAG=(--disable-rgb-randomizer)
fi
OBS_NORMALIZATION_FLAG=()
if [[ "${HDF5_NORMALIZE_OBS}" == "1" ]]; then
  OBS_NORMALIZATION_FLAG=(--hdf5-normalize-obs)
fi
BEST_VALIDATION_SAVE_FLAG=()
if [[ "${SAVE_BEST_VALIDATION}" == "0" ]]; then
  BEST_VALIDATION_SAVE_FLAG=(--disable-best-validation-save)
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
  "${BEST_VALIDATION_SAVE_FLAG[@]}" \
  --learning-rate "${LEARNING_RATE}" \
  --l2-regularization "${L2_REGULARIZATION}" \
  --actor-layer-dims "${ACTOR_LAYER_DIMS}" \
  --train-filter-key "${TRAIN_FILTER_KEY}" \
  --valid-filter-key "${VALID_FILTER_KEY}" \
  --gmm-modes "${GMM_MODES}" \
  --gaussian-min-std "${GAUSSIAN_MIN_STD}" \
  --action-transform "${ACTION_TRANSFORM}" \
  --mean-squash "${MEAN_SQUASH}" \
  --covariance-type "${COVARIANCE_TYPE}" \
  --seed "${TRAIN_SEED}" \
  --code-version "${CODE_VERSION}" \
  --regime "${RUN_REGIME}" \
  --discrete-bins "${DISCRETE_BINS}" \
  --discrete-loss-type "${DISCRETE_LOSS_TYPE}" \
  --wandb-project "${WANDB_PROJECT}" \
  "${RGB_RANDOMIZER_FLAG[@]}" \
  "${OBS_NORMALIZATION_FLAG[@]}" \
  ${LOG_WANDB:+--log-wandb}

cd "${REPO}/robomimic"
export MUJOCO_GL=egl
export USE_FLAX=0
export PYTHONPATH="${PWD}:${REPO}/scripts/quality:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export ROBOMIMIC_EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}"
export ROBOMIMIC_EARLY_STOPPING_MIN_EPOCH="${EARLY_STOPPING_MIN_EPOCH}"
export ROBOMIMIC_LOCAL_BEST_SYNC_INTERVAL="${LOCAL_BEST_SYNC_INTERVAL}"
LOCAL_BEST_DIR="$(mktemp -d "/tmp/density_best.${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}.XXXXXX")"
export ROBOMIMIC_LOCAL_BEST_DIR="${LOCAL_BEST_DIR}"
TRAIN_LOG=""
cleanup_local_best() {
  rm -rf -- "${LOCAL_BEST_DIR}"
  if [[ -n "${TRAIN_LOG}" ]]; then
    rm -f -- "${TRAIN_LOG}"
  fi
}
trap cleanup_local_best EXIT

echo "hostname=$(hostname)"
echo "task=${TASK_ID}/${TOTAL}"
echo "dataset_tag=${DATASET_TAG}"
echo "task_tag=${TASK_TAG}"
echo "dataset_hdf5=${DATASET_HDF5}"
echo "source_dataset_hdf5=${SOURCE_DATASET_HDF5}"
echo "stage_dataset_to_tmp=${STAGE_DATASET_TO_TMP}"
echo "action_source=${ACTION_SOURCE}"
echo "fold_tag=${FOLD_TAG}"
echo "train_filter_key=${TRAIN_FILTER_KEY}"
echo "valid_filter_key=${VALID_FILTER_KEY}"
echo "algo=${ALGO}"
echo "condition=${CONDITION}"
echo "gaussian_min_std=${GAUSSIAN_MIN_STD}"
echo "save_every_n_epochs=${SAVE_EVERY_N_EPOCHS}"
echo "save_best_validation=${SAVE_BEST_VALIDATION}"
echo "early_stopping_patience=${EARLY_STOPPING_PATIENCE}"
echo "early_stopping_min_epoch=${EARLY_STOPPING_MIN_EPOCH}"
echo "local_best_sync_interval=${LOCAL_BEST_SYNC_INTERVAL}"
echo "local_best_dir=${LOCAL_BEST_DIR}"
echo "latest_save_interval=${ROBOMIMIC_LATEST_SAVE_INTERVAL:-50}"
echo "action_transform=${ACTION_TRANSFORM}"
echo "mean_squash=${MEAN_SQUASH}"
echo "covariance_type=${COVARIANCE_TYPE}"
echo "train_seed=${TRAIN_SEED}"
echo "run_regime=${RUN_REGIME}"
echo "code_version=${CODE_VERSION}"
echo "variant_tag=${VARIANT_TAG}"
echo "disable_rgb_randomizer=${DISABLE_RGB_RANDOMIZER}"
echo "hdf5_normalize_obs=${HDF5_NORMALIZE_OBS}"
echo "run_name=${RUN_NAME}"
echo "run_dir=${RUN_DIR}"
echo "resume_requested=${RESUME_REQUESTED}"
echo "auto_resume_partition=${AUTO_RESUME_PARTITION}"
echo "resume_enabled=$([[ ${#RESUME_FLAG[@]} -gt 0 ]] && echo 1 || echo 0)"

TRAIN_LOG="$(mktemp /tmp/${TASK_TAG}_${DATASET_TAG}_${ALGO}_${CONDITION}_train.XXXXXX.log)"
python robomimic/scripts/train.py \
  --config "${CONFIG}" \
  --dataset "${DATASET_HDF5}" \
  --name "${RUN_NAME}" \
  "${RESUME_FLAG[@]}" 2>&1 | tee "${TRAIN_LOG}"

if grep -qE 'run failed with error:|EOF when reading a line|Traceback \\(most recent call last\\)' "${TRAIN_LOG}"; then
  echo "training command reported a failure; see ${TRAIN_LOG}" >&2
  exit 1
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "training did not create run directory: ${RUN_DIR}" >&2
  exit 1
fi
if ! find "${RUN_DIR}" -path '*/models/*.pth' -print -quit 2>/dev/null | grep -q .; then
  echo "training finished without producing a checkpoint under: ${RUN_DIR}" >&2
  exit 1
fi
{
  echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "hostname=$(hostname)"
  echo "dataset_tag=${DATASET_TAG}"
  echo "algo=${ALGO}"
  echo "condition=${CONDITION}"
  echo "fold_tag=${FOLD_TAG}"
} > "${RUN_DIR}/TRAIN_DONE"
