#!/bin/bash
# Closed-loop rollout eval for DemInf / MI-score filtered BC policies.
#
# Usage:
#   CONDA_ENV=robodiff sbatch --array=1-20%4 scripts/slurm/evaluate_deminf_mi_filtered_bc_rollouts_array.sh
#
# Optional env:
#   ALGO=gmm|discrete
#   DATASETS="ph_agentview 400_agentview 400_left_close_low"
#   DROP_PCTS="0 10 20 30 40"
#   N_ROLLOUTS=50 HORIZON=400
#   EPOCHS="600 800 1000 1200 1400 1600 1800 2000"
#   EPOCH_START=600 EPOCH_END=2000 EPOCH_STEP=200
#   SKIP_MISSING_EPOCHS=1
#   MIX_EVAL_CAMERA=agentview|left_close_low

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=deminf_mi_eval
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris9,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_deminf_mi_eval.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_deminf_mi_eval.err

set -euo pipefail

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-robodiff}"
conda activate "${CONDA_ENV}"

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
CKPT_ROOT="${CKPT_ROOT:-/iris/u/jasonyan/data/robomimic_outputs/deminf_mi_filtered_bc}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/robomimic_rollout_scores/deminf_mi_filtered_bc}"
ALGO="${ALGO:-gmm}"
DATASETS="${DATASETS:-ph_agentview 400_agentview 400_left_close_low}"
DROP_PCTS="${DROP_PCTS:-0 10 20 30 40}"
N_ROLLOUTS="${N_ROLLOUTS:-50}"
HORIZON="${HORIZON:-400}"
SEED="${SEED:-0}"
MIX_EVAL_CAMERA="${MIX_EVAL_CAMERA:-agentview}"
EPOCH_START="${EPOCH_START:-600}"
EPOCH_END="${EPOCH_END:-2000}"
EPOCH_STEP="${EPOCH_STEP:-200}"
SKIP_MISSING_EPOCHS="${SKIP_MISSING_EPOCHS:-0}"

if [[ "${ALGO}" != "gmm" && "${ALGO}" != "discrete" ]]; then
  echo "bad ALGO=${ALGO}; expected gmm or discrete" >&2
  exit 2
fi
if [[ "${MIX_EVAL_CAMERA}" != "agentview" && "${MIX_EVAL_CAMERA}" != "left_close_low" ]]; then
  echo "bad MIX_EVAL_CAMERA=${MIX_EVAL_CAMERA}; expected agentview or left_close_low" >&2
  exit 2
fi

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a DROP_ARRAY <<< "${DROP_PCTS}"
NUM_DATASETS="${#DATASET_ARRAY[@]}"
NUM_DROPS="${#DROP_ARRAY[@]}"
NUM_TASKS=$((NUM_DATASETS * NUM_DROPS))
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if (( TASK_ID < 1 || TASK_ID > NUM_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside 1..${NUM_TASKS}" >&2
  exit 2
fi

ZERO_BASED=$((TASK_ID - 1))
DATASET_IDX=$((ZERO_BASED / NUM_DROPS))
DROP_IDX=$((ZERO_BASED % NUM_DROPS))
DATASET="${DATASET_ARRAY[$DATASET_IDX]}"
DROP_PCT="${DROP_ARRAY[$DROP_IDX]}"
DROP_TAG=$(printf "drop_%02d" "${DROP_PCT}")
RUN_NAME="deminf_mi_bc_${ALGO}_${DATASET}_${DROP_TAG}_seed1"
RUN_ROOT="${CKPT_ROOT}/${RUN_NAME}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"

mkdir -p /iris/u/jasonyan/slurm "${OUT_DIR}"
cd "${REPO}"

python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python - <<'PY'
import robosuite
from packaging.version import Version

version = getattr(robosuite, "__version__", "unknown")
print("robosuite", version)
minimum = "${ROBOSUITE_MIN_VERSION:-}"
if minimum and Version(version) < Version(minimum):
    raise SystemExit(f"robosuite {version} < required {minimum}; set CONDA_ENV to a robosuite {minimum}+ env")
PY

mapfile -t ALL_CKPTS < <(find "${RUN_ROOT}" -path "*/models/model_epoch_*.pth" -type f | sort -V)
if [[ "${#ALL_CKPTS[@]}" -eq 0 ]]; then
  echo "missing checkpoints under ${RUN_ROOT}" >&2
  exit 1
fi

CKPTS=()
if [[ -z "${EPOCHS:-}" ]]; then
  EPOCHS=""
  epoch="${EPOCH_START}"
  while (( epoch <= EPOCH_END )); do
    EPOCHS="${EPOCHS} ${epoch}"
    epoch=$((epoch + EPOCH_STEP))
  done
fi

for epoch in ${EPOCHS}; do
  match=""
  for ckpt in "${ALL_CKPTS[@]}"; do
    if [[ "$(basename "${ckpt}")" == model_epoch_${epoch}* ]]; then
      match="${ckpt}"
      break
    fi
  done
  if [[ -z "${match}" ]]; then
    if [[ "${SKIP_MISSING_EPOCHS}" == "1" ]]; then
      echo "skipping missing epoch ${epoch} for ${RUN_NAME}" >&2
      continue
    else
      echo "missing epoch ${epoch} for ${RUN_NAME}" >&2
      echo "available checkpoints:" >&2
      printf '%s\n' "${ALL_CKPTS[@]}" >&2
      exit 1
    fi
  fi
  CKPTS+=("${match}")
done

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "no requested checkpoints are currently available for ${RUN_NAME}" >&2
  exit 1
fi

EXTRA_ARGS=()
EVAL_CAMERA="agentview"
if [[ "${DATASET}" == "400_left_close_low" ]]; then
  EXTRA_ARGS=(--left-close-low-as-agentview)
  EVAL_CAMERA="left_close_low_as_agentview"
elif [[ "${DATASET}" == "400_mix" && "${MIX_EVAL_CAMERA}" == "left_close_low" ]]; then
  EXTRA_ARGS=(--left-close-low-as-agentview)
  EVAL_CAMERA="left_close_low_as_agentview"
fi

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}/${NUM_TASKS}"
echo "conda_env=${CONDA_ENV}"
echo "algo=${ALGO}"
echo "dataset=${DATASET}"
echo "drop_pct=${DROP_PCT}"
echo "run_name=${RUN_NAME}"
echo "eval_camera=${EVAL_CAMERA}"
echo "n_rollouts=${N_ROLLOUTS}"
echo "horizon=${HORIZON}"
echo "epochs=${EPOCHS}"
printf 'checkpoints %s\n' "${#CKPTS[@]}"
printf '%s\n' "${CKPTS[@]}"

python scripts/quality/evaluate_robomimic_bc_checkpoint_rollouts.py \
  --dataset "${DATASET}" \
  --policy "${ALGO}" \
  --view "${EVAL_CAMERA}" \
  --run-name "${RUN_NAME}" \
  --checkpoints "${CKPTS[@]}" \
  --output-dir "${OUT_DIR}" \
  --n-rollouts "${N_ROLLOUTS}" \
  --horizon "${HORIZON}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
