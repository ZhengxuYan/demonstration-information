#!/bin/bash
# Train robomimic BC policies on DemInf / MI-score filtered camera-view datasets.
#
# Usage:
#   sbatch --array=1-20%4 scripts/slurm/train_deminf_mi_filtered_bc_policy_array.sh
#
# Optional env:
#   ALGO=gmm|discrete
#   DATASETS="ph_agentview 400_agentview 400_left_close_low 400_mix"
#   DROP_PCTS="0 10 20 30 40"
#   NUM_EPOCHS=2000

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=deminf_mi_bc
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_deminf_mi_bc.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_deminf_mi_bc.err

set -euo pipefail

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_ROOT="${DATASET_ROOT:-/iris/u/jasonyan/data/deminf_filtered_bc_datasets/mi_score}"
CONFIG_DIR="${CONFIG_DIR:-/iris/u/jasonyan/data/deminf_filtered_bc_datasets/configs/robomimic}"
OUTPUT_DIR="${OUTPUT_DIR:-/iris/u/jasonyan/data/robomimic_outputs/deminf_mi_filtered_bc}"
ALGO="${ALGO:-gmm}"
DATASETS="${DATASETS:-ph_agentview 400_agentview 400_left_close_low 400_mix}"
DROP_PCTS="${DROP_PCTS:-0 10 20 30 40}"

if [[ "${ALGO}" != "gmm" && "${ALGO}" != "discrete" ]]; then
  echo "bad ALGO=${ALGO}; expected gmm or discrete" >&2
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

DATASET_PATH="${DATASET_ROOT}/${DATASET}/${DROP_TAG}/image.hdf5"
RUN_NAME="deminf_mi_bc_${ALGO}_${DATASET}_${DROP_TAG}_seed1"
CONFIG="${CONFIG_DIR}/${RUN_NAME}.json"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "missing filtered dataset: ${DATASET_PATH}" >&2
  echo "Run scripts/quality/make_deminf_mi_filtered_bc_datasets.py first." >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${CONFIG_DIR}" "${OUTPUT_DIR}"

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}/${NUM_TASKS}"
echo "algo=${ALGO}"
echo "dataset=${DATASET}"
echo "drop_pct=${DROP_PCT}"
echo "dataset_path=${DATASET_PATH}"
echo "run_name=${RUN_NAME}"

cd "${REPO}"
python scripts/quality/write_policy_view_bc_config.py \
  --algo "${ALGO}" \
  --view agent_wrist \
  --repo "${REPO}" \
  --output "${CONFIG}" \
  --dataset "${DATASET_PATH}" \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-epochs "${NUM_EPOCHS:-2000}" \
  --enable-validation \
  --log-wandb \
  --wandb-project "${WANDB_PROJECT:-deminf-mi-filtered-bc}" \
  --l2-regularization "${L2_REGULARIZATION:-0.0}"

python - <<PY
import json
with open("${CONFIG}") as f:
    cfg = json.load(f)
print("config", "${CONFIG}")
print("run", cfg["experiment"]["name"])
print("data", cfg["train"]["data"])
print("output_dir", cfg["train"]["output_dir"])
print("validate", cfg["experiment"]["validate"])
print("filters", cfg["train"].get("hdf5_filter_key"), cfg["train"].get("hdf5_validation_filter_key"))
print("epochs", cfg["train"]["num_epochs"])
PY

cd "${REPO}/robomimic"
python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
if [[ "${ALGO}" == "discrete" ]]; then
  python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python robomimic/scripts/train.py \
  --config "${CONFIG}" \
  --dataset "${DATASET_PATH}" \
  --name "${RUN_NAME}"
