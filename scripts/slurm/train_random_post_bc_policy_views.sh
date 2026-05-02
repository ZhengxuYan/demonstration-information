#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=random_post_bc
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_random_post_bc.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_random_post_bc.err

set -euo pipefail

ALGO="${1:-}"
VIEW="${2:-}"
if [[ "${ALGO}" != "gmm" && "${ALGO}" != "discrete" ]]; then
  echo "Usage: sbatch $0 gmm|discrete agent_wrist|left_close_low_wrist"
  exit 2
fi
if [[ "${VIEW}" != "agent_wrist" && "${VIEW}" != "left_close_low_wrist" ]]; then
  echo "Usage: sbatch $0 gmm|discrete agent_wrist|left_close_low_wrist"
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
DATASET_ROOT=/iris/u/jasonyan/data/policy_view_experiments/random_post
CONFIG_DIR=/iris/u/jasonyan/data/policy_view_experiments/configs/robomimic
CONFIG="${CONFIG_DIR}/random_post_bc_${ALGO}_${VIEW}_seed1.json"
RUN_NAME="random_post_bc_${ALGO}_${VIEW}_seed1"
DATASET="${DATASET_ROOT}/random_post_${VIEW}_image.hdf5"

mkdir -p /iris/u/jasonyan/slurm "${CONFIG_DIR}"

cd "${REPO}"
python scripts/quality/write_policy_view_bc_config.py \
  --algo "${ALGO}" \
  --view "${VIEW}" \
  --repo "${REPO}" \
  --output "${CONFIG}" \
  --dataset "${DATASET}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-prefix random_post \
  --run-prefix random_post_bc \
  --suffix _seed1 \
  --output-dir /iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments \
  --num-epochs "${NUM_EPOCHS:-2000}" \
  --enable-validation \
  --log-wandb \
  --wandb-project "${WANDB_PROJECT:-policy-view-bc-random-post}" \
  --l2-regularization "${L2_REGULARIZATION:-0.0}"

python - <<PY
import json
with open("${CONFIG}") as f:
    cfg = json.load(f)
print("config", "${CONFIG}")
print("name", cfg["experiment"]["name"])
print("data", cfg["train"]["data"])
print("validate", cfg["experiment"]["validate"])
print("wandb", cfg["experiment"]["logging"]["log_wandb"], cfg["experiment"]["logging"].get("wandb_proj_name"))
print("filters", cfg["train"].get("hdf5_filter_key"), cfg["train"].get("hdf5_validation_filter_key"))
print("L2", cfg["algo"]["optim_params"]["policy"]["regularization"]["L2"])
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
  --dataset "${DATASET}" \
  --name "${RUN_NAME}"
