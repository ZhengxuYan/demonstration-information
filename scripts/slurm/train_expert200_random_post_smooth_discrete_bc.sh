#!/bin/bash
# Train smooth-target discrete BC policies for expert200 random-post policy views.
#
# Usage:
#   sbatch scripts/slurm/train_expert200_random_post_smooth_discrete_bc.sh agent_wrist
#   sbatch scripts/slurm/train_expert200_random_post_smooth_discrete_bc.sh left_close_low_wrist
#   sbatch scripts/slurm/train_expert200_random_post_smooth_discrete_bc.sh both

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=expert200_smooth_bc
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_expert200_smooth_bc.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_expert200_smooth_bc.err

set -euo pipefail

MODE="${1:-both}"
if [[ "${MODE}" != "agent_wrist" && "${MODE}" != "left_close_low_wrist" && "${MODE}" != "both" ]]; then
  echo "Usage: sbatch $0 agent_wrist|left_close_low_wrist|both" >&2
  exit 2
fi

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx
set -u

REPO=/iris/u/jasonyan/repos/demonstration-information
DATASET_ROOT="${DATASET_ROOT:-/iris/u/jasonyan/data/policy_view_experiments/expert200_random_post_bc}"
CONFIG_DIR=/iris/u/jasonyan/data/policy_view_experiments/configs/robomimic
OUTPUT_DIR=/iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments

mkdir -p /iris/u/jasonyan/slurm "${CONFIG_DIR}"
cd "${REPO}"

train_one() {
  local view="$1"
  local dataset="$2"
  local run_name="expert200_random_post_bc_discrete_smooth_${view}_seed1"
  local config="${CONFIG_DIR}/${run_name}.json"

  python scripts/quality/write_policy_view_bc_config.py \
    --algo discrete \
    --view "${view}" \
    --repo "${REPO}" \
    --output "${config}" \
    --dataset "${dataset}" \
    --run-name "${run_name}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-epochs "${NUM_EPOCHS:-2000}" \
    --enable-validation \
    --log-wandb \
    --wandb-project "${WANDB_PROJECT:-policy-view-bc-random-post}" \
    --l2-regularization "${L2_REGULARIZATION:-0.0}" \
    --discrete-loss-type soft_ce \
    --soft-sigma-bins "${SOFT_SIGMA_BINS:-1.5}" \
    --soft-truncate-bins "${SOFT_TRUNCATE_BINS:-6}"

  python - <<PY
import json
with open("${config}") as f:
    cfg = json.load(f)
print("config", "${config}")
print("run", cfg["experiment"]["name"])
print("data", cfg["train"]["data"])
print("discrete", cfg["algo"]["discrete"])
print("filters", cfg["train"].get("hdf5_filter_key"), cfg["train"].get("hdf5_validation_filter_key"))
PY

  cd "${REPO}/robomimic"
  python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
  python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
  export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2

  python robomimic/scripts/train.py \
    --config "${config}" \
    --dataset "${dataset}" \
    --name "${run_name}"

  cd "${REPO}"
}

if [[ "${MODE}" == "agent_wrist" || "${MODE}" == "both" ]]; then
  train_one agent_wrist "${DATASET_ROOT}/expert200_random_post_agent_wrist_image_abs.hdf5"
fi

if [[ "${MODE}" == "left_close_low_wrist" || "${MODE}" == "both" ]]; then
  train_one left_close_low_wrist "${DATASET_ROOT}/expert200_random_post_left_close_low_wrist_image_abs.hdf5"
fi
