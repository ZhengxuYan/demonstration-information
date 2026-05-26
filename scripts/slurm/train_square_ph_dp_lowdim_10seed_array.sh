#!/bin/bash
# Train Diffusion Policy lowdim state policy on Square PH over 10 seeds.
#
# Usage:
#   sbatch --array=1-10%5 scripts/slurm/train_square_ph_dp_lowdim_10seed_array.sh

#SBATCH --job-name=ph_dp_lowdim
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_ph_dp_lowdim.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_ph_dp_lowdim.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-robodiff}"
set -u

DP_REPO="${DP_REPO:-/iris/u/jasonyan/repos/diffusion_policy}"
TASK_CONFIG="${TASK_CONFIG:-square_lowdim_abs}"
DATASET_PATH="${DATASET_PATH:-/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/low_dim_abs.hdf5}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/diffusion_policy_outputs/square_ph_lowdim_abs_10seed}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-1}}"
RUN_DIR="${OUT_ROOT}/seed_${SEED}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "missing DATASET_PATH=${DATASET_PATH}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${RUN_DIR}"
cd "${DP_REPO}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${DP_REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "hostname=$(hostname)"
echo "seed=${SEED}"
echo "task_config=${TASK_CONFIG}"
echo "dataset_path=${DATASET_PATH}"
echo "run_dir=${RUN_DIR}"

python train.py \
  --config-name=train_diffusion_unet_lowdim_workspace \
  task="${TASK_CONFIG}" \
  task.dataset_path="${DATASET_PATH}" \
  task.dataset.dataset_path="${DATASET_PATH}" \
  task.env_runner.dataset_path="${DATASET_PATH}" \
  training.seed="${SEED}" \
  task.dataset.seed="${SEED}" \
  task.env_runner.n_test="${N_TEST:-50}" \
  task.env_runner.test_start_seed="${TEST_START_SEED:-100000}" \
  training.num_epochs="${DP_NUM_EPOCHS:-5000}" \
  training.rollout_every="${ROLLOUT_EVERY:-50}" \
  training.checkpoint_every="${CHECKPOINT_EVERY:-50}" \
  logging.mode="${WANDB_MODE:-disabled}" \
  training.device=cuda:0 \
  hydra.run.dir="${RUN_DIR}" \
  hydra.sweep.dir="${RUN_DIR}"
