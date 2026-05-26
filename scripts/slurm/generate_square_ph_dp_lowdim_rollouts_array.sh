#!/bin/bash
# Generate 200 rollouts from each seed's best Diffusion Policy checkpoint.
#
# Usage:
#   sbatch --array=1-10%5 scripts/slurm/generate_square_ph_dp_lowdim_rollouts_array.sh

#SBATCH --job-name=ph_dp_rollout
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_ph_dp_rollout.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_ph_dp_rollout.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-robodiff}"
set -u

DEMO_REPO="${DEMO_REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DP_REPO="${DP_REPO:-/iris/u/jasonyan/repos/diffusion_policy}"
RUN_ROOT="${RUN_ROOT:-/iris/u/jasonyan/data/diffusion_policy_outputs/square_ph_lowdim_abs_10seed}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-/iris/u/jasonyan/data/diffusion_policy_rollouts/square_ph_lowdim_abs_10seed}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-1}}"
RUN_DIR="${RUN_ROOT}/seed_${SEED}"
OUT_HDF5="${ROLLOUT_ROOT}/seed_${SEED}/rollouts.hdf5"

mkdir -p /iris/u/jasonyan/slurm "$(dirname "${OUT_HDF5}")"
cd "${DEMO_REPO}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${DP_REPO}:${DEMO_REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

BEST_CKPT="$(python scripts/quality/select_dp_best_checkpoints.py \
  --run-root "${RUN_ROOT}" \
  --glob "seed_${SEED}/checkpoints/*.ckpt" \
  --output-csv "${ROLLOUT_ROOT}/best_checkpoints_seed_${SEED}.csv" \
  | awk 'END{print $NF}')"

if [[ ! -f "${BEST_CKPT}" ]]; then
  echo "missing best checkpoint for seed=${SEED}: ${BEST_CKPT}" >&2
  exit 1
fi

echo "hostname=$(hostname)"
echo "seed=${SEED}"
echo "run_dir=${RUN_DIR}"
echo "best_ckpt=${BEST_CKPT}"
echo "out_hdf5=${OUT_HDF5}"

python scripts/quality/generate_dp_lowdim_rollouts.py \
  --checkpoint "${BEST_CKPT}" \
  --output "${OUT_HDF5}" \
  --n-rollouts "${N_ROLLOUTS:-200}" \
  --seed-start "$(( ${ROLLOUT_SEED_START:-200000} + SEED * 10000 ))" \
  --horizon "${HORIZON:-400}" \
  --device cuda:0 \
  --overwrite
