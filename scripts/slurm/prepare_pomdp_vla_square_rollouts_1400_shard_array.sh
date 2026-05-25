#!/bin/bash
# Render one raw POMDP-VLA Square rollout HDF5 into a shard with image observations.
#
# Usage:
#   SOURCE_ROOT=/iris/u/jasonyan/data/pomdp_vla_square_rollouts_raw \
#   sbatch --array=1-7%7 scripts/slurm/prepare_pomdp_vla_square_rollouts_1400_shard_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=prep_pvla_shard
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_prep_pvla_shard.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_prep_pvla_shard.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-robodiff}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
SOURCE_ROOT="${SOURCE_ROOT:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_raw}"
SHARD_ROOT="${SHARD_ROOT:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400_shards}"

mkdir -p /iris/u/jasonyan/slurm "${SHARD_ROOT}"
cd "${REPO}"

python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mapfile -t HDF5_FILES < <(find "${SOURCE_ROOT}" -maxdepth 1 -type f -name '*.hdf5' | sort)
if [[ "${#HDF5_FILES[@]}" -eq 0 ]]; then
  echo "no hdf5 files found under SOURCE_ROOT=${SOURCE_ROOT}" >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
IDX=$((TASK_ID - 1))
if [[ "${IDX}" -lt 0 || "${IDX}" -ge "${#HDF5_FILES[@]}" ]]; then
  echo "task_id=${TASK_ID} out of range for ${#HDF5_FILES[@]} files" >&2
  exit 1
fi

SRC_HDF5="${HDF5_FILES[${IDX}]}"
SHARD_NAME="$(basename "${SRC_HDF5}" .hdf5)"
OUT_HDF5="${SHARD_ROOT}/${SHARD_NAME}/image.hdf5"
mkdir -p "$(dirname "${OUT_HDF5}")"

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}/${#HDF5_FILES[@]}"
echo "src_hdf5=${SRC_HDF5}"
echo "out_hdf5=${OUT_HDF5}"

python scripts/quality/prepare_pomdp_vla_square_rollout_dataset.py \
  --input-hdf5 "${SRC_HDF5}" \
  --output "${OUT_HDF5}" \
  --valid-ratio 0.0 \
  --split-seed "${SPLIT_SEED:-1}" \
  --overwrite
