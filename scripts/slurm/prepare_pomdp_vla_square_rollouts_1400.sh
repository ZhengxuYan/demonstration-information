#!/bin/bash
# Merge Tian's POMDP-VLA Square rollouts into one robomimic image.hdf5.
#
# The source path is on /scr and should be accessed from iris10.
#
# Usage:
#   sbatch scripts/slurm/prepare_pomdp_vla_square_rollouts_1400.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=iris10
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=prep_pvla1400
#SBATCH --output=/iris/u/jasonyan/slurm/%j_prep_pvla1400.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_prep_pvla1400.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-robodiff}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
SOURCE_ROOT="${SOURCE_ROOT:-/scr/tiangao/pomdp_vla/square_rollouts}"
OUT_HDF5="${OUT_HDF5:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400/image.hdf5}"
MAX_DEMOS="${MAX_DEMOS:-1400}"

mkdir -p /iris/u/jasonyan/slurm "$(dirname "${OUT_HDF5}")"
cd "${REPO}"

python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "hostname=$(hostname)"
echo "source_root=${SOURCE_ROOT}"
echo "out_hdf5=${OUT_HDF5}"
echo "max_demos=${MAX_DEMOS}"

python scripts/quality/prepare_pomdp_vla_square_rollout_dataset.py \
  --input-root "${SOURCE_ROOT}" \
  --output "${OUT_HDF5}" \
  --max-demos "${MAX_DEMOS}" \
  --valid-ratio "${VALID_RATIO:-0.1}" \
  --split-seed "${SPLIT_SEED:-1}" \
  --overwrite
