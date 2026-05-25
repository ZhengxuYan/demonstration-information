#!/bin/bash
# Build RLDS/TFDS for the merged 1400 POMDP-VLA Square rollout dataset.
#
# Usage:
#   sbatch scripts/slurm/build_pomdp_vla_square_rollouts_1400_rlds.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --job-name=rlds_pvla1400
#SBATCH --output=/iris/u/jasonyan/slurm/%j_rlds_pvla1400.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_rlds_pvla1400.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_NAME="${DATASET_NAME:-pomdp_vla_square_rollouts_1400}"
DATASET_HDF5="${DATASET_HDF5:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400/image.hdf5}"
MANUAL_DIR="${MANUAL_DIR:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400_rlds_manual/${DATASET_NAME}}"
RLDS_ROOT="${RLDS_ROOT:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400_rlds}"

if [[ ! -f "${DATASET_HDF5}" ]]; then
  echo "missing DATASET_HDF5=${DATASET_HDF5}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${MANUAL_DIR}" "${RLDS_ROOT}/${DATASET_NAME}"
ln -sfn "${DATASET_HDF5}" "${MANUAL_DIR}/image.hdf5"

cd "${REPO}/rlds/robomimic"
rm -rf "${RLDS_ROOT}/${DATASET_NAME}/robo_mimic"
tfds build \
  --manual_dir "${MANUAL_DIR}" \
  --data_dir "${RLDS_ROOT}/${DATASET_NAME}"

echo "${RLDS_ROOT}/${DATASET_NAME}/robo_mimic/1.0.0"
