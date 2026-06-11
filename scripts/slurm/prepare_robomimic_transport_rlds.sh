#!/bin/bash
# Build RLDS/TFDS for the bimanual transport RoboMimic image dataset.
#
# Usage:
#   HDF5_PATH=/scr/tiangao/datasets/transport_mh_image_v15.hdf5 \
#   sbatch scripts/slurm/prepare_robomimic_transport_rlds.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=iris8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --job-name=prep_transport_rlds
#SBATCH --output=/iris/u/jasonyan/slurm/%j_prep_transport_rlds.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_prep_transport_rlds.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
HDF5_PATH="${HDF5_PATH:-/scr/tiangao/datasets/transport_mh_image_v15.hdf5}"
RLDS_ROOT="${RLDS_ROOT:-/iris/u/jasonyan/data/transport_mh_image_v15_rlds}"
MANUAL_DIR="${MANUAL_DIR:-/iris/u/jasonyan/data/transport_mh_image_v15_rlds_manual}"
EXPECTED_DEMOS="${EXPECTED_DEMOS:-300}"

mkdir -p /iris/u/jasonyan/slurm "${RLDS_ROOT}" "${MANUAL_DIR}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "hdf5_path=${HDF5_PATH}"
echo "rlds_root=${RLDS_ROOT}"
echo "manual_dir=${MANUAL_DIR}"

python scripts/data/validate_robomimic_transport_hdf5.py \
  --hdf5 "${HDF5_PATH}" \
  --expected-demos "${EXPECTED_DEMOS}"

ln -sfn "${HDF5_PATH}" "${MANUAL_DIR}/image.hdf5"

cd "${REPO}/rlds/robomimic_transport"
tfds build --overwrite \
  --manual_dir "${MANUAL_DIR}" \
  --data_dir "${RLDS_ROOT}"

echo "PREP_ROBOMIMIC_TRANSPORT_RLDS_OK"
echo "${RLDS_ROOT}/robo_mimic_transport/1.0.0"
