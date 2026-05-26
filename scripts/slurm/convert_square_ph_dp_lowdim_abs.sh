#!/bin/bash
# Convert Square PH delta-action robomimic HDF5 to absolute-action HDF5 for Diffusion Policy.
#
# Usage:
#   sbatch scripts/slurm/convert_square_ph_dp_lowdim_abs.sh

#SBATCH --job-name=conv_ph_abs
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/iris/u/jasonyan/slurm/%j_conv_ph_abs.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_conv_ph_abs.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-robodiff}"
set -u

DP_REPO="${DP_REPO:-/iris/u/jasonyan/repos/diffusion_policy}"
INPUT_HDF5="${INPUT_HDF5:-/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image.hdf5}"
OUTPUT_HDF5="${OUTPUT_HDF5:-/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/low_dim_abs.hdf5}"
EVAL_DIR="${EVAL_DIR:-/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/abs_conversion_eval}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ ! -f "${INPUT_HDF5}" ]]; then
  echo "missing INPUT_HDF5=${INPUT_HDF5}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "$(dirname "${OUTPUT_HDF5}")" "$(dirname "${EVAL_DIR}")"
cd "${DP_REPO}"

export PYTHONPATH="${DP_REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "hostname=$(hostname)"
echo "input_hdf5=${INPUT_HDF5}"
echo "output_hdf5=${OUTPUT_HDF5}"
echo "eval_dir=${EVAL_DIR}"
echo "num_workers=${NUM_WORKERS}"

rm -f "${OUTPUT_HDF5}"
rm -rf "${EVAL_DIR}"

python diffusion_policy/scripts/robomimic_dataset_conversion.py \
  --input "${INPUT_HDF5}" \
  --output "${OUTPUT_HDF5}" \
  --eval_dir "${EVAL_DIR}" \
  --num_workers "${NUM_WORKERS}"
