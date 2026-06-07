#!/bin/bash
# Download Transport PH low-dim robomimic data and convert delta actions to absolute actions.
#
# Usage:
#   sbatch scripts/slurm/prepare_transport_ph_dp_lowdim_abs.sh

#SBATCH --job-name=conv_trans_abs
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/iris/u/jasonyan/slurm/%j_conv_trans_abs.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_conv_trans_abs.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-robodiff}"
set -u

DP_REPO="${DP_REPO:-/iris/u/jasonyan/repos/diffusion_policy}"
DATA_DIR="${DATA_DIR:-/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/transport/ph}"
INPUT_HDF5="${INPUT_HDF5:-${DATA_DIR}/low_dim_v15.hdf5}"
OUTPUT_HDF5="${OUTPUT_HDF5:-${DATA_DIR}/low_dim_abs.hdf5}"
EVAL_DIR="${EVAL_DIR:-${DATA_DIR}/abs_conversion_eval}"
NUM_WORKERS="${NUM_WORKERS:-8}"
URL="${URL:-https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5/transport/ph/low_dim_v15.hdf5}"

mkdir -p /iris/u/jasonyan/slurm "${DATA_DIR}" "${EVAL_DIR}"

echo "hostname=$(hostname)"
echo "input_hdf5=${INPUT_HDF5}"
echo "output_hdf5=${OUTPUT_HDF5}"
echo "eval_dir=${EVAL_DIR}"

if [[ ! -f "${INPUT_HDF5}" ]]; then
  echo "downloading ${URL}"
  curl -L --fail --retry 5 --retry-delay 10 -o "${INPUT_HDF5}.tmp" "${URL}"
  mv "${INPUT_HDF5}.tmp" "${INPUT_HDF5}"
fi
ls -lh "${INPUT_HDF5}"

cd "${DP_REPO}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=/sailhome/jasonyan/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH="${DP_REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

rm -f "${OUTPUT_HDF5}"
rm -rf "${EVAL_DIR}"

python diffusion_policy/scripts/robomimic_dataset_conversion.py \
  --input "${INPUT_HDF5}" \
  --output "${OUTPUT_HDF5}" \
  --eval_dir "${EVAL_DIR}" \
  --num_workers "${NUM_WORKERS}"

ls -lh "${OUTPUT_HDF5}"
