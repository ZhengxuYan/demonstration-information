#!/bin/bash
# Filter 1000 best successful rollouts from the 10-seed Diffusion Policy rollout set.
#
# Usage:
#   sbatch scripts/slurm/filter_square_ph_dp_lowdim_rollouts_1000.sh

#SBATCH --job-name=filter_ph_dp
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/iris/u/jasonyan/slurm/%j_filter_ph_dp.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_filter_ph_dp.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-robodiff}"
set -u

DEMO_REPO="${DEMO_REPO:-/iris/u/jasonyan/repos/demonstration-information}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-/iris/u/jasonyan/data/diffusion_policy_rollouts/square_ph_lowdim_abs_10seed}"
OUT_HDF5="${OUT_HDF5:-/iris/u/jasonyan/data/diffusion_policy_rollouts/square_ph_lowdim_abs_10seed_filtered_1000/image.hdf5}"

mkdir -p /iris/u/jasonyan/slurm "$(dirname "${OUT_HDF5}")"
cd "${DEMO_REPO}"

mapfile -t INPUTS < <(find "${ROLLOUT_ROOT}" -maxdepth 2 -type f -name 'rollouts.hdf5' | sort)
if [[ "${#INPUTS[@]}" -eq 0 ]]; then
  echo "no rollout hdf5 files found under ${ROLLOUT_ROOT}" >&2
  exit 1
fi

echo "hostname=$(hostname)"
echo "num_inputs=${#INPUTS[@]}"
echo "out_hdf5=${OUT_HDF5}"

python scripts/quality/filter_robomimic_rollouts.py \
  --inputs "${INPUTS[@]}" \
  --output "${OUT_HDF5}" \
  --num-demos "${NUM_DEMOS:-1000}" \
  --valid-ratio "${VALID_RATIO:-0.1}" \
  --seed "${SPLIT_SEED:-1}" \
  --overwrite
