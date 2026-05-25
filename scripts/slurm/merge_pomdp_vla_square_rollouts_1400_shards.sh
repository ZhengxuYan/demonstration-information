#!/bin/bash
# Merge rendered POMDP-VLA Square rollout shards into one 1400-demo HDF5.
#
# Usage:
#   sbatch scripts/slurm/merge_pomdp_vla_square_rollouts_1400_shards.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name=merge_pvla1400
#SBATCH --output=/iris/u/jasonyan/slurm/%j_merge_pvla1400.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_merge_pvla1400.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-robodiff}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
SHARD_ROOT="${SHARD_ROOT:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400_shards}"
OUT_HDF5="${OUT_HDF5:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400/image.hdf5}"
MAX_DEMOS="${MAX_DEMOS:-1400}"

mkdir -p /iris/u/jasonyan/slurm "$(dirname "${OUT_HDF5}")"
cd "${REPO}"

python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "hostname=$(hostname)"
echo "shard_root=${SHARD_ROOT}"
echo "out_hdf5=${OUT_HDF5}"
echo "max_demos=${MAX_DEMOS}"

python scripts/quality/prepare_pomdp_vla_square_rollout_dataset.py \
  --input-root "${SHARD_ROOT}" \
  --glob "image.hdf5" \
  --output "${OUT_HDF5}" \
  --max-demos "${MAX_DEMOS}" \
  --valid-ratio "${VALID_RATIO:-0.1}" \
  --split-seed "${SPLIT_SEED:-1}" \
  --overwrite
