#!/bin/bash
# Build full / 75% / 50% / 25% BC datasets by dropping low DemInf-score episodes.
#
# Usage:
#   sbatch scripts/slurm/make_pomdp_vla_square_rollouts_1400_filtered_bc_datasets.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name=filter_pvla1400
#SBATCH --output=/iris/u/jasonyan/slurm/%j_filter_pvla1400.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_filter_pvla1400.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_NAME="${DATASET_NAME:-pomdp_vla_square_rollouts_1400}"
DATASET_HDF5="${DATASET_HDF5:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400/image.hdf5}"
SCORE_CSV="${SCORE_CSV:-/iris/u/jasonyan/data/deminf_outputs/pomdp_vla_square_rollouts_1400_scores_trained/image_proprio/episode_scores.csv}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400_filtered_bc_datasets/mi_score}"

if [[ ! -f "${DATASET_HDF5}" ]]; then
  echo "missing DATASET_HDF5=${DATASET_HDF5}" >&2
  exit 1
fi
if [[ ! -f "${SCORE_CSV}" ]]; then
  echo "missing SCORE_CSV=${SCORE_CSV}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

python scripts/quality/make_deminf_mi_filtered_bc_datasets.py \
  --episode-csv "${SCORE_CSV}" \
  --source-hdf5 "${DATASET_NAME}=${DATASET_HDF5}" \
  --output-root "${OUT_ROOT}" \
  --datasets "${DATASET_NAME}" \
  --drop-fractions 0 0.25 0.5 0.75 \
  --drop-side low \
  --score-universe scored \
  --valid-ratio "${VALID_RATIO:-0.1}" \
  --split-seed "${SPLIT_SEED:-1}" \
  --overwrite
