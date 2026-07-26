#!/bin/bash
# Export and validate the Wrench-on-Hook 0722 image+proprio density HDF5.

#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodelist=iris8
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=w0722_pomdp_h5
#SBATCH --output=/iris/u/jasonyan/slurm/%j_w0722_pomdp_h5.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_w0722_pomdp_h5.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
RAW_ROOT="${RAW_ROOT:-/scr/tiangao/wrench_on_hook_0722}"
LABELS_CSV="${LABELS_CSV:-/scr/tiangao/wrench_on_hook_0722_labels.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/iris/u/jasonyan/data/wrench_on_hook_0722_pomdp}"
OUTPUT_HDF5="${OUTPUT_HDF5:-${OUTPUT_ROOT}/wrench_on_hook_0722_image_proprio_cartvel7d.hdf5}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/wrench_on_hook_0722_manifest.csv}"
SEED="${SEED:-20260725}"

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"

cd "${REPO}"
mkdir -p "${OUTPUT_ROOT}" /iris/u/jasonyan/slurm

python scripts/quality/prepare_wrench_0722_image_proprio_density_hdf5.py \
  --root "${RAW_ROOT}" \
  --labels-csv "${LABELS_CSV}" \
  --output "${OUTPUT_HDF5}" \
  --seed "${SEED}" \
  --overwrite

python scripts/quality/validate_wrench_0722_density_hdf5.py \
  --input "${OUTPUT_HDF5}" \
  --raw-root "${RAW_ROOT}" \
  --contact-sheet "${OUTPUT_ROOT}/validation_contact_sheet.png"

python scripts/quality/export_density_hdf5_labels.py \
  --input "${OUTPUT_HDF5}" \
  --output "${MANIFEST}"

echo "WRENCH_0722_PREPARE_OK"
echo "output_hdf5=${OUTPUT_HDF5}"
echo "manifest=${MANIFEST}"
