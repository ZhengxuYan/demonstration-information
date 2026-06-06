#!/bin/bash
# Prepare Tian's random-view Square rollout dataset for original DemInf.
#
# The original DemInf VAE config should still use camera=agent. This script
# makes that safe by copying thirdperson_1_image into agentview_image before
# building RLDS, because the RoboMimic RLDS builder reads agentview_image.
#
# Usage:
#   sbatch scripts/slurm/prepare_square_randomviews_thirdperson1_deminf.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --job-name=prep_randview
#SBATCH --output=/iris/u/jasonyan/slurm/%j_prep_randview.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_prep_randview.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
SOURCE_HDF5="${SOURCE_HDF5:-/scr/tiangao/datasets/square_1400_rollouts_randomviews.hdf5}"
DATASET_NAME="${DATASET_NAME:-square_1400_rollouts_randomviews_thirdperson1}"
DATASET_HDF5="${DATASET_HDF5:-/iris/u/jasonyan/data/${DATASET_NAME}/image.hdf5}"
RLDS_ROOT="${RLDS_ROOT:-/iris/u/jasonyan/data/${DATASET_NAME}_rlds}"
MANUAL_DIR="${MANUAL_DIR:-/iris/u/jasonyan/data/${DATASET_NAME}_rlds_manual/${DATASET_NAME}}"
SOURCE_IMAGE_KEY="${SOURCE_IMAGE_KEY:-thirdperson_1_image}"
EXPECTED_DEMOS="${EXPECTED_DEMOS:-1400}"

mkdir -p /iris/u/jasonyan/slurm "${MANUAL_DIR}" "${RLDS_ROOT}/${DATASET_NAME}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "source_hdf5=${SOURCE_HDF5}"
echo "dataset_hdf5=${DATASET_HDF5}"
echo "source_image_key=${SOURCE_IMAGE_KEY}"
echo "dataset_name=${DATASET_NAME}"
echo "rlds_root=${RLDS_ROOT}"

python scripts/quality/alias_hdf5_image_key_to_agentview.py \
  --input "${SOURCE_HDF5}" \
  --output "${DATASET_HDF5}" \
  --source-image-key "${SOURCE_IMAGE_KEY}" \
  --target-image-key agentview_image \
  --expected-demos "${EXPECTED_DEMOS}" \
  --overwrite

python scripts/quality/verify_policy_view_dataset.py "${DATASET_HDF5}" \
  --expected-demos "${EXPECTED_DEMOS}" \
  --expected-action-dim 7 \
  --required-obs-key agentview_image \
  --required-obs-key robot0_eye_in_hand_image \
  --required-obs-key robot0_eef_pos \
  --required-obs-key robot0_eef_quat \
  --required-obs-key robot0_gripper_qpos \
  --expected-obs-shape agentview_image=84,84,3 \
  --expected-obs-shape robot0_eye_in_hand_image=84,84,3 \
  --expected-obs-shape robot0_eef_pos=3 \
  --expected-obs-shape robot0_eef_quat=4 \
  --expected-obs-shape robot0_gripper_qpos=2

ln -sfn "${DATASET_HDF5}" "${MANUAL_DIR}/image.hdf5"

cd "${REPO}/rlds/robomimic"
rm -rf "${RLDS_ROOT}/${DATASET_NAME}/robo_mimic"
tfds build \
  --manual_dir "${MANUAL_DIR}" \
  --data_dir "${RLDS_ROOT}/${DATASET_NAME}"

echo "${RLDS_ROOT}/${DATASET_NAME}/robo_mimic/1.0.0"
