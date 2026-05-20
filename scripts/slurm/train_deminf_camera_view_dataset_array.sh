#!/bin/bash
# Train DemInf VAEs for the camera-view datasets.
#
# Each array task trains one VAE:
#   dataset x {obs, action}
#
# Usage:
#   sbatch --array=1-8%6 scripts/slurm/train_deminf_camera_view_dataset_array.sh
#   DATASETS="ph_agentview 400_agentview 400_left_close_low" sbatch --array=1-6%6 scripts/slurm/train_deminf_camera_view_dataset_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris9,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --job-name=train_deminf_cam
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_train_deminf_cam.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_train_deminf_cam.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
RLDS_ROOT="${RLDS_ROOT:-/iris/u/jasonyan/data/deminf_camera_view_rlds}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/camera_view_train}"
DATASETS="${DATASETS:-ph_agentview 400_agentview 400_left_close_low 400_mix}"
SEED="${SEED:-1}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if [[ "${TASK_ID}" -lt 1 ]]; then
  echo "SLURM_ARRAY_TASK_ID must be >= 1. Got: ${TASK_ID}" >&2
  exit 2
fi

read -r -a DATASET_ARRAY <<< "${DATASETS}"
N_DATASETS="${#DATASET_ARRAY[@]}"
N_TASKS="$((N_DATASETS * 2))"
if [[ "${TASK_ID}" -gt "${N_TASKS}" ]]; then
  echo "Task ${TASK_ID} is out of range for ${N_DATASETS} datasets (${N_TASKS} tasks)." >&2
  exit 2
fi

DATASET_INDEX="$(((TASK_ID - 1) / 2))"
KIND_INDEX="$(((TASK_ID - 1) % 2))"
DATASET="${DATASET_ARRAY[DATASET_INDEX]}"
if [[ "${KIND_INDEX}" -eq 0 ]]; then
  KIND="obs"
  CONFIG_TYPE="s"
  CAMERA="both"
  NAME="${DATASET}_image_proprio_obs_vae_seed${SEED}"
else
  KIND="action"
  CONFIG_TYPE="a"
  CAMERA="wrist"
  NAME="${DATASET}_action_vae_seed${SEED}"
fi

RLDS="${RLDS_ROOT}/${DATASET}/robo_mimic/1.0.0"
if [[ ! -d "${RLDS}" ]]; then
  echo "RLDS directory does not exist: ${RLDS}" >&2
  exit 1
fi

DATASET_SPEC="${DATASET}=${RLDS}"
OUT="${OUT_ROOT}/${DATASET}"

mkdir -p /iris/u/jasonyan/slurm "${OUT}"

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}/${N_TASKS}"
echo "dataset=${DATASET}"
echo "kind=${KIND}"
echo "rlds=${RLDS}"
echo "out=${OUT}"
echo "name=${NAME}"

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx
set -u

cd "${REPO}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

python scripts/train.py \
  --config="configs/quality/vae_robomimic_image.py:${DATASET},${CONFIG_TYPE},${SEED},${CAMERA},${DATASET_SPEC}" \
  --path "${OUT}" \
  --name "${NAME}"
