#!/bin/bash
# Score camera-view datasets with the DemInf VAEs trained on the same dataset.
#
# Usage:
#   sbatch --array=1-4%4 scripts/slurm/score_trained_deminf_camera_view_dataset_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris9,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --job-name=score_tr_deminf
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_score_tr_deminf.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_score_tr_deminf.err

set -euo pipefail

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
RLDS_ROOT="${RLDS_ROOT:-/iris/u/jasonyan/data/deminf_camera_view_rlds}"
TRAIN_ROOT="${TRAIN_ROOT:-/iris/u/jasonyan/data/deminf_outputs/camera_view_train}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/camera_view_scores_trained}"
DATASETS="${DATASETS:-ph_agentview 400_agentview 400_left_close_low 400_mix}"
MODE="${MODE:-image_proprio}"

if [[ "${MODE}" != "image_proprio" ]]; then
  echo "Only MODE=image_proprio is supported by this script. Got: ${MODE}" >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
read -r -a DATASET_ARRAY <<< "${DATASETS}"
N_TASKS="${#DATASET_ARRAY[@]}"
if [[ "${TASK_ID}" -lt 1 || "${TASK_ID}" -gt "${N_TASKS}" ]]; then
  echo "Task ${TASK_ID} is out of range for ${N_TASKS} datasets." >&2
  exit 2
fi

DATASET="${DATASET_ARRAY[$((TASK_ID - 1))]}"
RLDS="${RLDS_ROOT}/${DATASET}/robo_mimic/1.0.0"
TRAIN_DIR="${TRAIN_ROOT}/${DATASET}"
OUTPUT="${OUT_ROOT}/${MODE}/${DATASET}"

if [[ ! -d "${RLDS}" ]]; then
  echo "RLDS directory does not exist: ${RLDS}" >&2
  exit 1
fi
if [[ ! -d "${TRAIN_DIR}" ]]; then
  echo "Train directory does not exist: ${TRAIN_DIR}" >&2
  exit 1
fi

OBS_CKPT="$(find "${TRAIN_DIR}" -maxdepth 1 -type d -name "${DATASET}_image_proprio_obs_vae_seed1_*" | sort | tail -1)"
ACTION_CKPT="$(find "${TRAIN_DIR}" -maxdepth 1 -type d -name "${DATASET}_action_vae_seed1_*" | sort | tail -1)"

if [[ -z "${OBS_CKPT}" || ! -d "${OBS_CKPT}/100000" ]]; then
  echo "Missing completed obs checkpoint for ${DATASET}: ${OBS_CKPT}" >&2
  exit 1
fi
if [[ -z "${ACTION_CKPT}" || ! -d "${ACTION_CKPT}/100000" ]]; then
  echo "Missing completed action checkpoint for ${DATASET}: ${ACTION_CKPT}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${OUTPUT}"

echo "hostname=$(hostname)"
echo "dataset=${DATASET}"
echo "mode=${MODE}"
echo "obs_ckpt=${OBS_CKPT}"
echo "action_ckpt=${ACTION_CKPT}"
echo "rlds=${RLDS}"
echo "output=${OUTPUT}"

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx
set -u

cd "${REPO}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

python scripts/quality/estimate_quality_combined_robomimic.py \
  --obs_ckpt "${OBS_CKPT}" \
  --action_ckpt "${ACTION_CKPT}" \
  --square_dataset_name "${DATASET}" \
  --square_path_override "${RLDS}" \
  --output "${OUTPUT}" \
  --batch_size 1024
