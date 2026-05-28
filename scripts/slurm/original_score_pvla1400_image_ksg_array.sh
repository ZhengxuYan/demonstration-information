#!/bin/bash
# Score the 1400 rollout RLDS dataset with original-style VAE KSG.
#
# Usage:
#   sbatch --array=1-1%1 scripts/slurm/original_score_pvla1400_image_ksg_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=orig_pvla_ksg
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_orig_pvla_ksg.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_orig_pvla_ksg.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_NAME="${DATASET_NAME:-pomdp_vla_square_rollouts_1400}"
VAE_ROOT="${VAE_ROOT:-/iris/u/jasonyan/data/deminf_outputs/pomdp_vla_square_rollouts_1400_original_pipeline}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/deminf_outputs/pomdp_vla_square_rollouts_1400_original_pipeline_scores}"
SEEDS="${SEEDS:-1}"

read -r -a SEED_ARRAY <<< "${SEEDS}"
NUM_SEEDS="${#SEED_ARRAY[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if (( TASK_ID < 1 || TASK_ID > NUM_SEEDS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside 1..${NUM_SEEDS}" >&2
  exit 2
fi

SEED="${SEED_ARRAY[$((TASK_ID - 1))]}"
OBS_CKPT="${VAE_ROOT}/config-vae_robomimic_image_env-${DATASET_NAME}_type-s_seed-${SEED}/100000"
ACTION_CKPT="${VAE_ROOT}/config-vae_robomimic_image_env-${DATASET_NAME}_type-a_seed-${SEED}/50000"
OUT_DIR="${SCORE_ROOT}/${DATASET_NAME}/ksg/seed-${SEED}"

if [[ ! -d "${OBS_CKPT}" ]]; then
  echo "missing obs checkpoint: ${OBS_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${ACTION_CKPT}" ]]; then
  echo "missing action checkpoint: ${ACTION_CKPT}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${OUT_DIR}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}/${NUM_SEEDS}"
echo "dataset=${DATASET_NAME}"
echo "seed=${SEED}"
echo "obs_ckpt=${OBS_CKPT}"
echo "action_ckpt=${ACTION_CKPT}"
echo "out_dir=${OUT_DIR}"

python scripts/quality/estimate_quality.py \
  --estimator=ksg \
  --obs_ckpt="${OBS_CKPT}" \
  --action_ckpt="${ACTION_CKPT}" \
  --batch_size="${BATCH_SIZE:-1024}" \
  --path="${OUT_DIR}"
