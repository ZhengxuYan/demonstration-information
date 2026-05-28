#!/bin/bash
# Score upstream OpenX Robomimic-MH datasets with VAE KSG.
#
# This is the KSG subset of tools/generate_quality_sweep.py --mode image:
#   obs checkpoint:    VAE type s, step 100000
#   action checkpoint: VAE type a, step 50000
#
# Usage:
#   sbatch --array=1-9%3 scripts/slurm/original_score_mh_image_ksg_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=orig_mh_ksg
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_orig_mh_ksg.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_orig_mh_ksg.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
VAE_ROOT="${VAE_ROOT:-/iris/u/jasonyan/data/deminf_outputs/robomimic_image_agent_original_repro}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/deminf_outputs/robomimic_image_agent_inference}"
ENVS="${ENVS:-lift/mh can/mh square/mh}"
SEEDS="${SEEDS:-1 2 3}"

read -r -a ENV_ARRAY <<< "${ENVS}"
read -r -a SEED_ARRAY <<< "${SEEDS}"
NUM_ENVS="${#ENV_ARRAY[@]}"
NUM_SEEDS="${#SEED_ARRAY[@]}"
NUM_TASKS=$((NUM_ENVS * NUM_SEEDS))
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if (( TASK_ID < 1 || TASK_ID > NUM_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside 1..${NUM_TASKS}" >&2
  exit 2
fi

ZERO_BASED=$((TASK_ID - 1))
ENV_IDX=$((ZERO_BASED / NUM_SEEDS))
SEED_IDX=$((ZERO_BASED % NUM_SEEDS))

ENV_NAME="${ENV_ARRAY[$ENV_IDX]}"
SEED="${SEED_ARRAY[$SEED_IDX]}"
ENV_TAG="${ENV_NAME//\//_}"
OBS_CKPT="${VAE_ROOT}/config-vae_robomimic_image_env-${ENV_TAG}_type-s_seed-${SEED}/100000"
ACTION_CKPT="${VAE_ROOT}/config-vae_robomimic_image_env-${ENV_TAG}_type-a_seed-${SEED}/50000"
OUT_DIR="${SCORE_ROOT}/${ENV_TAG}/ksg/seed-${SEED}"

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
echo "task_id=${TASK_ID}/${NUM_TASKS}"
echo "env=${ENV_NAME}"
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
