#!/bin/bash
# Train the upstream OpenX Robomimic-MH image VAEs.
#
# This follows configs/quality/sweep_robomimic_image_vae.json:
#   env in lift/mh can/mh square/mh
#   type in s a
#   seed in 1 2 3
#
# Usage:
#   sbatch --array=1-18%6 scripts/slurm/original_train_mh_image_vaes_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=orig_mh_vae
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_orig_mh_vae.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_orig_mh_vae.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/robomimic_image_agent_original_repro}"
ENVS="${ENVS:-lift/mh can/mh square/mh}"
TYPES="${TYPES:-s a}"
SEEDS="${SEEDS:-1 2 3}"

read -r -a ENV_ARRAY <<< "${ENVS}"
read -r -a TYPE_ARRAY <<< "${TYPES}"
read -r -a SEED_ARRAY <<< "${SEEDS}"
NUM_ENVS="${#ENV_ARRAY[@]}"
NUM_TYPES="${#TYPE_ARRAY[@]}"
NUM_SEEDS="${#SEED_ARRAY[@]}"
NUM_TASKS=$((NUM_ENVS * NUM_TYPES * NUM_SEEDS))
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if (( TASK_ID < 1 || TASK_ID > NUM_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside 1..${NUM_TASKS}" >&2
  exit 2
fi

ZERO_BASED=$((TASK_ID - 1))
ENV_IDX=$((ZERO_BASED / (NUM_TYPES * NUM_SEEDS)))
REM=$((ZERO_BASED % (NUM_TYPES * NUM_SEEDS)))
TYPE_IDX=$((REM / NUM_SEEDS))
SEED_IDX=$((REM % NUM_SEEDS))

ENV_NAME="${ENV_ARRAY[$ENV_IDX]}"
CONFIG_TYPE="${TYPE_ARRAY[$TYPE_IDX]}"
SEED="${SEED_ARRAY[$SEED_IDX]}"
ENV_TAG="${ENV_NAME//\//_}"
RUN_NAME="config-vae_robomimic_image_env-${ENV_TAG}_type-${CONFIG_TYPE}_seed-${SEED}"

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}/${NUM_TASKS}"
echo "env=${ENV_NAME}"
echo "type=${CONFIG_TYPE}"
echo "seed=${SEED}"
echo "run_name=${RUN_NAME}"
echo "out_root=${OUT_ROOT}"

python scripts/train.py \
  --config="configs/quality/vae_robomimic_image.py:${ENV_NAME},${CONFIG_TYPE},${SEED},agent" \
  --path="${OUT_ROOT}" \
  --name="${RUN_NAME}" \
  --project="${WANDB_PROJECT:-original-mh-deminf-repro}" \
  --include_timestamp=false
