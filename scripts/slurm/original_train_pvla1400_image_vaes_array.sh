#!/bin/bash
# Train original-style OpenX image VAEs on the 1400 rollout RLDS dataset.
#
# Uses the upstream Robomimic image VAE architecture with agent view.
#
# Usage:
#   sbatch --array=1-2%2 scripts/slurm/original_train_pvla1400_image_vaes_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=orig_pvla_vae
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_orig_pvla_vae.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_orig_pvla_vae.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_NAME="${DATASET_NAME:-pomdp_vla_square_rollouts_1400}"
RLDS_PATH="${RLDS_PATH:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400_rlds/pomdp_vla_square_rollouts_1400/robo_mimic/1.0.0}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/pomdp_vla_square_rollouts_1400_original_pipeline}"
TYPES="${TYPES:-s a}"
SEEDS="${SEEDS:-1}"

read -r -a TYPE_ARRAY <<< "${TYPES}"
read -r -a SEED_ARRAY <<< "${SEEDS}"
NUM_TYPES="${#TYPE_ARRAY[@]}"
NUM_SEEDS="${#SEED_ARRAY[@]}"
NUM_TASKS=$((NUM_TYPES * NUM_SEEDS))
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if (( TASK_ID < 1 || TASK_ID > NUM_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside 1..${NUM_TASKS}" >&2
  exit 2
fi

ZERO_BASED=$((TASK_ID - 1))
TYPE_IDX=$((ZERO_BASED / NUM_SEEDS))
SEED_IDX=$((ZERO_BASED % NUM_SEEDS))

CONFIG_TYPE="${TYPE_ARRAY[$TYPE_IDX]}"
SEED="${SEED_ARRAY[$SEED_IDX]}"
RUN_NAME="config-vae_robomimic_image_env-${DATASET_NAME}_type-${CONFIG_TYPE}_seed-${SEED}"

if [[ ! -f "${RLDS_PATH}/dataset_info.json" ]]; then
  echo "missing RLDS dataset_info.json under ${RLDS_PATH}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}/${NUM_TASKS}"
echo "dataset=${DATASET_NAME}"
echo "rlds_path=${RLDS_PATH}"
echo "type=${CONFIG_TYPE}"
echo "seed=${SEED}"
echo "run_name=${RUN_NAME}"
echo "out_root=${OUT_ROOT}"

python scripts/train.py \
  --config="configs/quality/vae_robomimic_image.py:${DATASET_NAME},${CONFIG_TYPE},${SEED},agent,${RLDS_PATH}" \
  --path="${OUT_ROOT}" \
  --name="${RUN_NAME}" \
  --project="${WANDB_PROJECT:-original-pvla1400-deminf}" \
  --include_timestamp=false
