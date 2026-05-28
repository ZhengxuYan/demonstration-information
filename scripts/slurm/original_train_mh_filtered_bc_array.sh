#!/bin/bash
# Train upstream OpenX filtered BC on Robomimic-MH image datasets.
#
# This calls configs/bc/robomimic_image_filter.py with:
#   env, percentile, estimator, seed
#
# By default this runs the config default percentile 50 for all MH envs / seeds.
# Override PERCENTILES to sweep more filters, e.g.:
#   PERCENTILES="0 10 20 30 40 50 60 70 80 90"
#
# Usage:
#   sbatch --array=1-9%3 scripts/slurm/original_train_mh_filtered_bc_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=orig_mh_bc
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_orig_mh_bc.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_orig_mh_bc.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/robomimic_image_filter_bc_original_repro}"
ENVS="${ENVS:-lift/mh can/mh square/mh}"
PERCENTILES="${PERCENTILES:-50}"
ESTIMATOR="${ESTIMATOR:-ksg}"
SEEDS="${SEEDS:-1 2 3}"

read -r -a ENV_ARRAY <<< "${ENVS}"
read -r -a PERCENTILE_ARRAY <<< "${PERCENTILES}"
read -r -a SEED_ARRAY <<< "${SEEDS}"
NUM_ENVS="${#ENV_ARRAY[@]}"
NUM_PERCENTILES="${#PERCENTILE_ARRAY[@]}"
NUM_SEEDS="${#SEED_ARRAY[@]}"
NUM_TASKS=$((NUM_ENVS * NUM_PERCENTILES * NUM_SEEDS))
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if (( TASK_ID < 1 || TASK_ID > NUM_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside 1..${NUM_TASKS}" >&2
  exit 2
fi

ZERO_BASED=$((TASK_ID - 1))
ENV_IDX=$((ZERO_BASED / (NUM_PERCENTILES * NUM_SEEDS)))
REM=$((ZERO_BASED % (NUM_PERCENTILES * NUM_SEEDS)))
PERCENTILE_IDX=$((REM / NUM_SEEDS))
SEED_IDX=$((REM % NUM_SEEDS))

ENV_NAME="${ENV_ARRAY[$ENV_IDX]}"
PERCENTILE="${PERCENTILE_ARRAY[$PERCENTILE_IDX]}"
SEED="${SEED_ARRAY[$SEED_IDX]}"
ENV_TAG="${ENV_NAME//\//_}"
SCORE_PKL="/iris/u/jasonyan/data/deminf_outputs/robomimic_image_inference/${ENV_TAG}/${ESTIMATOR}/seed-${SEED}/${ENV_TAG}.pkl"
RUN_NAME="config-robomimic_image_filter_env-${ENV_TAG}_percentile-${PERCENTILE}_estimator-${ESTIMATOR}_seed-${SEED}"

if [[ ! -f "${SCORE_PKL}" ]]; then
  echo "missing score pkl: ${SCORE_PKL}" >&2
  echo "Run scripts/slurm/original_score_mh_image_ksg_array.sh first." >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}/${NUM_TASKS}"
echo "env=${ENV_NAME}"
echo "percentile=${PERCENTILE}"
echo "estimator=${ESTIMATOR}"
echo "seed=${SEED}"
echo "score_pkl=${SCORE_PKL}"
echo "run_name=${RUN_NAME}"
echo "out_root=${OUT_ROOT}"

python scripts/train.py \
  --config="configs/bc/robomimic_image_filter.py:${ENV_NAME},${PERCENTILE},${ESTIMATOR},${SEED}" \
  --path="${OUT_ROOT}" \
  --name="${RUN_NAME}" \
  --project="${WANDB_PROJECT:-original-mh-bc-repro}" \
  --include_timestamp=false
