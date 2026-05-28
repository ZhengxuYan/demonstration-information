#!/bin/bash
# Train original-style OpenX filtered BC on the 1400 rollout RLDS dataset.
#
# Usage:
#   PERCENTILES="50" sbatch --array=1-1%1 scripts/slurm/original_train_pvla1400_filtered_bc_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=orig_pvla_bc
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_orig_pvla_bc.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_orig_pvla_bc.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LD_LIBRARY_PATH="/sailhome/jasonyan/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
DATASET_NAME="${DATASET_NAME:-pomdp_vla_square_rollouts_1400}"
RLDS_PATH="${RLDS_PATH:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400_rlds/pomdp_vla_square_rollouts_1400/robo_mimic/1.0.0}"
HDF5_PATH="${HDF5_PATH:-/iris/u/jasonyan/data/pomdp_vla_square_rollouts_1400/image.hdf5}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/deminf_outputs/pomdp_vla_square_rollouts_1400_original_pipeline_scores}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/pomdp_vla_square_rollouts_1400_original_pipeline_bc}"
PERCENTILES="${PERCENTILES:-50}"
ESTIMATOR="${ESTIMATOR:-ksg}"
SEEDS="${SEEDS:-1}"

read -r -a PERCENTILE_ARRAY <<< "${PERCENTILES}"
read -r -a SEED_ARRAY <<< "${SEEDS}"
NUM_PERCENTILES="${#PERCENTILE_ARRAY[@]}"
NUM_SEEDS="${#SEED_ARRAY[@]}"
NUM_TASKS=$((NUM_PERCENTILES * NUM_SEEDS))
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if (( TASK_ID < 1 || TASK_ID > NUM_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside 1..${NUM_TASKS}" >&2
  exit 2
fi

ZERO_BASED=$((TASK_ID - 1))
PERCENTILE_IDX=$((ZERO_BASED / NUM_SEEDS))
SEED_IDX=$((ZERO_BASED % NUM_SEEDS))

PERCENTILE="${PERCENTILE_ARRAY[$PERCENTILE_IDX]}"
SEED="${SEED_ARRAY[$SEED_IDX]}"
SCORE_PKL="${SCORE_ROOT}/${DATASET_NAME}/${ESTIMATOR}/seed-${SEED}/${DATASET_NAME}.pkl"
RUN_NAME="config-robomimic_image_filter_env-${DATASET_NAME}_percentile-${PERCENTILE}_estimator-${ESTIMATOR}_seed-${SEED}"

if [[ ! -f "${SCORE_PKL}" ]]; then
  echo "missing score pkl: ${SCORE_PKL}" >&2
  echo "Run scripts/slurm/original_score_pvla1400_image_ksg_array.sh first." >&2
  exit 1
fi
if [[ ! -f "${HDF5_PATH}" ]]; then
  echo "missing HDF5 dataset: ${HDF5_PATH}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}/${NUM_TASKS}"
echo "dataset=${DATASET_NAME}"
echo "rlds_path=${RLDS_PATH}"
echo "hdf5_path=${HDF5_PATH}"
echo "percentile=${PERCENTILE}"
echo "estimator=${ESTIMATOR}"
echo "seed=${SEED}"
echo "score_pkl=${SCORE_PKL}"
echo "run_name=${RUN_NAME}"
echo "out_root=${OUT_ROOT}"

python scripts/train.py \
  --config="configs/bc/robomimic_image_filter.py:${DATASET_NAME},${PERCENTILE},${ESTIMATOR},${SEED},${SCORE_ROOT},${RLDS_PATH},${HDF5_PATH}" \
  --path="${OUT_ROOT}" \
  --name="${RUN_NAME}" \
  --project="${WANDB_PROJECT:-original-pvla1400-bc}" \
  --include_timestamp=false
