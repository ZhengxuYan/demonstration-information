#!/bin/bash
# Train DemInf/OpenX image-based diffusion policies for pen-in-cup human label filters.
#
# Usage:
#   sbatch --array=1-6%3 scripts/slurm/train_pen_in_cup_dp_human_label_array.sh
#
# Array mapping with defaults:
#   1 observability drop25
#   2 observability drop50
#   3 observability drop75
#   4 optimality drop25
#   5 optimality drop50
#   6 optimality drop75

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=pic_dp_hlabel
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_pic_dp_hlabel.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_pic_dp_hlabel.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LD_LIBRARY_PATH="/sailhome/jasonyan/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
RLDS_PATH="${RLDS_PATH:-/iris/u/jasonyan/data/droid_pen_in_cup_06072026_rlds/droid_pen_in_cup/1.0.0}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/droid_pen_in_cup_06072026_human_label_scores}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_dp_pen_in_cup_06072026}"
ESTIMATORS="${ESTIMATORS:-observability optimality}"
DROP_PERCENTS="${DROP_PERCENTS:-25 50 75}"
SEED="${SEED:-1}"
PROJECT="${WANDB_PROJECT:-deminf-dp-pen-in-cup}"

read -r -a ESTIMATOR_ARRAY <<< "${ESTIMATORS}"
read -r -a DROP_ARRAY <<< "${DROP_PERCENTS}"
NUM_TASKS=$(( ${#ESTIMATOR_ARRAY[@]} * ${#DROP_ARRAY[@]} ))
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if (( TASK_ID < 1 || TASK_ID > NUM_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside 1..${NUM_TASKS}" >&2
  exit 2
fi

ZERO_INDEX=$((TASK_ID - 1))
ESTIMATOR="${ESTIMATOR_ARRAY[$((ZERO_INDEX / ${#DROP_ARRAY[@]}))]}"
DROP_PERCENT="${DROP_ARRAY[$((ZERO_INDEX % ${#DROP_ARRAY[@]}))]}"
SCORE_PKL="${SCORE_ROOT}/pen_in_cup/${ESTIMATOR}/${ESTIMATOR}_scores.pkl"
RUN_NAME="pen_in_cup_dp_${ESTIMATOR}_drop_${DROP_PERCENT}_seed${SEED}"

if [[ ! -f "${RLDS_PATH}/dataset_info.json" ]]; then
  echo "missing RLDS dataset_info.json under ${RLDS_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SCORE_PKL}" ]]; then
  echo "missing score pkl: ${SCORE_PKL}" >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}/${NUM_TASKS}"
echo "estimator=${ESTIMATOR}"
echo "drop_percent=${DROP_PERCENT}"
echo "seed=${SEED}"
echo "rlds_path=${RLDS_PATH}"
echo "score_root=${SCORE_ROOT}"
echo "score_pkl=${SCORE_PKL}"
echo "out_root=${OUT_ROOT}"
echo "run_name=${RUN_NAME}"

python scripts/train.py \
  --config="configs/bc/droid_pen_in_cup_dp_random_drop.py:pen_in_cup,${DROP_PERCENT},${ESTIMATOR},${SEED},${SCORE_ROOT},${RLDS_PATH}" \
  --path="${OUT_ROOT}" \
  --name="${RUN_NAME}" \
  --project="${PROJECT}" \
  --include_timestamp=false
