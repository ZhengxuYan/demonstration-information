#!/bin/bash
# Train DemInf/OpenX image-based diffusion policies for pen-in-cup random episode drops.
#
# Usage:
#   sbatch --array=1-4%4 scripts/slurm/train_pen_in_cup_dp_random_drop_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=pic_dp_drop
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_pic_dp_drop.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_pic_dp_drop.err

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
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/droid_pen_in_cup_06072026_random_drop_scores}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_dp_pen_in_cup_06072026}"
DROP_PERCENTS="${DROP_PERCENTS:-0 25 50 75}"
SEED="${SEED:-1}"
PROJECT="${WANDB_PROJECT:-deminf-dp-pen-in-cup}"
RESUME="${RESUME:-0}"

read -r -a DROP_ARRAY <<< "${DROP_PERCENTS}"
NUM_TASKS="${#DROP_ARRAY[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

if (( TASK_ID < 1 || TASK_ID > NUM_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside 1..${NUM_TASKS}" >&2
  exit 2
fi

DROP_PERCENT="${DROP_ARRAY[$((TASK_ID - 1))]}"
SCORE_PKL="${SCORE_ROOT}/pen_in_cup/random/seed-${SEED}/random_drop_$(printf '%02d' "${DROP_PERCENT}")_seed${SEED}.pkl"
if [[ "${DROP_PERCENT}" == "0" ]]; then
  RUN_NAME="pen_in_cup_dp_full_seed${SEED}"
else
  RUN_NAME="pen_in_cup_dp_random_drop_${DROP_PERCENT}_seed${SEED}"
fi

if [[ ! -f "${RLDS_PATH}/dataset_info.json" ]]; then
  echo "missing RLDS dataset_info.json under ${RLDS_PATH}" >&2
  echo "Run scripts/slurm/prepare_droid_pen_in_cup_rlds.sh first." >&2
  exit 1
fi
if [[ "${DROP_PERCENT}" != "0" && ! -f "${SCORE_PKL}" ]]; then
  echo "missing score pkl: ${SCORE_PKL}" >&2
  echo "Run scripts/slurm/prepare_droid_pen_in_cup_rlds.sh first." >&2
  exit 1
fi

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}/${NUM_TASKS}"
echo "drop_percent=${DROP_PERCENT}"
echo "seed=${SEED}"
echo "rlds_path=${RLDS_PATH}"
echo "score_root=${SCORE_ROOT}"
echo "score_pkl=${SCORE_PKL}"
echo "out_root=${OUT_ROOT}"
echo "run_name=${RUN_NAME}"
echo "resume=${RESUME}"

TRAIN_ARGS=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "TRUE" ]]; then
  TRAIN_ARGS+=(--resume)
fi

python scripts/train.py \
  --config="configs/bc/droid_pen_in_cup_dp_random_drop.py:pen_in_cup,${DROP_PERCENT},random,${SEED},${SCORE_ROOT},${RLDS_PATH}" \
  --path="${OUT_ROOT}" \
  --name="${RUN_NAME}" \
  --project="${PROJECT}" \
  --include_timestamp=false \
  "${TRAIN_ARGS[@]}"
