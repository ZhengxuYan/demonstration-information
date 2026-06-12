#!/bin/bash
# Train DemInf VAEs on the 06/10 pen-in-cup DROID RLDS dataset.
#
# Usage:
#   sbatch --array=1-2%2 scripts/slurm/train_pen_in_cup_0610_deminf_vaes_array.sh
#   RESUME=1 sbatch --array=1-2%2 scripts/slurm/train_pen_in_cup_0610_deminf_vaes_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=pic0610_vae
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8,iliad1,iliad2,iliad3,iliad4
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_pic0610_vae.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_pic0610_vae.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
RLDS_PATH="${RLDS_PATH:-/iris/u/jasonyan/data/droid_pen_in_cup_06102026_89_rlds/droid_pen_in_cup/1.0.0}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_vae_pen_in_cup_06102026_89}"
PROJECT="${WANDB_PROJECT:-pen-in-cup-deminf}"
SEED="${SEED:-1}"

TYPES=("s" "a")
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK_ID < 1 || TASK_ID > ${#TYPES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside 1..${#TYPES[@]}" >&2
  exit 2
fi
CONFIG_TYPE="${TYPES[$((TASK_ID - 1))]}"
RUN_NAME="pen_in_cup_${CONFIG_TYPE}_vae_seed${SEED}"

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "task_id=${TASK_ID}"
echo "type=${CONFIG_TYPE}"
echo "rlds_path=${RLDS_PATH}"
echo "out_root=${OUT_ROOT}"
echo "run_name=${RUN_NAME}"
echo "resume=${RESUME:-0}"

RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

python scripts/train.py \
  --config="configs/quality/vae_droid.py:pen_in_cup,${CONFIG_TYPE},${SEED},${RLDS_PATH}" \
  --path="${OUT_ROOT}" \
  --name="${RUN_NAME}" \
  --project="${PROJECT}" \
  --include_timestamp=false \
  "${RESUME_ARGS[@]}"
