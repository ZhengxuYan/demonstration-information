#!/bin/bash
# Train transport DemInf VAEs over shoulder camera views and full 14D actions.
#
# Usage:
#   sbatch --array=1-2%2 scripts/slurm/train_transport_mh_image_deminf_vaes_array.sh
#   RESUME=1 sbatch --array=1-2%2 scripts/slurm/train_transport_mh_image_deminf_vaes_array.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=transport_vae
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%A_%a_transport_vae.out
#SBATCH --error=/iris/u/jasonyan/slurm/%A_%a_transport_vae.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
RLDS_PATH="${RLDS_PATH:-/iris/u/jasonyan/data/transport_mh_image_v15_rlds/robo_mimic_transport/1.0.0}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/transport_mh_image_v15}"
PROJECT="${WANDB_PROJECT:-transport-deminf}"
SEED="${SEED:-1}"
CAMERA="${CAMERA:-both}"

TYPES=("s" "a")
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK_ID < 1 || TASK_ID > ${#TYPES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} outside 1..${#TYPES[@]}" >&2
  exit 2
fi
CONFIG_TYPE="${TYPES[$((TASK_ID - 1))]}"
RUN_NAME="transport_mh_${CAMERA}_${CONFIG_TYPE}_vae_seed${SEED}"

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
  --config="configs/quality/vae_robomimic_transport.py:transport/mh,${CONFIG_TYPE},${SEED},${CAMERA},transport_mh=${RLDS_PATH}" \
  --path="${OUT_ROOT}" \
  --name="${RUN_NAME}" \
  --project="${PROJECT}" \
  --include_timestamp=false \
  "${RESUME_ARGS[@]}"
