#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=ph_bc_views
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_ph_bc_views.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_ph_bc_views.err

set -euo pipefail

ALGO="${1:-}"
VIEW="${2:-}"
if [[ "${ALGO}" != "gmm" && "${ALGO}" != "discrete" ]]; then
  echo "Usage: sbatch $0 gmm|discrete agent_wrist|left_close_low_wrist"
  exit 2
fi
if [[ "${VIEW}" != "agent_wrist" && "${VIEW}" != "left_close_low_wrist" ]]; then
  echo "Usage: sbatch $0 gmm|discrete agent_wrist|left_close_low_wrist"
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
CONFIG_DIR=/iris/u/jasonyan/data/policy_view_experiments/configs/robomimic
CONFIG="${CONFIG_DIR}/square_ph_bc_${ALGO}_${VIEW}_200_seed1.json"
RUN_NAME="square_ph_bc_${ALGO}_${VIEW}_200_seed1"

case "${VIEW}" in
  agent_wrist)
    DATASET=/iris/u/jasonyan/data/policy_view_experiments/square_ph/square_ph_agent_wrist_image.hdf5
    ;;
  left_close_low_wrist)
    DATASET=/iris/u/jasonyan/data/policy_view_experiments/square_ph/square_ph_left_close_low_wrist_image.hdf5
    ;;
esac

mkdir -p /iris/u/jasonyan/slurm "${CONFIG_DIR}"

cd "${REPO}"
python scripts/quality/write_policy_view_bc_config.py \
  --algo "${ALGO}" \
  --view "${VIEW}" \
  --repo "${REPO}" \
  --output "${CONFIG}"

cd "${REPO}/robomimic"
python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
if [[ "${ALGO}" == "discrete" ]]; then
  python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"
fi

export MUJOCO_GL=egl
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python robomimic/scripts/train.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --name "${RUN_NAME}"
