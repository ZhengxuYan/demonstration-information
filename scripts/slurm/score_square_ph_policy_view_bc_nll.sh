#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=ph_bc_view_nll
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_ph_bc_view_nll.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_ph_bc_view_nll.err

set -euo pipefail

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
DATA_ROOT=/iris/u/jasonyan/data/policy_view_experiments/square_ph
CKPT_ROOT=/iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments
OUT_DIR=/iris/u/jasonyan/data/robomimic_policy_scores/square_ph_policy_view_bc

mkdir -p /iris/u/jasonyan/slurm "${OUT_DIR}"
cd "${REPO}"

python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export MUJOCO_GL=egl
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

latest_ckpt() {
  local run_name="$1"
  local ckpt
  ckpt="$(find "${CKPT_ROOT}/${run_name}" -path "*/models/model_epoch_2000.pth" -type f | sort | tail -n 1)"
  if [[ -z "${ckpt}" ]]; then
    echo "missing checkpoint for ${run_name}" >&2
    return 1
  fi
  echo "${ckpt}"
}

score_one() {
  local algo="$1"
  local view="$2"
  local dataset="$3"
  local run_name="square_ph_bc_${algo}_${view}_200_seed1"
  local ckpt
  ckpt="$(latest_ckpt "${run_name}")"
  echo "scoring ${run_name}"
  echo "  checkpoint: ${ckpt}"
  echo "  dataset:    ${dataset}"
  python scripts/quality/score_robomimic_policy_nll.py \
    --checkpoint "${ckpt}" \
    --dataset "${dataset}" \
    --output "${OUT_DIR}" \
    --name "${algo}_${view}_epoch_2000" \
    --batch-size 128
}

score_one gmm agent_wrist "${DATA_ROOT}/square_ph_agent_wrist_image.hdf5"
score_one gmm left_close_low_wrist "${DATA_ROOT}/square_ph_left_close_low_wrist_image.hdf5"
score_one discrete agent_wrist "${DATA_ROOT}/square_ph_agent_wrist_image.hdf5"
score_one discrete left_close_low_wrist "${DATA_ROOT}/square_ph_left_close_low_wrist_image.hdf5"
