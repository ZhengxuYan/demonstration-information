#!/bin/bash
#SBATCH --job-name=ph_dp_views
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_ph_dp_views.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_ph_dp_views.err

set -euo pipefail

VIEW="${1:-}"
if [[ "${VIEW}" != "agent_wrist" && "${VIEW}" != "left_close_low_wrist" ]]; then
  echo "Usage: sbatch $0 agent_wrist|left_close_low_wrist"
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate robodiff

DEMO_REPO=/iris/u/jasonyan/repos/demonstration-information
DP_REPO=/iris/u/jasonyan/repos/diffusion_policy
OUT_ROOT=/iris/u/jasonyan/data/diffusion_policy_outputs/policy_view_experiments
mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"

python "${DEMO_REPO}/scripts/setup/install_diffusion_policy_view_tasks.py" \
  --dp-repo "${DP_REPO}" \
  --data-root /iris/u/jasonyan/data/policy_view_experiments

case "${VIEW}" in
  agent_wrist)
    TASK=square_ph_dp_agent_wrist_abs_50_seed42
    ;;
  left_close_low_wrist)
    TASK=square_ph_dp_left_close_low_wrist_abs_50_seed42
    ;;
esac
RUN_NAME="${TASK}"

cd "${DP_REPO}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task="${TASK}" \
  training.num_epochs="${DP_NUM_EPOCHS:-51}" \
  logging.mode=disabled \
  training.device=cuda:0 \
  hydra.run.dir="${OUT_ROOT}/${RUN_NAME}" \
  hydra.sweep.dir="${OUT_ROOT}/${RUN_NAME}"
