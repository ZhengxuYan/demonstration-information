#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=square_ph_bc_nll
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_square_ph_bc_nll.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_square_ph_bc_nll.err

set -euo pipefail

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
DATASET=/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/square/ph/image.hdf5
OUT_DIR=/iris/u/jasonyan/data/robomimic_policy_scores/square_ph_bc_wrist_proprio

GMM_CKPT=/iris/u/jasonyan/data/robomimic_outputs/square_ph_bc_gmm_wrist_proprio_seed1_v2/20260420182629/models/model_epoch_2000.pth
DISCRETE_CKPT=/iris/u/jasonyan/data/robomimic_outputs/square_ph_bc_discrete_wrist_proprio_seed1_v2/20260420182629/models/model_epoch_2000.pth

mkdir -p /iris/u/jasonyan/slurm "${OUT_DIR}"

cd "${REPO}"

python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export MUJOCO_GL=egl
export PYTHONPATH="${REPO}/robomimic:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python scripts/quality/score_robomimic_policy_nll.py \
  --checkpoint "${GMM_CKPT}" \
  --dataset "${DATASET}" \
  --output "${OUT_DIR}" \
  --name gmm_bc_epoch_2000_v2 \
  --batch-size 128

python scripts/quality/score_robomimic_policy_nll.py \
  --checkpoint "${DISCRETE_CKPT}" \
  --dataset "${DATASET}" \
  --output "${OUT_DIR}" \
  --name discrete_bc_epoch_2000_v2 \
  --batch-size 128
