#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=square_nll
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_square_nll.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_square_nll.err

set -euo pipefail

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
DATASET=/iris/u/jasonyan/data/robomimic/square/mh/image.hdf5
OUT_DIR=/iris/u/jasonyan/data/robomimic_policy_scores/square_mh_no_object

GMM_CKPT=/iris/u/jasonyan/data/robomimic_outputs/square_mh_tf_gmm_agent_wrist_no_object_seed1/20260414172517/models/model_epoch_200.pth
DISCRETE_CKPT=/iris/u/jasonyan/data/robomimic_outputs/square_mh_tf_discrete_agent_wrist_no_object_seed1/20260414172525/models/model_epoch_200.pth

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
  --name gmm_epoch_200 \
  --batch-size 128

python scripts/quality/score_robomimic_policy_nll.py \
  --checkpoint "${DISCRETE_CKPT}" \
  --dataset "${DATASET}" \
  --output "${OUT_DIR}" \
  --name discrete_epoch_200 \
  --batch-size 128
