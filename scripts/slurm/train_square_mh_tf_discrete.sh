#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=square_tf_disc
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_square_tf_disc.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_square_tf_disc.err

set -euo pipefail

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO=/iris/u/jasonyan/repos/demonstration-information
DATASET=/iris/u/jasonyan/data/robomimic/square/mh/image.hdf5
CONFIG=${REPO}/configs/robomimic/square_mh_bc_transformer_discrete_agent_wrist.json

mkdir -p /iris/u/jasonyan/slurm

cd "${REPO}/robomimic"

python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export MUJOCO_GL=egl
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python robomimic/scripts/train.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --name square_mh_tf_discrete_agent_wrist_no_object_seed1
