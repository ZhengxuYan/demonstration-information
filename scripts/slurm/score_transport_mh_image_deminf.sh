#!/bin/bash
# Score transport demos with trained DemInf VAEs.
#
# Usage:
#   sbatch scripts/slurm/score_transport_mh_image_deminf.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=iris8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=transport_score
#SBATCH --output=/iris/u/jasonyan/slurm/%j_transport_score.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_transport_score.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
HDF5_PATH="${HDF5_PATH:-/scr/tiangao/datasets/transport_mh_image_v15.hdf5}"
VAE_ROOT="${VAE_ROOT:-/iris/u/jasonyan/data/deminf_outputs/transport_mh_image_v15}"
OBS_CKPT="${OBS_CKPT:-${VAE_ROOT}/transport_mh_both_s_vae_seed1}"
ACTION_CKPT="${ACTION_CKPT:-${VAE_ROOT}/transport_mh_both_a_vae_seed1}"
OUT_ROOT="${OUT_ROOT:-/iris/u/jasonyan/data/deminf_outputs/transport_mh_image_v15_scores}"

mkdir -p /iris/u/jasonyan/slurm "${OUT_ROOT}"
cd "${REPO}"

echo "hostname=$(hostname)"
echo "hdf5_path=${HDF5_PATH}"
echo "obs_ckpt=${OBS_CKPT}"
echo "action_ckpt=${ACTION_CKPT}"
echo "out_root=${OUT_ROOT}"

python scripts/data/validate_robomimic_transport_hdf5.py \
  --hdf5 "${HDF5_PATH}" \
  --expected-demos 300

python scripts/quality/score_robomimic_hdf5.py \
  --transport \
  --camera both \
  --obs_ckpt "${OBS_CKPT}" \
  --action_ckpt "${ACTION_CKPT}" \
  --dataset "transport_mh_image_v15=0=${HDF5_PATH}" \
  --batch_size "${BATCH_SIZE:-1024}" \
  --output "${OUT_ROOT}"

echo "SCORE_TRANSPORT_MH_IMAGE_DEMINF_OK"
