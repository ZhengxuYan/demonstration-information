#!/bin/bash
# Download Transport PH low-dim robomimic data and convert delta actions to absolute actions.
#
# Usage:
#   sbatch scripts/slurm/prepare_transport_ph_dp_lowdim_abs.sh

#SBATCH --job-name=conv_trans_abs
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/iris/u/jasonyan/slurm/%j_conv_trans_abs.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_conv_trans_abs.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-robodiff}"
set -u

DP_REPO="${DP_REPO:-/iris/u/jasonyan/repos/diffusion_policy}"
DATA_DIR="${DATA_DIR:-/iris/u/jasonyan/data/diffusion_policy/robomimic/datasets/transport/ph}"
INPUT_HDF5="${INPUT_HDF5:-${DATA_DIR}/low_dim_v15.hdf5}"
COMPAT_HDF5="${COMPAT_HDF5:-${DATA_DIR}/low_dim_v15_robodiff_compat.hdf5}"
OUTPUT_HDF5="${OUTPUT_HDF5:-${DATA_DIR}/low_dim_abs.hdf5}"
EVAL_DIR="${EVAL_DIR:-${DATA_DIR}/abs_conversion_eval}"
NUM_WORKERS="${NUM_WORKERS:-8}"
URL="${URL:-https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5/transport/ph/low_dim_v15.hdf5}"

mkdir -p /iris/u/jasonyan/slurm "${DATA_DIR}" "${EVAL_DIR}"

echo "hostname=$(hostname)"
echo "input_hdf5=${INPUT_HDF5}"
echo "output_hdf5=${OUTPUT_HDF5}"
echo "eval_dir=${EVAL_DIR}"

if [[ ! -f "${INPUT_HDF5}" ]]; then
  echo "downloading ${URL}"
  curl -L --fail --retry 5 --retry-delay 10 -o "${INPUT_HDF5}.tmp" "${URL}"
  mv "${INPUT_HDF5}.tmp" "${INPUT_HDF5}"
fi
ls -lh "${INPUT_HDF5}"

python - <<PY
import h5py
import json
import shutil

src = "${INPUT_HDF5}"
dst = "${COMPAT_HDF5}"
shutil.copy2(src, dst)
with h5py.File(dst, "r+") as f:
    env_args = f["data"].attrs["env_args"]
    if isinstance(env_args, bytes):
        env_args = env_args.decode()
    meta = json.loads(env_args)
    env_kwargs = meta["env_kwargs"]
    # Robosuite 1.5 datasets include this kwarg, but the robodiff env used for
    # these DP runs is older and rejects it during env construction.
    env_kwargs.pop("lite_physics", None)
    # Robosuite 1.5 stores two-arm controllers as a composite BASIC controller.
    # The robodiff environment uses the older per-robot OSC_POSE config format.
    controller = env_kwargs.get("controller_configs")
    if isinstance(controller, dict) and controller.get("type") == "BASIC":
        body_parts = controller.get("body_parts", {})
        osc_pose = dict(body_parts.get("right") or next(iter(body_parts.values())))
        osc_pose.pop("input_ref_frame", None)
        osc_pose.pop("gripper", None)
        env_kwargs["controller_configs"] = osc_pose
    f["data"].attrs["env_args"] = json.dumps(meta, indent=4)
print(dst)
PY

cd "${DP_REPO}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=/sailhome/jasonyan/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH="${DP_REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

rm -f "${OUTPUT_HDF5}"
rm -rf "${EVAL_DIR}"

python diffusion_policy/scripts/robomimic_dataset_conversion.py \
  --input "${COMPAT_HDF5}" \
  --output "${OUTPUT_HDF5}" \
  --eval_dir "${EVAL_DIR}" \
  --num_workers "${NUM_WORKERS}"

ls -lh "${OUTPUT_HDF5}"
