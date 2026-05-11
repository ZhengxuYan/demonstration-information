#!/bin/bash
# Train non-abs BC baselines and state-only BC policies.
#
# Usage:
#   sbatch scripts/slurm/train_nonabs_bc_retraining_and_state_only.sh prepare_expert200 all all
#   sbatch scripts/slurm/train_nonabs_bc_retraining_and_state_only.sh original discrete_smooth ph
#   sbatch scripts/slurm/train_nonabs_bc_retraining_and_state_only.sh original gmm expert200
#   sbatch scripts/slurm/train_nonabs_bc_retraining_and_state_only.sh state_only discrete_smooth all

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=nonabs_bc
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8
#SBATCH --output=/iris/u/jasonyan/slurm/%j_nonabs_bc.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_nonabs_bc.err

set -euo pipefail

MODE="${1:-all}"
POLICY="${2:-all}"
DATASET_SCOPE="${3:-all}"

if [[ "${MODE}" != "all" && "${MODE}" != "original" && "${MODE}" != "state_only" && "${MODE}" != "prepare_expert200" ]]; then
  echo "Usage: sbatch $0 all|original|state_only|prepare_expert200 all|gmm|discrete|discrete_smooth all|ph|mh|expert200" >&2
  exit 2
fi
if [[ "${POLICY}" != "all" && "${POLICY}" != "gmm" && "${POLICY}" != "discrete" && "${POLICY}" != "discrete_smooth" ]]; then
  echo "Usage: sbatch $0 all|original|state_only|prepare_expert200 all|gmm|discrete|discrete_smooth all|ph|mh|expert200" >&2
  exit 2
fi
if [[ "${DATASET_SCOPE}" != "all" && "${DATASET_SCOPE}" != "ph" && "${DATASET_SCOPE}" != "mh" && "${DATASET_SCOPE}" != "expert200" ]]; then
  echo "Usage: sbatch $0 all|original|state_only|prepare_expert200 all|gmm|discrete|discrete_smooth all|ph|mh|expert200" >&2
  exit 2
fi

source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate openx

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
CONFIG_DIR="${CONFIG_DIR:-/iris/u/jasonyan/data/policy_view_experiments/configs/robomimic}"
OUTPUT_DIR="${OUTPUT_DIR:-/iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments}"
PH_ROOT="${PH_ROOT:-/iris/u/jasonyan/data/policy_view_experiments/square_ph}"
MH_DATASET="${MH_DATASET:-/iris/u/jasonyan/data/robomimic/square/mh/image.hdf5}"
EXPERT_ROOT="${EXPERT_ROOT:-/iris/u/jasonyan/data/policy_view_experiments/expert200_random_post_bc}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"
WANDB_PROJECT="${WANDB_PROJECT:-policy-view-bc-random-post}"
L2_REGULARIZATION="${L2_REGULARIZATION:-0.0}"
SOFT_SIGMA_BINS="${SOFT_SIGMA_BINS:-1.5}"
SOFT_TRUNCATE_BINS="${SOFT_TRUNCATE_BINS:-6}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"

mkdir -p /iris/u/jasonyan/slurm "${CONFIG_DIR}"
cd "${REPO}"

python scripts/quality/make_expert200_nonabs_bc_datasets.py \
  --src-root "${EXPERT_ROOT}" \
  --dst-root "${EXPERT_ROOT}" \
  ${OVERWRITE_EXPERT200_NONABS:+--overwrite}

if [[ "${MODE}" == "prepare_expert200" ]]; then
  exit 0
fi

cd "${REPO}/robomimic"
python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"
python "${REPO}/scripts/setup/patch_robomimic_discrete_action.py"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

want_policy() {
  local policy="$1"
  [[ "${POLICY}" == "all" || "${POLICY}" == "${policy}" ]]
}

want_dataset() {
  local dataset="$1"
  [[ "${DATASET_SCOPE}" == "all" || "${DATASET_SCOPE}" == "${dataset}" ]]
}

policy_algo() {
  local policy="$1"
  if [[ "${policy}" == "gmm" ]]; then
    echo "gmm"
  else
    echo "discrete"
  fi
}

write_and_train() {
  local dataset_name="$1"
  local policy="$2"
  local view="$3"
  local obs_mode="$4"
  local dataset="$5"
  local run_name="$6"
  local config="${CONFIG_DIR}/${run_name}.json"
  local algo
  algo="$(policy_algo "${policy}")"

  cd "${REPO}"
  cmd=(
    python scripts/quality/write_policy_view_bc_config.py
    --algo "${algo}"
    --view "${view}"
    --repo "${REPO}"
    --output "${config}"
    --dataset "${dataset}"
    --run-name "${run_name}"
    --output-dir "${OUTPUT_DIR}"
    --num-epochs "${NUM_EPOCHS}"
    --obs-mode "${obs_mode}"
    --enable-validation
    --l2-regularization "${L2_REGULARIZATION}"
  )
  if [[ "${ENABLE_WANDB}" == "1" ]]; then
    cmd+=(--log-wandb --wandb-project "${WANDB_PROJECT}")
  fi
  if [[ "${policy}" == "discrete_smooth" ]]; then
    cmd+=(
      --discrete-loss-type soft_ce
      --soft-sigma-bins "${SOFT_SIGMA_BINS}"
      --soft-truncate-bins "${SOFT_TRUNCATE_BINS}"
    )
  fi
  "${cmd[@]}"

  python - <<PY
import json
with open("${config}") as f:
    cfg = json.load(f)
print("dataset_name", "${dataset_name}")
print("policy", "${policy}")
print("run", cfg["experiment"]["name"])
print("data", cfg["train"]["data"])
print("obs_mode", "${obs_mode}")
print("low_dim", cfg["observation"]["modalities"]["obs"]["low_dim"])
print("rgb", cfg["observation"]["modalities"]["obs"]["rgb"])
print("cache", cfg["train"].get("hdf5_cache_mode"))
print("filters", cfg["train"].get("hdf5_filter_key"), cfg["train"].get("hdf5_validation_filter_key"))
if "${algo}" == "discrete":
    print("discrete", cfg["algo"]["discrete"])
PY

  cd "${REPO}/robomimic"
  python robomimic/scripts/train.py \
    --config "${config}" \
    --dataset "${dataset}" \
    --name "${run_name}"
}

train_original() {
  local policy="$1"
  local algo_name="$policy"
  if [[ "${policy}" == "discrete_smooth" ]]; then
    algo_name="discrete_smooth"
  fi
  if want_dataset ph; then
    write_and_train square_ph "${policy}" agent_wrist image_state \
      "${PH_ROOT}/square_ph_agent_wrist_image.hdf5" \
      "square_ph_bc_nonabs_${algo_name}_agent_wrist_seed1"
    write_and_train square_ph "${policy}" left_close_low_wrist image_state \
      "${PH_ROOT}/square_ph_left_close_low_wrist_image.hdf5" \
      "square_ph_bc_nonabs_${algo_name}_left_close_low_wrist_seed1"
  fi
  if want_dataset expert200; then
    write_and_train expert200_random_post "${policy}" agent_wrist image_state \
      "${EXPERT_ROOT}/expert200_random_post_agent_wrist_image.hdf5" \
      "expert200_random_post_bc_nonabs_${algo_name}_agent_wrist_seed1"
    write_and_train expert200_random_post "${policy}" left_close_low_wrist image_state \
      "${EXPERT_ROOT}/expert200_random_post_left_close_low_wrist_image.hdf5" \
      "expert200_random_post_bc_nonabs_${algo_name}_left_close_low_wrist_seed1"
  fi
}

train_state_only() {
  local policy="$1"
  local algo_name="$policy"
  if [[ "${policy}" == "discrete_smooth" ]]; then
    algo_name="discrete_smooth"
  fi
  if want_dataset ph; then
    write_and_train square_ph "${policy}" agent_wrist state_only \
      "${PH_ROOT}/square_ph_agent_wrist_image.hdf5" \
      "square_ph_bc_nonabs_state_only_${algo_name}_seed1"
  fi
  if want_dataset mh; then
    write_and_train square_mh "${policy}" agent_wrist state_only \
      "${MH_DATASET}" \
      "square_mh_bc_nonabs_state_only_${algo_name}_seed1"
  fi
  if want_dataset expert200; then
    write_and_train expert200_random_post "${policy}" agent_wrist state_only \
      "${EXPERT_ROOT}/expert200_random_post_agent_wrist_image.hdf5" \
      "expert200_random_post_bc_nonabs_state_only_${algo_name}_seed1"
  fi
}

for policy in gmm discrete discrete_smooth; do
  if ! want_policy "${policy}"; then
    continue
  fi
  if [[ "${MODE}" == "all" || "${MODE}" == "original" ]]; then
    train_original "${policy}"
  fi
  if [[ "${MODE}" == "all" || "${MODE}" == "state_only" ]]; then
    train_state_only "${policy}"
  fi
done
