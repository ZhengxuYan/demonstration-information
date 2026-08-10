#!/bin/bash
# Train one selected final-density-score Threading filtered-BC policy.

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${REPO:-/iris/u/${USER}/repos/demonstration-information}"
ROBOMIMIC_REPO="${ROBOMIMIC_REPO:-${REPO}/robomimic}"
ROBOSUITE_REPO="${ROBOSUITE_REPO:-/iris/u/${USER}/repos/robosuite-pomdp}"
ROOT="${ROOT:-/iris/u/${USER}/data/threading_d0_final_score_filtered_bc_20260808}"
RESULT_ROOT="${RESULT_ROOT:-/iris/u/${USER}/data/threading_d0_final_score_filtered_bc_results_20260808}"
SOURCE_HDF5="${SOURCE_HDF5:-/iris/u/${USER}/data/threading_d0_final200_abs_delta_20260730/hdf5/image_final200_joint_absolute_fixedobs_contiguous.hdf5}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/iris/u/${USER}/slurm}"
CONDA_ROOT="${CONDA_ROOT:-/iris/u/${USER}/miniforge3}"
MANIFEST="${MANIFEST:-${ROOT}/config_manifest.csv}"
SELECTIONS="${SELECTIONS:-${ROOT}/score_selections.json}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-1}}"
EXPECTED_TASKS="${EXPECTED_TASKS:-54}"
CONDA_ENV="${CONDA_ENV:-openx}"

set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

IFS=$'\t' read -r CONFIG RUN_NAME RESULT_PATH FILTER_KEY < <(
  python - "${MANIFEST}" "${TASK_ID}" "${EXPECTED_TASKS}" <<'PY'
import csv, sys
manifest, task_id, expected = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(manifest, newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != expected:
    raise SystemExit(f"expected {expected} tasks, found {len(rows)}")
row = next((r for r in rows if int(r["task_id"]) == task_id), None)
if row is None:
    raise SystemExit(f"missing task_id={task_id}")
print("\t".join([row["config_path"], row["run_name"], row["result_path"], row["filter_key"]]))
PY
)

if [[ -f "${RESULT_PATH}" ]]; then
  rm -rf "${RESULT_ROOT}/.resume/${RUN_NAME}"
  echo "already_complete task=${TASK_ID} result=${RESULT_PATH}"
  exit 0
fi

mkdir -p "${RESULT_ROOT}" "${RESULT_ROOT}/failures" "${SLURM_LOG_DIR}"

CACHE_ROOT="/tmp/${USER}_threading_d0_filtered_bc_cache_20260808"
CACHE_HDF5="${CACHE_ROOT}/image_final200_joint_absolute_filtered_bc.hdf5"
CACHE_READY="${CACHE_ROOT}/READY.v1"
CACHE_LOCK="${CACHE_ROOT}/stage.lock"
mkdir -p "${CACHE_ROOT}"

# Stagger first-use copies across nodes; subsequent tasks on a node reuse the cache.
if [[ ! -f "${CACHE_READY}" ]]; then
  STAGE_DELAY=$(( ((TASK_ID - 1) % 10) * 90 ))
  echo "stage_wait_seconds=${STAGE_DELAY} task=${TASK_ID} host=$(hostname)"
  sleep "${STAGE_DELAY}"
fi

exec 9>"${CACHE_LOCK}"
flock 9
if [[ ! -f "${CACHE_READY}" || ! -f "${CACHE_HDF5}" ]]; then
  TEMP_HDF5="${CACHE_HDF5}.tmp.${SLURM_JOB_ID:-manual}.${TASK_ID}"
  rm -f "${TEMP_HDF5}"
  echo "stage_begin source=${SOURCE_HDF5} host=$(hostname)"
  cp "${SOURCE_HDF5}" "${TEMP_HDF5}"
  python - "${TEMP_HDF5}" "${SELECTIONS}" <<'PY'
import json, sys
import h5py
import numpy as np

hdf5_path, selection_path = sys.argv[1:]
payload = json.load(open(selection_path))
with h5py.File(hdf5_path, "r+") as dataset:
    masks = dataset.require_group("mask")
    demos = set(dataset["data"].keys())
    if len(demos) != payload["expected_demos"]:
        raise ValueError(f"expected {payload['expected_demos']} demos, found {len(demos)}")
    for name, selected in payload["selections"].items():
        if not set(selected).issubset(demos):
            raise ValueError(f"{name}: selection references missing demos")
        values = np.asarray([value.encode("utf-8") for value in selected])
        if name in masks:
            del masks[name]
        masks.create_dataset(name, data=values)
    dataset.flush()
PY
  mv "${TEMP_HDF5}" "${CACHE_HDF5}"
  printf 'source=%s\nprepared_at=%s\n' "${SOURCE_HDF5}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${CACHE_READY}"
  echo "stage_complete bytes=$(stat -c %s "${CACHE_HDF5}") host=$(hostname)"
else
  echo "stage_cache_hit host=$(hostname)"
fi
flock -u 9

LOCAL_WORK="/tmp/${USER}_threading_d0_filtered_bc_runs/${SLURM_JOB_ID:-manual}_${TASK_ID}"
LOCAL_CONFIG="${LOCAL_WORK}/config.json"
LOCAL_LOG="${LOCAL_WORK}/train.log"
LOCAL_OUTPUT="${LOCAL_WORK}/output"
mkdir -p "${LOCAL_WORK}" "${LOCAL_OUTPUT}"

PREEMPTIBLE=0
RUN_OUTPUT="${LOCAL_OUTPUT}"
RESUME_FLAG=()
case "${SLURM_JOB_PARTITION:-}" in
  sc-loprio|iris|iliad-lo)
    PREEMPTIBLE=1
    RUN_OUTPUT="${RESULT_ROOT}/.resume"
    mkdir -p "${RUN_OUTPUT}"
    if find "${RUN_OUTPUT}/${RUN_NAME}" -name last.pth -print -quit 2>/dev/null | grep -q .; then
      RESUME_FLAG=(--resume)
    elif [[ -d "${RUN_OUTPUT}/${RUN_NAME}" ]]; then
      # A launch that exits before its first checkpoint can leave only empty
      # logs/videos directories. They are not resumable and otherwise trigger
      # robomimic's interactive overwrite prompt on the next batch attempt.
      rm -rf "${RUN_OUTPUT:?}/${RUN_NAME}"
    fi
    ;;
esac

cleanup() {
  rm -rf "${LOCAL_WORK}"
}
trap cleanup EXIT TERM INT

python - "${CONFIG}" "${LOCAL_CONFIG}" "${CACHE_HDF5}" "${RUN_OUTPUT}" <<'PY'
import json, sys
source, destination, dataset, output = sys.argv[1:]
config = json.load(open(source))
config["train"]["data"] = dataset
config["train"]["output_dir"] = output
config["experiment"]["save"]["enabled"] = False
config["experiment"]["logging"]["terminal_output_to_txt"] = False
config["experiment"]["logging"]["log_tb"] = False
config["experiment"]["logging"]["log_wandb"] = False
with open(destination, "w") as handle:
    json.dump(config, handle, indent=2)
PY

cd "${ROBOMIMIC_REPO}"
python "${REPO}/scripts/setup/patch_robomimic_optional_diffusion.py"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="/sailhome/${USER}/.mujoco/mujoco210/bin:/afs/cs.stanford.edu/u/${USER}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export USE_FLAX=0
export ROBOMIMIC_SKIP_MUJOCO_PY=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH="${ROBOMIMIC_REPO}:${ROBOSUITE_REPO}:${REPO}/scripts/quality:${PYTHONPATH:-}"
if [[ "${PREEMPTIBLE}" == "1" ]]; then
  # Sparse, temporary shared resume state for jobs that can move to another node.
  export ROBOMIMIC_LATEST_SAVE_INTERVAL=200
else
  # Non-preemptible jobs need no intermediate resume writes; the final local
  # checkpoint is deleted together with LOCAL_WORK.
  export ROBOMIMIC_LATEST_SAVE_INTERVAL=1000000000
fi

echo "train_begin task=${TASK_ID}/${EXPECTED_TASKS} run=${RUN_NAME} filter=${FILTER_KEY} host=$(hostname) preemptible=${PREEMPTIBLE} resume=$([[ ${#RESUME_FLAG[@]} -gt 0 ]] && echo 1 || echo 0)"
python robomimic/scripts/train.py \
  --config "${LOCAL_CONFIG}" \
  --dataset "${CACHE_HDF5}" \
  --name "${RUN_NAME}" \
  "${RESUME_FLAG[@]}" >"${LOCAL_LOG}" 2>&1 &
TRAIN_PID=$!

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 300
  if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    STATUS="$(python - "${LOCAL_LOG}" <<'PY'
import re, sys
try:
    text = open(sys.argv[1], errors="ignore").read()
except OSError:
    text = ""
epochs = [int(x) for x in re.findall(r"(?:Train\s+)?Epoch\s+(\d+)", text)]
print(max(epochs) if epochs else 0)
PY
)"
    echo "train_progress task=${TASK_ID} epoch=${STATUS} host=$(hostname)"
  fi
done

set +e
wait "${TRAIN_PID}"
TRAIN_RC=$?
set -e
if [[ "${TRAIN_RC}" != "0" ]]; then
  tail -n 200 "${LOCAL_LOG}" > "${RESULT_ROOT}/failures/task_${TASK_ID}_${SLURM_JOB_ID:-manual}.log"
  echo "train_failed task=${TASK_ID} rc=${TRAIN_RC}" >&2
  exit "${TRAIN_RC}"
fi

TEMP_RESULT="${RESULT_PATH}.tmp.${SLURM_JOB_ID:-manual}.${TASK_ID}"
set +e
python - "${LOCAL_LOG}" "${MANIFEST}" "${TASK_ID}" "${TEMP_RESULT}" <<'PY'
import csv, json, math, re, sys
from datetime import datetime, timezone

log_path, manifest_path, task_id, output_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
text = open(log_path, errors="ignore").read()
pattern = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+Rollouts took.*?"
    r'"Success_Rate"\s*:\s*(?P<success>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)',
    re.DOTALL,
)
rollouts = {}
for match in pattern.finditer(text):
    epoch, success = int(match.group("epoch")), float(match.group("success"))
    if not math.isfinite(success) or not 0 <= success <= 1:
        raise ValueError((epoch, success))
    rollouts[epoch] = success
epochs = [int(x) for x in re.findall(r"(?:Train\s+)?Epoch\s+(\d+)", text)]
max_epoch = max(epochs) if epochs else 0
if max_epoch < 600 or 600 not in rollouts:
    raise RuntimeError(f"missing epoch-600 result: max_epoch={max_epoch}, rollout_epochs={sorted(rollouts)}")
with open(manifest_path, newline="") as handle:
    row = next(r for r in csv.DictReader(handle) if int(r["task_id"]) == task_id)
best_epoch, best_success = max(rollouts.items(), key=lambda item: (item[1], item[0]))
payload = {
    **row,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
    "hostname": __import__("socket").gethostname(),
    "persistent_checkpoint_files_kept": 0,
    "max_epoch": max_epoch,
    "final_epoch": 600,
    "final_success_rate": rollouts[600],
    "best_rollout_epoch": best_epoch,
    "best_success_rate": best_success,
    "rollouts": [{"epoch": e, "success_rate": rollouts[e]} for e in sorted(rollouts)],
}
with open(output_path, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
PARSE_RC=$?
set -e
if [[ "${PARSE_RC}" != "0" ]]; then
  tail -n 240 "${LOCAL_LOG}" > "${RESULT_ROOT}/failures/task_${TASK_ID}_${SLURM_JOB_ID:-manual}_parse.log"
  rm -f "${TEMP_RESULT}"
  echo "result_parse_failed task=${TASK_ID} rc=${PARSE_RC}" >&2
  exit "${PARSE_RC}"
fi
mv "${TEMP_RESULT}" "${RESULT_PATH}"
if [[ "${PREEMPTIBLE}" == "1" ]]; then
  rm -rf "${RUN_OUTPUT:?}/${RUN_NAME}"
fi
echo "train_complete task=${TASK_ID} result=${RESULT_PATH} persistent_checkpoints_kept=0"
