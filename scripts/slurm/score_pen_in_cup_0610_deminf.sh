#!/bin/bash
# Compute DemInf scores for the 06/10 pen-in-cup DROID RLDS dataset.
#
# Usage:
#   sbatch scripts/slurm/score_pen_in_cup_0610_deminf.sh

#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=pic0610_deminf
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8,iliad1,iliad2,iliad3,iliad4
#SBATCH --output=/iris/u/jasonyan/slurm/%j_pic0610_deminf.out
#SBATCH --error=/iris/u/jasonyan/slurm/%j_pic0610_deminf.err

set -euo pipefail

set +u
source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openx}"
set -u

REPO="${REPO:-/iris/u/jasonyan/repos/demonstration-information}"
VAE_ROOT="${VAE_ROOT:-/iris/u/jasonyan/data/deminf_vae_pen_in_cup_06102026_89}"
OBS_CKPT="${OBS_CKPT:-${VAE_ROOT}/pen_in_cup_s_vae_seed1/100000}"
ACTION_CKPT="${ACTION_CKPT:-${VAE_ROOT}/pen_in_cup_a_vae_seed1/100000}"
RAW_SCORE_ROOT="${RAW_SCORE_ROOT:-/iris/u/jasonyan/data/deminf_outputs/pen_in_cup_06102026_89_deminf_raw}"
SCORE_ROOT="${SCORE_ROOT:-/iris/u/jasonyan/data/droid_pen_in_cup_06102026_89_deminf_scores}"
ESTIMATOR="${ESTIMATOR:-deminf}"
export RAW_SCORE_ROOT SCORE_ROOT ESTIMATOR

mkdir -p /iris/u/jasonyan/slurm "${RAW_SCORE_ROOT}" "${SCORE_ROOT}/pen_in_cup/${ESTIMATOR}"
cd "${REPO}"

if [[ ! -d "${OBS_CKPT}" ]]; then
  echo "missing completed obs VAE checkpoint: ${OBS_CKPT}" >&2
  exit 2
fi
if [[ ! -d "${ACTION_CKPT}" ]]; then
  echo "missing completed action VAE checkpoint: ${ACTION_CKPT}" >&2
  exit 2
fi

echo "hostname=$(hostname)"
echo "obs_ckpt=${OBS_CKPT}"
echo "action_ckpt=${ACTION_CKPT}"
echo "raw_score_root=${RAW_SCORE_ROOT}"
echo "score_root=${SCORE_ROOT}"
echo "estimator=${ESTIMATOR}"

python scripts/quality/estimate_quality.py \
  --obs_ckpt="${OBS_CKPT}" \
  --action_ckpt="${ACTION_CKPT}" \
  --batch_size="${BATCH_SIZE:-1024}" \
  --estimator="${DEMINF_ESTIMATOR:-ksg}" \
  --path="${RAW_SCORE_ROOT}"

python - <<'PY'
import os
import pickle
from pathlib import Path

raw_score_root = Path(os.environ["RAW_SCORE_ROOT"])
score_root = Path(os.environ["SCORE_ROOT"])
estimator = os.environ["ESTIMATOR"]
raw_pkl = raw_score_root / "pen_in_cup.pkl"
out_pkl = score_root / "pen_in_cup" / estimator / f"{estimator}_scores.pkl"

with raw_pkl.open("rb") as f:
    scores = pickle.load(f)

if "ep_idx" not in scores:
    raise KeyError(f"{raw_pkl} missing ep_idx scores; keys={sorted(scores)}")

ep_scores = {int(k): float(v) for k, v in scores["ep_idx"].items()}
out_pkl.parent.mkdir(parents=True, exist_ok=True)
with out_pkl.open("wb") as f:
    pickle.dump({"ep_idx": ep_scores}, f)

print(f"raw_score_pkl={raw_pkl}")
print(f"filter_score_pkl={out_pkl}")
print(f"num_episodes={len(ep_scores)}")
print(f"score_min={min(ep_scores.values()):.6f}")
print(f"score_max={max(ep_scores.values()):.6f}")
PY

echo "SCORE_PEN_IN_CUP_0610_DEMINF_OK"
