#!/usr/bin/env python3
"""Manage image+proprio POMDP density runs for Square MH and Threading D1."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py


REPO = "/iris/u/jasonyan/repos/demonstration-information"
DATA = "/iris/u/jasonyan/data"
REGIMES = (
    ("normal", "", "train", "valid", "score_all"),
    ("fold0", "fold0", "fold0_train", "fold0_valid", "fold0_score"),
    ("fold1", "fold1", "fold1_train", "fold1_valid", "fold1_score"),
)
ALGOS = ("gaussian", "gmm")
CONDITIONS = ("image_proprio", "action_prior")
LIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING", "SUSPENDED"}
BAD = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL"}
REQUIRED_MASKS = {
    "train", "valid", "score_all", "fold0_train", "fold0_valid", "fold0_score",
    "fold1_train", "fold1_valid", "fold1_score",
}


@dataclass(frozen=True)
class Stage:
    key: str
    dataset_tag: str
    action_dim: int
    source_hdf5: str
    hdf5: str
    manifest: str
    run_prefix: str
    out_root: str
    config_root: str
    score_root: str
    report_root: str
    title: str
    description: str
    prepare_mode: str
    labels_csv: str
    score_seed: int


STAGES = (
    Stage(
        key="square_7d",
        dataset_tag="square_mh_300",
        action_dim=7,
        source_hdf5=f"{DATA}/policy_view_experiments/square_mh/square_mh_agent_wrist_image.hdf5",
        hdf5=f"{DATA}/pomdp_image_proprio_20260723/square_mh_300_image_proprio_7d.hdf5",
        manifest=f"{DATA}/pomdp_image_proprio_20260723/square_mh_300_manifest.csv",
        run_prefix="square_mh_pomdp_image_proprio_7d",
        out_root=f"{DATA}/robomimic_outputs/pomdp_image_proprio_20260723",
        config_root=f"{DATA}/pomdp_image_proprio_20260723/configs",
        score_root=f"{DATA}/pomdp_image_proprio_20260723/scores/square_mh_300_7d",
        report_root=f"{DATA}/pomdp_image_proprio_20260723/reports/square_mh_300_7d",
        title="Square MH Image + Proprio POMDP Scores - 7D",
        description="Conditional models use two RGB views and end-effector proprio. Actions use dimensions 0-6.",
        prepare_mode="square",
        labels_csv=f"{REPO}/observability_annotations.csv",
        score_seed=20260723,
    ),
    Stage(
        key="square_6d",
        dataset_tag="square_mh_300",
        action_dim=6,
        source_hdf5=f"{DATA}/pomdp_image_proprio_20260723/square_mh_300_image_proprio_7d.hdf5",
        hdf5=f"{DATA}/pomdp_image_proprio_20260723/square_mh_300_image_proprio_6d.hdf5",
        manifest=f"{DATA}/pomdp_image_proprio_20260723/square_mh_300_manifest.csv",
        run_prefix="square_mh_pomdp_image_proprio_6d",
        out_root=f"{DATA}/robomimic_outputs/pomdp_image_proprio_20260723",
        config_root=f"{DATA}/pomdp_image_proprio_20260723/configs",
        score_root=f"{DATA}/pomdp_image_proprio_20260723/scores/square_mh_300_6d",
        report_root=f"{DATA}/pomdp_image_proprio_20260723/reports/square_mh_300_6d",
        title="Square MH Image + Proprio POMDP Scores - 6D",
        description="Conditional models use two RGB views and end-effector proprio. Actions use dimensions 0-5.",
        prepare_mode="subset",
        labels_csv=f"{REPO}/observability_annotations.csv",
        score_seed=20260723,
    ),
    Stage(
        key="d1_8d",
        dataset_tag="threading_d1_manual200",
        action_dim=8,
        source_hdf5=f"{DATA}/threading_d1_joint_position_manual_200_full100_partial100_20260723",
        hdf5=f"{DATA}/pomdp_image_proprio_20260723/threading_d1_manual200_image_proprio_8d.hdf5",
        manifest=f"{DATA}/pomdp_image_proprio_20260723/threading_d1_manual200_manifest.csv",
        run_prefix="threading_d1_manual200_pomdp_image_proprio_8d",
        out_root=f"{DATA}/robomimic_outputs/pomdp_image_proprio_20260723",
        config_root=f"{DATA}/pomdp_image_proprio_20260723/configs",
        score_root=f"{DATA}/pomdp_image_proprio_20260723/scores/threading_d1_manual200_8d",
        report_root=f"{DATA}/pomdp_image_proprio_20260723/reports/threading_d1_manual200_8d",
        title="Threading D1 Manual200 Image + Proprio POMDP Scores - 8D",
        description="Conditional models use two RGB views and end-effector proprio. Actions are seven absolute joint-position targets plus gripper.",
        prepare_mode="threading",
        labels_csv=f"{DATA}/threading_d1_joint_position_manual_200_full100_partial100_20260723/annotations.csv",
        score_seed=20260723,
    ),
    Stage(
        key="d1_7d",
        dataset_tag="threading_d1_manual200",
        action_dim=7,
        source_hdf5=f"{DATA}/pomdp_image_proprio_20260723/threading_d1_manual200_image_proprio_8d.hdf5",
        hdf5=f"{DATA}/pomdp_image_proprio_20260723/threading_d1_manual200_image_proprio_7d.hdf5",
        manifest=f"{DATA}/pomdp_image_proprio_20260723/threading_d1_manual200_manifest.csv",
        run_prefix="threading_d1_manual200_pomdp_image_proprio_7d",
        out_root=f"{DATA}/robomimic_outputs/pomdp_image_proprio_20260723",
        config_root=f"{DATA}/pomdp_image_proprio_20260723/configs",
        score_root=f"{DATA}/pomdp_image_proprio_20260723/scores/threading_d1_manual200_7d",
        report_root=f"{DATA}/pomdp_image_proprio_20260723/reports/threading_d1_manual200_7d",
        title="Threading D1 Manual200 Image + Proprio POMDP Scores - 7D Arm Only",
        description="Conditional models use two RGB views and end-effector proprio. Actions are the seven absolute arm joint-position targets; gripper is excluded.",
        prepare_mode="subset",
        labels_csv=f"{DATA}/threading_d1_joint_position_manual_200_full100_partial100_20260723/annotations.csv",
        score_seed=20260723,
    ),
)


@dataclass(frozen=True)
class Tier:
    name: str
    account: str
    partition: str
    slots: int
    preemptible: bool
    gres: str = "gpu:1"
    extra: tuple[str, ...] = ()


IRIS_EXCLUDE = "iris1,iris2,iris3,iris4,iris-hgx-1,iris-hgx-2,iris-hp-z8"
ILIAD_EXCLUDE = "iliad1,iliad2,iliad3,iliad4,iliad-hgx-1"
SC_EXCLUDE = (
    "iris-hgx-1,iris-hgx-2,iliad-hgx-1,pasteur-hgx-1,pasteur-hgx-2,"
    "tiger-hgx-1,cocoflops-hgx-1"
)
TIERS = (
    Tier("iris_hi", "iris", "iris-hi", 6, False, extra=("--exclude", IRIS_EXCLUDE)),
    Tier("iliad", "iliad", "iliad", 8, False, extra=("--exclude", ILIAD_EXCLUDE)),
    Tier("iris", "iris", "iris", 10, True, extra=("--exclude", IRIS_EXCLUDE)),
    Tier("iliad_lo", "iliad", "iliad-lo", 16, True, extra=("--exclude", ILIAD_EXCLUDE)),
    Tier("sc_l40s", "iliad", "sc-loprio", 4, True, "gpu:l40s:1", ("--exclude", SC_EXCLUDE)),
    Tier("sc_a6000", "iliad", "sc-loprio", 4, True, "gpu:a6000:1", ("--exclude", SC_EXCLUDE)),
    Tier("sc_a40", "iliad", "sc-loprio", 4, True, "gpu:a40:1", ("--exclude", SC_EXCLUDE)),
    Tier("sc_a5000", "iliad", "sc-loprio", 4, True, "gpu:a5000:1", ("--exclude", SC_EXCLUDE)),
)


@dataclass
class JobRecord:
    job_id: str | None = None
    state: str = "WAITING"
    attempts: int = 0
    tier: int | None = None
    submitted_at: float = 0.0


@dataclass
class ManagerState:
    jobs: dict[str, JobRecord] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, default=Path(f"{DATA}/pomdp_image_proprio_20260723/manager.json"))
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-active-gpu", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--pending-migrate-seconds", type=int, default=1800)
    parser.add_argument(
        "--stages",
        default=",".join(stage.key for stage in STAGES),
        help="Comma-separated stage keys to manage.",
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], check: bool = True) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"command failed: {shlex.join(cmd)}\n{detail}")
    return proc.stdout.strip()


def quote(value: str) -> str:
    return shlex.quote(str(value))


def env_command(env: dict[str, str], script: str) -> str:
    assignments = " ".join(f"{key}={quote(value)}" for key, value in env.items())
    return f"cd {quote(REPO)} && {assignments} bash {quote(script)}"


def load_state(path: Path) -> ManagerState:
    if not path.exists():
        return ManagerState()
    raw = json.loads(path.read_text())
    return ManagerState(jobs={key: JobRecord(**value) for key, value in raw.get("jobs", {}).items()})


def save_state(path: Path, state: ManagerState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps({"jobs": {key: asdict(value) for key, value in state.jobs.items()}}, indent=2, sort_keys=True)
            + "\n"
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def record(state: ManagerState, key: str) -> JobRecord:
    return state.jobs.setdefault(key, JobRecord())


def job_state(job_id: str | None) -> str | None:
    if not job_id:
        return None
    out = run(["sacct", "-j", job_id, "-n", "-P", "-o", "JobIDRaw,State"], check=False)
    for line in reversed(out.splitlines()):
        if line.startswith(job_id + "|"):
            return line.split("|", 1)[1].split()[0]
    out = run(["squeue", "-h", "-j", job_id, "-o", "%T"], check=False)
    return out.splitlines()[0].strip() if out.strip() else None


def pending_reason(job_id: str | None) -> str:
    if not job_id:
        return ""
    out = run(["squeue", "-h", "-j", job_id, "-t", "PD", "-o", "%R"], check=False)
    return out.splitlines()[0].strip() if out.strip() else ""


def expected_job_name(key: str) -> str | None:
    parts = key.split(":")
    if parts[0] == "train" and len(parts) == 5:
        _, stage_key, regime, algo, condition = parts
        return f"pip_{stage_key}_{regime}_{algo}_{condition}"
    if parts[0] == "score" and len(parts) == 4:
        _, stage_key, regime, algo = parts
        return f"pip_{stage_key}_{regime}_{algo}_score"
    return None


def live_jobs_named(name: str) -> list[tuple[str, str]]:
    output = run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%A|%j|%T"], check=False)
    jobs = []
    for line in output.splitlines():
        fields = line.split("|", 2)
        if len(fields) == 3 and fields[1] == name and fields[2] in LIVE:
            jobs.append((fields[0], fields[2]))
    return jobs


def infer_live_job_tier(job_id: str) -> int | None:
    output = run(["squeue", "-h", "-j", job_id, "-o", "%P|%b"], check=False)
    if not output.strip():
        return None
    partition, gres = output.splitlines()[0].split("|", 1)
    partition = partition.rstrip("*")
    if partition == "iris-hi":
        return 0
    if partition == "iliad":
        return 1
    if partition == "iris":
        return 2
    if partition == "iliad-lo":
        return 3
    if partition == "sc-loprio":
        lowered = gres.lower()
        for idx, token in ((4, "l40s"), (5, "a6000"), (6, "a40"), (7, "a5000")):
            if token in lowered:
                return idx
    return None


def adopt_and_prune_live_job(key: str, item: JobRecord) -> bool:
    name = expected_job_name(key)
    if name is None:
        return False
    jobs = live_jobs_named(name)
    if not jobs:
        return False
    jobs.sort(key=lambda pair: (pair[1] != "RUNNING", int(pair[0])))
    keep_id, keep_state = jobs[0]
    for duplicate_id, _ in jobs[1:]:
        run(["scancel", duplicate_id], check=False)
    item.job_id = keep_id
    item.state = keep_state
    inferred_tier = infer_live_job_tier(keep_id)
    if inferred_tier is not None:
        item.tier = inferred_tier
    return True


def partition_counts() -> dict[str, int]:
    out = run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%P|%T"], check=False)
    counts: dict[str, int] = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        partition, status = line.split("|", 1)
        if status in LIVE:
            partition = partition.rstrip("*")
            counts[partition] = counts.get(partition, 0) + 1
    return counts


def choose_tier(start: int | None = None) -> int:
    counts = partition_counts()
    first = start or 0
    for idx in range(first, len(TIERS)):
        tier = TIERS[idx]
        if tier.partition == "sc-loprio" or counts.get(tier.partition, 0) < tier.slots:
            return idx
    return len(TIERS) - 1


def hdf5_ready(stage: Stage) -> bool:
    path = Path(stage.hdf5)
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            if int(handle.attrs.get("action_dim", -1)) != stage.action_dim:
                return False
            if not REQUIRED_MASKS.issubset(set(handle.get("mask", {}))):
                return False
            first = next(iter(handle["data"]))
            return handle[f"data/{first}/actions"].shape[1] == stage.action_dim
    except (OSError, KeyError, StopIteration):
        return False


def run_name(stage: Stage, regime: str, algo: str, condition: str) -> str:
    middle = "single_image_proprio_none" + (f"_{regime}" if regime != "normal" else "")
    return f"{stage.run_prefix}_{stage.dataset_tag}_{middle}_{algo}_{condition}_seed1"


def train_done(stage: Stage, regime: str, algo: str, condition: str) -> bool:
    run_dir = Path(stage.out_root) / run_name(stage, regime, algo, condition)
    if (run_dir / "TRAIN_DONE").is_file():
        return True
    # A full checkpoint is authoritative even if a node-local logging failure
    # prevents the wrapper from writing its completion sentinel.
    return any(run_dir.glob("*/models/model_epoch_2000.pth"))


def score_path(stage: Stage, regime: str) -> Path:
    return Path(stage.score_root) / "{algo}" / regime / "threading_pomdp_6_scores.csv"


def score_done(stage: Stage, regime: str, algo: str) -> bool:
    return Path(str(score_path(stage, regime)).format(algo=algo)).is_file()


def report_done(stage: Stage) -> bool:
    return (Path(stage.report_root) / "index.html").is_file()


def artifact_done(key: str) -> bool:
    parts = key.split(":")
    kind, stage_key = parts[:2]
    stage = next((item for item in STAGES if item.key == stage_key), None)
    if stage is None:
        return False
    if kind == "prepare":
        return hdf5_ready(stage)
    if kind == "train":
        _, _, regime, algo, condition = parts
        return train_done(stage, regime, algo, condition)
    if kind == "score":
        _, _, regime, algo = parts
        return score_done(stage, regime, algo)
    if kind == "report":
        return report_done(stage)
    raise ValueError(key)


def refresh(state: ManagerState) -> None:
    for key, item in state.jobs.items():
        if artifact_done(key):
            item.state = "COMPLETED"
            continue
        if adopt_and_prune_live_job(key, item):
            continue
        status = job_state(item.job_id)
        if status in LIVE:
            item.state = status
        elif status in BAD:
            item.state = status
        elif status == "COMPLETED":
            item.state = "WAITING"
            item.job_id = None
        elif item.state in LIVE:
            item.state = "UNKNOWN"


def submit_cpu(name: str, command: str, wall_time: str = "02:00:00") -> str:
    return run([
        "sbatch", "--parsable", "--account=iliad", "--partition=sc-freecpu",
        "--cpus-per-task=4", "--mem=24G", "--time", wall_time,
        "--job-name", name,
        "--output", f"/iris/u/jasonyan/slurm/%j_{name}.out",
        "--error", f"/iris/u/jasonyan/slurm/%j_{name}.err",
        "--wrap", f"bash -lc {quote(command)}",
    ])


def submit_prepare(stage: Stage) -> str:
    common = (
        f"source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh && conda activate openx && "
        f"cd {quote(REPO)} && "
    )
    if stage.prepare_mode == "square":
        command = (
            common + "python scripts/quality/prepare_image_proprio_density_hdf5.py "
            f"--source-type square --input {quote(stage.source_hdf5)} --output {quote(stage.hdf5)} "
            f"--labels-csv {quote(stage.labels_csv)} --seed 20260723 --overwrite && "
            f"python scripts/quality/export_density_hdf5_labels.py --input {quote(stage.hdf5)} "
            f"--output {quote(stage.manifest)}"
        )
        return submit_cpu(f"pip_{stage.key}_h5", command, "08:00:00")
    if stage.prepare_mode == "threading":
        command = (
            common + "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 && "
            "python scripts/quality/prepare_image_proprio_density_hdf5.py "
            f"--source-type threading --input {quote(stage.source_hdf5)} --output {quote(stage.hdf5)} "
            "--env-name Threading_D1 --robosuite-root /iris/u/jasonyan/repos/robosuite-pomdp "
            "--seed 20260723 --overwrite && "
            f"python scripts/quality/export_density_hdf5_labels.py --input {quote(stage.hdf5)} "
            f"--output {quote(stage.manifest)}"
        )
        return gpu_sbatch(TIERS[1], f"pip_{stage.key}_h5", command, "12:00:00", "64G", 8)
    if stage.prepare_mode == "subset":
        command = (
            common + "python scripts/quality/subset_density_hdf5_action_dims.py "
            f"--input {quote(stage.source_hdf5)} --output {quote(stage.hdf5)} "
            f"--action-dims {quote(','.join(str(idx) for idx in range(stage.action_dim)))} --overwrite && "
            f"python scripts/quality/export_density_hdf5_labels.py --input {quote(stage.hdf5)} "
            f"--output {quote(stage.manifest)}"
        )
        return submit_cpu(f"pip_{stage.key}_h5", command, "08:00:00")
    raise ValueError(stage.prepare_mode)


def gpu_sbatch(tier: Tier, name: str, command: str, wall_time: str, mem: str, cpus: int) -> str:
    cmd = [
        "sbatch", "--parsable", f"--account={tier.account}", f"--partition={tier.partition}",
        "--job-name", name, "--gres", tier.gres, "--cpus-per-task", str(cpus),
        "--mem", mem, "--time", wall_time,
        "--output", f"/iris/u/jasonyan/slurm/%j_{name}.out",
        "--error", f"/iris/u/jasonyan/slurm/%j_{name}.err",
    ]
    if tier.preemptible:
        cmd.append("--requeue")
    cmd.extend(tier.extra)
    cmd.extend(["--wrap", f"bash -lc {quote(command)}"])
    return run(cmd)


def submit_train(stage: Stage, regime: str, fold_tag: str, train_key: str, valid_key: str, algo: str, condition: str, item: JobRecord) -> str:
    start_tier = item.tier
    if item.attempts > 0 and item.state in BAD and start_tier is not None:
        start_tier = min(start_tier + 1, len(TIERS) - 1)
    item.tier = choose_tier(start_tier)
    tier = TIERS[item.tier]
    env = {
        "REPO": REPO, "TASK_TAG": "threading_action_dim", "RUN_PREFIX": stage.run_prefix,
        "DATASET_TAG": stage.dataset_tag, "DATASET_HDF5": stage.hdf5,
        "OUT_ROOT": stage.out_root, "CONFIG_ROOT": stage.config_root,
        "ACTION_SOURCE": "image_proprio", "ACTION_TARGET": "single", "ACTION_NORMALIZATION": "none",
        "ALGOS": algo, "CONDITIONS": condition, "FOLD_TAG": fold_tag,
        "TRAIN_FILTER_KEY": train_key, "VALID_FILTER_KEY": valid_key,
        "NUM_EPOCHS": "2000", "BATCH_SIZE": "64", "EPOCH_STEPS": "100",
        "VALIDATION_STEPS": "25", "SAVE_EVERY_N_EPOCHS": "50",
        "GAUSSIAN_MIN_STD": "0.0001", "GMM_MODES": "5",
        "RESUME": "1" if tier.preemptible or item.attempts > 0 else "0",
        "WANDB_PROJECT": "threading-action-dim-density",
    }
    name = f"pip_{stage.key}_{regime}_{algo}_{condition}"
    return gpu_sbatch(tier, name, env_command(env, f"{REPO}/scripts/slurm/train_pen_in_cup_density_models_array.sh"), "24:00:00", "64G", 10)


def submit_score(stage: Stage, regime: str, fold_tag: str, filter_key: str, algo: str, item: JobRecord) -> str:
    start_tier = item.tier
    if item.attempts > 0 and item.state in BAD and start_tier is not None:
        start_tier = min(start_tier + 1, len(TIERS) - 1)
    item.tier = choose_tier(start_tier)
    tier = TIERS[item.tier]
    env = {
        "REPO": REPO, "DATASET_TAG": stage.dataset_tag, "DATASET_HDF5": stage.hdf5,
        "OUT_ROOT": stage.out_root, "SCORE_ROOT": stage.score_root, "RUN_PREFIX": stage.run_prefix,
        "ALGO": algo, "REGIME": regime, "FOLD_TAG": fold_tag, "FILTER_KEY": filter_key,
        "ACTION_SOURCE": "image_proprio", "CONDITIONAL_CONDITION": "image_proprio",
        "M": "16", "K": "512", "SEED": str(stage.score_seed),
    }
    name = f"pip_{stage.key}_{regime}_{algo}_score"
    return gpu_sbatch(tier, name, env_command(env, f"{REPO}/scripts/slurm/score_threading_pomdp_6.sh"), "12:00:00", "64G", 8)


def submit_report(stage: Stage) -> str:
    commands = [
        "set -euo pipefail",
        "source /iris/u/jasonyan/miniforge3/etc/profile.d/conda.sh",
        "conda activate openx",
        f"cd {quote(REPO)}",
    ]
    for algo in ALGOS:
        commands.append(
            "python scripts/quality/merge_threading_pomdp_fold_scores.py "
            f"--input {quote(stage.score_root + '/' + algo + '/fold0/threading_pomdp_6_scores.csv')} "
            f"--input {quote(stage.score_root + '/' + algo + '/fold1/threading_pomdp_6_scores.csv')} "
            f"--output {quote(stage.score_root + '/' + algo + '/2fold/threading_pomdp_6_scores.csv')}"
        )
    commands.append(
        "python scripts/quality/build_threading_pomdp_report.py "
        f"--manifest {quote(stage.manifest)} --score-root {quote(stage.score_root)} "
        f"--output {quote(stage.report_root)} --title {quote(stage.title)} "
        f"--description {quote(stage.description)}"
    )
    return submit_cpu(f"pip_{stage.key}_report", " && ".join(commands), "02:00:00")


def active_gpu_jobs(state: ManagerState) -> int:
    return sum(key.startswith(("train:", "score:")) and item.state in LIVE for key, item in state.jobs.items())


def can_retry(item: JobRecord, args: argparse.Namespace) -> bool:
    return item.state not in LIVE and item.state != "COMPLETED" and item.attempts < args.max_attempts


def submit_record(item: JobRecord, job_id: str) -> None:
    item.job_id = job_id
    item.state = "PENDING"
    item.attempts += 1
    item.submitted_at = time.time()


def migrate_pending(state: ManagerState, args: argparse.Namespace) -> None:
    now = time.time()
    for key, item in state.jobs.items():
        if item.state != "PENDING" or not key.startswith(("train:", "score:")):
            continue
        reason = pending_reason(item.job_id)
        unavailable = any(token in reason for token in ("ReqNodeNotAvail", "UnavailableNodes", "DOWN", "DRAINED"))
        resource_wait = any(token in reason for token in ("Priority", "Resources")) and now - item.submitted_at >= args.pending_migrate_seconds
        if not unavailable and not resource_wait:
            continue
        if item.tier is not None and item.tier < len(TIERS) - 1:
            old_job_id = str(item.job_id)
            run(["scancel", old_job_id], check=False)
            for _ in range(15):
                if job_state(old_job_id) not in LIVE:
                    break
                time.sleep(2)
            if job_state(old_job_id) in LIVE:
                continue
            item.tier += 1
            item.job_id = None
            item.state = "WAITING"


def ensure_stage(state: ManagerState, stage: Stage, args: argparse.Namespace) -> None:
    prep_key = f"prepare:{stage.key}"
    prep = record(state, prep_key)
    if stage.prepare_mode == "subset" and not Path(stage.source_hdf5).is_file():
        return
    if not hdf5_ready(stage) and can_retry(prep, args):
        try:
            submit_record(prep, submit_prepare(stage))
        except RuntimeError as exc:
            prep.state = "WAITING"
            print(f"prepare submission deferred for {prep_key}: {exc}", flush=True)
        return
    if not hdf5_ready(stage):
        return
    prep.state = "COMPLETED"

    for regime, fold_tag, train_key, valid_key, _ in REGIMES:
        for algo in ALGOS:
            for condition in CONDITIONS:
                key = f"train:{stage.key}:{regime}:{algo}:{condition}"
                item = record(state, key)
                if adopt_and_prune_live_job(key, item):
                    continue
                if train_done(stage, regime, algo, condition):
                    item.state = "COMPLETED"
                    continue
                if active_gpu_jobs(state) >= args.max_active_gpu:
                    break
                if can_retry(item, args):
                    try:
                        submit_record(
                            item,
                            submit_train(stage, regime, fold_tag, train_key, valid_key, algo, condition, item),
                        )
                    except RuntimeError as exc:
                        item.state = "WAITING"
                        print(f"train submission deferred for {key}: {exc}", flush=True)

    refresh(state)
    for regime, _, _, _, filter_key in REGIMES:
        fold_tag = "" if regime == "normal" else regime
        for algo in ALGOS:
            key = f"score:{stage.key}:{regime}:{algo}"
            item = record(state, key)
            if adopt_and_prune_live_job(key, item):
                continue
            if score_done(stage, regime, algo):
                item.state = "COMPLETED"
                continue
            pair_done = all(train_done(stage, regime, algo, condition) for condition in CONDITIONS)
            if pair_done and active_gpu_jobs(state) < args.max_active_gpu and can_retry(item, args):
                try:
                    submit_record(item, submit_score(stage, regime, fold_tag, filter_key, algo, item))
                except RuntimeError as exc:
                    item.state = "WAITING"
                    print(f"score submission deferred for {key}: {exc}", flush=True)

    all_scores = all(score_done(stage, regime, algo) for regime, *_ in REGIMES for algo in ALGOS)
    report_key = f"report:{stage.key}"
    report = record(state, report_key)
    if report_done(stage):
        report.state = "COMPLETED"
    elif all_scores and can_retry(report, args):
        try:
            submit_record(report, submit_report(stage))
        except RuntimeError as exc:
            report.state = "WAITING"
            print(f"report submission deferred for {report_key}: {exc}", flush=True)


def status_line(state: ManagerState) -> str:
    counts: dict[str, int] = {}
    for item in state.jobs.values():
        counts[item.state] = counts.get(item.state, 0) + 1
    stage = next((item.key for item in STAGES if not report_done(item)), "complete")
    return f"stage={stage} jobs={counts}"


def main() -> None:
    args = parse_args()
    requested = [value.strip() for value in re.split(r"[:,]", args.stages) if value.strip()]
    known = {stage.key: stage for stage in STAGES}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}; choices={sorted(known)}")
    stages = [known[key] for key in requested]
    state = load_state(args.state_file)
    while True:
        refresh(state)
        migrate_pending(state, args)
        remaining = [stage for stage in stages if not report_done(stage)]
        if not remaining:
            save_state(args.state_file, state)
            print("all stages completed", flush=True)
            return
        for stage in remaining:
            ensure_stage(state, stage, args)
        refresh(state)
        save_state(args.state_file, state)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), status_line(state), flush=True)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
