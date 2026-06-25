#!/usr/bin/env python3
"""Maintain the wrench-to-hook density training and scoring queue on Slurm.

The manager submits at most 48 model-training jobs across normal and 2-fold
runs. It prioritizes all 06/13 model training before 06/15. Preemptible tiers
are submitted with RESUME=1 and --requeue; if a job exits unsuccessfully before
the expected checkpoint or score file appears, the manager resubmits it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DATASETS = {
    "0613_98": "/iris/u/jasonyan/data/droid_wrench_to_hook_06132026_98_rlds/droid_pen_in_cup/1.0.0",
    "0615_96": "/iris/u/jasonyan/data/droid_wrench_on_hook_06152026_96_rlds/droid_pen_in_cup/1.0.0",
}
LABELS = {
    "0613_98": "/iris/u/jasonyan/data/wrench_to_hook_06132026_annotations_renumbered_98.csv",
    "0615_96": "/iris/u/jasonyan/data/wrench_on_hook_06152026_annotations.csv",
}
LABEL_COLUMNS = ("observability", "optimality")
ALGOS = ("gaussian", "gmm")
CONDITIONS = ("image_state", "image", "state", "action_prior")
REGIMES = (
    ("normal", "", "train", "valid", ""),
    ("fold0", "fold0", "fold0_train", "fold0_valid", "fold0_score"),
    ("fold1", "fold1", "fold1_train", "fold1_valid", "fold1_score"),
)

TERMINAL_BAD = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL"}
TERMINAL_OK = {"COMPLETED"}
LIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING", "SUSPENDED"}
IRIS5_PLUS = ("--exclude", "iris1,iris2,iris3,iris4,iris-hp-z8")
ILIAD5_PLUS = ("--exclude", "iliad1,iliad2,iliad3,iliad4")


@dataclass(frozen=True)
class Tier:
    name: str
    account: str
    partition: str
    slots: int
    preemptible: bool
    gres: str = "gpu:1"
    extra_args: tuple[str, ...] = ()


TIERS = (
    Tier("iris_hi", "iris", "iris-hi", 6, False, "gpu:1", IRIS5_PLUS),
    Tier("iris", "iris", "iris", 10, True, "gpu:1", IRIS5_PLUS),
    Tier("iliad", "iliad", "iliad", 8, False, "gpu:1", ILIAD5_PLUS),
    Tier("iliad_lo", "iliad", "iliad-lo", 8, True, "gpu:1", ILIAD5_PLUS),
    Tier("sc_loprio_h200", "iliad", "sc-loprio", 4, True, "gpu:h200:1"),
    Tier("sc_loprio_a100", "iliad", "sc-loprio", 4, True, "gpu:a100:1"),
    Tier("sc_loprio_l40s", "iliad", "sc-loprio", 2, True, "gpu:l40s:1"),
    Tier("sc_loprio_a6000", "iliad", "sc-loprio", 2, True, "gpu:a6000:1"),
    Tier("sc_loprio_a40", "iliad", "sc-loprio", 1, True, "gpu:a40:1"),
    Tier("sc_loprio_a5000", "iliad", "sc-loprio", 1, True, "gpu:a5000:1"),
)


@dataclass
class Task:
    key: str
    dataset: str
    regime: str
    fold_tag: str
    algo: str
    condition: str
    train_filter: str
    valid_filter: str
    score_filter: str
    tier_index: int
    train_job_id: str | None = None
    train_state: str = "WAITING"
    train_attempts: int = 0
    train_tier_cursor: int | None = None
    train_submitted_at: float = 0.0
    score_job_id: str | None = None
    score_state: str = "WAITING"
    score_attempts: int = 0
    score_tier_cursor: int | None = None
    score_submitted_at: float = 0.0


@dataclass
class ManagerState:
    prepare_jobs: dict[str, str] = field(default_factory=dict)
    prepare_states: dict[str, str] = field(default_factory=dict)
    tasks: dict[str, Task] = field(default_factory=dict)
    eval_jobs: dict[str, str] = field(default_factory=dict)
    eval_states: dict[str, str] = field(default_factory=dict)


def run(cmd: list[str], check: bool = True) -> str:
    proc = subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout.strip()


def load_state(path: Path) -> ManagerState:
    if not path.exists():
        return ManagerState()
    raw = json.loads(path.read_text())
    return ManagerState(
        prepare_jobs=raw.get("prepare_jobs", {}),
        prepare_states=raw.get("prepare_states", {}),
        tasks={key: Task(**value) for key, value in raw.get("tasks", {}).items()},
        eval_jobs=raw.get("eval_jobs", {}),
        eval_states=raw.get("eval_states", {}),
    )


def save_state(path: Path, state: ManagerState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "prepare_jobs": state.prepare_jobs,
        "prepare_states": state.prepare_states,
        "tasks": {key: asdict(task) for key, task in state.tasks.items()},
        "eval_jobs": state.eval_jobs,
        "eval_states": state.eval_states,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def hdf5_path(args: argparse.Namespace, dataset: str) -> Path:
    return Path(args.dataset_root) / f"{args.task_tag}_{dataset}_{args.action_target}_{args.action_source}_{args.action_normalization}.hdf5"


def hdf5_ready(args: argparse.Namespace, dataset: str) -> bool:
    path = hdf5_path(args, dataset)
    if not path.exists():
        return False
    code = (
        "import h5py,sys; "
        f"p={str(path)!r}; "
        "f=h5py.File(p,'r'); "
        "need={'train','valid','all','fold0_train','fold0_valid','fold0_score','fold1_train','fold1_valid','fold1_score'}; "
        "ok=f.attrs.get('action_source','')=='cartesian_velocity' and f.attrs.get('action_normalization','')=='none' and need.issubset(set(f['mask'].keys())); "
        "sys.exit(0 if ok else 1)"
    )
    return subprocess.run(["python", "-c", code]).returncode == 0


def build_tasks() -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    tier_index = 0
    for dataset in ("0613_98", "0615_96"):
        for regime, fold_tag, train_filter, valid_filter, score_filter in REGIMES:
            for algo in ALGOS:
                for condition in CONDITIONS:
                    key = f"{dataset}:{regime}:{algo}:{condition}"
                    tasks[key] = Task(
                        key=key,
                        dataset=dataset,
                        regime=regime,
                        fold_tag=fold_tag,
                        algo=algo,
                        condition=condition,
                        train_filter=train_filter,
                        valid_filter=valid_filter,
                        score_filter=score_filter,
                        tier_index=tier_index,
                    )
                    tier_index += 1
    return tasks


def tier_for(task: Task) -> Tier:
    idx = task.tier_index
    for tier in TIERS:
        if idx < tier.slots:
            return tier
        idx -= tier.slots
    return TIERS[-1]


def tier_position_for(task: Task) -> int:
    idx = task.tier_index
    for pos, tier in enumerate(TIERS):
        if idx < tier.slots:
            return pos
        idx -= tier.slots
    return len(TIERS) - 1


def tier_for_phase(task: Task, phase: str) -> Tier:
    cursor = task.train_tier_cursor if phase == "train" else task.score_tier_cursor
    if cursor is None:
        cursor = tier_position_for(task)
    return TIERS[min(cursor, len(TIERS) - 1)]


def tier_capacity() -> int:
    return sum(tier.slots for tier in TIERS)


def job_state(job_id: str | None) -> str | None:
    if not job_id:
        return None
    try:
        out = run(["sacct", "-j", str(job_id), "-n", "-P", "-o", "JobIDRaw,State"], check=False)
    except FileNotFoundError:
        return None
    states = []
    for line in out.splitlines():
        if not line.strip():
            continue
        raw_id, state = line.split("|", 1)
        if raw_id == str(job_id):
            states.append(state.split()[0])
    if states:
        return states[-1]
    out = run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], check=False)
    return out.splitlines()[0].strip() if out.strip() else None


def live_jobs_by_name() -> dict[str, list[tuple[str, str]]]:
    out = run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%A|%j|%T"], check=False)
    jobs: dict[str, list[tuple[str, str]]] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        job_id, name, state = line.split("|", 2)
        jobs.setdefault(name, []).append((job_id, state))
    return jobs


def pick_live_job(jobs: list[tuple[str, str]]) -> tuple[str, str] | None:
    for wanted_state in ("RUNNING", "PENDING"):
        for job_id, state in jobs:
            if state == wanted_state:
                return job_id, state
    return jobs[0] if jobs else None


def prune_duplicate_live_jobs() -> None:
    for name, jobs in live_jobs_by_name().items():
        if not name.startswith("wth_") or name == "wth_den_mgr":
            continue
        keep = pick_live_job(jobs)
        if not keep:
            continue
        keep_id, _ = keep
        for job_id, state in jobs:
            if job_id != keep_id and state == "PENDING":
                scancel(job_id)


def pending_reason(job_id: str | None) -> str:
    if not job_id:
        return ""
    out = run(["squeue", "-h", "-j", str(job_id), "-t", "PD", "-o", "%R"], check=False)
    return out.splitlines()[0].strip() if out.strip() else ""


def scancel(job_id: str | None) -> None:
    if job_id:
        run(["scancel", str(job_id)], check=False)


def run_name(args: argparse.Namespace, task: Task) -> str:
    middle = f"{args.action_target}_{args.action_source}_{args.action_normalization}"
    if task.fold_tag:
        middle = f"{middle}_{task.fold_tag}"
    return f"{args.run_prefix}_{task.dataset}_{middle}_{task.algo}_{task.condition}_seed1"


def checkpoint_done(args: argparse.Namespace, task: Task) -> bool:
    run_dir = Path(args.out_root) / run_name(args, task)
    return any(run_dir.glob("*/models/*best*validation*.pth")) or any(run_dir.glob("*/models/*.pth"))


def train_done_marker(args: argparse.Namespace, task: Task) -> bool:
    run_dir = Path(args.out_root) / run_name(args, task)
    return (run_dir / "TRAIN_DONE").is_file()


def score_output(args: argparse.Namespace, task: Task) -> Path:
    recipe = f"{task.dataset}/{args.action_target}_{args.action_source}_{args.action_normalization}"
    if task.fold_tag:
        recipe += f"/{task.fold_tag}"
    return Path(args.score_root) / recipe / task.algo / args.ckpt_mode / f"{task.condition}.pkl"


def score_done(args: argparse.Namespace, task: Task) -> bool:
    return score_output(args, task).is_file()


def normal_group_score_done(args: argparse.Namespace, dataset: str, algo: str) -> bool:
    for condition in CONDITIONS:
        key = f"{dataset}:normal:{algo}:{condition}"
        if key not in build_tasks():
            return False
        task = build_tasks()[key]
        if not score_output(args, task).is_file():
            return False
    return True


def twofold_group_score_done(args: argparse.Namespace, dataset: str, algo: str) -> bool:
    tasks = build_tasks()
    for fold in ("fold0", "fold1"):
        for condition in CONDITIONS:
            task = tasks[f"{dataset}:{fold}:{algo}:{condition}"]
            if not score_output(args, task).is_file():
                return False
    return True


def sbatch_base(tier: Tier, job_name: str, wall_time: str) -> list[str]:
    cmd = [
        "sbatch",
        "--parsable",
        "--account",
        tier.account,
        "--partition",
        tier.partition,
        "--job-name",
        job_name,
        "--gres",
        tier.gres,
        "--cpus-per-task",
        "12",
        "--mem",
        "96G",
        "--time",
        wall_time,
    ]
    if tier.preemptible:
        cmd.append("--requeue")
    cmd.extend(tier.extra_args)
    return cmd


def env_prefix(env: dict[str, str]) -> str:
    return " ".join(f"{key}={sh_quote(value)}" for key, value in env.items())


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def submit_prepare(args: argparse.Namespace, dataset: str) -> str:
    env = {
        "REPO": args.repo,
        "TASK_TAG": args.task_tag,
        "DATASET_TAG": dataset,
        "RLDS_PATH": DATASETS[dataset],
        "OUT_ROOT": args.dataset_root,
        "OUTPUT": str(hdf5_path(args, dataset)),
        "ACTION_SOURCE": args.action_source,
        "ACTION_TARGET": args.action_target,
        "ACTION_NORMALIZATION": args.action_normalization,
        "NUM_FOLDS": "2",
    }
    cmd = [
        "sbatch",
        "--parsable",
        "--account",
        "iris",
        "--partition",
        "iris-hi",
        "--job-name",
        f"wth_h5_{dataset}",
        *IRIS5_PLUS,
        "--cpus-per-task",
        "8",
        "--mem",
        "96G",
        "--time",
        "08:00:00",
        "--wrap",
        f"cd {sh_quote(args.repo)} && {env_prefix(env)} bash {sh_quote(args.repo + '/scripts/slurm/prepare_wrench_to_hook_density_hdf5.sh')}",
    ]
    return run(cmd)


def submit_train(args: argparse.Namespace, task: Task) -> str:
    tier = tier_for_phase(task, "train")
    env = {
        "REPO": args.repo,
        "TASK_TAG": args.task_tag,
        "RUN_PREFIX": args.run_prefix,
        "DATASET_TAG": task.dataset,
        "DATASET_HDF5": str(hdf5_path(args, task.dataset)),
        "OUT_ROOT": args.out_root,
        "CONFIG_ROOT": args.config_root,
        "ACTION_SOURCE": args.action_source,
        "ACTION_TARGET": args.action_target,
        "ACTION_NORMALIZATION": args.action_normalization,
        "ALGOS": task.algo,
        "CONDITIONS": task.condition,
        "FOLD_TAG": task.fold_tag,
        "TRAIN_FILTER_KEY": task.train_filter,
        "VALID_FILTER_KEY": task.valid_filter,
        "RESUME": "1" if tier.preemptible or task.train_attempts > 0 or checkpoint_done(args, task) else "0",
        "WANDB_PROJECT": "wrench-to-hook-density",
    }
    cmd = sbatch_base(tier, train_job_name(task), args.train_time)
    cmd.extend(
        [
            "--wrap",
            f"cd {sh_quote(args.repo)} && {env_prefix(env)} bash {sh_quote(args.repo + '/scripts/slurm/train_wrench_to_hook_density_model.sh')}",
        ]
    )
    return run(cmd)


def submit_score(args: argparse.Namespace, task: Task) -> str:
    tier = tier_for_phase(task, "score")
    env = {
        "REPO": args.repo,
        "TASK_TAG": args.task_tag,
        "RUN_PREFIX": args.run_prefix,
        "DATASET_TAG": task.dataset,
        "DATASET_HDF5": str(hdf5_path(args, task.dataset)),
        "OUT_ROOT": args.out_root,
        "SCORE_ROOT": args.score_root,
        "ACTION_SOURCE": args.action_source,
        "ACTION_TARGET": args.action_target,
        "ACTION_NORMALIZATION": args.action_normalization,
        "ALGOS": task.algo,
        "CONDITIONS": task.condition,
        "FOLD_TAG": task.fold_tag,
        "SCORE_FILTER_KEY": task.score_filter,
        "CKPT_MODE": args.ckpt_mode,
        "RESUME": "1",
    }
    cmd = sbatch_base(tier, score_job_name(task), args.score_time)
    cmd.extend(
        [
            "--wrap",
            f"cd {sh_quote(args.repo)} && {env_prefix(env)} bash {sh_quote(args.repo + '/scripts/slurm/score_wrench_to_hook_density_model.sh')}",
        ]
    )
    return run(cmd)


def train_job_name(task: Task) -> str:
    return f"wth_{task.dataset}_{task.regime}_{task.algo}_{task.condition}"


def score_job_name(task: Task) -> str:
    return f"wth_score_{task.dataset}_{task.regime}_{task.algo}_{task.condition}"


def submit_eval(args: argparse.Namespace, dataset: str, algo: str, label_column: str, twofold: bool) -> str:
    if twofold:
        script = "merge_combine_eval_density_2fold_scores.sh"
        name = f"wth_2f_eval_{dataset}_{algo}_{label_column}"
    else:
        script = "combine_eval_pen_in_cup_density_scores.sh"
        name = f"wth_eval_{dataset}_{algo}_{label_column}"
    env = {
        "REPO": args.repo,
        "TASK_TAG": args.task_tag,
        "DATASET_TAG": dataset,
        "LABELS_CSV": LABELS[dataset],
        "SCORE_ROOT": args.score_root,
        "EVAL_ROOT": args.eval_root,
        "ACTION_SOURCE": args.action_source,
        "ACTION_TARGET": args.action_target,
        "ACTION_NORMALIZATION": args.action_normalization,
        "ALGO": algo,
        "CKPT_MODE": args.ckpt_mode,
        "LABEL_COLUMN": label_column,
        "HIGHER_IS_BETTER": "0",
    }
    cmd = [
        "sbatch",
        "--parsable",
        "--account",
        "iris",
        "--partition",
        "iris-hi",
        "--job-name",
        name,
        *IRIS5_PLUS,
        "--cpus-per-task",
        "2",
        "--mem",
        "16G",
        "--time",
        "02:00:00",
        "--wrap",
        f"cd {sh_quote(args.repo)} && {env_prefix(env)} bash {sh_quote(args.repo + '/scripts/slurm/' + script)}",
    ]
    return run(cmd)


def active_jobs(state: ManagerState) -> int:
    count = 0
    for task in state.tasks.values():
        if task.train_state in LIVE:
            count += 1
    return count


def ensure_tasks(state: ManagerState) -> None:
    if state.tasks:
        return
    state.tasks = build_tasks()


def refresh(args: argparse.Namespace, state: ManagerState) -> None:
    for dataset, job_id in list(state.prepare_jobs.items()):
        if hdf5_ready(args, dataset):
            state.prepare_states[dataset] = "COMPLETED"
            continue
        state.prepare_states[dataset] = job_state(job_id) or "UNKNOWN"
    live_jobs = live_jobs_by_name()
    for task in state.tasks.values():
        if not train_done_marker(args, task):
            live_train = pick_live_job(live_jobs.get(train_job_name(task), []))
            if live_train and (not task.train_job_id or task.train_state not in LIVE):
                task.train_job_id, task.train_state = live_train
        live_score = pick_live_job(live_jobs.get(score_job_name(task), []))
        if live_score and (not task.score_job_id or task.score_state not in LIVE):
            task.score_job_id, task.score_state = live_score
        train_state = job_state(task.train_job_id) if task.train_job_id else None
        has_checkpoint = checkpoint_done(args, task)
        if train_done_marker(args, task) or (train_state in TERMINAL_OK and has_checkpoint):
            task.train_state = "COMPLETED"
        elif train_state:
            task.train_state = train_state
        elif task.train_state == "COMPLETED":
            # Older manager versions treated any checkpoint as completion. Downgrade
            # those checkpoint-only entries so they resume instead of being scored.
            task.train_state = "WAITING"
        if score_done(args, task):
            task.score_state = "COMPLETED"
        elif task.score_job_id:
            task.score_state = job_state(task.score_job_id) or "UNKNOWN"
    for key, job_id in list(state.eval_jobs.items()):
        state.eval_states[key] = job_state(job_id) or state.eval_states.get(key, "UNKNOWN")


def step(args: argparse.Namespace, state: ManagerState) -> None:
    ensure_tasks(state)
    prune_duplicate_live_jobs()
    for dataset in DATASETS:
        if hdf5_ready(args, dataset):
            state.prepare_states[dataset] = "COMPLETED"
        elif state.prepare_states.get(dataset) not in LIVE:
            state.prepare_jobs[dataset] = submit_prepare(args, dataset)
            state.prepare_states[dataset] = "PENDING"

    refresh(args, state)
    migrate_pending_jobs(args, state)

    # Submit training in manifest order: all 06/13 tasks are before 06/15 tasks.
    for task in state.tasks.values():
        if active_jobs(state) >= min(args.max_train_jobs, tier_capacity()):
            break
        if state.prepare_states.get(task.dataset) != "COMPLETED":
            continue
        if task.train_state == "COMPLETED":
            continue
        if task.train_state in LIVE:
            continue
        if task.train_state in TERMINAL_BAD or task.train_job_id is None or task.train_state in {"WAITING", "UNKNOWN"}:
            if task.train_attempts >= args.max_attempts:
                continue
            if task.train_tier_cursor is None:
                task.train_tier_cursor = tier_position_for(task)
            task.train_attempts += 1
            task.train_job_id = submit_train(args, task)
            task.train_submitted_at = time.time()
            task.train_state = "PENDING"

    refresh(args, state)
    migrate_pending_jobs(args, state)

    # Submit scoring only after the corresponding train model is complete.
    for task in state.tasks.values():
        if task.train_state != "COMPLETED" or task.score_state == "COMPLETED" or task.score_state in LIVE:
            continue
        if task.score_state in TERMINAL_BAD or task.score_job_id is None or task.score_state in {"WAITING", "UNKNOWN"}:
            if task.score_attempts >= args.max_attempts:
                continue
            if task.score_tier_cursor is None:
                task.score_tier_cursor = tier_position_for(task)
            task.score_attempts += 1
            task.score_job_id = submit_score(args, task)
            task.score_submitted_at = time.time()
            task.score_state = "PENDING"

    refresh(args, state)
    migrate_pending_jobs(args, state)

    for dataset in DATASETS:
        for algo in ALGOS:
            for label_column in LABEL_COLUMNS:
                key = f"normal:{dataset}:{algo}:{label_column}"
                if key not in state.eval_jobs and normal_group_score_done(args, dataset, algo):
                    state.eval_jobs[key] = submit_eval(args, dataset, algo, label_column, twofold=False)
                    state.eval_states[key] = "PENDING"

                key = f"2fold:{dataset}:{algo}:{label_column}"
                if key not in state.eval_jobs and twofold_group_score_done(args, dataset, algo):
                    state.eval_jobs[key] = submit_eval(args, dataset, algo, label_column, twofold=True)
                    state.eval_states[key] = "PENDING"


def should_migrate_pending(args: argparse.Namespace, job_id: str | None, submitted_at: float) -> bool:
    reason = pending_reason(job_id)
    if not reason:
        return False
    if (
        "ReqNodeNotAvail" in reason
        or "UnavailableNodes" in reason
        or "DOWN" in reason
        or "DRAINED" in reason
    ):
        return True
    age = time.time() - submitted_at if submitted_at else float("inf")
    return age >= args.pending_migrate_seconds and ("Priority" in reason or "Resources" in reason)


def migrate_pending_jobs(args: argparse.Namespace, state: ManagerState) -> None:
    for task in state.tasks.values():
        if task.train_state == "PENDING" and should_migrate_pending(args, task.train_job_id, task.train_submitted_at):
            cursor = task.train_tier_cursor if task.train_tier_cursor is not None else tier_position_for(task)
            if cursor < len(TIERS) - 1:
                scancel(task.train_job_id)
                task.train_tier_cursor = cursor + 1
                task.train_job_id = None
                task.train_state = "WAITING"
        if task.score_state == "PENDING" and should_migrate_pending(args, task.score_job_id, task.score_submitted_at):
            cursor = task.score_tier_cursor if task.score_tier_cursor is not None else tier_position_for(task)
            if cursor < len(TIERS) - 1:
                scancel(task.score_job_id)
                task.score_tier_cursor = cursor + 1
                task.score_job_id = None
                task.score_state = "WAITING"


def summary(state: ManagerState) -> str:
    counts: dict[str, int] = {}
    score_counts: dict[str, int] = {}
    for task in state.tasks.values():
        counts[task.train_state] = counts.get(task.train_state, 0) + 1
        score_counts[task.score_state] = score_counts.get(task.score_state, 0) + 1
    eval_counts: dict[str, int] = {}
    for value in state.eval_states.values():
        eval_counts[value] = eval_counts.get(value, 0) + 1
    return f"prepare={state.prepare_states} train={counts} score={score_counts} eval={eval_counts}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/iris/u/jasonyan/repos/demonstration-information")
    parser.add_argument("--task-tag", default="wrench_to_hook")
    parser.add_argument("--run-prefix", default="wrench_to_hook")
    parser.add_argument("--action-source", default="cartesian_velocity")
    parser.add_argument("--action-target", default="single")
    parser.add_argument("--action-normalization", default="none")
    parser.add_argument("--dataset-root", default="/iris/u/jasonyan/data/wrench_to_hook_density_datasets")
    parser.add_argument("--out-root", default="/iris/u/jasonyan/data/robomimic_outputs/wrench_to_hook_density")
    parser.add_argument("--config-root", default="/iris/u/jasonyan/data/wrench_to_hook_density_configs")
    parser.add_argument("--score-root", default="/iris/u/jasonyan/data/wrench_to_hook_density_scores")
    parser.add_argument("--eval-root", default="/iris/u/jasonyan/data/wrench_to_hook_density_eval")
    parser.add_argument("--state-file", type=Path, default=Path("/iris/u/jasonyan/data/wrench_to_hook_density_queue_state.json"))
    parser.add_argument("--max-train-jobs", type=int, default=48)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--pending-migrate-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--train-time", default="24:00:00")
    parser.add_argument("--score-time", default="08:00:00")
    parser.add_argument("--ckpt-mode", default="best_validation")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = load_state(args.state_file)
    while True:
        step(args, state)
        save_state(args.state_file, state)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), summary(state), flush=True)
        all_train_done = all(task.train_state == "COMPLETED" for task in state.tasks.values())
        all_score_done = all(task.score_state == "COMPLETED" for task in state.tasks.values())
        all_eval_done = len(state.eval_states) == 16 and all(value == "COMPLETED" for value in state.eval_states.values())
        if args.once or (all_train_done and all_score_done and all_eval_done):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
