#!/usr/bin/env python3
"""Export status for the 48 Square / Threading image-proprio density runs."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import Counter
from pathlib import Path


STAGES = {
    "square_7d": ("square_mh_pomdp_image_proprio_7d", "square_mh_300"),
    "square_6d": ("square_mh_pomdp_image_proprio_6d", "square_mh_300"),
    "d1_8d": ("threading_d1_manual200_pomdp_image_proprio_8d", "threading_d1_manual200"),
    "d1_7d": ("threading_d1_manual200_pomdp_image_proprio_7d", "threading_d1_manual200"),
}
REGIMES = ("normal", "fold0", "fold1")
ALGOS = ("gaussian", "gmm")
CONDITIONS = ("image_proprio", "action_prior")
ANY_EPOCH_RE = re.compile(r"model_epoch_(\d+)")
BEST_RE = re.compile(r"model_epoch_(\d+)_best_validation_([-+0-9.eE]+)\.pth$")
RESUME_RE = re.compile(r"resuming training from epoch\s+(\d+)", re.IGNORECASE)
LOG_EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s+Memory Usage")


def command(args: list[str]) -> str:
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def read_edges(path: Path, edge_bytes: int = 2_000_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            head = handle.read(min(edge_bytes, size))
            if size > edge_bytes:
                handle.seek(max(0, size - edge_bytes))
                tail = handle.read(edge_bytes)
            else:
                tail = b""
        return (head + b"\n" + tail).decode(errors="ignore")
    except OSError:
        return ""


def elapsed_seconds(value: str) -> int:
    if not value:
        return 0
    if "-" in value:
        days, value = value.split("-", 1)
    else:
        days = "0"
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, *parts
    else:
        return 0
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def expected_runs(root: Path) -> list[dict[str, str | Path]]:
    runs = []
    for stage, (prefix, dataset_tag) in STAGES.items():
        for regime in REGIMES:
            middle = "single_image_proprio_none" + (f"_{regime}" if regime != "normal" else "")
            for algo in ALGOS:
                for condition in CONDITIONS:
                    run_name = f"{prefix}_{dataset_tag}_{middle}_{algo}_{condition}_seed1"
                    runs.append(
                        {
                            "stage": stage,
                            "regime": regime,
                            "algo": algo,
                            "condition": condition,
                            "job_name": f"pip_{stage}_{regime}_{algo}_{condition}",
                            "run_dir": root / run_name,
                        }
                    )
    return runs


def queue_rows() -> dict[str, list[dict[str, str]]]:
    text = command(["squeue", "-h", "-u", "jasonyan", "-o", "%i|%j|%T|%M|%S|%N|%R"])
    rows: dict[str, list[dict[str, str]]] = {}
    for line in text.splitlines():
        parts = line.split("|", 6)
        if len(parts) != 7 or not parts[1].startswith("pip_"):
            continue
        job_id, name, state, elapsed, start, node, reason = parts
        rows.setdefault(name, []).append(
            {
                "job_id": job_id,
                "state": state,
                "elapsed": elapsed,
                "start": start,
                "node": node,
                "reason": reason,
            }
        )
    return rows


def history_rows() -> dict[str, list[dict[str, str]]]:
    text = command(
        [
            "sacct",
            "-X",
            "-S",
            "2026-07-22",
            "-u",
            "jasonyan",
            "-n",
            "-P",
            "-o",
            "JobIDRaw,JobName%100,State,Elapsed,Start,End,NodeList,ExitCode,Restarts",
        ]
    )
    rows: dict[str, list[dict[str, str]]] = {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 9 or not parts[1].startswith("pip_"):
            continue
        job_id, name, state, elapsed, start, end, node, exit_code, restarts = parts[:9]
        rows.setdefault(name, []).append(
            {
                "job_id": job_id,
                "state": state.split()[0],
                "elapsed": elapsed,
                "start": start,
                "end": end,
                "node": node,
                "exit_code": exit_code,
                "restarts": restarts,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/iris/u/jasonyan/data/robomimic_outputs/pomdp_image_proprio_20260723"),
    )
    parser.add_argument(
        "--slurm-root",
        type=Path,
        default=Path("/iris/u/jasonyan/slurm"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/iris/u/jasonyan/data/pomdp_image_proprio_20260723/pomdp_run_status_20260723.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = queue_rows()
    history = history_rows()
    rows = []

    for spec in expected_runs(args.root):
        run_dir = Path(spec["run_dir"])
        checkpoints = list(run_dir.glob("*/models/*.pth")) if run_dir.exists() else []
        checkpoint_epochs = [
            int(match.group(1))
            for path in checkpoints
            if (match := ANY_EPOCH_RE.search(path.name))
        ]
        best = [
            (float(match.group(2)), int(match.group(1)))
            for path in checkpoints
            if (match := BEST_RE.match(path.name))
        ]
        best_val, best_epoch = min(best) if best else ("", "")

        job_name = str(spec["job_name"])
        job_history = sorted(history.get(job_name, []), key=lambda item: (item["start"], item["job_id"]))
        active_jobs = sorted(queue.get(job_name, []), key=lambda item: item["job_id"])
        active = active_jobs[-1] if active_jobs else None

        log_epochs: list[int] = []
        resume_epochs: list[int] = []
        finished = False
        for job in job_history:
            for path in args.slurm_root.glob(f"{job['job_id']}_{job_name}.*"):
                text = read_edges(path)
                log_epochs.extend(int(value) for value in LOG_EPOCH_RE.findall(text))
                resume_epochs.extend(int(value) for value in RESUME_RE.findall(text))
                finished = finished or "finished run successfully" in text

        all_epochs = checkpoint_epochs + log_epochs
        current_epoch: int | str = max(all_epochs) if all_epochs else ""
        full_checkpoint = any(run_dir.glob("*/models/model_epoch_2000.pth"))
        if full_checkpoint:
            current_epoch = 2000

        submission_ids = {job["job_id"] for job in job_history}
        restart_count = sum(
            int(job["restarts"])
            for job in job_history
            if str(job["restarts"]).isdigit()
        )
        resume_attempted = bool(resume_epochs or restart_count or len(submission_ids) > 1)
        resume_from: int | str = max(resume_epochs) if resume_epochs else ""

        if full_checkpoint:
            status = "COMPLETED"
        elif active:
            status = active["state"]
        elif current_epoch != "":
            status = "WAITING_RESUME"
        else:
            status = "NOT_STARTED"

        if not resume_attempted:
            resume_status = "not_needed"
        elif resume_from != "" and current_epoch != "" and int(current_epoch) > int(resume_from):
            resume_status = "success"
        elif full_checkpoint:
            resume_status = "success"
        elif active and active["state"] == "PENDING":
            resume_status = "waiting"
        elif active and active["state"] == "RUNNING" and resume_from != "":
            resume_status = "started_not_advanced"
        else:
            resume_status = "unconfirmed"

        latest_history = job_history[-1] if job_history else {}
        current_job_id = active["job_id"] if active else latest_history.get("job_id", "")
        elapsed = active["elapsed"] if active else latest_history.get("elapsed", "")
        start_time = active["start"] if active else latest_history.get("start", "")
        node_or_reason = (
            (active["node"] if active["state"] == "RUNNING" else active["reason"])
            if active
            else latest_history.get("node", "")
        )

        rows.append(
            {
                "stage": spec["stage"],
                "regime": spec["regime"],
                "algo": spec["algo"],
                "condition": spec["condition"],
                "status": status,
                "job_id": current_job_id,
                "current_job_elapsed": elapsed,
                "current_job_start": start_time,
                "node_or_reason": node_or_reason,
                "job_submissions": len(submission_ids),
                "slurm_restarts": restart_count,
                "resume_from_epoch": resume_from,
                "resume_status": resume_status,
                "current_epoch": current_epoch,
                "best_epoch": best_epoch,
                "best_validation_loss": best_val,
                "finished_log": finished,
                "run_dir": run_dir,
            }
        )

    fields = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)

    print(args.output)
    print("status", dict(Counter(row["status"] for row in rows)))
    print("resume", dict(Counter(row["resume_status"] for row in rows)))
    for row in rows:
        print(
            "|".join(
                str(row[key])
                for key in (
                    "stage",
                    "regime",
                    "algo",
                    "condition",
                    "status",
                    "current_job_elapsed",
                    "resume_from_epoch",
                    "resume_status",
                    "current_epoch",
                    "best_epoch",
                )
            )
        )


if __name__ == "__main__":
    main()
