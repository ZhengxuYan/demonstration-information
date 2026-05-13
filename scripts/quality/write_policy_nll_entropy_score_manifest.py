#!/usr/bin/env python3
"""Discover policy checkpoints and write scoring / report manifests."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


DATASETS = ("square_ph", "square_mh", "expert200_random_post")
POLICIES = ("gmm", "discrete", "discrete_smooth")
VIEWS = ("agent_wrist", "left_close_low_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments"))
    parser.add_argument("--score-root", type=Path, default=Path("/iris/u/jasonyan/data/robomimic_policy_scores/policy_nll_entropy"))
    parser.add_argument("--manifest", type=Path, required=True, help="Scoring manifest CSV to write.")
    parser.add_argument("--report-manifest", type=Path, required=True, help="Image/robot score-pair manifest CSV to write.")
    parser.add_argument("--ph-root", type=Path, default=Path("/iris/u/jasonyan/data/policy_view_experiments/square_ph"))
    parser.add_argument("--mh-root", type=Path, default=Path("/iris/u/jasonyan/data/policy_view_experiments/square_mh"))
    parser.add_argument("--expert-root", type=Path, default=Path("/iris/u/jasonyan/data/policy_view_experiments/expert200_random_post_bc"))
    parser.add_argument("--rollout-root", type=Path, default=Path("/iris/u/jasonyan/data/robomimic_rollout_scores"))
    parser.add_argument("--include-missing", action="store_true", help="Include missing rows in report comments on stdout.")
    return parser.parse_args()


def dataset_path(dataset: str, view: str, args: argparse.Namespace) -> Path:
    if dataset == "square_ph":
        return args.ph_root / f"square_ph_{view}_image.hdf5"
    if dataset == "square_mh":
        return args.mh_root / f"square_mh_{view}_image.hdf5"
    if dataset == "expert200_random_post":
        return args.expert_root / f"expert200_random_post_{view}_image_abs.hdf5"
    raise ValueError(dataset)


def policy_in_name(name: str, policy: str) -> bool:
    if policy == "discrete_smooth":
        return "discrete_smooth" in name
    if policy == "discrete":
        return "discrete" in name and "discrete_smooth" not in name
    return "gmm" in name


def discover_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.iterdir() if path.is_dir()]


def run_checkpoints(run_dir: Path) -> list[Path]:
    return [p for p in sorted(run_dir.glob("*/models/*.pth")) if p.name.endswith(".pth")]


def find_run(
    runs: list[Path],
    dataset: str,
    policy: str,
    view: str | None,
    state_only: bool,
) -> tuple[Path | None, bool]:
    candidates = []
    for path in runs:
        name = path.name
        if dataset not in name or not policy_in_name(name, policy):
            continue
        if state_only != ("state_only" in name):
            continue
        if view is not None and view not in name:
            continue
        candidates.append(path)
    if not candidates:
        return None, False
    candidates_with_ckpts = [path for path in candidates if run_checkpoints(path)]
    if not candidates_with_ckpts:
        candidates.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
        return candidates[0], False
    candidates = candidates_with_ckpts
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return candidates[0], True


def epoch_num(path: Path) -> int | None:
    match = re.search(r"model_epoch_(\d+)", path.name)
    return int(match.group(1)) if match else None


def best_success_checkpoint(run_dir: Path, rollout_root: Path) -> Path | None:
    if not rollout_root.exists():
        return None
    summaries = [p for p in rollout_root.rglob("*summary.csv") if run_dir.name in p.name or run_dir.name in str(p.parent)]
    best: tuple[float, Path] | None = None
    for summary in summaries:
        with summary.open(newline="") as f:
            for row in csv.DictReader(f):
                if not row or row.get("Epoch") in (None, "Epoch"):
                    continue
                try:
                    success = float(row.get("Success_Rate", "nan"))
                except ValueError:
                    continue
                checkpoint = row.get("Checkpoint")
                if checkpoint:
                    ckpt = Path(checkpoint)
                else:
                    epoch = int(row["Epoch"])
                    matches = list(run_dir.glob(f"*/models/model_epoch_{epoch}.pth"))
                    ckpt = matches[0] if matches else Path()
                if ckpt and ckpt.exists() and (best is None or success > best[0]):
                    best = (success, ckpt)
    return None if best is None else best[1]


def select_checkpoints(run_dir: Path, rollout_root: Path) -> dict[str, Path]:
    checkpoints = run_checkpoints(run_dir)
    if not checkpoints:
        return {}
    by_label: dict[str, Path] = {}
    for ckpt in checkpoints:
        name = ckpt.name
        if "best_validation" in name or "valid_best" in name:
            by_label.setdefault("best_validation", ckpt)
    epoch_ckpts = [(epoch_num(p), p) for p in checkpoints]
    epoch_ckpts = [(e, p) for e, p in epoch_ckpts if e is not None]
    epoch_ckpts.sort(key=lambda item: item[0])
    if epoch_ckpts:
        by_label["final"] = epoch_ckpts[-1][1]
        indices = sorted(set(round((len(epoch_ckpts) - 1) * q) for q in (0.25, 0.5, 0.75)))
        for q_label, idx in zip(("quartile_25", "quartile_50", "quartile_75"), indices):
            by_label[q_label] = epoch_ckpts[idx][1]
    elif "last.pth" in {p.name for p in checkpoints}:
        by_label["final"] = next(p for p in checkpoints if p.name == "last.pth")
    best_success = best_success_checkpoint(run_dir, rollout_root)
    if best_success is not None:
        by_label["best_success"] = best_success
    return by_label


def score_name(dataset: str, policy: str, view: str, baseline: str, checkpoint_label: str) -> str:
    return f"{dataset}_{policy}_{view}_{baseline}_{checkpoint_label}"


def score_output_dir(score_root: Path, dataset: str, policy: str, view: str) -> Path:
    return score_root / dataset / policy / view


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.output_root)
    score_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, str]] = []
    missing: dict[str, list[str]] = defaultdict(list)

    for dataset in DATASETS:
        for policy in POLICIES:
            robot_run, robot_has_ckpts = find_run(runs, dataset, policy, view=None, state_only=True)
            if robot_run is None:
                missing[f"{dataset}/{policy}"].append("state_only")
                continue
            if not robot_has_ckpts:
                missing[f"{dataset}/{policy}"].append(f"state_only_no_checkpoints:{robot_run.name}")
                continue
            robot_ckpts = select_checkpoints(robot_run, args.rollout_root)
            for view in VIEWS:
                image_run, image_has_ckpts = find_run(runs, dataset, policy, view=view, state_only=False)
                if image_run is None:
                    missing[f"{dataset}/{policy}/{view}"].append("image_run")
                    continue
                if not image_has_ckpts:
                    missing[f"{dataset}/{policy}/{view}"].append(f"image_run_no_checkpoints:{image_run.name}")
                    continue
                image_ckpts = select_checkpoints(image_run, args.rollout_root)
                labels = sorted(set(image_ckpts) & set(robot_ckpts))
                if "best_success" in image_ckpts and "best_success" not in labels:
                    fallback_robot = robot_ckpts.get("final") or robot_ckpts.get("best_validation")
                    if fallback_robot is not None:
                        robot_ckpts["best_success"] = fallback_robot
                        labels.append("best_success")
                        labels = sorted(set(labels))
                if not labels:
                    missing[f"{dataset}/{policy}/{view}"].append("matching_checkpoints")
                    continue
                for checkpoint_label in labels:
                    data_path = dataset_path(dataset, view, args)
                    out_dir = score_output_dir(args.score_root, dataset, policy, view)
                    image_name = score_name(dataset, policy, view, "image_robot", checkpoint_label)
                    robot_name = score_name(dataset, policy, view, "robot", checkpoint_label)
                    for baseline, ckpt, name in (
                        ("image_robot", image_ckpts[checkpoint_label], image_name),
                        ("robot", robot_ckpts[checkpoint_label], robot_name),
                    ):
                        score_rows.append(
                            {
                                "dataset": dataset,
                                "policy": policy,
                                "view": view,
                                "baseline": baseline,
                                "checkpoint_label": checkpoint_label,
                                "checkpoint": str(ckpt),
                                "dataset_path": str(data_path),
                                "output": str(out_dir),
                                "name": name,
                            }
                        )
                    report_rows.append(
                        {
                            "dataset": dataset,
                            "policy": policy,
                            "view": view,
                            "checkpoint_label": checkpoint_label,
                            "image_score": str(out_dir / f"{image_name}.pkl"),
                            "robot_score": str(out_dir / f"{robot_name}.pkl"),
                        }
                    )

    write_csv(
        args.manifest,
        score_rows,
        ["dataset", "policy", "view", "baseline", "checkpoint_label", "checkpoint", "dataset_path", "output", "name"],
    )
    write_csv(
        args.report_manifest,
        report_rows,
        ["dataset", "policy", "view", "checkpoint_label", "image_score", "robot_score"],
    )
    print(f"wrote {args.manifest} ({len(score_rows)} score rows)")
    print(f"wrote {args.report_manifest} ({len(report_rows)} report rows)")
    if args.include_missing:
        for key, values in sorted(missing.items()):
            print(f"missing {key}: {', '.join(values)}")


if __name__ == "__main__":
    main()
