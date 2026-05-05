#!/usr/bin/env python3
"""Create a symlinked `/scr/jasonyan` index for policy-view artifacts.

Default mode is dry-run. Pass --apply on iris10 to create symlinks and manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Artifact:
    artifact_type: str
    original_path: str
    symlink_path: str
    run_name: str
    view: str
    algo: str
    label: str
    size_bytes: int | None
    mtime: float | None
    checksum_sha256: str | None = None


VIEWS = ("agent_wrist", "left_close_low_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, default=Path("/scr/jasonyan/policy_view_artifacts"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/iris/u/jasonyan/data/policy_view_experiments/expert200_random_post_bc"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/iris/u/jasonyan/data/robomimic_outputs/policy_view_experiments"))
    parser.add_argument("--score-root", type=Path, default=Path("/iris/u/jasonyan/data/robomimic_policy_scores/expert200_random_post_bc"))
    parser.add_argument("--knn-root", type=Path, default=Path("/iris/u/jasonyan/data/knn_entropy/expert200_random_post"))
    parser.add_argument("--apply", action="store_true", help="Create symlinks and manifests. Default only prints a dry-run.")
    parser.add_argument("--checksum", action="store_true", help="Compute sha256 for regular files. Can be slow.")
    return parser.parse_args()


def stat_info(path: Path) -> tuple[int | None, float | None]:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None, None
    return (int(st.st_size) if path.is_file() else None, float(st.st_mtime))


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_view(name: str) -> str:
    for view in VIEWS:
        if view in name:
            return view
    return "unknown"


def infer_algo(name: str) -> str:
    if "discrete_smooth" in name or "smooth_discrete" in name:
        return "discrete_smooth"
    if "discrete" in name:
        return "discrete"
    if "gmm" in name:
        return "gmm"
    if "qwen" in name:
        return "qwen3_vl"
    if "sa_vae" in name:
        return "sa_vae"
    if "bc_latent" in name:
        return "bc_latent"
    return "unknown"


def safe_rel_name(path: Path) -> str:
    return path.name.replace("/", "_")


def add_artifact(rows: list[Artifact], args, artifact_type: str, original: Path, symlink: Path, run_name: str, label: str) -> None:
    size, mtime = stat_info(original)
    rows.append(
        Artifact(
            artifact_type=artifact_type,
            original_path=str(original),
            symlink_path=str(symlink),
            run_name=run_name,
            view=infer_view(run_name + " " + original.name),
            algo=infer_algo(run_name + " " + original.name),
            label=label,
            size_bytes=size,
            mtime=mtime,
            checksum_sha256=sha256(original) if args.checksum else None,
        )
    )


def collect(args) -> list[Artifact]:
    rows: list[Artifact] = []
    target = args.target_root

    for path in sorted(args.dataset_root.glob("expert200_random_post_*_image_abs.hdf5")):
        add_artifact(
            rows,
            args,
            "dataset",
            path,
            target / "datasets" / "expert200_random_post" / path.name,
            path.stem,
            "dataset",
        )

    for run_dir in sorted(args.checkpoint_root.glob("expert200_random_post_bc_*")):
        if not run_dir.is_dir():
            continue
        algo = infer_algo(run_dir.name)
        if algo not in {"gmm", "discrete", "discrete_smooth"}:
            continue
        for ckpt in sorted(run_dir.glob("*/models/*.pth")):
            label = ckpt.stem
            add_artifact(
                rows,
                args,
                "checkpoint",
                ckpt,
                target / "checkpoints" / "expert200_random_post" / algo / run_dir.name / ckpt.name,
                run_dir.name,
                label,
            )

    for score_file in sorted(args.score_root.glob("expert200_random_post_bc_*/*")):
        if not score_file.is_file() or score_file.suffix not in {".pkl", ".csv"}:
            continue
        run_name = score_file.parent.name
        add_artifact(
            rows,
            args,
            "score",
            score_file,
            target / "scores" / "expert200_random_post" / run_name / score_file.name,
            run_name,
            score_file.stem,
        )

    knn_specs = [
        ("bc_latent", args.knn_root.glob("*_bc_latent")),
        ("sa_vae", args.knn_root.glob("*sa_vae*")),
        ("qwen3_vl", (args.knn_root / "qwen3_vl").glob("*") if (args.knn_root / "qwen3_vl").exists() else []),
    ]
    for algo, iterator in knn_specs:
        for path in sorted(iterator):
            if not path.is_dir():
                continue
            add_artifact(
                rows,
                args,
                "knn",
                path,
                target / "knn" / "expert200_random_post" / algo / path.name,
                path.name,
                "review_dir",
            )

    rows.sort(key=lambda row: (row.artifact_type, row.algo, row.view, row.run_name, row.label))
    return rows


def create_symlinks(rows: list[Artifact]) -> None:
    for row in rows:
        src = Path(row.original_path)
        dst = Path(row.symlink_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            if os.readlink(dst) == str(src):
                continue
            raise FileExistsError(f"{dst} already points to {os.readlink(dst)}, not {src}")
        if dst.exists():
            raise FileExistsError(f"{dst} exists and is not a symlink")
        os.symlink(src, dst)


def write_manifests(rows: list[Artifact], target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    dicts = [asdict(row) for row in rows]
    csv_path = target_root / "manifest.csv"
    json_path = target_root / "manifest.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dicts[0].keys()) if dicts else list(Artifact.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(dicts)
    json_path.write_text(json.dumps(dicts, indent=2) + "\n")
    print(csv_path)
    print(json_path)


def main() -> None:
    args = parse_args()
    rows = collect(args)
    print(f"planned artifacts: {len(rows)}")
    for row in rows[:25]:
        print(f"{row.artifact_type:10s} {row.original_path} -> {row.symlink_path}")
    if len(rows) > 25:
        print(f"... {len(rows) - 25} more")
    if args.apply:
        create_symlinks(rows)
        write_manifests(rows, args.target_root)
    else:
        print("dry-run only; pass --apply to create symlinks and manifests")


if __name__ == "__main__":
    main()
