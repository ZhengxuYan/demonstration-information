#!/usr/bin/env python3
"""Plot empirical action distributions against action-prior model samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "robomimic"))
sys.path.insert(0, str(REPO / "scripts" / "quality"))

from score_robomimic_policy_nll import (  # noqa: E402
    load_algo,
    make_loader,
    policy_distribution_for_batch,
)
import robomimic.utils.tensor_utils as TensorUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


DIM_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


@dataclass(frozen=True)
class Task:
    dataset: str
    regime: str
    algo: str
    train_filter: str
    valid_filter: str
    score_filter: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, default=Path("/iris/u/jasonyan/data/wrench_to_hook_density_queue_state.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/iris/u/jasonyan/data/wrench_to_hook_density_datasets"))
    parser.add_argument("--out-root", type=Path, default=Path("/iris/u/jasonyan/data/robomimic_outputs/wrench_to_hook_density"))
    parser.add_argument("--output-dir", type=Path, default=Path("/iris/u/jasonyan/data/wrench_to_hook_action_prior_fit_report"))
    parser.add_argument("--task-prefix", default="wrench_to_hook")
    parser.add_argument("--action-target", default="single")
    parser.add_argument("--action-source", default="cartesian_velocity")
    parser.add_argument("--action-normalization", default="none")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def dataset_path(args: argparse.Namespace, dataset: str) -> Path:
    recipe = f"{args.action_target}_{args.action_source}_{args.action_normalization}"
    return args.dataset_root / f"{args.task_prefix}_{dataset}_{recipe}.hdf5"


def run_name(args: argparse.Namespace, task: Task) -> str:
    middle = f"{args.action_target}_{args.action_source}_{args.action_normalization}"
    if task.regime != "normal":
        middle = f"{middle}_{task.regime}"
    return f"{args.task_prefix}_{task.dataset}_{middle}_{task.algo}_action_prior_seed1"


def best_checkpoint_for_run(run_dir: Path) -> Path:
    ckpts = list(run_dir.glob("*/models/*best_validation*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No best-validation checkpoint under {run_dir}")

    def key(path: Path) -> tuple[float, int]:
        loss_match = re.search(r"best_validation_([-+0-9.eE]+)\.pth$", path.name)
        epoch_match = re.search(r"epoch_(\d+)", path.name)
        loss = float(loss_match.group(1)) if loss_match else math.inf
        epoch = int(epoch_match.group(1)) if epoch_match else -1
        return loss, -epoch

    return min(ckpts, key=key)


def last_checkpoint_for_run(run_dir: Path) -> Path:
    time_dirs = [path for path in run_dir.iterdir() if path.is_dir()]
    if not time_dirs:
        raise FileNotFoundError(f"No timestamped run directory under {run_dir}")
    latest_time_dir = sorted(time_dirs)[-1]
    last = latest_time_dir / "last.pth"
    if last.is_file():
        return last
    ckpts = list((latest_time_dir / "models").glob("model_epoch_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No last or epoch checkpoint under {latest_time_dir}")

    def epoch(path: Path) -> int:
        match = re.search(r"epoch_(\d+)", path.name)
        return int(match.group(1)) if match else -1

    return max(ckpts, key=epoch)


def load_tasks(state_file: Path) -> list[Task]:
    raw = json.loads(state_file.read_text())
    tasks = []
    for value in raw["tasks"].values():
        if value["condition"] != "action_prior":
            continue
        if value["train_state"] != "COMPLETED":
            continue
        tasks.append(
            Task(
                dataset=value["dataset"],
                regime=value["regime"],
                algo=value["algo"],
                train_filter=value["train_filter"],
                valid_filter=value["valid_filter"],
                score_filter=value["score_filter"],
            )
        )
    return sorted(tasks, key=lambda t: (t.dataset, t.regime, t.algo))


def read_actions(path: Path, filter_key: str, max_samples: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    arrays = []
    with h5py.File(path, "r") as f:
        demos = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["mask"][filter_key][:]]
        for demo in demos:
            arrays.append(np.asarray(f["data"][demo]["actions"], dtype=np.float32))
    actions = np.concatenate(arrays, axis=0)
    if len(actions) > max_samples:
        idx = rng.choice(len(actions), size=max_samples, replace=False)
        actions = actions[np.sort(idx)]
    return actions


def sample_model(checkpoint: Path, dataset: Path, filter_key: str, max_samples: int, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    torch_device = torch.device(device) if device != "auto" else TorchUtils.get_torch_device(try_to_use_cuda=True)
    algo, config = load_algo(checkpoint, dataset, torch_device)
    dataset_obj, loader = make_loader(config, batch_size=batch_size, num_workers=0, filter_key=filter_key)
    samples = []
    actions = []
    with torch.no_grad():
        for batch in loader:
            input_batch = algo.process_batch_for_training(batch)
            input_batch = algo.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)
            dist = policy_distribution_for_batch(algo, input_batch)
            sample = dist.sample()
            samples.append(TensorUtils.to_numpy(sample).astype(np.float32))
            actions.append(TensorUtils.to_numpy(input_batch["actions"]).astype(np.float32))
            if sum(len(x) for x in samples) >= max_samples:
                break
    del dataset_obj
    return np.concatenate(samples, axis=0)[:max_samples], np.concatenate(actions, axis=0)[:max_samples]


def js_distance(a: np.ndarray, b: np.ndarray, bins: np.ndarray) -> float:
    pa, _ = np.histogram(a, bins=bins)
    pb, _ = np.histogram(b, bins=bins)
    pa = pa.astype(np.float64) + 1e-12
    pb = pb.astype(np.float64) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    return float(0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m)))


def plot_task(
    task: Task,
    train_actions: np.ndarray,
    eval_actions: np.ndarray,
    best_samples: np.ndarray,
    last_samples: np.ndarray,
    output: Path,
    bins_count: int,
) -> list[dict[str, object]]:
    rows = []
    fig, axes = plt.subplots(7, 1, figsize=(13, 17), sharex=False)
    fig.suptitle(f"{task.dataset} {task.regime} {task.algo} action_prior: empirical actions vs model samples", fontsize=16)
    for dim, ax in enumerate(axes):
        combined = np.concatenate([train_actions[:, dim], eval_actions[:, dim], best_samples[:, dim], last_samples[:, dim]])
        lo = float(np.percentile(combined, 0.2))
        hi = float(np.percentile(combined, 99.8))
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        bins = np.linspace(lo, hi, bins_count + 1)
        ax.hist(train_actions[:, dim], bins=bins, density=True, alpha=0.35, label=f"train {task.train_filter}", color="#4C78A8")
        eval_label = task.score_filter or task.valid_filter
        ax.hist(eval_actions[:, dim], bins=bins, density=True, alpha=0.35, label=f"heldout {eval_label}", color="#F58518")
        ax.hist(best_samples[:, dim], bins=bins, density=True, histtype="step", linewidth=2.0, label="best-val samples", color="#54A24B")
        ax.hist(last_samples[:, dim], bins=bins, density=True, histtype="step", linewidth=2.0, label="last samples", color="#B279A2")
        ax.set_title(f"dim {dim}: {DIM_NAMES[dim]}")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        rows.append(
            {
                "dataset": task.dataset,
                "regime": task.regime,
                "algo": task.algo,
                "dim": dim,
                "name": DIM_NAMES[dim],
                "train_mean": float(train_actions[:, dim].mean()),
                "heldout_mean": float(eval_actions[:, dim].mean()),
                "best_mean": float(best_samples[:, dim].mean()),
                "last_mean": float(last_samples[:, dim].mean()),
                "train_std": float(train_actions[:, dim].std()),
                "heldout_std": float(eval_actions[:, dim].std()),
                "best_std": float(best_samples[:, dim].std()),
                "last_std": float(last_samples[:, dim].std()),
                "js_best_vs_train": js_distance(best_samples[:, dim], train_actions[:, dim], bins),
                "js_best_vs_heldout": js_distance(best_samples[:, dim], eval_actions[:, dim], bins),
                "js_last_vs_train": js_distance(last_samples[:, dim], train_actions[:, dim], bins),
                "js_last_vs_heldout": js_distance(last_samples[:, dim], eval_actions[:, dim], bins),
            }
        )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return rows


def write_index(output_dir: Path, image_names: list[str], rows: list[dict[str, object]]) -> None:
    by_task = {}
    for row in rows:
        key = (row["dataset"], row["regime"], row["algo"])
        by_task.setdefault(key, []).append(row)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Action Prior Fit Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:28px;color:#17201a}img{max-width:100%;border:1px solid #ddd}table{border-collapse:collapse;margin:16px 0 32px;width:100%}td,th{border-bottom:1px solid #ddd;padding:6px 8px;text-align:right}td:first-child,th:first-child{text-align:left}h2{margin-top:36px}</style>",
        "</head><body><h1>Action Prior Fit Report</h1>",
        "<p>Histograms compare empirical train actions, heldout/score actions, samples from the best-validation checkpoint, and samples from the last checkpoint. Lower JS means closer marginal fit.</p>",
    ]
    for name in image_names:
        stem = Path(name).stem
        parts.append(f"<h2>{stem}</h2><img src='{name}'>")
    parts.append("<h2>Per-dimension JS Summary</h2>")
    for key, task_rows in by_task.items():
        parts.append(f"<h3>{key[0]} {key[1]} {key[2]}</h3><table><tr><th>dim</th><th>name</th><th>JS best/train</th><th>JS best/heldout</th><th>JS last/train</th><th>JS last/heldout</th><th>train mean</th><th>best mean</th><th>last mean</th><th>train std</th><th>best std</th><th>last std</th></tr>")
        for row in task_rows:
            parts.append(
                "<tr><td>{dim}</td><td>{name}</td><td>{js_best_vs_train:.4f}</td><td>{js_best_vs_heldout:.4f}</td><td>{js_last_vs_train:.4f}</td><td>{js_last_vs_heldout:.4f}</td><td>{train_mean:.4f}</td><td>{best_mean:.4f}</td><td>{last_mean:.4f}</td><td>{train_std:.4f}</td><td>{best_std:.4f}</td><td>{last_std:.4f}</td></tr>".format(**row)
            )
        parts.append("</table>")
    parts.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(parts))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.state_file)
    all_rows = []
    image_names = []
    for task in tasks:
        data_path = dataset_path(args, task.dataset)
        run_dir = args.out_root / run_name(args, task)
        best_ckpt = best_checkpoint_for_run(run_dir)
        last_ckpt = last_checkpoint_for_run(run_dir)
        heldout_filter = task.score_filter or task.valid_filter
        print(f"{task.dataset} {task.regime} {task.algo}: best={best_ckpt} last={last_ckpt} heldout={heldout_filter}", flush=True)
        train_actions = read_actions(data_path, task.train_filter, args.max_samples)
        heldout_actions = read_actions(data_path, heldout_filter, args.max_samples)
        best_samples, _ = sample_model(best_ckpt, data_path, task.train_filter, args.max_samples, args.batch_size, args.device)
        last_samples, _ = sample_model(last_ckpt, data_path, task.train_filter, args.max_samples, args.batch_size, args.device)
        image_name = f"{task.dataset}_{task.regime}_{task.algo}_action_prior_fit.png"
        rows = plot_task(task, train_actions, heldout_actions, best_samples, last_samples, args.output_dir / image_name, args.bins)
        image_names.append(image_name)
        all_rows.extend(rows)
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    write_index(args.output_dir, image_names, all_rows)
    print(f"wrote {args.output_dir / 'index.html'}")
    print(f"wrote {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
