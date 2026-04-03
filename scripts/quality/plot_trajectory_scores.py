"""
Plot sample-level score traces for selected trajectories from a quality-estimation pickle.

Example:

python scripts/quality/plot_trajectory_scores.py \
    --scores /path/to/square_mh.pkl \
    --ep_idx 20 41 247 \
    --output_dir /tmp/trajectory_score_plots
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="Path to a quality-estimation pickle file.")
    parser.add_argument("--ep_idx", type=int, nargs="+", required=True, help="Episode ids to visualize.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save plots.")
    return parser.parse_args()


def load_scores(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    required = {"sample_score", "sample_ep_idx", "sample_step_idx"}
    missing = required - set(data)
    if missing:
        raise ValueError(
            f"{path} is missing {sorted(missing)}. Re-run quality estimation after updating the scoring script."
        )
    return data


def summarize_episode(data: dict, ep_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    mask = data["sample_ep_idx"] == ep_idx
    if not np.any(mask):
        raise ValueError(f"Episode {ep_idx} not found in sample-level score data.")

    step_idx = np.asarray(data["sample_step_idx"][mask])
    sample_score = np.asarray(data["sample_score"][mask])
    order = np.argsort(step_idx)
    step_idx = step_idx[order]
    sample_score = sample_score[order]

    unique_steps = np.unique(step_idx)
    step_mean = np.array([sample_score[step_idx == step].mean() for step in unique_steps])
    step_std = np.array([sample_score[step_idx == step].std() for step in unique_steps])

    human_label = None
    if "quality_by_ep_idx" in data:
        human_label = data["quality_by_ep_idx"].get(ep_idx)
    return unique_steps, step_mean, step_std, human_label


def plot_episode(data: dict, ep_idx: int, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    steps, step_mean, step_std, human_label = summarize_episode(data, ep_idx)
    trajectory_score = data["ep_idx"].get(ep_idx) if "ep_idx" in data else None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, step_mean, color="#0f5b5c", linewidth=2, label="mean sample score")
    if np.any(step_std > 0):
        ax.fill_between(steps, step_mean - step_std, step_mean + step_std, color="#0f5b5c", alpha=0.18, label="std")
    ax.set_xlabel("step index")
    ax.set_ylabel("score")
    title = f"demo_{ep_idx}"
    if trajectory_score is not None:
        title += f" | traj score={trajectory_score:.4f}"
    if human_label is not None:
        title += f" | label={human_label:.1f}"
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"demo_{ep_idx:04d}_score_trace.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    data = load_scores(args.scores)
    for ep_idx in args.ep_idx:
        out_path = plot_episode(data, ep_idx, args.output_dir)
        print(out_path)


if __name__ == "__main__":
    main()
