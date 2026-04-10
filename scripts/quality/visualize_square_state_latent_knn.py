"""
Visualize k-NN neighborhoods in the Square observation-latent space for selected frames.

This script loads a trained Square wrist observation VAE, encodes manually chosen
query frames plus a reference pool from RoboMimic `image.hdf5`, computes
k-nearest neighbors in latent space, and writes static figures plus CSV metadata.

Example:

python scripts/quality/visualize_square_state_latent_knn.py \
  --obs_ckpt /iris/u/jasonyan/data/deminf_outputs/robomimic_image/square_mh_wrist_obs_vae_seed1 \
  --reference_hdf5 /iris/u/jasonyan/data/robomimic/square/image.hdf5 \
  --query_hdf5 /iris/u/jasonyan/data/fb_demos/forward_grab_image.hdf5 \
  --query 20=0,60,120 \
  --query 41=30,90 \
  --k 8 \
  --camera wrist \
  --output_dir /tmp/square_state_latent_knn
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPLCONFIGDIR = os.path.join(tempfile.gettempdir(), "matplotlib-codex")
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from openx.data.utils import NormalizationType
from openx.utils.evaluate import load_checkpoint


CAMERA_KEY_BY_NAME = {
    "wrist": "robot0_eye_in_hand_image",
    "agent": "agentview_image",
}


@dataclass(frozen=True)
class QueryFrame:
    demo: int
    frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs_ckpt", required=True, help="Path to the trained Square wrist observation VAE.")
    parser.add_argument(
        "--reference_hdf5",
        type=Path,
        required=True,
        help="Path to the reference RoboMimic image.hdf5 file used for k-NN and PCA.",
    )
    parser.add_argument(
        "--query_hdf5",
        type=Path,
        default=None,
        help="Optional query RoboMimic image.hdf5 file. Defaults to --reference_hdf5.",
    )
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Manual query specification: demo=frame[,frame...]. Repeat for multiple demos.",
    )
    parser.add_argument("--k", type=int, default=8, help="Number of neighbors to return per query.")
    parser.add_argument(
        "--max_one_per_demo",
        action="store_true",
        help="If set, keep at most one neighbor from each demo in the returned top-k list.",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for figures and CSV output.")
    parser.add_argument(
        "--camera",
        choices=sorted(CAMERA_KEY_BY_NAME),
        default="wrist",
        help="Camera stream used for thumbnails in the output figures. Latent encoding always uses wrist images.",
    )
    parser.add_argument(
        "--max_reference_frames",
        type=int,
        default=None,
        help="Optional maximum number of reference frames used for k-NN and PCA.",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for latent encoding.")
    return parser.parse_args()


def parse_queries(specs: list[str]) -> list[QueryFrame]:
    queries: list[QueryFrame] = []
    for spec in specs:
        demo_part, frame_part = spec.split("=", 1)
        demo = int(demo_part)
        frames = [int(value) for value in frame_part.split(",") if value.strip()]
        if not frames:
            raise ValueError(f"Query spec {spec!r} does not contain any frame indices.")
        queries.extend(QueryFrame(demo=demo, frame=frame) for frame in frames)
    return queries


def _normalize(x, mode: str, mean, std, low, high):
    if mode == NormalizationType.NONE:
        return x
    if mode == NormalizationType.GAUSSIAN:
        return np.divide(x - mean, std, out=np.zeros_like(x), where=std != 0)
    if mode == NormalizationType.BOUNDS:
        x = np.clip(x, low, high)
        denom = high - low
        return 2 * np.divide(x - low, denom, out=np.zeros_like(x), where=denom != 0) - 1
    if mode == NormalizationType.BOUNDS_5STDV:
        low = np.maximum(low, mean - 5 * std)
        high = np.minimum(high, mean + 5 * std)
        x = np.clip(x, low, high)
        denom = high - low
        return 2 * np.divide(x - low, denom, out=np.zeros_like(x), where=denom != 0) - 1
    raise ValueError(f"Invalid normalization mode: {mode}")


def normalize_tree(tree: dict, structure: dict, stats: dict) -> dict:
    def _child_stats(current_stats: dict, current_key: str) -> dict:
        if all(name in current_stats for name in ("mean", "std", "min", "max")):
            return {name: current_stats[name][current_key] for name in ("mean", "std", "min", "max")}
        if current_key in current_stats:
            return current_stats[current_key]
        raise KeyError(f"Could not find stats for key {current_key}")

    def _leaf_stat(current_stats: dict, stat_name: str, current_key: str):
        if stat_name in current_stats:
            return current_stats[stat_name][current_key]
        if current_key in current_stats and stat_name in current_stats[current_key]:
            return current_stats[current_key][stat_name]
        raise KeyError(f"Could not find {stat_name} stats for key {current_key}")

    out = {}
    for key, substructure in structure.items():
        if isinstance(substructure, dict):
            out[key] = normalize_tree(tree[key], substructure, _child_stats(stats, key))
        else:
            out[key] = _normalize(
                np.asarray(tree[key], dtype=np.float32),
                str(substructure),
                np.asarray(_leaf_stat(stats, "mean", key), dtype=np.float32),
                np.asarray(_leaf_stat(stats, "std", key), dtype=np.float32),
                np.asarray(_leaf_stat(stats, "min", key), dtype=np.float32),
                np.asarray(_leaf_stat(stats, "max", key), dtype=np.float32),
            )
    return out


def stats_subtree(stats: dict, key: str) -> dict:
    return {name: value[key] for name, value in stats.items() if name in ("mean", "std", "min", "max")}


def concatenate_ordered(tree: dict) -> np.ndarray:
    flat = []
    for value in tree.values():
        if isinstance(value, dict):
            flat.append(concatenate_ordered(value))
        else:
            flat.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(flat, axis=-1)


def load_hdf5_observations(
    hdf5_path: Path,
    obs_structure: dict,
    dataset_statistics: dict,
    max_frames: int | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    ref_state = []
    ref_wrist = []
    ref_demo_idx = []
    ref_frame_idx = []

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda name: int(name.split("_")[-1]))
        for demo_name in demos:
            demo_idx = int(demo_name.split("_")[-1])
            obs_grp = f["data"][demo_name]["obs"]
            raw_state = {
                "EE_POS": obs_grp["robot0_eef_pos"][:].astype(np.float32),
                "EE_QUAT": obs_grp["robot0_eef_quat"][:].astype(np.float32),
                "GRIPPER": obs_grp["robot0_gripper_qpos"][:, :1].astype(np.float32),
            }
            normalized_state = normalize_tree(raw_state, obs_structure["state"], stats_subtree(dataset_statistics, "state"))
            state_array = concatenate_ordered(normalized_state)
            wrist_array = obs_grp["robot0_eye_in_hand_image"][:].astype(np.float32) / 255.0

            for frame_idx in range(state_array.shape[0]):
                ref_state.append(state_array[frame_idx])
                ref_wrist.append(wrist_array[frame_idx])
                ref_demo_idx.append(demo_idx)
                ref_frame_idx.append(frame_idx)

    ref_observation = {
        "state": np.asarray(ref_state, dtype=np.float32),
        "image": {"wrist": np.asarray(ref_wrist, dtype=np.float32)},
    }
    ref_demo_idx = np.asarray(ref_demo_idx, dtype=np.int32)
    ref_frame_idx = np.asarray(ref_frame_idx, dtype=np.int32)

    if max_frames is not None and max_frames < ref_observation["state"].shape[0]:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(ref_observation["state"].shape[0], size=max_frames, replace=False))
        ref_observation = {
            "state": ref_observation["state"][keep],
            "image": {"wrist": ref_observation["image"]["wrist"][keep]},
        }
        ref_demo_idx = ref_demo_idx[keep]
        ref_frame_idx = ref_frame_idx[keep]

    return ref_observation, ref_demo_idx, ref_frame_idx


def encode_observations(obs_alg, obs_state, observation: dict[str, object], batch_size: int) -> np.ndarray:
    predict = jax.jit(lambda batch, rng: obs_alg.predict(obs_state, batch, rng))
    outputs = []
    base_rng = jax.random.key(0)
    total = observation["state"].shape[0]
    for batch_idx, start in enumerate(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        batch = {
            "observation": {
                "state": jax.device_put(observation["state"][start:end, None, :]),
                "image": {"wrist": jax.device_put(observation["image"]["wrist"][start:end, None, ...])},
            }
        }
        rng = jax.random.fold_in(base_rng, batch_idx)
        outputs.append(np.asarray(predict(batch, rng)))
    return np.concatenate(outputs, axis=0)


def fit_pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    centered = x - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return mean, components


def project_pca(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (x - mean) @ components


def load_frame_image(hdf5_path: Path, demo_idx: int, frame_idx: int, camera: str) -> np.ndarray:
    camera_key = CAMERA_KEY_BY_NAME[camera]
    with h5py.File(hdf5_path, "r") as f:
        frame = f["data"][f"demo_{demo_idx}"]["obs"][camera_key][frame_idx]
    return np.asarray(frame, dtype=np.uint8)


def load_query_observations(
    hdf5_path: Path,
    obs_structure: dict,
    dataset_statistics: dict,
    queries: list[QueryFrame],
) -> dict[str, object]:
    query_state = []
    query_wrist = []
    with h5py.File(hdf5_path, "r") as f:
        for query in queries:
            demo_key = f"demo_{query.demo}"
            if demo_key not in f["data"]:
                raise ValueError(f"Query demo={query.demo} does not exist in {hdf5_path}")
            obs_grp = f["data"][demo_key]["obs"]
            num_frames = obs_grp["robot0_eef_pos"].shape[0]
            if query.frame < 0 or query.frame >= num_frames:
                raise ValueError(
                    f"Query demo={query.demo}, frame={query.frame} is out of range for {hdf5_path} "
                    f"(num_frames={num_frames})"
                )
            raw_state = {
                "EE_POS": obs_grp["robot0_eef_pos"][query.frame : query.frame + 1].astype(np.float32),
                "EE_QUAT": obs_grp["robot0_eef_quat"][query.frame : query.frame + 1].astype(np.float32),
                "GRIPPER": obs_grp["robot0_gripper_qpos"][query.frame : query.frame + 1, :1].astype(np.float32),
            }
            normalized_state = normalize_tree(raw_state, obs_structure["state"], stats_subtree(dataset_statistics, "state"))
            query_state.append(concatenate_ordered(normalized_state)[0])
            query_wrist.append(obs_grp["robot0_eye_in_hand_image"][query.frame].astype(np.float32) / 255.0)
    return {
        "state": np.asarray(query_state, dtype=np.float32),
        "image": {"wrist": np.asarray(query_wrist, dtype=np.float32)},
    }


def find_neighbors(
    query: QueryFrame,
    query_latent: np.ndarray,
    ref_latents: np.ndarray,
    ref_demo_idx: np.ndarray,
    ref_frame_idx: np.ndarray,
    k: int,
    exclude_same_demo: bool,
    max_one_per_demo: bool,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(ref_latents - query_latent[None, :], axis=1)
    distances = distances.copy()
    if exclude_same_demo:
        distances[ref_demo_idx == query.demo] = np.inf
    order = np.argsort(distances)
    valid = order[np.isfinite(distances[order])]
    if max_one_per_demo:
        keep_list = []
        seen_demos: set[int] = set()
        for idx in valid.tolist():
            demo_idx = int(ref_demo_idx[idx])
            if demo_idx in seen_demos:
                continue
            keep_list.append(idx)
            seen_demos.add(demo_idx)
            if len(keep_list) == k:
                break
        if len(keep_list) < k:
            raise ValueError(
                f"Requested k={k} with --max_one_per_demo, but only {len(keep_list)} unique-demo neighbors are available."
            )
        keep = np.asarray(keep_list, dtype=np.int32)
    else:
        if valid.shape[0] < k:
            raise ValueError(f"Requested k={k}, but only {valid.shape[0]} valid neighbors are available.")
        keep = valid[:k]
    return keep, distances[keep]


def plot_query_figure(
    out_path: Path,
    query: QueryFrame,
    query_projection: np.ndarray,
    ref_projection: np.ndarray,
    ref_demo_idx: np.ndarray,
    ref_frame_idx: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    query_hdf5_path: Path,
    reference_hdf5_path: Path,
    camera: str,
) -> None:
    cols = max(2, math.ceil(math.sqrt(len(neighbor_indices))))
    neighbor_rows = math.ceil(len(neighbor_indices) / cols)

    fig = plt.figure(figsize=(4.8 + 2.6 * cols, 5.8 + 2.0 * neighbor_rows))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.6], wspace=0.25)

    ax_scatter = fig.add_subplot(outer[0, 0])
    ax_scatter.scatter(ref_projection[:, 0], ref_projection[:, 1], s=6, alpha=0.12, color="#6f6a63", rasterized=True)
    ax_scatter.scatter(
        ref_projection[neighbor_indices, 0],
        ref_projection[neighbor_indices, 1],
        s=42,
        color="#c96c33",
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
        label="neighbors",
    )
    ax_scatter.scatter(
        [query_projection[0]],
        [query_projection[1]],
        s=90,
        color="#b9472c",
        marker="x",
        linewidths=2.0,
        zorder=4,
        label="query",
    )
    for rank, ref_idx in enumerate(neighbor_indices, start=1):
        ax_scatter.text(
            ref_projection[ref_idx, 0],
            ref_projection[ref_idx, 1],
            str(rank),
            fontsize=8,
            color="#1f1b17",
            ha="center",
            va="center",
            zorder=5,
        )
    ax_scatter.set_title(f"Latent PCA: demo {query.demo}, frame {query.frame}")
    ax_scatter.set_xlabel("PC1")
    ax_scatter.set_ylabel("PC2")
    ax_scatter.legend(frameon=False, loc="best")

    right = outer[0, 1].subgridspec(1 + neighbor_rows, cols, hspace=0.35, wspace=0.2)
    ax_query = fig.add_subplot(right[0, :])
    ax_query.imshow(load_frame_image(query_hdf5_path, query.demo, query.frame, camera))
    ax_query.set_title(f"Query: demo {query.demo}, frame {query.frame}", fontsize=11)
    ax_query.axis("off")

    for idx in range(neighbor_rows * cols):
        row = 1 + idx // cols
        col = idx % cols
        ax = fig.add_subplot(right[row, col])
        if idx < len(neighbor_indices):
            ref_idx = neighbor_indices[idx]
            neighbor_demo = int(ref_demo_idx[ref_idx])
            neighbor_frame = int(ref_frame_idx[ref_idx])
            ax.imshow(load_frame_image(reference_hdf5_path, neighbor_demo, neighbor_frame, camera))
            ax.set_title(
                f"#{idx + 1}: d{neighbor_demo} f{neighbor_frame}\n"
                f"dist={neighbor_distances[idx]:.4f}",
                fontsize=9,
            )
        ax.axis("off")

    fig.suptitle(f"Observation-latent k-NN for demo {query.demo}, frame {query.frame}", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    queries = parse_queries(args.query)
    query_hdf5 = args.query_hdf5 or args.reference_hdf5
    args.output_dir.mkdir(parents=True, exist_ok=True)

    obs_alg, obs_state, dataset_statistics, obs_config = load_checkpoint(args.obs_ckpt)
    if "mean" not in dataset_statistics and len(dataset_statistics) == 1:
        dataset_statistics = next(iter(dataset_statistics.values()))
    obs_structure = obs_config.structure["observation"].to_dict()

    ref_observation, ref_demo_idx, ref_frame_idx = load_hdf5_observations(
        args.reference_hdf5,
        obs_structure=obs_structure,
        dataset_statistics=dataset_statistics,
        max_frames=args.max_reference_frames,
    )
    ref_latents = encode_observations(obs_alg, obs_state, ref_observation, args.batch_size)

    query_observation = load_query_observations(query_hdf5, obs_structure, dataset_statistics, queries)
    query_latents = encode_observations(obs_alg, obs_state, query_observation, args.batch_size)

    pca_mean, pca_components = fit_pca_2d(ref_latents)
    ref_projection = project_pca(ref_latents, pca_mean, pca_components)
    query_projection = project_pca(query_latents, pca_mean, pca_components)

    query_equals_reference = query_hdf5.resolve() == args.reference_hdf5.resolve()
    csv_rows = []
    for query_idx, query in enumerate(queries):
        neighbor_indices, neighbor_distances = find_neighbors(
            query,
            query_latents[query_idx],
            ref_latents,
            ref_demo_idx,
            ref_frame_idx,
            args.k,
            exclude_same_demo=query_equals_reference,
            max_one_per_demo=args.max_one_per_demo,
        )
        out_path = args.output_dir / f"demo_{query.demo:04d}_frame_{query.frame:04d}_neighbors.png"
        plot_query_figure(
            out_path=out_path,
            query=query,
            query_projection=query_projection[query_idx],
            ref_projection=ref_projection,
            ref_demo_idx=ref_demo_idx,
            ref_frame_idx=ref_frame_idx,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            query_hdf5_path=query_hdf5,
            reference_hdf5_path=args.reference_hdf5,
            camera=args.camera,
        )

        for rank, (ref_idx, distance) in enumerate(zip(neighbor_indices, neighbor_distances, strict=False), start=1):
            csv_rows.append(
                {
                    "query_demo": query.demo,
                    "query_frame": query.frame,
                    "neighbor_rank": rank,
                    "neighbor_demo": int(ref_demo_idx[ref_idx]),
                    "neighbor_frame": int(ref_frame_idx[ref_idx]),
                    "latent_distance": float(distance),
                }
            )

    csv_path = args.output_dir / "neighbors.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_demo",
                "query_frame",
                "neighbor_rank",
                "neighbor_demo",
                "neighbor_frame",
                "latent_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "obs_ckpt": os.path.abspath(args.obs_ckpt),
        "reference_hdf5": str(args.reference_hdf5.resolve()),
        "query_hdf5": str(query_hdf5.resolve()),
        "camera": args.camera,
        "k": args.k,
        "batch_size": args.batch_size,
        "max_reference_frames": args.max_reference_frames,
        "num_reference_frames": int(ref_observation["state"].shape[0]),
        "latent_dim": int(ref_latents.shape[-1]),
        "query_equals_reference": query_equals_reference,
        "exclude_same_demo": query_equals_reference,
        "max_one_per_demo": args.max_one_per_demo,
        "queries": [{"demo": query.demo, "frame": query.frame} for query in queries],
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(csv_path)
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
