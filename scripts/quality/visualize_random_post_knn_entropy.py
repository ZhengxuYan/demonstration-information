#!/usr/bin/env python3
"""Build static kNN review pages over BC policy latents with NLL entropy."""

from __future__ import annotations

import argparse
import csv
import html
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt

import robomimic.utils.tensor_utils as TensorUtils

from score_robomimic_policy_nll import index_metadata, load_algo, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--score-pkl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-key", choices=["agentview_image", "left_close_low_image"], required=True)
    parser.add_argument("--run-label", type=str, required=True)
    parser.add_argument(
        "--latent-npz",
        type=Path,
        default=None,
        help="Optional external latent file with arrays latent, ep_idx, step_idx, demo_key.",
    )
    parser.add_argument("--latent-label", type=str, default="BC policy observation encoder")
    parser.add_argument("--filter-key", type=str, default=None)
    parser.add_argument("--query", action="append", default=[], help="Explicit query as demo_id:frame_id, e.g. 12:47")
    parser.add_argument("--num-queries", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_scores(path: Path) -> dict[tuple[int, int], float]:
    with path.open("rb") as f:
        scores = pickle.load(f)
    out = {}
    for ep, step, score in zip(scores["sample_ep_idx"], scores["sample_step_idx"], scores["sample_score"]):
        out[(int(ep), int(step))] = float(score)
    return out


def extract_policy_latents(args: argparse.Namespace):
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo, config = load_algo(args.checkpoint, args.dataset, device)
    dataset, loader = make_loader(config, args.batch_size, 0, args.filter_key)

    latents = []
    ep_idxs = []
    step_idxs = []
    demo_keys = []
    policy = algo.nets["policy"]
    encoder = policy.nets["encoder"]

    with torch.no_grad():
        for batch in loader:
            indices = np.asarray(TensorUtils.to_numpy(batch["index"]))
            input_batch = algo.process_batch_for_training(batch)
            input_batch = algo.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)
            goal = input_batch.get("goal_obs")
            if goal is None:
                latent = encoder(obs=input_batch["obs"])
            else:
                latent = encoder(obs=input_batch["obs"], goal=goal)
            latent_np = TensorUtils.to_numpy(latent).astype(np.float32)
            ep, step, demos = index_metadata(dataset, indices)
            latents.append(latent_np)
            ep_idxs.append(ep)
            step_idxs.append(step)
            demo_keys.append(demos)

    return (
        np.concatenate(latents, axis=0),
        np.concatenate(ep_idxs, axis=0),
        np.concatenate(step_idxs, axis=0),
        np.concatenate(demo_keys, axis=0),
    )


def load_external_latents(path: Path):
    data = np.load(path, allow_pickle=True)
    required = {"latent", "ep_idx", "step_idx", "demo_key"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"{path} missing arrays: {sorted(missing)}")
    return (
        np.asarray(data["latent"], dtype=np.float32),
        np.asarray(data["ep_idx"], dtype=np.int64),
        np.asarray(data["step_idx"], dtype=np.int64),
        np.asarray(data["demo_key"]).astype(str),
    )


def normalized_knn(latents: np.ndarray, query_index: int, top_k: int) -> list[tuple[int, float, float]]:
    norms = np.linalg.norm(latents, axis=1, keepdims=True)
    normalized = latents / np.maximum(norms, 1e-12)
    query = normalized[query_index]
    cosine = normalized @ query
    dist = np.linalg.norm(normalized - query[None, :], axis=1)
    order = np.argsort(dist)
    out = []
    for idx in order:
        if int(idx) == int(query_index):
            continue
        out.append((int(idx), float(dist[idx]), float(cosine[idx])))
        if len(out) >= top_k:
            break
    return out


def choose_queries(args, ep_idxs, step_idxs, score_map):
    index_by_pair = {(int(ep), int(step)): idx for idx, (ep, step) in enumerate(zip(ep_idxs, step_idxs))}
    queries = []
    for raw in args.query:
        ep_s, step_s = raw.split(":", 1)
        pair = (int(ep_s), int(step_s))
        if pair not in index_by_pair:
            raise KeyError(f"query {raw} is not present in scored transitions")
        queries.append(index_by_pair[pair])
    if queries:
        return queries

    scored = []
    for idx, pair in enumerate(zip(ep_idxs, step_idxs)):
        score = score_map.get((int(pair[0]), int(pair[1])))
        if score is not None:
            scored.append((score, idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[: args.num_queries]]


def read_frame(dataset_path: Path, demo_key: str, image_key: str, frame: int) -> np.ndarray:
    with h5py.File(dataset_path, "r") as f:
        return f[f"data/{demo_key}/obs/{image_key}"][frame]


def save_frame(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image)


def write_outputs(args, ep_idxs, step_idxs, demo_keys, score_map, queries, rows):
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "knn_entropy.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_ep",
                "query_step",
                "query_entropy",
                "rank",
                "neighbor_ep",
                "neighbor_step",
                "neighbor_entropy",
                "distance",
                "cosine",
                "neighbor_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    sections = []
    for q_idx in queries:
        q_ep = int(ep_idxs[q_idx])
        q_step = int(step_idxs[q_idx])
        q_score = score_map.get((q_ep, q_step), float("nan"))
        q_img = f"frames/query_{q_ep:04d}_{q_step:04d}.png"
        save_frame(args.output / q_img, read_frame(args.dataset, str(demo_keys[q_idx]), args.view_key, q_step))
        cards = [f"<img class='frame query' src='{q_img}'><div class='meta'>query demo {q_ep} frame {q_step}<br>NLL {q_score:.3f}</div>"]
        for row in [r for r in rows if r["query_ep"] == q_ep and r["query_step"] == q_step]:
            cards.append(
                "<div class='neighbor'>"
                f"<img class='frame' src='{html.escape(row['neighbor_image'])}'>"
                f"<div class='meta'>#{row['rank']} demo {row['neighbor_ep']} frame {row['neighbor_step']}<br>"
                f"NLL {float(row['neighbor_entropy']):.3f} dist {float(row['distance']):.3f} cos {float(row['cosine']):.3f}</div>"
                "</div>"
            )
        sections.append(f"<section><h2>Query demo {q_ep} frame {q_step}</h2><div class='grid'>{''.join(cards)}</div></section>")

    html_path = args.output / "index.html"
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{html.escape(args.run_label)} kNN Entropy</title>
<style>
body {{ margin: 0; padding: 32px; font-family: ui-sans-serif, system-ui; background: #f4f0e8; color: #1f2522; }}
h1 {{ font-size: 34px; margin: 0 0 8px; }}
section {{ margin: 28px 0; padding: 20px; background: #fffaf0; border: 1px solid #d8cbb5; border-radius: 18px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 14px; align-items: start; }}
.frame {{ width: 100%; border-radius: 12px; border: 2px solid #26352c; background: #ddd; }}
.query {{ border-color: #b55224; }}
.meta {{ font-size: 12px; line-height: 1.35; margin-top: 6px; }}
</style></head><body>
<h1>{html.escape(args.run_label)} kNN Entropy</h1>
<p>Latent space: {html.escape(args.latent_label)}. Distance: L2 after unit normalization; cosine shown for ranking audit. CSV: <a href='knn_entropy.csv'>knn_entropy.csv</a></p>
{''.join(sections)}
</body></html>
""",
        encoding="utf-8",
    )
    print(csv_path)
    print(html_path)


def main() -> None:
    args = parse_args()
    score_map = load_scores(args.score_pkl)
    if args.latent_npz is None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --latent-npz is provided")
        latents, ep_idxs, step_idxs, demo_keys = extract_policy_latents(args)
    else:
        latents, ep_idxs, step_idxs, demo_keys = load_external_latents(args.latent_npz)
    queries = choose_queries(args, ep_idxs, step_idxs, score_map)

    rows = []
    for q_idx in queries:
        q_ep = int(ep_idxs[q_idx])
        q_step = int(step_idxs[q_idx])
        for rank, (n_idx, dist, cosine) in enumerate(normalized_knn(latents, q_idx, args.top_k), start=1):
            n_ep = int(ep_idxs[n_idx])
            n_step = int(step_idxs[n_idx])
            img_rel = f"frames/neighbor_q{q_ep:04d}_{q_step:04d}_r{rank:02d}_d{n_ep:04d}_{n_step:04d}.png"
            save_frame(args.output / img_rel, read_frame(args.dataset, str(demo_keys[n_idx]), args.view_key, n_step))
            rows.append(
                {
                    "query_ep": q_ep,
                    "query_step": q_step,
                    "query_entropy": score_map.get((q_ep, q_step), float("nan")),
                    "rank": rank,
                    "neighbor_ep": n_ep,
                    "neighbor_step": n_step,
                    "neighbor_entropy": score_map.get((n_ep, n_step), float("nan")),
                    "distance": dist,
                    "cosine": cosine,
                    "neighbor_image": img_rel,
                }
            )
    write_outputs(args, ep_idxs, step_idxs, demo_keys, score_map, queries, rows)


if __name__ == "__main__":
    main()
