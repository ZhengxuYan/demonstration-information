#!/usr/bin/env python3
"""Build a kNN review page from Qwen3-VL frame embeddings."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt


VIEW_KEYS = {
    "agent_wrist": ("agentview_image", "robot0_eye_in_hand_image"),
    "left_close_low_wrist": ("left_close_low_image", "robot0_eye_in_hand_image"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-npz", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", choices=sorted(VIEW_KEYS), required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--query", action="append", default=[], help="Explicit query demo_id:frame_id.")
    parser.add_argument("--num-queries", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--allow-same-demo", action="store_true")
    return parser.parse_args()


def load_latents(path: Path):
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
        {key: data[key].tolist() for key in data.files if key not in required},
    )


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def choose_queries(args, ep_idxs: np.ndarray, step_idxs: np.ndarray) -> list[int]:
    index_by_pair = {(int(ep), int(step)): idx for idx, (ep, step) in enumerate(zip(ep_idxs, step_idxs))}
    out = []
    for raw in args.query:
        ep_s, step_s = raw.split(":", 1)
        pair = (int(ep_s), int(step_s))
        if pair not in index_by_pair:
            raise KeyError(f"query {raw} is not present in {args.latent_npz}")
        out.append(index_by_pair[pair])
    if out:
        return out
    if ep_idxs.size <= args.num_queries:
        return list(range(ep_idxs.size))
    keep = np.linspace(0, ep_idxs.size - 1, args.num_queries).round().astype(int)
    return [int(idx) for idx in keep]


def knn(latents: np.ndarray, ep_idxs: np.ndarray, query_index: int, top_k: int, allow_same_demo: bool):
    normalized = normalize(latents)
    query = normalized[query_index]
    cosine = normalized @ query
    order = np.argsort(-cosine)
    out = []
    query_ep = int(ep_idxs[query_index])
    used_eps = set() if allow_same_demo else {query_ep}
    for idx in order:
        if int(idx) == int(query_index):
            continue
        ep_idx = int(ep_idxs[idx])
        if ep_idx in used_eps:
            continue
        if not allow_same_demo:
            used_eps.add(ep_idx)
        out.append((int(idx), float(cosine[idx]), float(1.0 - cosine[idx])))
        if len(out) >= top_k:
            break
    return out


def read_composite(dataset: Path, demo_key: str, step_idx: int, image_keys: tuple[str, str]) -> np.ndarray:
    with h5py.File(dataset, "r") as f:
        obs = f[f"data/{demo_key}/obs"]
        left = np.asarray(obs[image_keys[0]][step_idx], dtype=np.uint8)
        right = np.asarray(obs[image_keys[1]][step_idx], dtype=np.uint8)
    if left.shape[0] != right.shape[0]:
        raise ValueError("Composite frame inputs must have matching heights for review rendering.")
    return np.concatenate([left, right], axis=1)


def save_frame(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image)


def write_outputs(args, meta, ep_idxs, step_idxs, demo_keys, queries, rows) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "qwen3_vl_knn.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "query_ep",
            "query_step",
            "rank",
            "neighbor_ep",
            "neighbor_step",
            "cosine",
            "cosine_distance",
            "neighbor_image",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    prompt = meta.get("prompt", "")
    model_name = meta.get("model_name", "")
    sections = []
    for q_idx in queries:
        q_ep = int(ep_idxs[q_idx])
        q_step = int(step_idxs[q_idx])
        q_img = f"frames/query_{q_ep:04d}_{q_step:04d}.png"
        save_frame(args.output / q_img, read_composite(args.dataset, str(demo_keys[q_idx]), q_step, VIEW_KEYS[args.view]))
        cards = [f"<img class='frame query' src='{q_img}'><div class='meta'>query demo {q_ep} frame {q_step}</div>"]
        for row in [r for r in rows if r["query_ep"] == q_ep and r["query_step"] == q_step]:
            cards.append(
                "<div class='neighbor'>"
                f"<img class='frame' src='{html.escape(row['neighbor_image'])}'>"
                f"<div class='meta'>#{row['rank']} demo {row['neighbor_ep']} frame {row['neighbor_step']}<br>"
                f"cos {float(row['cosine']):.4f}</div></div>"
            )
        sections.append(f"<section><h2>Query demo {q_ep} frame {q_step}</h2><div class='grid'>{''.join(cards)}</div></section>")

    html_path = args.output / "index.html"
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{html.escape(args.run_label)} Qwen3-VL kNN</title>
<style>
body {{ margin: 0; padding: 32px; font-family: ui-sans-serif, system-ui; background: #f5f6f2; color: #18201c; }}
h1 {{ font-size: 32px; margin: 0 0 8px; }}
.note {{ max-width: 1100px; color: #5a635e; line-height: 1.45; }}
section {{ margin: 24px 0; padding: 18px; background: #fffefa; border: 1px solid #d5dbd2; border-radius: 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 14px; align-items: start; }}
.frame {{ width: 100%; border-radius: 6px; border: 2px solid #2a342e; background: #ddd; }}
.query {{ border-color: #a7471d; }}
.meta {{ font-size: 12px; line-height: 1.35; margin-top: 6px; }}
</style></head><body>
<h1>{html.escape(args.run_label)} Qwen3-VL kNN</h1>
<p class='note'>Model: {html.escape(str(model_name))}. Distance: cosine similarity over Qwen3-VL embeddings; same-demo neighbors are {'allowed' if args.allow_same_demo else 'excluded'}. CSV: <a href='qwen3_vl_knn.csv'>qwen3_vl_knn.csv</a></p>
<p class='note'>Prompt: {html.escape(str(prompt))}</p>
{''.join(sections)}
</body></html>
""",
        encoding="utf-8",
    )
    print(csv_path)
    print(html_path)


def main() -> None:
    args = parse_args()
    latents, ep_idxs, step_idxs, demo_keys, meta = load_latents(args.latent_npz)
    queries = choose_queries(args, ep_idxs, step_idxs)
    rows = []
    for q_idx in queries:
        q_ep = int(ep_idxs[q_idx])
        q_step = int(step_idxs[q_idx])
        for rank, (n_idx, cosine, distance) in enumerate(
            knn(latents, ep_idxs, q_idx, args.top_k, args.allow_same_demo), start=1
        ):
            n_ep = int(ep_idxs[n_idx])
            n_step = int(step_idxs[n_idx])
            img_rel = f"frames/neighbor_q{q_ep:04d}_{q_step:04d}_r{rank:02d}_d{n_ep:04d}_{n_step:04d}.png"
            save_frame(args.output / img_rel, read_composite(args.dataset, str(demo_keys[n_idx]), n_step, VIEW_KEYS[args.view]))
            rows.append(
                {
                    "query_ep": q_ep,
                    "query_step": q_step,
                    "rank": rank,
                    "neighbor_ep": n_ep,
                    "neighbor_step": n_step,
                    "cosine": cosine,
                    "cosine_distance": distance,
                    "neighbor_image": img_rel,
                }
            )
    write_outputs(args, meta, ep_idxs, step_idxs, demo_keys, queries, rows)


if __name__ == "__main__":
    main()
