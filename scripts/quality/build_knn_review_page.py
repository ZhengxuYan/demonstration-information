"""
Build a static HTML review page for latent k-NN query results.

Example:

python scripts/quality/build_knn_review_page.py \
  --results-dir /Users/jasonyan/Desktop/demonstration-information/fb_forward_knn_vs_square \
  --query-hdf5 /Users/jasonyan/Desktop/demonstration-information/fb_demos/forward_grab/1775520924_6683068/image.hdf5 \
  --reference-hdf5 /Users/jasonyan/Desktop/demonstration-information/fb_demos/forward_grab/1775520924_6683068/image.hdf5
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


CAMERA_KEY_BY_NAME = {
    "wrist": "robot0_eye_in_hand_image",
    "agent": "agentview_image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing neighbors.csv and summary.json.")
    parser.add_argument("--query-hdf5", type=Path, required=True, help="HDF5 used for query frames.")
    parser.add_argument("--reference-hdf5", type=Path, required=True, help="HDF5 used for neighbor frames.")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path. Defaults to <results-dir>/index.html.")
    parser.add_argument(
        "--camera",
        choices=sorted(CAMERA_KEY_BY_NAME),
        default=None,
        help="Camera stream for thumbnails. Defaults to summary.json camera or wrist.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_neighbors(path: Path) -> list[dict[str, object]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["query_demo"] = int(row["query_demo"])
        row["query_frame"] = int(row["query_frame"])
        row["neighbor_rank"] = int(row["neighbor_rank"])
        row["neighbor_demo"] = int(row["neighbor_demo"])
        row["neighbor_frame"] = int(row["neighbor_frame"])
        row["latent_distance"] = float(row["latent_distance"])
    return rows


def save_frame_png(hdf5_path: Path, demo_idx: int, frame_idx: int, camera: str, out_path: Path) -> None:
    camera_key = CAMERA_KEY_BY_NAME[camera]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "r") as f:
        frame = np.asarray(f["data"][f"demo_{demo_idx}"]["obs"][camera_key][frame_idx], dtype=np.uint8)
    imageio.imwrite(out_path, frame)


def export_assets(
    rows: list[dict[str, object]],
    results_dir: Path,
    query_hdf5: Path,
    reference_hdf5: Path,
    camera: str,
) -> dict[tuple[int, int], dict[str, str]]:
    assets: dict[tuple[int, int], dict[str, str]] = {}
    needed = set()
    for row in rows:
        needed.add(("query", row["query_demo"], row["query_frame"]))
        needed.add(("ref", row["neighbor_demo"], row["neighbor_frame"]))

    for kind, demo_idx, frame_idx in sorted(needed):
        rel = Path("assets") / kind / f"demo_{demo_idx:04d}_frame_{frame_idx:04d}.png"
        out_path = results_dir / rel
        source_hdf5 = query_hdf5 if kind == "query" else reference_hdf5
        if not out_path.exists():
            save_frame_png(source_hdf5, demo_idx, frame_idx, camera, out_path)
        assets[(demo_idx, frame_idx)] = {"kind": kind, "src": rel.as_posix()}

    return assets


def build_query_cards(
    rows: list[dict[str, object]],
    assets: dict[tuple[int, int], dict[str, str]],
    results_dir: Path,
) -> str:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["query_demo"], row["query_frame"])].append(row)

    cards = []
    for (query_demo, query_frame), group in sorted(grouped.items()):
        group.sort(key=lambda row: row["neighbor_rank"])
        query_asset = assets[(query_demo, query_frame)]["src"]
        pca_plot = f"demo_{query_demo:04d}_frame_{query_frame:04d}_neighbors.png"
        pca_plot_html = ""
        if (results_dir / pca_plot).exists():
            pca_plot_html = f'<img class="pca-plot" src="{html.escape(pca_plot)}" alt="PCA view for demo {query_demo} frame {query_frame}">'

        neighbor_html = []
        for row in group:
            neighbor_asset = assets[(row["neighbor_demo"], row["neighbor_frame"])]["src"]
            neighbor_html.append(
                f"""
                <article class="neighbor-card">
                  <img src="{html.escape(neighbor_asset)}" alt="neighbor {row['neighbor_rank']}">
                  <div class="neighbor-meta">
                    <strong>#{row['neighbor_rank']}</strong>
                    <span>demo {row['neighbor_demo']}</span>
                    <span>frame {row['neighbor_frame']}</span>
                    <span>dist {row['latent_distance']:.4f}</span>
                  </div>
                </article>
                """
            )

        cards.append(
            f"""
            <section class="query-section">
              <div class="query-head">
                <div class="query-summary">
                  <h2>demo {query_demo}, frame {query_frame}</h2>
                  <p>Query frame with its latent nearest neighbors, ordered by increasing Euclidean distance.</p>
                </div>
              </div>
              <div class="query-layout">
                <div class="query-column">
                  <div class="query-frame-card">
                    <img class="query-frame" src="{html.escape(query_asset)}" alt="query frame">
                    <div class="query-frame-meta">
                      <strong>Query</strong>
                      <span>demo {query_demo}</span>
                      <span>frame {query_frame}</span>
                    </div>
                  </div>
                  {pca_plot_html}
                </div>
                <div class="neighbor-grid">
                  {''.join(neighbor_html)}
                </div>
              </div>
            </section>
            """
        )
    return "".join(cards)


def build_page(results_dir: Path, summary: dict, cards_html: str, output: Path) -> None:
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>latent k-NN review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe5;
      --panel: #fffaf0;
      --ink: #1f1b17;
      --muted: #6d675f;
      --border: #ddd3c2;
      --accent: #0f5b5c;
      --warm: #c96c33;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
    }}
    header {{
      padding: 22px 24px 18px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 250, 240, 0.94);
      position: sticky;
      top: 0;
      z-index: 1;
      backdrop-filter: blur(10px);
    }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    .lede {{ margin: 0; color: var(--muted); max-width: 920px; line-height: 1.5; }}
    .meta {{
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
    }}
    .chip {{
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 999px;
      padding: 6px 10px;
    }}
    main {{ padding: 24px; display: grid; gap: 26px; }}
    .query-section {{
      background: rgba(255, 250, 240, 0.75);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 10px 28px rgba(0,0,0,0.05);
      overflow: hidden;
    }}
    .query-head {{ padding: 18px 20px 8px; }}
    .query-head h2 {{ margin: 0 0 4px; font-size: 24px; }}
    .query-head p {{ margin: 0; color: var(--muted); }}
    .query-layout {{
      display: grid;
      grid-template-columns: minmax(320px, 420px) 1fr;
      gap: 18px;
      padding: 14px 18px 20px;
      align-items: start;
    }}
    .query-column {{ display: grid; gap: 14px; }}
    .query-frame-card, .pca-plot {{
      width: 100%;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
    }}
    .query-frame {{ display: block; width: 100%; image-rendering: auto; }}
    .query-frame-meta {{
      padding: 10px 12px 12px;
      display: grid;
      gap: 4px;
      font-size: 14px;
    }}
    .pca-plot {{ display: block; }}
    .neighbor-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 12px;
    }}
    .neighbor-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 6px 18px rgba(0,0,0,0.04);
    }}
    .neighbor-card img {{ display: block; width: 100%; }}
    .neighbor-meta {{
      padding: 10px 12px 12px;
      display: grid;
      gap: 3px;
      font-size: 13px;
    }}
    .neighbor-meta strong {{ color: var(--warm); }}
    @media (max-width: 980px) {{
      .query-layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Latent k-NN Review</h1>
    <p class="lede">Query frames and their nearest neighbors in the learned observation latent space. Each section shows the exact query frame, its neighbor frames, and the original PCA summary figure from the k-NN run.</p>
    <div class="meta">
      <span class="chip">camera: {html.escape(str(summary.get("camera", "wrist")))}</span>
      <span class="chip">k: {int(summary.get("k", 0))}</span>
      <span class="chip">reference frames: {int(summary.get("num_reference_frames", 0))}</span>
      <span class="chip">latent dim: {int(summary.get("latent_dim", 0))}</span>
    </div>
  </header>
  <main>
    {cards_html}
  </main>
</body>
</html>
"""
    output.write_text(html_doc)


def main() -> None:
    args = parse_args()
    output = args.output or (args.results_dir / "index.html")
    summary = load_summary(args.results_dir / "summary.json")
    camera = args.camera or summary.get("camera", "wrist")
    rows = load_neighbors(args.results_dir / "neighbors.csv")
    assets = export_assets(rows, args.results_dir, args.query_hdf5, args.reference_hdf5, camera)
    cards_html = build_query_cards(rows, assets, args.results_dir)
    build_page(args.results_dir, summary, cards_html, output)
    print(output)


if __name__ == "__main__":
    main()
