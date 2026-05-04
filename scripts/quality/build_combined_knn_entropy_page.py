#!/usr/bin/env python3
"""Build a combined review page from multiple kNN entropy result directories."""

from __future__ import annotations

import argparse
import csv
import html
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Result spec label=/path/to/knn_dir. The directory must contain knn_entropy.csv and frames/.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Expert200 Random-Post BC Latent kNN")
    return parser.parse_args()


def parse_result_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid --result {raw!r}; expected label=/path/to/dir")
    label, path = raw.split("=", 1)
    return label, Path(path)


def read_rows(label: str, path: Path) -> list[dict]:
    csv_path = path / "knn_entropy.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["label"] = label
            row["source_dir"] = path
            row["query_ep"] = int(row["query_ep"])
            row["query_step"] = int(row["query_step"])
            row["query_entropy"] = float(row["query_entropy"])
            row["rank"] = int(row["rank"])
            row["neighbor_ep"] = int(row["neighbor_ep"])
            row["neighbor_step"] = int(row["neighbor_step"])
            row["neighbor_entropy"] = float(row["neighbor_entropy"])
            row["distance"] = float(row["distance"])
            row["cosine"] = float(row["cosine"])
            if row["query_ep"] == row["neighbor_ep"]:
                raise AssertionError(f"{csv_path}: same-demo neighbor in row {row}")
            rows.append(row)
    return rows


def copy_asset(src_dir: Path, rel: str, dst_root: Path, label: str) -> str:
    src = src_dir / rel
    if not src.exists():
        raise FileNotFoundError(src)
    dst = dst_root / "assets" / label / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.relative_to(dst_root).as_posix()


def fmt(value: float) -> str:
    return f"{value:.3f}"


def stats_block(label: str, rows: list[dict]) -> str:
    query_keys = {(r["query_ep"], r["query_step"]) for r in rows}
    query_entropy = [r["query_entropy"] for r in rows if r["rank"] == 1]
    neighbor_entropy = [r["neighbor_entropy"] for r in rows]
    avg_q = sum(query_entropy) / len(query_entropy)
    avg_n = sum(neighbor_entropy) / len(neighbor_entropy)
    return (
        "<div class='stat'>"
        f"<span>{html.escape(label)}</span>"
        f"<b>{len(query_keys)}</b><small>queries</small>"
        f"<b>{fmt(avg_q)}</b><small>mean query NLL</small>"
        f"<b>{fmt(avg_n)}</b><small>mean neighbor NLL</small>"
        "</div>"
    )


def build_section(label: str, rows: list[dict], output: Path) -> str:
    by_query: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_query[(row["query_ep"], row["query_step"])].append(row)

    cards = []
    for (query_ep, query_step), query_rows in sorted(
        by_query.items(), key=lambda item: (-item[1][0]["query_entropy"], item[0])
    ):
        query_rows = sorted(query_rows, key=lambda r: r["rank"])
        source_dir = query_rows[0]["source_dir"]
        query_rel = f"frames/query_{query_ep:04d}_{query_step:04d}.png"
        query_img = copy_asset(source_dir, query_rel, output, label)
        neighbor_html = []
        for row in query_rows:
            img = copy_asset(source_dir, row["neighbor_image"], output, label)
            neighbor_html.append(
                "<article class='neighbor'>"
                f"<img src='{html.escape(img)}' alt='neighbor demo {row['neighbor_ep']} frame {row['neighbor_step']}'>"
                "<div>"
                f"<strong>#{row['rank']} demo {row['neighbor_ep']} : {row['neighbor_step']}</strong>"
                f"<span>NLL {fmt(row['neighbor_entropy'])}</span>"
                f"<span>dist {fmt(row['distance'])} | cos {fmt(row['cosine'])}</span>"
                "</div>"
                "</article>"
            )
        cards.append(
            "<section class='query-card'>"
            "<div class='query-head'>"
            f"<img src='{html.escape(query_img)}' alt='query demo {query_ep} frame {query_step}'>"
            "<div>"
            f"<p class='eyebrow'>{html.escape(label)}</p>"
            f"<h2>demo {query_ep} | frame {query_step}</h2>"
            f"<p class='nll'>query NLL <b>{fmt(query_rows[0]['query_entropy'])}</b></p>"
            "</div>"
            "</div>"
            f"<div class='neighbors'>{''.join(neighbor_html)}</div>"
            "</section>"
        )
    return "".join(cards)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    result_rows = []
    for raw in args.result:
        label, path = parse_result_spec(raw)
        result_rows.append((label, read_rows(label, path)))

    combined_csv = args.output / "combined_knn_entropy.csv"
    with combined_csv.open("w", newline="") as f:
        fieldnames = [
            "label",
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
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, rows in result_rows:
            for row in rows:
                writer.writerow({k: row[k] for k in fieldnames})

    nav = "".join(
        f"<a href='#{html.escape(label)}'>{html.escape(label.replace('_', ' '))}</a>" for label, _ in result_rows
    )
    stats = "".join(stats_block(label.replace("_", " "), rows) for label, rows in result_rows)
    sections = "".join(
        f"<div class='view-section' id='{html.escape(label)}'><h1>{html.escape(label.replace('_', ' '))}</h1>{build_section(label, rows, args.output)}</div>"
        for label, rows in result_rows
    )

    (args.output / "index.html").write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{html.escape(args.title)}</title>
<style>
:root {{
  --ink: #18221d;
  --muted: #657065;
  --paper: #fff8ea;
  --line: #d7c6aa;
  --accent: #b94f24;
  --field: #e7dfca;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(185,79,36,.20), transparent 34rem),
    linear-gradient(135deg, #f7ead0, #d9e2ce 55%, #f6f0e2);
  font-family: Avenir Next, Gill Sans, Trebuchet MS, sans-serif;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 5;
  padding: 20px 28px;
  backdrop-filter: blur(18px);
  background: rgba(255,248,234,.82);
  border-bottom: 1px solid var(--line);
}}
.title {{ display: flex; justify-content: space-between; gap: 20px; align-items: end; }}
.title h1 {{ margin: 0; font-size: clamp(30px, 4vw, 56px); line-height: .95; letter-spacing: -.04em; }}
.title p {{ margin: 0; max-width: 720px; color: var(--muted); line-height: 1.45; }}
nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
nav a {{
  color: var(--ink);
  text-decoration: none;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 12px;
  background: rgba(255,255,255,.38);
}}
main {{ padding: 28px; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-bottom: 32px; }}
.stat {{
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 6px 16px;
  padding: 18px;
  background: rgba(255,248,234,.76);
  border: 1px solid var(--line);
  border-radius: 20px;
  box-shadow: 0 18px 50px rgba(58,43,24,.08);
}}
.stat span {{ grid-column: 1 / -1; color: var(--accent); text-transform: uppercase; letter-spacing: .08em; font-size: 12px; }}
.stat b {{ font-size: 24px; }}
.stat small {{ color: var(--muted); align-self: center; }}
.view-section > h1 {{ margin: 42px 0 16px; font-size: 28px; letter-spacing: -.02em; }}
.query-card {{
  margin: 18px 0;
  padding: 18px;
  display: grid;
  grid-template-columns: minmax(210px, 280px) 1fr;
  gap: 18px;
  background: rgba(255,248,234,.86);
  border: 1px solid var(--line);
  border-radius: 24px;
  box-shadow: 0 20px 60px rgba(58,43,24,.08);
}}
.query-head img, .neighbor img {{
  width: 100%;
  display: block;
  border-radius: 16px;
  border: 2px solid var(--ink);
  background: var(--field);
}}
.eyebrow {{ margin: 12px 0 6px; color: var(--accent); text-transform: uppercase; letter-spacing: .08em; font-size: 12px; }}
.query-head h2 {{ margin: 0; font-size: 26px; letter-spacing: -.03em; }}
.nll {{ margin: 8px 0 0; color: var(--muted); }}
.nll b {{ color: var(--ink); font-size: 22px; }}
.neighbors {{ display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 12px; }}
.neighbor {{
  padding: 10px;
  border-radius: 18px;
  background: rgba(231,223,202,.65);
  border: 1px solid rgba(24,34,29,.12);
}}
.neighbor strong {{ display: block; margin-top: 8px; font-size: 13px; }}
.neighbor span {{ display: block; color: var(--muted); font-size: 12px; line-height: 1.35; }}
@media (max-width: 900px) {{
  .title {{ display: block; }}
  .query-card {{ grid-template-columns: 1fr; }}
  .neighbors {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
}}
</style>
</head>
<body>
<header>
  <div class='title'>
    <h1>{html.escape(args.title)}</h1>
    <p>Cross-demo nearest neighbors only. Each query excludes its own demo and shows at most one frame from any other demo. Ranking uses unit-normalized latent L2 distance; cosine and NLL are shown for audit.</p>
  </div>
  <nav>{nav}<a href='combined_knn_entropy.csv'>CSV</a></nav>
</header>
<main>
  <div class='stats'>{stats}</div>
  {sections}
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(args.output / "index.html")
    print(combined_csv)


if __name__ == "__main__":
    main()
