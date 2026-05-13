#!/usr/bin/env python3
"""Build a compact success-rate report for Expert200 random-post policies."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


DP_RUNS = {
    "agent_wrist": "expert200_dp_agent_wrist_abs_212",
    "left_close_low_wrist": "expert200_dp_left_close_low_wrist_abs_212",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bc-root",
        type=Path,
        default=Path("/iris/u/jasonyan/data/robomimic_rollout_scores/expert200_random_post_bc_success"),
    )
    parser.add_argument(
        "--dp-root",
        type=Path,
        default=Path("/iris/u/jasonyan/data/diffusion_policy_outputs/policy_view_experiments"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/iris/u/jasonyan/data/robomimic_rollout_scores/expert200_random_post_success_report"),
    )
    return parser.parse_args()


def read_json_lines(path: Path) -> list[dict[str, object]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def final_float(rows: list[dict[str, object]], key: str) -> float | None:
    for row in reversed(rows):
        if key in row:
            return float(row[key])
    return None


def dp_rows(dp_root: Path) -> list[dict[str, str]]:
    rows = []
    for view, run_name in DP_RUNS.items():
        logs = read_json_lines(dp_root / run_name / "logs.json.txt")
        final = logs[-1] if logs else {}
        rewards = [float(v) for k, v in final.items() if str(k).startswith("test/sim_max_reward_")]
        rows.append(
            {
                "method": "diffusion_policy",
                "policy": "dp",
                "view": view,
                "run_name": run_name,
                "checkpoint_label": "final",
                "epoch": str(final.get("epoch", "")),
                "num_rollouts": str(len(rewards)),
                "num_success": str(int(sum(1 for v in rewards if v > 0))),
                "success_rate": f"{final_float(logs, 'test/mean_score') or 0.0:.6g}",
                "return": "",
                "horizon": "",
                "checkpoint": str(dp_root / run_name / "checkpoints" / "latest.ckpt"),
            }
        )
    return rows


def bc_rows(bc_root: Path) -> list[dict[str, str]]:
    rows = []
    for summary in sorted(bc_root.glob("*/*_summary.csv")):
        with summary.open(newline="") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "method": "bc",
                        "policy": row.get("Policy", ""),
                        "view": row.get("View", ""),
                        "run_name": row.get("Run_Name", ""),
                        "checkpoint_label": row.get("Checkpoint_Label", ""),
                        "epoch": row.get("Epoch", ""),
                        "num_rollouts": row.get("Num_Rollouts", ""),
                        "num_success": row.get("Num_Success", ""),
                        "success_rate": row.get("Success_Rate", ""),
                        "return": row.get("Return", ""),
                        "horizon": row.get("Horizon", ""),
                        "checkpoint": row.get("Checkpoint", ""),
                    }
                )
    return rows


def best_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best = {}
    for row in rows:
        key = (row["method"], row["policy"], row["view"], row["run_name"])
        try:
            score = float(row["success_rate"])
        except ValueError:
            score = -1.0
        if key not in best or score > float(best[key]["success_rate"] or -1):
            best[key] = row
    return sorted(best.values(), key=lambda r: (r["method"], r["policy"], r["view"], r["run_name"]))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "method",
        "policy",
        "view",
        "run_name",
        "checkpoint_label",
        "epoch",
        "num_rollouts",
        "num_success",
        "success_rate",
        "return",
        "horizon",
        "checkpoint",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in ["method", "policy", "view", "checkpoint", "success", "n"])
    body = []
    for row in rows:
        ckpt = row["checkpoint_label"] or row["epoch"]
        success = row["success_rate"]
        n = f"{row['num_success']}/{row['num_rollouts']}" if row["num_rollouts"] else ""
        body.append(
            "<tr>"
            f"<td>{html.escape(row['method'])}</td>"
            f"<td>{html.escape(row['policy'])}</td>"
            f"<td>{html.escape(row['view'])}</td>"
            f"<td>{html.escape(ckpt)}</td>"
            f"<td>{html.escape(success)}</td>"
            f"<td>{html.escape(n)}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(path: Path, rows: list[dict[str, str]], best: list[dict[str, str]]) -> None:
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Expert200 Random-Post Success Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17201b; }}
    h1 {{ font-size: 22px; margin: 0 0 16px; }}
    h2 {{ font-size: 16px; margin: 24px 0 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d9e0db; padding: 7px 8px; text-align: left; }}
    th {{ background: #f3f6f4; }}
  </style>
</head>
<body>
  <h1>Expert200 Random-Post Success Report</h1>
  <p><a href="all_success.csv">all_success.csv</a> · <a href="best_success.csv">best_success.csv</a></p>
  <h2>Best Per Run</h2>
  {table(best)}
  <h2>All Evaluated Checkpoints</h2>
  {table(rows)}
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = dp_rows(args.dp_root) + bc_rows(args.bc_root)
    best = best_rows(rows)
    write_csv(args.output / "all_success.csv", rows)
    write_csv(args.output / "best_success.csv", best)
    write_html(args.output / "index.html", rows, best)
    print(f"wrote {args.output / 'all_success.csv'}")
    print(f"wrote {args.output / 'best_success.csv'}")
    print(f"wrote {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
