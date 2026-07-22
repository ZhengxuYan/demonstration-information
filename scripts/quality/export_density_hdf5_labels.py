#!/usr/bin/env python3
"""Export episode labels stored in a density HDF5 to a small CSV manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with h5py.File(args.input, "r") as handle:
        for demo_key in handle["data"]:
            demo = handle[f"data/{demo_key}"]
            rows.append(
                {
                    "ep_idx": int(demo.attrs["ep_idx"]),
                    "episode": str(demo.attrs.get("episode", demo_key)),
                    "label": str(demo.attrs["label"]),
                }
            )
    rows.sort(key=lambda row: int(row["ep_idx"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ep_idx", "episode", "label"))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
