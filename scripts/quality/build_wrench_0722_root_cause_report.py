#!/usr/bin/env python3
"""Build the Wrench 0722 density-score root-cause report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import imageio_ffmpeg
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCORES = (
    "neg_h_data_cond",
    "neg_h_model_cond",
    "mi_data_direct",
    "mi_data_mc_marginal",
    "mi_model_direct",
    "mi_model_mc_marginal",
)
COMPONENTS = (
    "neg_h_data_cond",
    "log_prior_data",
    "log_mc_marginal_data",
    "neg_h_model_cond",
    "log_prior_model",
    "log_mc_marginal_model",
)
CONDITIONS = (
    "proprio_euler",
    "exterior_proprio_euler",
    "wrist_proprio_euler",
    "image_proprio_euler",
)
REGIMES = ("normal", "2fold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--baseline-score-root", type=Path, required=True)
    parser.add_argument(
        "--gmm-score-root",
        type=Path,
        help="Stable min-std GMM scores; defaults to baseline-score-root.",
    )
    parser.add_argument("--control-score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--video-review",
        type=Path,
        default=Path(
            "/iris/u/jasonyan/data/wrench_on_hook_0722_pomdp/"
            "score_failure_review_20260729"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def read_score(root: Path, algo: str, regime: str, sample: bool) -> pd.DataFrame:
    name = (
        "threading_pomdp_6_sample_scores.csv"
        if sample
        else "threading_pomdp_6_scores.csv"
    )
    if regime == "normal":
        return pd.read_csv(root / algo / "normal" / name)
    frames = [
        pd.read_csv(root / algo / fold / name).assign(source_fold=fold)
        for fold in ("fold0", "fold1")
    ]
    merged = pd.concat(frames, ignore_index=True)
    keys = ["ep_idx", "step_idx"] if sample else ["ep_idx"]
    if merged.duplicated(keys).any():
        raise ValueError(f"{root}/{algo}/2fold contains duplicate {keys}")
    return merged


def add_components(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["log_prior_data"] = result.neg_h_data_cond - result.mi_data_direct
    result["log_mc_marginal_data"] = (
        result.neg_h_data_cond - result.mi_data_mc_marginal
    )
    result["log_prior_model"] = result.neg_h_model_cond - result.mi_model_direct
    result["log_mc_marginal_model"] = (
        result.neg_h_model_cond - result.mi_model_mc_marginal
    )
    return result


def auc_1_vs_3(values: pd.Series, labels: pd.Series) -> float:
    low = np.asarray(values[labels == 1], dtype=float)
    high = np.asarray(values[labels == 3], dtype=float)
    if not len(low) or not len(high):
        return float("nan")
    return float(
        (
            (high[:, None] > low[None, :]).sum()
            + 0.5 * (high[:, None] == low[None, :]).sum()
        )
        / (len(high) * len(low))
    )


def bootstrap_metric(
    frame: pd.DataFrame,
    value: str,
    metric: str,
    count: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    groups = {label: group for label, group in frame.groupby("label")}
    for _ in range(count):
        sampled = pd.concat(
            [
                group.iloc[rng.integers(0, len(group), size=len(group))]
                for group in groups.values()
            ],
            ignore_index=True,
        )
        if metric == "spearman":
            estimates.append(spearmanr(sampled[value], sampled.label).statistic)
        else:
            estimates.append(auc_1_vs_3(sampled[value], sampled.label))
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def association_rows(
    frame: pd.DataFrame,
    values: tuple[str, ...],
    source: str,
    algo: str,
    regime: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    for value in values:
        rho = float(spearmanr(frame[value], frame.label).statistic)
        auc = auc_1_vs_3(frame[value], frame.label)
        rho_low, rho_high = bootstrap_metric(
            frame, value, "spearman", bootstrap_samples, seed
        )
        auc_low, auc_high = bootstrap_metric(
            frame, value, "auc", bootstrap_samples, seed + 1
        )
        row: dict[str, object] = {
            "source": source,
            "algo": algo,
            "regime": regime,
            "value": value,
            "spearman": rho,
            "spearman_ci_low": rho_low,
            "spearman_ci_high": rho_high,
            "auc_label1_vs3": auc,
            "auc_ci_low": auc_low,
            "auc_ci_high": auc_high,
        }
        for label in (1, 2, 3):
            row[f"mean_label_{label}"] = float(frame.loc[frame.label == label, value].mean())
        rows.append(row)
        for day, day_frame in frame.groupby("day"):
            rows.append(
                {
                    "source": f"{source}:within_day",
                    "algo": algo,
                    "regime": regime,
                    "value": value,
                    "day": day,
                    "spearman": float(
                        spearmanr(day_frame[value], day_frame.label).statistic
                    ),
                    "auc_label1_vs3": auc_1_vs_3(
                        day_frame[value], day_frame.label
                    ),
                }
            )
    return rows


def trimmed_mean(values: pd.Series) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    trim = int(0.1 * len(array))
    return float(array[trim : len(array) - trim].mean()) if trim else float(array.mean())


def sliding_extreme(values: pd.Series, window: int = 31) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    width = min(window, len(array))
    means = np.convolve(array, np.ones(width) / width, mode="valid")
    return float(means.max()), float(means.min())


def build_phase_rows(
    samples: pd.DataFrame,
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    hdf5: h5py.File,
    algo: str,
    regime: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_lookup = labels.set_index("episode")
    episode_rows = []
    phase_rows = []
    for ep_idx, group in samples.groupby("ep_idx"):
        ep_idx = int(ep_idx)
        group = group.sort_values("step_idx").reset_index(drop=True)
        meta = manifest.loc[manifest.ep_idx == ep_idx].iloc[0]
        demo = hdf5[f"data/demo_{ep_idx}"]
        source_steps = np.asarray(demo["source_step_index"][:], dtype=int)
        gripper = np.asarray(demo["obs/robot0_gripper_qpos"][:], dtype=float).reshape(-1)
        if len(group) != len(source_steps):
            raise ValueError(f"demo_{ep_idx}: score/HDF5 transition mismatch")
        progress_bin = np.minimum(9, (np.arange(len(group)) * 10 // len(group)))
        middle_source = None
        if meta.episode in label_lookup.index:
            middle_source = int(label_lookup.loc[meta.episode, "middle_frame"])
        middle_local = (
            int(np.argmin(np.abs(source_steps - middle_source)))
            if middle_source is not None
            else len(group) // 2
        )
        gripper_events = np.argsort(np.abs(np.diff(gripper, prepend=gripper[0])))[-2:]
        windows = {
            "all": np.ones(len(group), dtype=bool),
            "middle_pm15": np.abs(np.arange(len(group)) - middle_local) <= 15,
            "gripper_pm15": np.any(
                np.abs(np.arange(len(group))[:, None] - gripper_events[None, :]) <= 15,
                axis=1,
            ),
        }
        base = {
            "algo": algo,
            "regime": regime,
            "ep_idx": ep_idx,
            "episode": meta.episode,
            "label": int(meta.label),
            "day": str(meta.episode).split("_", 1)[0],
            "num_steps": len(group),
            "middle_source_frame": middle_source,
            "middle_local_step": middle_local,
        }
        for score in SCORES:
            sliding_max, sliding_min = sliding_extreme(group[score])
            episode_rows.extend(
                [
                    {
                        **base,
                        "score": score,
                        "aggregation": "mean",
                        "value": float(group[score].mean()),
                    },
                    {
                        **base,
                        "score": score,
                        "aggregation": "median",
                        "value": float(group[score].median()),
                    },
                    {
                        **base,
                        "score": score,
                        "aggregation": "trim10",
                        "value": trimmed_mean(group[score]),
                    },
                    {
                        **base,
                        "score": score,
                        "aggregation": "sliding31_max",
                        "value": sliding_max,
                    },
                    {
                        **base,
                        "score": score,
                        "aggregation": "sliding31_min",
                        "value": sliding_min,
                    },
                ]
            )
            for window_name, mask in windows.items():
                if window_name == "all":
                    continue
                episode_rows.append(
                    {
                        **base,
                        "score": score,
                        "aggregation": window_name,
                        "value": float(group.loc[mask, score].mean()),
                    }
                )
            for phase in range(10):
                phase_rows.append(
                    {
                        **base,
                        "score": score,
                        "phase_decile": phase,
                        "value": float(group.loc[progress_bin == phase, score].mean()),
                    }
                )
    return pd.DataFrame(episode_rows), pd.DataFrame(phase_rows)


def merge_control_samples(
    root: Path, condition: str, regime: str
) -> pd.DataFrame:
    return read_score(root / condition, "gmm", regime, sample=True)


def build_control_rows(
    root: Path,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transition_frames = []
    episode_frames = []
    for regime in REGIMES:
        by_condition = {
            condition: merge_control_samples(root, condition, regime)
            for condition in CONDITIONS
        }
        merged = by_condition["proprio_euler"][
            ["ep_idx", "step_idx", "neg_h_data_cond"]
        ].rename(columns={"neg_h_data_cond": "log_prob_proprio"})
        for condition in CONDITIONS[1:]:
            current = by_condition[condition][
                ["ep_idx", "step_idx", "neg_h_data_cond"]
            ].rename(columns={"neg_h_data_cond": f"log_prob_{condition}"})
            merged = merged.merge(
                current, on=["ep_idx", "step_idx"], validate="one_to_one"
            )
            merged[f"visual_gain_{condition}"] = (
                merged[f"log_prob_{condition}"] - merged.log_prob_proprio
            )
        merged["regime"] = regime
        merged = merged.merge(manifest, on="ep_idx", validate="many_to_one")
        merged["day"] = merged.episode.str.split("_", n=1).str[0]
        transition_frames.append(merged)
        numeric = [
            column
            for column in merged
            if column.startswith(("log_prob_", "visual_gain_"))
        ]
        episode = merged.groupby("ep_idx")[numeric].mean().reset_index()
        episode["regime"] = regime
        episode = episode.merge(manifest, on="ep_idx", validate="one_to_one")
        episode["day"] = episode.episode.str.split("_", n=1).str[0]
        episode_frames.append(episode)
    return (
        pd.concat(transition_frames, ignore_index=True),
        pd.concat(episode_frames, ignore_index=True),
    )


def build_counterfactual_rows(root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for regime in REGIMES:
        if regime == "normal":
            frame = pd.read_csv(
                root / "image_counterfactual/normal/episode_image_counterfactuals.csv"
            )
        else:
            frame = pd.concat(
                [
                    pd.read_csv(
                        root
                        / f"image_counterfactual/{fold}/episode_image_counterfactuals.csv"
                    )
                    for fold in ("fold0", "fold1")
                ],
                ignore_index=True,
            )
            if frame.duplicated("ep_idx").any():
                raise ValueError("Counterfactual folds overlap")
        frame["regime"] = regime
        frame = frame.merge(manifest, on="ep_idx", validate="one_to_one")
        frame["day"] = frame.episode.str.split("_", n=1).str[0]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_counterfactual_transition_rows(
    root: Path, manifest: pd.DataFrame
) -> pd.DataFrame:
    frames = []
    for regime in REGIMES:
        if regime == "normal":
            frame = pd.read_csv(
                root
                / "image_counterfactual/normal/transition_image_counterfactuals.csv"
            )
        else:
            frame = pd.concat(
                [
                    pd.read_csv(
                        root
                        / f"image_counterfactual/{fold}/"
                        "transition_image_counterfactuals.csv"
                    )
                    for fold in ("fold0", "fold1")
                ],
                ignore_index=True,
            )
            if frame.duplicated(["ep_idx", "step_idx"]).any():
                raise ValueError("Counterfactual transition folds overlap")
        frame["regime"] = regime
        frame = frame.merge(manifest, on="ep_idx", validate="many_to_one")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_associations(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[
        (frame.source == "baseline")
        & frame.algo.isin(["gaussian", "gmm"])
    ].copy()
    selected["key"] = selected.algo + " " + selected.regime
    pivot = selected.pivot(index="value", columns="key", values="spearman").loc[
        list(COMPONENTS)
    ]
    plt.figure(figsize=(10, 4.8))
    plt.imshow(pivot, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    plt.colorbar(label="Spearman with ordinal label")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    for row in range(len(pivot.index)):
        for col in range(len(pivot.columns)):
            plt.text(col, row, f"{pivot.iloc[row, col]:+.2f}", ha="center", va="center")
    plt.title("Existing score components vs label")
    save_figure(output)


def plot_phase(frame: pd.DataFrame, output: Path) -> None:
    rows = []
    for (regime, score, phase), group in frame.groupby(
        ["regime", "score", "phase_decile"]
    ):
        rows.append(
            {
                "regime": regime,
                "score": score,
                "phase": phase,
                "rho": spearmanr(group.value, group.label).statistic,
            }
        )
    stats = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, regime in zip(axes, REGIMES):
        pivot = stats[stats.regime == regime].pivot(
            index="score", columns="phase", values="rho"
        ).loc[list(SCORES)]
        image = axis.imshow(pivot, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
        axis.set_title(regime)
        axis.set_xticks(range(10), [f"{10*i}-{10*(i+1)}%" for i in range(10)], rotation=45, ha="right")
        axis.set_yticks(range(len(SCORES)), SCORES)
    fig.colorbar(image, ax=axes, label="Spearman with label", shrink=0.8)
    fig.suptitle("GMM baseline score signal by trajectory phase")
    save_figure(output)


def plot_control(frame: pd.DataFrame, output: Path) -> None:
    values = [
        "visual_gain_exterior_proprio_euler",
        "visual_gain_wrist_proprio_euler",
        "visual_gain_image_proprio_euler",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, regime in zip(axes, REGIMES):
        subset = frame[frame.regime == regime]
        x = np.arange(3)
        width = 0.24
        for offset, label in enumerate((1, 2, 3)):
            means = [
                subset.loc[subset.label == label, value].mean() for value in values
            ]
            axis.bar(x + (offset - 1) * width, means, width, label=f"label {label}")
        axis.axhline(0, color="#222", linewidth=0.8)
        axis.set_xticks(x, ["exterior", "wrist", "both"])
        axis.set_title(regime)
        axis.set_ylabel("mean visual gain (nats / transition)")
    axes[1].legend()
    fig.suptitle("Conditional log-likelihood gain over proprio-only")
    save_figure(output)


def plot_counterfactual(frame: pd.DataFrame, output: Path) -> None:
    values = [
        "delta_shuffle_exterior",
        "delta_shuffle_wrist",
        "delta_shuffle_both",
        "delta_temporal_shift",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for axis, regime in zip(axes, REGIMES):
        subset = frame[frame.regime == regime]
        x = np.arange(len(values))
        width = 0.24
        for offset, label in enumerate((1, 2, 3)):
            means = [
                subset.loc[subset.label == label, value].mean() for value in values
            ]
            axis.bar(x + (offset - 1) * width, means, width, label=f"label {label}")
        axis.axhline(0, color="#222", linewidth=0.8)
        axis.set_xticks(x, ["shuffle ext", "shuffle wrist", "shuffle both", "time shift"], rotation=20)
        axis.set_title(regime)
        axis.set_ylabel("correct logprob - perturbed logprob")
    axes[1].legend()
    fig.suptitle("Does the dual-camera model use its images?")
    save_figure(output)


def html_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    shown = frame[columns].copy()
    for column in shown.select_dtypes(include=[np.number]):
        shown[column] = shown[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    return shown.to_html(index=False, escape=True, border=0)


def build_transition_review(
    transitions: pd.DataFrame,
    episode_aggregation: pd.DataFrame,
    output: Path,
    video_root: Path,
    dataset: Path,
    per_tail: int = 2,
) -> None:
    episode_mean = episode_aggregation[
        episode_aggregation.aggregation == "mean"
    ]
    selections = []
    for (regime, score), group in episode_mean.groupby(["regime", "score"]):
        selections.extend(
            group[group.label == 1]
            .nlargest(per_tail, "value")
            .assign(case="high-score label 1")
            [["regime", "score", "ep_idx", "case"]]
            .to_dict("records")
        )
        selections.extend(
            group[group.label == 3]
            .nsmallest(per_tail, "value")
            .assign(case="low-score label 3")
            [["regime", "score", "ep_idx", "case"]]
            .to_dict("records")
        )
    selection_frame = pd.DataFrame(selections)
    selected = sorted(selection_frame.ep_idx.unique())
    trace_columns = [
        *SCORES,
        "log_prior_data",
        "log_mc_marginal_data",
        "visual_gain_exterior_proprio_euler",
        "visual_gain_wrist_proprio_euler",
        "visual_gain_image_proprio_euler",
        "delta_shuffle_exterior",
        "delta_shuffle_wrist",
        "delta_shuffle_both",
        "delta_temporal_shift",
    ]
    rows = []
    for (regime, ep_idx), group in transitions[
        transitions.ep_idx.isin(selected)
    ].groupby(["regime", "ep_idx"]):
        group = group.sort_values("step_idx")
        available = [
            column for column in trace_columns if column in group and group[column].notna().any()
        ]
        rows.append(
            {
                "regime": regime,
                "ep_idx": int(ep_idx),
                "episode": str(group.episode.iloc[0]),
                "label": int(group.label.iloc[0]),
                "video": f"video_review/videos/ep_{int(ep_idx):03d}.mp4",
                "steps": group.step_idx.astype(int).tolist(),
                "traces": {
                    column: group[column].astype(float).round(6).tolist()
                    for column in available
                },
                "selections": selection_frame[
                    (selection_frame.regime == regime)
                    & (selection_frame.ep_idx == ep_idx)
                ][["score", "case"]].to_dict("records"),
            }
        )
    video_root.mkdir(parents=True, exist_ok=True)
    missing = sorted(
        {
            row["ep_idx"]
            for row in rows
            if not (video_root / f"ep_{row['ep_idx']:03d}.mp4").is_file()
        }
    )
    if missing:
        with h5py.File(dataset, "r") as hdf5:
            for ep_idx in missing:
                render_review_video(
                    hdf5,
                    ep_idx,
                    video_root / f"ep_{ep_idx:03d}.mp4",
                )
    payload = json.dumps(
        {"rows": rows, "scores": list(SCORES), "traces": trace_columns},
        separators=(",", ":"),
    )
    page = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;font:13px Arial;color:#17201d;background:#f4f6f4}.bar{position:sticky;top:0;background:white;padding:10px;border-bottom:1px solid #ccd5cf;z-index:2}
select{margin-right:10px;padding:5px}.list{padding:10px;display:grid;gap:12px}.card{background:white;border:1px solid #d7ddd9;padding:10px}.top{display:grid;grid-template-columns:42% 1fr;gap:12px}
video{width:100%;aspect-ratio:32/9;background:#111}.trace{margin-bottom:7px}.name{font-size:11px;color:#5c6b63}svg{display:block;width:100%;height:70px;background:#fafbfa}
.line{fill:none;stroke:#16708f;stroke-width:1.8}.head{stroke:#111}.meta{color:#5c6b63;margin:5px 0}@media(max-width:850px){.top{grid-template-columns:1fr}}</style></head>
<body><div class="bar">Regime <select id="regime"><option>normal</option><option>2fold</option></select>
Selection score <select id="score"></select></div><div id="list" class="list"></div><script>
const DATA=__PAYLOAD__; const score=document.querySelector("#score"), regime=document.querySelector("#regime");
DATA.scores.forEach(x=>score.add(new Option(x,x)));
const fmt=x=>Number.isFinite(x)?x.toFixed(3):"";
function chart(values,name){const finite=values.filter(Number.isFinite),lo=Math.min(...finite),hi=Math.max(...finite),span=hi-lo||1;
 const pts=values.map((v,i)=>`${i/(values.length-1||1)*700},${64-(v-lo)/span*58}`).join(" ");
 return `<div class="trace"><div class="name">${name} · ${fmt(lo)} to ${fmt(hi)}</div><svg viewBox="0 0 700 70"><polyline class="line" points="${pts}"/><line class="head" y1="0" y2="70"/></svg></div>`}
function render(){const rows=DATA.rows.filter(r=>r.regime===regime.value&&r.selections.some(s=>s.score===score.value));
 document.querySelector("#list").innerHTML=rows.map(r=>`<div class="card"><b>ep ${r.ep_idx} · label ${r.label}</b><div class="meta">${r.episode} · ${r.selections.map(x=>x.score+" / "+x.case).join(", ")}</div>
 <div class="top"><video controls preload="metadata" src="${r.video}"></video><div>${Object.entries(r.traces).map(([n,v])=>chart(v,n)).join("")}</div></div></div>`).join("");
 document.querySelectorAll(".card").forEach(card=>{const v=card.querySelector("video"),heads=card.querySelectorAll(".head");
 v.addEventListener("timeupdate",()=>{const x=v.duration?v.currentTime/v.duration*700:0;heads.forEach(h=>{h.setAttribute("x1",x);h.setAttribute("x2",x)})})})}
score.onchange=regime.onchange=render;render();</script></body></html>""".replace(
        "__PAYLOAD__", payload
    )
    (output / "controlled_transition_review.html").write_text(page)


def render_review_video(
    hdf5: h5py.File,
    ep_idx: int,
    output: Path,
    fps: float = 15.0,
) -> None:
    demo = hdf5[f"data/demo_{ep_idx}"]
    exterior = demo["obs/agentview_image"]
    wrist = demo["obs/robot0_eye_in_hand_image"]
    if exterior.shape != wrist.shape:
        raise ValueError(f"demo_{ep_idx}: camera shape mismatch")
    frames, height, width, channels = exterior.shape
    if channels != 3:
        raise ValueError(f"demo_{ep_idx}: expected RGB images")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width * 2}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    for index in range(frames):
        frame = np.concatenate([exterior[index], wrist[index]], axis=1)
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    stderr = (
        process.stderr.read().decode("utf-8", errors="replace")
        if process.stderr
        else ""
    )
    if process.wait():
        raise RuntimeError(f"ffmpeg failed for demo_{ep_idx}:\n{stderr[-2000:]}")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()
    if args.gmm_score_root is None:
        args.gmm_score_root = args.baseline_score_root
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    labels = pd.read_csv(args.labels_csv)
    enriched = manifest.merge(
        labels[["episode", "middle_frame", "image_file", "source_video"]],
        on="episode",
        how="left",
        validate="one_to_one",
    )
    if enriched.middle_frame.isna().any():
        raise ValueError("Missing middle_frame for one or more episodes")
    enriched["day"] = enriched.episode.str.split("_", n=1).str[0]
    expected_labels = {1: 60, 2: 85, 3: 155}
    observed_labels = enriched.label.value_counts().sort_index().to_dict()
    if len(enriched) != 300 or observed_labels != expected_labels:
        raise ValueError(
            f"Unexpected manifest coverage: episodes={len(enriched)}, "
            f"labels={observed_labels}"
        )
    write_csv(args.output / "manifest_enriched.csv", enriched)

    association = []
    phase_frames = []
    episode_aggregation_frames = []
    baseline_transition_frames = []
    with h5py.File(args.dataset, "r") as hdf5:
        for algo in ("gaussian", "gmm"):
            score_root = (
                args.gmm_score_root if algo == "gmm" else args.baseline_score_root
            )
            for regime in REGIMES:
                episode = add_components(
                    read_score(score_root, algo, regime, sample=False)
                ).merge(enriched, on="ep_idx", validate="one_to_one")
                association.extend(
                    association_rows(
                        episode,
                        COMPONENTS,
                        "baseline",
                        algo,
                        regime,
                        args.bootstrap_samples,
                        args.seed,
                    )
                )
                if algo == "gmm":
                    samples = add_components(
                        read_score(
                            score_root, algo, regime, sample=True
                        )
                    )
                    baseline_transition = samples.merge(
                        enriched, on="ep_idx", validate="many_to_one"
                    )
                    baseline_transition["regime"] = regime
                    baseline_transition_frames.append(baseline_transition)
                    episode_agg, phase = build_phase_rows(
                        samples, enriched, labels, hdf5, algo, regime
                    )
                    episode_aggregation_frames.append(episode_agg)
                    phase_frames.append(phase)

    association_frame = pd.DataFrame(association)
    episode_aggregation = pd.concat(episode_aggregation_frames, ignore_index=True)
    phase_frame = pd.concat(phase_frames, ignore_index=True)
    baseline_transition = pd.concat(
        baseline_transition_frames, ignore_index=True
    )
    transition_control, episode_control = build_control_rows(
        args.control_score_root, enriched
    )
    counterfactual = build_counterfactual_rows(args.control_score_root, enriched)
    counterfactual_transition = build_counterfactual_transition_rows(
        args.control_score_root, enriched
    )
    transition_analysis = baseline_transition.merge(
        transition_control.drop(
            columns=[
                column
                for column in enriched.columns
                if column in transition_control and column != "ep_idx"
            ]
        ),
        on=["regime", "ep_idx", "step_idx"],
        validate="one_to_one",
    ).merge(
        counterfactual_transition[
            [
                "regime",
                "ep_idx",
                "step_idx",
                *[
                    column
                    for column in counterfactual_transition
                    if column.startswith("delta_")
                ],
            ]
        ],
        on=["regime", "ep_idx", "step_idx"],
        validate="one_to_one",
    )

    aggregation_stats = []
    for (regime, score, aggregation), group in episode_aggregation.groupby(
        ["regime", "score", "aggregation"]
    ):
        aggregation_stats.append(
            {
                "regime": regime,
                "score": score,
                "aggregation": aggregation,
                "spearman": float(spearmanr(group.value, group.label).statistic),
                "auc_label1_vs3": auc_1_vs_3(group.value, group.label),
            }
        )
    aggregation_stats = pd.DataFrame(aggregation_stats)

    control_stats = []
    control_values = [
        column
        for column in episode_control
        if column.startswith(("log_prob_", "visual_gain_"))
    ]
    for regime, group in episode_control.groupby("regime"):
        control_stats.extend(
            association_rows(
                group,
                tuple(control_values),
                "controlled_condition",
                "gmm",
                regime,
                args.bootstrap_samples,
                args.seed,
            )
        )
    control_stats = pd.DataFrame(control_stats)

    counterfactual_stats = []
    delta_values = [
        column for column in counterfactual if column.startswith("delta_")
    ]
    for regime, group in counterfactual.groupby("regime"):
        counterfactual_stats.extend(
            association_rows(
                group,
                tuple(delta_values),
                "image_counterfactual",
                "gmm",
                regime,
                args.bootstrap_samples,
                args.seed,
            )
        )
    counterfactual_stats = pd.DataFrame(counterfactual_stats)
    all_associations = pd.concat(
        [association_frame, control_stats, counterfactual_stats],
        ignore_index=True,
        sort=False,
    )

    finite_frames = [
        episode_aggregation.select_dtypes(include=[np.number]),
        phase_frame.select_dtypes(include=[np.number]),
        transition_analysis.select_dtypes(include=[np.number]),
        counterfactual.select_dtypes(include=[np.number]),
    ]
    if any(not np.isfinite(frame.to_numpy()).all() for frame in finite_frames):
        raise ValueError("Non-finite value in report inputs")

    write_csv(args.output / "episode_analysis.csv", episode_aggregation)
    write_csv(args.output / "transition_analysis.csv", transition_analysis)
    write_csv(args.output / "phase_analysis.csv", phase_frame)
    write_csv(args.output / "camera_ablation.csv", counterfactual)
    write_csv(args.output / "association_metrics.csv", all_associations)
    write_csv(args.output / "aggregation_metrics.csv", aggregation_stats)

    plot_associations(all_associations, args.output / "score_component_associations.png")
    plot_phase(phase_frame, args.output / "phase_associations.png")
    plot_control(episode_control, args.output / "visual_gain_by_condition.png")
    plot_counterfactual(counterfactual, args.output / "image_counterfactuals.png")

    if args.video_review.is_dir():
        shutil.copytree(
            args.video_review,
            args.output / "video_review",
            dirs_exist_ok=True,
        )
        build_transition_review(
            transition_analysis,
            episode_aggregation,
            args.output,
            args.output / "video_review/videos",
            args.dataset,
        )

    best_control = control_stats.loc[
        control_stats.auc_label1_vs3.sub(0.5).abs().idxmax()
    ]
    strongest_phase = aggregation_stats.loc[
        aggregation_stats.spearman.abs().idxmax()
    ]
    strongest_counterfactual = counterfactual_stats.loc[
        counterfactual_stats.auc_label1_vs3.sub(0.5).abs().idxmax()
    ]
    baseline_gmm = association_frame[
        (association_frame.algo == "gmm")
        & (association_frame.source == "baseline")
    ]
    max_baseline_rho = float(baseline_gmm.spearman.abs().max())
    heldout_aggregation = aggregation_stats[
        aggregation_stats.regime == "2fold"
    ]
    strongest_heldout_aggregation = heldout_aggregation.loc[
        heldout_aggregation.spearman.abs().idxmax()
    ]
    heldout_counterfactual = counterfactual[
        counterfactual.regime == "2fold"
    ]
    counterfactual_means = {
        column: float(heldout_counterfactual[column].mean())
        for column in delta_values
    }
    counterfactual_rhos = {
        column: float(
            spearmanr(
                heldout_counterfactual[column],
                heldout_counterfactual.label,
            ).statistic
        )
        for column in delta_values
    }
    root_causes = pd.DataFrame(
        [
            {
                "hypothesis": "Numerical instability",
                "status": "not supported",
                "evidence": "All GMM episode, transition, phase, and counterfactual values are finite.",
            },
            {
                "hypothesis": "The conditional model ignores images",
                "status": "not supported",
                "evidence": (
                    "Held-out mean log-likelihood drops after image perturbation: "
                    + ", ".join(
                        f"{key.removeprefix('delta_')}={value:.2f}"
                        for key, value in counterfactual_means.items()
                    )
                    + " nats/transition."
                ),
            },
            {
                "hypothesis": "Whole-episode averaging hides a stable phase signal",
                "status": "weak evidence only",
                "evidence": (
                    f"Best exploratory held-out aggregation is "
                    f"{strongest_heldout_aggregation.aggregation}/"
                    f"{strongest_heldout_aggregation.score}, "
                    f"rho={strongest_heldout_aggregation.spearman:+.3f}; "
                    "windows were not selected on an independent label set."
                ),
            },
            {
                "hypothesis": "Image dependence tracks the observability label",
                "status": "not supported",
                "evidence": (
                    "Held-out image-perturbation Spearman values are "
                    + ", ".join(
                        f"{key.removeprefix('delta_')}={value:+.3f}"
                        for key, value in counterfactual_rhos.items()
                    )
                    + "."
                ),
            },
            {
                "hypothesis": "Density signal and label target are misaligned",
                "status": "supported",
                "evidence": (
                    f"Stable-GMM score components have max |rho|={max_baseline_rho:.3f}; "
                    "the model uses images, but image dependence is nearly label-independent."
                ),
            },
        ]
    )
    write_csv(args.output / "root_cause_evidence.csv", root_causes)

    findings = [
        f"Existing GMM scores remain weak: maximum absolute Spearman is {max_baseline_rho:.3f}.",
        (
            f"Strongest tested episode aggregation is {strongest_phase.aggregation} / "
            f"{strongest_phase.score} / {strongest_phase.regime}: "
            f"rho={strongest_phase.spearman:+.3f}."
        ),
        (
            f"Strongest controlled input statistic is {best_control.value} / "
            f"{best_control.regime}: AUC={best_control.auc_label1_vs3:.3f}."
        ),
        (
            f"Strongest image perturbation statistic is {strongest_counterfactual.value} / "
            f"{strongest_counterfactual.regime}: "
            f"AUC={strongest_counterfactual.auc_label1_vs3:.3f}."
        ),
    ]
    (args.output / "findings.json").write_text(
        json.dumps({"findings": findings}, indent=2) + "\n"
    )

    controls_summary = control_stats[
        control_stats.source == "controlled_condition"
    ].sort_values(["regime", "auc_label1_vs3"], ascending=[True, False])
    camera_summary = counterfactual_stats.sort_values(
        ["regime", "auc_label1_vs3"], ascending=[True, False]
    )
    aggregation_summary = aggregation_stats.sort_values(
        ["regime", "spearman"], ascending=[True, False]
    )
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wrench 0722 score root-cause analysis</title>
<style>
body{{margin:0;font:15px Arial,sans-serif;color:#18211d;background:#f6f8f6}}main{{max-width:1500px;margin:auto;padding:24px}}
h1{{font-size:30px;margin:0 0 6px}}h2{{margin-top:34px}}.meta{{color:#617068}}.findings{{background:white;border:1px solid #d9e0dc;padding:14px 20px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.panel{{background:white;border:1px solid #d9e0dc;padding:14px}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px;border-bottom:1px solid #e1e5e2;text-align:right}}
th:first-child,td:first-child{{text-align:left}}iframe{{width:100%;height:900px;border:1px solid #d9e0dc;background:white}}
.links a{{margin-right:14px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Wrench 0722 score root-cause analysis</h1>
<div class="meta">300 episodes · labels 1 partial, 2 intermediate, 3 full · GMM 5 modes · min_std=0.01 · normal and held-out 2-fold</div>
<h2>Findings</h2><div class="findings"><ul>{''.join(f'<li>{html.escape(item)}</li>' for item in findings)}</ul></div>
<h2>Root-cause evidence</h2><div class="panel">{root_causes.to_html(index=False, escape=True, border=0)}</div>
<h2>Existing score decomposition</h2><div class="panel"><img src="score_component_associations.png"></div>
<h2>Episode aggregation and task phase</h2><div class="grid"><div class="panel"><img src="phase_associations.png"></div>
<div class="panel">{html_table(aggregation_summary.head(20), ['regime','score','aggregation','spearman','auc_label1_vs3'])}</div></div>
<h2>Controlled observation inputs</h2><div class="grid"><div class="panel"><img src="visual_gain_by_condition.png"></div>
<div class="panel">{html_table(controls_summary, ['regime','value','spearman','auc_label1_vs3','auc_ci_low','auc_ci_high'])}</div></div>
<h2>Image counterfactuals</h2><div class="grid"><div class="panel"><img src="image_counterfactuals.png"></div>
<div class="panel">{html_table(camera_summary, ['regime','value','spearman','auc_label1_vs3','auc_ci_low','auc_ci_high'])}</div></div>
<h2>Controlled transition review</h2><iframe src="controlled_transition_review.html"></iframe>
<h2>Original six-score review</h2><iframe src="video_review/index.html"></iframe>
<h2>Data</h2><div class="panel links">
<a href="episode_analysis.csv">episode analysis</a><a href="transition_analysis.csv">transition analysis</a>
<a href="phase_analysis.csv">phase analysis</a><a href="camera_ablation.csv">camera ablation</a>
<a href="association_metrics.csv">association metrics</a><a href="manifest_enriched.csv">enriched manifest</a>
<a href="root_cause_evidence.csv">root-cause evidence</a>
</div></main></body></html>"""
    (args.output / "index.html").write_text(body)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
