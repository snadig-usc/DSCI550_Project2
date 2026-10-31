"""Plot the completed Xinchen Geng baseline's historical CORR diagnostics.

These are reproducible static research figures. Each score is calculated on
one complete date, and cumulative scores are additive diagnostics, not returns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL = "B0_small_main"
SPLITS = ("tuning", "holdout")
COLORS = {"tuning": "#2E5F9E", "holdout": "#C76D20"}
LABELS = {"tuning": "Tuning period", "holdout": "Historical holdout (previously observed)"}
STYLES = {"tuning": "-", "holdout": "--"}
INK = "#26323E"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def date_ticks(ax, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Retain actual boundary dates and readable annual interior anchors."""
    anchors = [start]
    for year in range(start.year + 1, end.year + 1):
        candidate = pd.Timestamp(year=year, month=1, day=1)
        if (candidate - start).days > 110 and (end - candidate).days > 110:
            anchors.append(candidate)
    anchors.append(end)
    ax.set_xticks(mdates.date2num(anchors))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_xlim(start, end)
    ax.tick_params(axis="x", labelrotation=0, pad=8)
    ax.get_xticklabels()[-1].set_horizontalalignment("right")


def axis_style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BAC2CA")
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(axis="y", color="#E3E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#677583", linewidth=0.9, zorder=1)


def read_sources(output: Path):
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise ValueError("Figures require a completed baseline run")
    scores = pd.read_csv(output / "per_date_metrics.csv", parse_dates=["date"])
    summary = pd.read_csv(output / "metrics.csv")
    required = {"split", "date", "model", "n_rows", "CORR", "NCORR"}
    if not required.issubset(scores.columns):
        raise ValueError(f"Missing score columns: {sorted(required - set(scores.columns))}")
    if set(scores["model"]) != {MODEL} or set(scores["split"]) != set(SPLITS):
        raise ValueError("Figures expect only the baseline on tuning and holdout")
    if scores.duplicated(["split", "date"]).any():
        raise ValueError("Duplicate per-date baseline scores")
    if scores["date"].isna().any() or not np.isfinite(scores["CORR"]).all():
        raise ValueError("Invalid dates or CORR scores")
    if scores["n_rows"].le(0).any():
        raise ValueError("Empty scored date")
    for split in SPLITS:
        group = scores[scores["split"].eq(split)]
        reported = summary[
            summary["split"].eq(split)
            & summary["model"].eq(MODEL)
            & summary["metric"].eq("CORR")
        ]
        if len(reported) != 1:
            raise ValueError(f"Expected one CORR summary for {split}")
        if not np.isclose(group["CORR"].mean(), float(reported.iloc[0]["mean"]), atol=1e-12, rtol=0):
            raise ValueError(f"CORR mean and summary disagree for {split}")
        expected = run["splits"][split]
        if len(group) != expected["dates"] or int(group["n_rows"].sum()) != expected["rows"]:
            raise ValueError(f"Score coverage and run manifest disagree for {split}")
    return run, scores.sort_values(["split", "date"])


def render(output: Path) -> None:
    run, scores = read_sources(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.titleweight": "semibold", "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    groups = {split: scores[scores["split"].eq(split)].sort_values("date") for split in SPLITS}
    sample_rows = int(run["train_sample_rows"])
    low = min(0.0, float(scores["CORR"].min()))
    high = max(0.0, float(scores["CORR"].max()))
    pad = max((high - low) * 0.09, 0.002)

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.8), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.81, bottom=0.15, hspace=0.51)
    fig.suptitle("LightGBM baseline: historical CORR by date", x=0.09, y=0.96, ha="left", fontsize=20, weight="semibold")
    fig.text(0.09, 0.91, f"{sample_rows:,} training rows | small numeric feature set | target_jupiter_60", fontsize=11, color="#596775")
    for ax, split in zip(axes, SPLITS):
        group = groups[split]
        axis_style(ax)
        ax.plot(group["date"], group["CORR"], color=COLORS[split], linestyle=STYLES[split], linewidth=1.45)
        ax.set_ylim(low - pad, high + pad)
        ax.set_ylabel("Numerai CORR")
        ax.set_title(f"{LABELS[split]} | {len(group)} dates | mean {group['CORR'].mean():.5f}", loc="left", fontsize=12, pad=14)
        date_ticks(ax, group["date"].min(), group["date"].max())
    axes[-1].set_xlabel("Observation date", labelpad=10)
    fig.text(0.09, 0.055, "Fixed historical reproduction; the holdout was observed in earlier experiments.", fontsize=10, color="#596775")
    fig.text(0.09, 0.031, "Each point uses all eligible rows on that date. CORR is a score, not an investment return.", fontsize=10, color="#596775")
    fig.savefig(figures / "per_date_corr.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.8, 7.4))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.77, bottom=0.18)
    fig.suptitle("LightGBM baseline: cumulative historical CORR", x=0.09, y=0.96, ha="left", fontsize=20, weight="semibold")
    fig.text(0.09, 0.905, "Additive score accumulation, reset to zero at the start of each period", fontsize=11, color="#596775")
    axis_style(ax)
    for split in SPLITS:
        group = groups[split]
        cumulative = group["CORR"].cumsum()
        # Include a zero anchor immediately before the first scored date.
        dates = pd.concat([pd.Series([group["date"].iloc[0] - pd.Timedelta(days=1)]), group["date"]], ignore_index=True)
        values = np.concatenate([[0.0], cumulative.to_numpy()])
        ax.plot(dates, values, color=COLORS[split], linestyle=STYLES[split], linewidth=2,
                label=f"{LABELS[split]} ({len(group)} dates)")
        ax.plot(group["date"].iloc[-1], cumulative.iloc[-1], "o", color=COLORS[split], markersize=5)
        ax.annotate(f"{cumulative.iloc[-1]:.3f}", (group["date"].iloc[-1], cumulative.iloc[-1]),
                    xytext=(-7, 9), textcoords="offset points", ha="right", color=COLORS[split], fontsize=11)
    ax.set_ylabel("Additive cumulative CORR")
    ax.set_xlabel("Observation date", labelpad=10)
    date_ticks(ax, scores["date"].min() - pd.Timedelta(days=1), scores["date"].max())
    # At most six calendar ticks keep the complete historical range legible.
    start, end = scores["date"].min() - pd.Timedelta(days=1), scores["date"].max()
    interior = [pd.Timestamp(year=year, month=1, day=1) for year in range(start.year + 1, end.year + 1, 2)
                if (pd.Timestamp(year=year, month=1, day=1) - start).days > 110
                and (end - pd.Timestamp(year=year, month=1, day=1)).days > max(200, (end - start).days * 0.12)]
    ax.set_xticks(mdates.date2num([start, *interior, end]))
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=10, handlelength=3)
    fig.text(0.09, 0.065, "Historical development/reproduction: the holdout was observed in earlier experiments.", fontsize=10, color="#596775")
    fig.text(0.09, 0.036, "The periods cover different dates. Cumulative CORR is not a portfolio value or investment return.", fontsize=10, color="#596775")
    fig.savefig(figures / "cumulative_corr.png", dpi=180)
    plt.close(fig)

    metadata = {
        "source_files": {name: sha256(output / name) for name in ["per_date_metrics.csv", "metrics.csv", "run.json"]},
        "plot_script_sha256": sha256(Path(__file__)),
        "model": MODEL, "metric": "CORR", "aggregation": "one score per complete observation date",
        "figure_files": {name: sha256(figures / name) for name in ["per_date_corr.png", "cumulative_corr.png"]},
        "cumulative_definition": "Within each split, chronological additive sum of per-date CORR; reset to zero. Not an investment return.",
        "holdout_status": "Previously observed in earlier project experiments; historical development/reproduction evidence.",
        "date_counts": {split: len(groups[split]) for split in SPLITS},
    }
    (output / "figure_sources.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved baseline figures: {figures}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/xinchen_baseline_20261030")
    args = parser.parse_args()
    render(args.output.resolve())


if __name__ == "__main__":
    main()
