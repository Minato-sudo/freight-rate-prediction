"""Figures for the written report.

Run with:  python -m src.figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import REPORTS
from .features import LINEAR_FEATURES, city_coordinates, daily_market_table, engineer, load_frames
from .model import _design, fit_trimmed_ols

# Validated categorical slots 1-3 plus neutral ink.
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

plt.rcParams.update({
    # Matches the Charter family used by the LaTeX report, so figures and body
    # text share one typeface.
    "font.family": "serif",
    "font.serif": ["Bitstream Charter", "Charter", "DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": "#9a9a95",
    "axes.labelcolor": MUTED,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def _style(axis):
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)


def figure_corruption(ratio: np.ndarray, path):
    figure, axis = plt.subplots(figsize=(7.0, 2.8))
    bins = np.linspace(0, 6, 121)
    clean = (ratio > 0.7) & (ratio < 1.4)
    axis.hist(ratio[clean], bins=bins, color=BLUE, label="kept for training")
    axis.hist(ratio[~clean], bins=bins, color=RED, label="excluded (corrupted label)")
    axis.set_yscale("log")
    axis.set_xlabel("posted rate divided by model expectation")
    axis.set_ylabel("loads (log scale)")
    axis.set_xlim(0, 6)
    for edge in (0.7, 1.4):
        axis.axvline(edge, color=MUTED, linewidth=1.0, linestyle="--")
    for position, text in [(0.575, "empty: 0.50 to 0.86"), (1.78, "empty: 1.21 to 2.19")]:
        axis.annotate(text, xy=(position, 1.5), rotation=90, ha="center",
                      va="bottom", fontsize=7.5, color=MUTED)
    axis.legend(frameon=False, fontsize=8, loc="upper right",
                bbox_to_anchor=(1.0, 1.02))
    _style(axis)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def figure_trend(monthly: pd.Series, path):
    figure, axis = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(len(monthly))
    values = monthly.to_numpy() * 100
    slope, intercept = np.polyfit(x, values, 1)
    axis.plot(x, slope * x + intercept, color=ORANGE, linewidth=2.0,
              label="linear trend carried into Nov and Dec")
    axis.plot(x, values, color=BLUE, linewidth=2.0, marker="o", markersize=5,
              label="monthly mean residual")
    axis.axhline(0, color=MUTED, linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([m[-2:] for m in monthly.index], fontsize=8)
    axis.set_xlabel("month of 2025")
    axis.set_ylabel("residual (%)")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    _style(axis)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def figure_split(results: dict, path):
    figure, axis = plt.subplots(figsize=(7.0, 2.8))
    random_folds = [r["clean_MAE"] for r in results["random_kfold"]]
    forward_folds = [r["clean_MAE"] for r in results["forward_chaining"]]
    labels = ["random K-fold\n(what it looks like)",
              "forward chaining\n(what it actually is)"]
    means = [np.mean(random_folds), np.mean(forward_folds)]
    axis.bar(labels, means, color=[AQUA, BLUE], width=0.4, zorder=2)
    for i, folds in enumerate([random_folds, forward_folds]):
        axis.scatter(np.full(len(folds), i + 0.14), folds, color=INK, s=13,
                     zorder=3, label="individual folds" if i == 0 else None)
    for i, value in enumerate(means):
        axis.annotate(f"${value:,.0f}", (i - 0.1, value / 2), ha="center",
                      va="center", fontsize=11, fontweight="bold", color="white")
    axis.set_ylabel("holdout MAE ($)")
    axis.set_ylim(0, max(forward_folds) * 1.18)
    axis.set_xlim(-0.6, 1.6)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    _style(axis)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def figure_ablation(table: pd.DataFrame, path):
    data = table.sort_values("MAE")
    figure, axis = plt.subplots(figsize=(7.0, 2.9))
    colors = [BLUE if v.strip() == "full hybrid" else "#b9c6d4" for v in data.variant]
    names = []
    for value in data.variant:
        text = value.strip()
        names.append("without " + text[5:].strip() if text.startswith("- no ")
                     else text)
    bars = axis.barh(names, data.MAE, color=colors, height=0.6)
    for bar, value in zip(bars, data.MAE):
        axis.annotate(f"{value:,.0f}", (value, bar.get_y() + bar.get_height() / 2),
                      textcoords="offset points", xytext=(4, 0), va="center",
                      fontsize=8.5, color=INK)
    axis.set_xlabel("holdout MAE in dollars (Sep to Oct holdout; lower is better)")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.set_xlim(0, data.MAE.max() * 1.15)
    axis.tick_params(axis="y", labelsize=8.5)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main():
    import json

    REPORTS.mkdir(parents=True, exist_ok=True)
    train_raw, validation_raw = load_frames()
    daily = daily_market_table(train_raw, validation_raw)
    weight_fill = float(train_raw["weight"].abs().median())
    train = engineer(train_raw, daily, weight_fill=weight_fill)
    target = np.log(train["posted_rate"].to_numpy(dtype=float))

    coefficients, _ = fit_trimmed_ols(train, target, LINEAR_FEATURES)
    ratio = np.exp(target - _design(train, LINEAR_FEATURES) @ coefficients)
    figure_corruption(ratio, REPORTS / "fig_corruption.png")

    timeless = [c for c in LINEAR_FEATURES if c != "days_elapsed"]
    coefficients, keep = fit_trimmed_ols(train, target, timeless)
    residual = target - _design(train, timeless) @ coefficients
    monthly = pd.Series(residual[keep]).groupby(
        train.loc[keep].date.dt.to_period("M").astype(str).to_numpy()).mean()
    figure_trend(monthly, REPORTS / "fig_trend.png")

    results = json.loads((REPORTS / "validation_results.json").read_text())
    figure_split(results, REPORTS / "fig_split.png")
    figure_ablation(pd.read_csv(REPORTS / "ablation.csv"), REPORTS / "fig_ablation.png")
    print("wrote 4 figures to", REPORTS)


if __name__ == "__main__":
    main()
