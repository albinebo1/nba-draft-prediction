"""
Exploratory Data Analysis figures for the NBA Draft Bust/Star task.

Generates five plots saved to ``outputs/figures/``:
  1. eda_class_distribution.png — class balance bar chart
  2. eda_pick_vs_ws.png         — draft pick number vs career WS, colored by label
  3. eda_boxplots.png           — boxplots of key features by class
  4. eda_correlation.png        — correlation heatmap of numeric features
  5. eda_missingness.png        — missing-data heatmap

Can be run standalone (``python src/eda.py``) or imported by ``main.py``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "nba_draft_processed.csv"

CLASS_ORDER = ["Bust", "Solid", "Star"]
CLASS_COLORS = {"Bust": "#d95f02", "Solid": "#7570b3", "Star": "#1b9e77"}

NUMERIC_FEATURES = [
    "draft_pick",
    "height_no_shoes",
    "weight",
    "wingspan",
    "standing_reach",
    "hand_length",
    "hand_width",
    "standing_vertical",
    "vertical_leap_max",
    "lane_agility",
    "shuttle_run",
    "sprint",
    "bench_press",
]


def _save(name: str) -> Path:
    out = FIG_DIR / name
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_class_distribution(df: pd.DataFrame) -> Path:
    counts = df["career_label"].value_counts().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(counts.index, counts.values, color=[CLASS_COLORS[c] for c in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{int(val)}\n({val/counts.sum()*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Class distribution — career_label")
    ax.set_ylabel("Players")
    ax.set_ylim(0, counts.values.max() * 1.18)
    sns.despine()
    return _save("eda_class_distribution.png")


def plot_pick_vs_ws(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for cls in CLASS_ORDER:
        sub = df[df["career_label"] == cls]
        ax.scatter(sub["draft_pick"], sub["career_ws"], alpha=0.55, s=20,
                   color=CLASS_COLORS[cls], label=cls)
    ax.set_xlabel("Draft pick (#1 = first overall)")
    ax.set_ylabel("Career Win Shares")
    ax.set_title("Draft pick vs career Win Shares, colored by label")
    ax.axhline(50, ls="--", color="gray", alpha=0.5, label="Star threshold (WS ≥ 50)")
    ax.axhline(15, ls=":", color="gray", alpha=0.5, label="Solid threshold (WS ≥ 15)")
    ax.legend(loc="upper right")
    sns.despine()
    return _save("eda_pick_vs_ws.png")


def plot_boxplots(df: pd.DataFrame) -> Path:
    cols = ["draft_pick", "height_no_shoes", "wingspan", "vertical_leap_max"]
    available = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(1, len(available), figsize=(4.0 * len(available), 4.2))
    if len(available) == 1:
        axes = [axes]
    for ax, col in zip(axes, available):
        sns.boxplot(
            data=df, x="career_label", y=col, order=CLASS_ORDER,
            hue="career_label", palette=CLASS_COLORS, legend=False, ax=ax,
        )
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Feature distributions by career label", y=1.02)
    sns.despine()
    return _save("eda_boxplots.png")


def plot_correlation(df: pd.DataFrame) -> Path:
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr = df[numeric].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Numeric feature correlation matrix")
    plt.xticks(rotation=45, ha="right")
    return _save("eda_correlation.png")


def plot_missingness(df: pd.DataFrame) -> Path:
    cols = [c for c in NUMERIC_FEATURES + ["position"] if c in df.columns]
    mask = df[cols].isna().astype(int)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(mask, cbar=False, yticklabels=False, cmap=["#f0f0f0", "#d95f02"], ax=ax)
    ax.set_title("Missing data (orange = missing) — features × players")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Player (row order)")
    plt.xticks(rotation=45, ha="right")
    return _save("eda_missingness.png")


def run_eda(df: pd.DataFrame | None = None) -> dict[str, Path]:
    """Generate all EDA plots. Returns {name: path} for each saved figure."""
    if df is None:
        df = pd.read_csv(PROCESSED_CSV)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "class_distribution": plot_class_distribution(df),
        "pick_vs_ws": plot_pick_vs_ws(df),
        "boxplots": plot_boxplots(df),
        "correlation": plot_correlation(df),
        "missingness": plot_missingness(df),
    }
    for name, p in outputs.items():
        print(f"  [eda] {name} -> {p.name}")
    return outputs


if __name__ == "__main__":
    run_eda()
