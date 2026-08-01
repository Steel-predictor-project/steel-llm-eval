"""Generate charts for the docs / GitHub Pages site from the scored results.

Outputs PNGs into docs/assets/. Reads results/scores.csv, results/raw_*.csv,
data/benchmark.csv and data/reference_model_predictions.csv.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "docs" / "assets"
BENCH = pd.read_csv(ROOT / "data" / "benchmark.csv")

INK = "#1a1a2e"
ACCENT = "#e94560"
BLUE = "#0f3460"
GREY = "#b0b0b8"
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#eeeeee",
    "axes.axisbelow": True,
})


def short(name: str) -> str:
    return name.split("/")[-1] if "/" in name else name


def load_runs():
    runs = {}
    for p in sorted(glob.glob(str(RESULTS / "raw_*.csv"))):
        df = pd.read_csv(p)
        runs[str(df["model"].iloc[0])] = df
    runs["steel-predictor (reference ML)"] = pd.read_csv(ROOT / "data" / "reference_model_predictions.csv")
    return runs


def chart_mean_leaderboard(scores):
    s = scores[scores["mean_spearman"].notna()].sort_values("mean_spearman")
    labels = [short(m) for m in s["model"]]
    vals = s["mean_spearman"].to_numpy()
    colors = [ACCENT if "reference" in m else BLUE for m in s["model"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean Spearman ρ vs. lab measurements (higher = better)")
    ax.set_title("Overall: ranking steels by predicted properties", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "leaderboard_mean.png")
    plt.close(fig)


def chart_grouped(scores):
    s = scores[scores["mean_spearman"].notna()].sort_values("mean_spearman", ascending=False)
    labels = [short(m) for m in s["model"]]
    edge = s["edge_retention_spearman"].to_numpy()
    tough = s["toughness_spearman"].to_numpy()
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w / 2, edge, w, label="Edge retention (CATRA)", color=BLUE)
    ax.bar(x + w / 2, tough, w, label="Toughness (Charpy)", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(0, 1.05)
    ax.set_title("Edge retention is easy; toughness is hard", color=INK, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "edge_vs_tough_bars.png")
    plt.close(fig)


def chart_scatter(runs, model, prop, truth_col, fname, title):
    df = runs[model].merge(BENCH[["steel_name", truth_col]], on="steel_name", how="inner")
    df = df[df[prop].notna() & df[truth_col].notna()]
    x = df[truth_col].to_numpy(dtype=float)
    y = df[prop].to_numpy(dtype=float)
    rho = spearmanr(y, x).correlation
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=42, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.6)
    # annotate a few notable steels
    for name in ["Maxamet", "CPM MagnaCut", "CPM 3V", "S110V", "15V", "1095", "AEB-L", "CPM S30V"]:
        r = df[df["steel_name"] == name]
        if len(r):
            ax.annotate(name, (float(r[truth_col].iloc[0]), float(r[prop].iloc[0])),
                        fontsize=7.5, color=INK, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(f"Measured: {truth_col}")
    ax.set_ylabel(f"{short(model)} predicted score (1-10)")
    ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / fname)
    plt.close(fig)


def chart_edge_tough_map(scores):
    s = scores[scores["mean_spearman"].notna()]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for _, r in s.iterrows():
        is_ref = "reference" in r["model"]
        ax.scatter(r["edge_retention_spearman"], r["toughness_spearman"],
                   s=90, color=ACCENT if is_ref else BLUE, zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(short(r["model"]), (r["edge_retention_spearman"], r["toughness_spearman"]),
                    fontsize=8, xytext=(6, -2), textcoords="offset points", color=INK)
    ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1, zorder=1)
    ax.set_xlabel("Edge-retention ρ (CATRA)")
    ax.set_ylabel("Toughness ρ (Charpy)")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Every model is better at edge retention than toughness", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "edge_tough_map.png")
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(RESULTS / "scores.csv")
    runs = load_runs()
    chart_mean_leaderboard(scores)
    chart_grouped(scores)
    chart_edge_tough_map(scores)
    chart_scatter(runs, "anthropic/claude-sonnet-5", "edge_retention", "catra_tcc_mm",
                  "scatter_edge_claude.png", "Claude Sonnet — edge retention vs. CATRA")
    chart_scatter(runs, "anthropic/claude-sonnet-5", "toughness", "charpy_ftlbs",
                  "scatter_tough_claude.png", "Claude Sonnet — toughness vs. Charpy")
    print("charts written to", ASSETS)
    for p in sorted(ASSETS.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
