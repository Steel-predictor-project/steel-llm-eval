"""Score all model runs against the objective measurements and build a leaderboard.

Ground truth:
    edge_retention  <- CATRA total-cards-cut / TCC (mm)   (higher = better)
    toughness       <- Charpy impact energy (ft-lbs)      (higher = better)

Metrics (per property, then averaged):
    spearman   : rank correlation between predicted score and measured value
                 (scale-free — the headline metric).
    kendall    : Kendall's tau-b rank correlation.
    pairwise   : fraction of steel pairs ordered correctly (ties excluded).
    norm_mae   : MAE after min-max normalizing the measurement to 1-10
                 (calibration sanity check; secondary).

Baselines included automatically:
    steel-predictor (reference ML)  : the project's purpose-built model — the bar.
    baseline-constant               : predicts the mean for every steel (floor).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "benchmark.csv"
REFERENCE = ROOT / "data" / "reference_model_predictions.csv"
RESULTS = ROOT / "results"

PROPS = {
    "edge_retention": "catra_tcc_mm",
    "toughness": "charpy_ftlbs",
}


def pairwise_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    n = len(truth)
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if truth[i] == truth[j] or pred[i] == pred[j]:
                continue
            total += 1
            if (pred[i] > pred[j]) == (truth[i] > truth[j]):
                correct += 1
    return correct / total if total else float("nan")


def norm_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    lo, hi = truth.min(), truth.max()
    if hi == lo:
        return float("nan")
    truth10 = 1 + 9 * (truth - lo) / (hi - lo)
    return float(np.mean(np.abs(pred - truth10)))


def score_property(preds: pd.DataFrame, bench: pd.DataFrame, prop: str, truth_col: str):
    merged = preds.merge(bench[["steel_name", truth_col]], on="steel_name", how="inner")
    merged = merged[merged[prop].notna() & merged[truth_col].notna()]
    if len(merged) < 4:
        return None
    pred = merged[prop].to_numpy(dtype=float)
    truth = merged[truth_col].to_numpy(dtype=float)
    rho = spearmanr(pred, truth).correlation
    tau = kendalltau(pred, truth).correlation
    return {
        "n": len(merged),
        "spearman": round(float(rho), 3),
        "kendall": round(float(tau), 3),
        "pairwise": round(pairwise_accuracy(pred, truth), 3),
        "norm_mae": round(norm_mae(pred, truth), 3),
    }


def constant_baseline(bench: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in bench.iterrows():
        rows.append({"steel_name": r["steel_name"], "edge_retention": 5.5, "toughness": 5.5})
    return pd.DataFrame(rows)


def collect_runs():
    runs = {}
    for path in sorted(glob.glob(str(RESULTS / "raw_*.csv"))):
        df = pd.read_csv(path)
        label = df["model"].iloc[0] if "model" in df.columns and len(df) else Path(path).stem
        runs[str(label)] = df
    if REFERENCE.exists():
        runs["steel-predictor (reference ML)"] = pd.read_csv(REFERENCE)
    return runs


def main() -> int:
    bench = pd.read_csv(BENCHMARK)
    runs = collect_runs()
    runs["baseline-constant"] = constant_baseline(bench)

    records = []
    for label, preds in runs.items():
        row = {"model": label}
        spearmans = []
        for prop, truth_col in PROPS.items():
            s = score_property(preds, bench, prop, truth_col)
            if s is None:
                continue
            for k, v in s.items():
                row[f"{prop}_{k}"] = v
            spearmans.append(s["spearman"])
        row["mean_spearman"] = round(float(np.mean(spearmans)), 3) if spearmans else float("nan")
        records.append(row)

    scores = pd.DataFrame(records).sort_values("mean_spearman", ascending=False)
    scores.to_csv(RESULTS / "scores.csv", index=False)

    # Leaderboard markdown
    lines = ["# Leaderboard — LLMs predicting knife-steel properties from composition", ""]
    lines.append("Ranked by mean Spearman rank correlation vs. objective lab measurements "
                 "(CATRA edge retention, Charpy toughness). Higher is better; "
                 "1.0 = perfect ordering, 0.0 = random.")
    lines.append("")
    er = "| Model | Edge ρ (n) | Edge pairwise | Tough ρ (n) | Tough pairwise | **Mean ρ** |"
    lines.append(er)
    lines.append("|" + "---|" * 6)
    for _, r in scores.iterrows():
        def g(c):
            return "-" if c not in r or pd.isna(r[c]) else r[c]
        edge_n = "-" if "edge_retention_n" not in r or pd.isna(r["edge_retention_n"]) else int(r["edge_retention_n"])
        tough_n = "-" if "toughness_n" not in r or pd.isna(r["toughness_n"]) else int(r["toughness_n"])
        lines.append(
            f"| {r['model']} | {g('edge_retention_spearman')} ({edge_n}) | "
            f"{g('edge_retention_pairwise')} | {g('toughness_spearman')} ({tough_n}) | "
            f"{g('toughness_pairwise')} | **{g('mean_spearman')}** |"
        )
    lines.append("")
    lines.append(f"_Edge retention n={int(bench['catra_tcc_mm'].notna().sum())} steels (CATRA), "
                 f"toughness n={int(bench['charpy_ftlbs'].notna().sum())} steels (Charpy)._")
    (RESULTS / "leaderboard.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {RESULTS/'leaderboard.md'} and {RESULTS/'scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
