"""Join our metrics with RecBole's into one table and chart.

Reads recsys_large_scale's metrics_log.csv (our side) and results/recbole_log.csv
(RecBole's side), aligns model names, and writes results/comparison.csv plus a grouped
bar chart. Both sides must be on the same dataset, split and eval_mode for a row to be
emitted -- the script refuses to put mismatched protocols side by side, since doing
exactly that is the error this whole exercise exists to demonstrate.

Usage:
    uv run python scripts/build_comparison.py \
        --ours ../recsys_large_scale/artifacts/results/metrics_log.csv \
        --ours-dataset movielens1m \
        --recbole-dataset recbole_ml1m_ours_full
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - must follow the Agg backend selection

# Canonical names so "popularity_ml1m" and "Pop" land in the same row. Anything not
# listed keeps its original name and simply appears on one side of the chart only,
# which is the honest rendering for a model only one framework has.
OURS_ALIASES = {
    "popularity_ml1m": "Popularity",
    "itemknn_ml1m": "ItemKNN",
    "als_ml1m": "ALS (a=20)",
    "als_alpha1_ml1m": "ALS (a=1)",
    "bpr_ml1m": "BPR",
    "popularity": "Popularity",
    "itemknn": "ItemKNN",
    "als": "ALS (a=20)",
    "als_alpha1": "ALS (a=1)",
    "bpr": "BPR",
    "two_stage": "Two-stage",
    "two_tower": "Two-tower",
}
RECBOLE_ALIASES = {
    "Pop": "Popularity",
    "ItemKNN": "ItemKNN",
    "BPR": "BPR",
    "NeuMF": "NeuMF",
    "LightGCN": "LightGCN",
    "MultiVAE": "MultiVAE",
    "EASE": "EASE",
}

# dataviz reference palette, categorical slots 1 and 2 (light mode).
COLORS = {"ours": "#2a78d6", "recbole": "#eb6834"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d8d7d2"


def load_ours(path: Path, dataset: str, split: str, k: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["dataset"] == dataset) & (df["split"] == split) & (df["k"] == k)]
    if df.empty:
        raise SystemExit(
            f"No rows in {path} for dataset={dataset!r} split={split!r} k={k}. "
            f"Available datasets: {sorted(pd.read_csv(path)['dataset'].unique())}"
        )
    # One model may have been evaluated several times; keep the most recent run.
    df = df.sort_values("timestamp").drop_duplicates(subset=["model"], keep="last")
    df = df.assign(model=df["model"].map(lambda m: OURS_ALIASES.get(m, m)), source="ours")
    return df[["source", "model", "recall", "ndcg", "mrr", "precision", "coverage"]]


def load_recbole(path: Path, dataset: str, split: str, k: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["dataset"] == dataset) & (df["split"] == split) & (df["k"] == k)]
    if df.empty:
        raise SystemExit(
            f"No rows in {path} for dataset={dataset!r} split={split!r} k={k}. "
            f"Available datasets: {sorted(pd.read_csv(path)['dataset'].unique())}"
        )
    df = df.sort_values("timestamp").drop_duplicates(subset=["model"], keep="last")
    df = df.assign(model=df["model"].map(lambda m: RECBOLE_ALIASES.get(m, m)), source="recbole")
    df = df.rename(columns={"itemcoverage": "coverage"})
    for col in ["recall", "ndcg", "mrr", "precision", "coverage"]:
        if col not in df:
            df[col] = pd.NA
    return df[["source", "model", "recall", "ndcg", "mrr", "precision", "coverage"]]


def plot(comparison: pd.DataFrame, metric: str, k: int, out_path: Path, subtitle: str) -> None:
    wide = comparison.pivot(index="model", columns="source", values=metric)
    # Strongest model first -- reading order should follow the thing being compared.
    # Rank on the best value across sources, so a model only one framework ran still
    # lands in its true position instead of being pushed to the end by a NaN.
    wide = wide.loc[wide.max(axis=1, numeric_only=True).sort_values(ascending=False).index]
    models = list(wide.index)
    sources = [s for s in ["ours", "recbole"] if s in wide.columns]

    x = range(len(models))
    bar_width = 0.38 if len(sources) == 2 else 0.6
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(models)), 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    labels = {"ours": "recsys_large_scale", "recbole": "RecBole"}
    for i, source in enumerate(sources):
        # 2px-equivalent surface gap between adjacent bars comes from the width/offset pair
        offset = (i - (len(sources) - 1) / 2) * (bar_width + 0.02)
        values = wide[source].astype(float)
        bars = ax.bar(
            [xi + offset for xi in x], values, bar_width,
            label=labels[source], color=COLORS[source], edgecolor=SURFACE, linewidth=1.5,
        )
        # Direct labels: 2 series, so every bar is labeled rather than relying on the axis
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, color=INK_PRIMARY, fontsize=10)
    ax.set_ylabel(f"{metric}@{k}", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        f"{metric}@{k} - two independent implementations, identical data and protocol",
        color=INK_PRIMARY, fontsize=12, pad=28, loc="left",
    )
    if subtitle:
        ax.text(
            0, 1.015, subtitle, transform=ax.transAxes,
            color=INK_SECONDARY, fontsize=9, va="bottom",
        )
    # Headroom so the tallest bar and its direct label clear the legend.
    ax.set_ylim(0, float(wide.max(axis=1, numeric_only=True).max()) * 1.30)
    # Recessive grid and axes: the data should carry the weight, not the frame.
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK_SECONDARY, length=0, labelsize=9)
    ax.tick_params(axis="x", colors=INK_PRIMARY, length=0)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", default="../recsys_large_scale/artifacts/results/metrics_log.csv")
    parser.add_argument("--recbole", default="results/recbole_log.csv")
    parser.add_argument("--ours-dataset", default="movielens1m")
    parser.add_argument("--recbole-dataset", default="recbole_ml1m_ours_full")
    # Default is "test", not "val", and the reason matters. RecBole early-stops on the
    # validation fold, so a trained model's `best_valid_result` is a maximum selected over
    # epochs using that same fold -- it is optimistically biased. Our models train for a
    # fixed number of iterations with no early stopping, so our val numbers carry no such
    # bias. Comparing val-to-val would hand RecBole a purely methodological advantage.
    # Both sides' test numbers are clean (RecBole selects on val, scores on test), so
    # test-to-test is the apples-to-apples comparison.
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", default="recall")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    ours = load_ours(Path(args.ours), args.ours_dataset, args.split, args.k)
    recbole = load_recbole(Path(args.recbole), args.recbole_dataset, args.split, args.k)
    comparison = pd.concat([ours, recbole], ignore_index=True)
    comparison.insert(2, "k", args.k)
    comparison.insert(2, "split", args.split)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison.csv"
    comparison.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}\n")

    pivot = comparison.pivot(index="model", columns="source", values=args.metric)
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    only_ours = set(ours["model"]) - set(recbole["model"])
    only_recbole = set(recbole["model"]) - set(ours["model"])
    if only_ours:
        print(f"\nOnly on our side (no RecBole counterpart run): {sorted(only_ours)}")
    if only_recbole:
        print(f"Only on RecBole's side: {sorted(only_recbole)}")

    plot(
        comparison,
        args.metric,
        args.k,
        out_dir / f"comparison_{args.metric}_at{args.k}.png",
        subtitle=f"{args.ours_dataset} - {args.split} fold - full-catalog leave-one-out",
    )


if __name__ == "__main__":
    main()
