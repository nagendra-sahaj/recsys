"""Popularity-bias vs. personalization: what fraction of each model's top-K
recommendations come from the catalog's top-1% most popular items (by train
interaction count), reported alongside Recall/NDCG/novelty/coverage in one table so the
accuracy-vs-personalization tradeoff is visible directly, instead of needing to
cross-reference separate metric runs.

Reuses the already-trained Phase 5b part-1 models (loo_train-fitted) and the standard
loo_test evaluation -- this isn't a new training condition, just a new metric
(popularity_bias, now one of evaluation.harness.EvalRow's fields) computed from the
same evaluation pass evaluate() already does for every other metric.

Usage:
    uv run python scripts/run_popularity_bias_eval.py \
        --data-config configs/data_hm.yaml \
        --eval-config configs/eval_default.yaml \
        --models-dir artifacts/models
"""

import argparse
import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from recsys_large_scale.data.preprocess import compute_item_counts
from recsys_large_scale.evaluation.harness import evaluate
from recsys_large_scale.utils.config import load_config
from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)

POPULARITY_BIAS_LOG_FIELDS = [
    "timestamp",
    "dataset",
    "model",
    "k",
    "n_users",
    "recall",
    "ndcg",
    "popularity_bias",
    "novelty",
    "coverage",
]

MODELS = ["popularity_hm", "itemknn_hm", "als_hm", "bpr_hm"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--popularity-bias-log", default="artifacts/results/popularity_bias_log.csv")
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    eval_config = load_config(args.eval_config)
    processed_dir = Path(data_config["processed_dir"])

    train = pl.read_parquet(processed_dir / "loo_train.parquet")
    test = pl.read_parquet(processed_dir / "loo_test.parquet")
    id_maps = json.loads((processed_dir / "id_maps.json").read_text())
    n_users, n_items = id_maps["n_users"], id_maps["n_items"]

    item_counts = compute_item_counts(train, n_items)
    item_pop_prob = item_counts / item_counts.sum()

    log_path = Path(args.popularity_bias_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POPULARITY_BIAS_LOG_FIELDS)
        if write_header:
            writer.writeheader()

        for model_name in MODELS:
            model_path = Path(args.models_dir) / f"{model_name}.pkl"
            if not model_path.exists():
                logger.warning(f"skipping {model_name}: {model_path} not found")
                continue
            with open(model_path, "rb") as mf:
                model = pickle.load(mf)
            item_embeddings = getattr(model, "item_embeddings", None)

            rows = evaluate(
                model=model,
                train=train,
                eval_set=test,
                n_users=n_users,
                n_items=n_items,
                item_pop_prob=item_pop_prob,
                k_values=eval_config["k_values"],
                batch_size=eval_config.get("batch_size", 2000),
                item_embeddings=item_embeddings,
            )
            for row in rows:
                logger.info(
                    f"{model_name} k={row.k:>3} recall={row.recall:.4f} ndcg={row.ndcg:.4f} "
                    f"popularity_bias={row.popularity_bias:.4f} novelty={row.novelty:.2f} "
                    f"coverage={row.coverage:.4f}"
                )
                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "dataset": data_config["dataset"],
                        "model": model_name,
                        "k": row.k,
                        "n_users": row.n_users,
                        "recall": row.recall,
                        "ndcg": row.ndcg,
                        "popularity_bias": row.popularity_bias,
                        "novelty": row.novelty,
                        "coverage": row.coverage,
                    }
                )

    logger.info(f"Wrote popularity-bias results to {log_path}")


if __name__ == "__main__":
    main()
