"""Evaluate a fitted model on a processed split and append results to the metrics log.

Usage:
    uv run python scripts/evaluate_model.py \
        --data-config configs/data_movielens25m.yaml \
        --eval-config configs/eval_default.yaml \
        --model-artifact artifacts/models/popularity.pkl \
        --split val
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

METRICS_LOG_FIELDS = [
    "timestamp",
    "dataset",
    "model",
    "split",
    "k",
    "n_users",
    "recall",
    "precision",
    "ndcg",
    "mrr",
    "map",
    "coverage",
    "novelty",
    "intra_list_similarity",
]

# Sampled-negative runs go to their own log rather than metrics_log.csv: they need an
# extra eval_mode column, and mixing protocols in one file is exactly the error this
# whole comparison exercise exists to expose.
PROTOCOL_LOG_FIELDS = METRICS_LOG_FIELDS + ["eval_mode"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--eval-mode",
        default="full",
        help="'full' for full-catalog ranking (default, what every existing number uses), "
        "or 'uni<N>' (e.g. uni100) to rank against N sampled negatives instead.",
    )
    parser.add_argument("--metrics-log", default=None)
    args = parser.parse_args()

    is_full = args.eval_mode == "full"
    fields = METRICS_LOG_FIELDS if is_full else PROTOCOL_LOG_FIELDS
    default_log = "metrics_log.csv" if is_full else "protocol_log.csv"
    metrics_log = args.metrics_log or f"artifacts/results/{default_log}"

    data_config = load_config(args.data_config)
    eval_config = load_config(args.eval_config)
    processed_dir = Path(data_config["processed_dir"])

    train = pl.read_parquet(processed_dir / "loo_train.parquet")
    eval_set = pl.read_parquet(processed_dir / f"loo_{args.split}.parquet")
    id_maps = json.loads((processed_dir / "id_maps.json").read_text())
    n_users, n_items = id_maps["n_users"], id_maps["n_items"]

    item_counts = compute_item_counts(train, n_items)
    item_pop_prob = item_counts / item_counts.sum()

    with open(args.model_artifact, "rb") as f:
        model = pickle.load(f)
    model_name = Path(args.model_artifact).stem
    item_embeddings = getattr(model, "item_embeddings", None)

    rows = evaluate(
        model=model,
        train=train,
        eval_set=eval_set,
        n_users=n_users,
        n_items=n_items,
        item_pop_prob=item_pop_prob,
        k_values=eval_config["k_values"],
        batch_size=eval_config.get("batch_size", 2000),
        item_embeddings=item_embeddings,
        eval_mode=args.eval_mode,
        seed=data_config.get("random_seed", 42),
    )

    for row in rows:
        ils = f"{row.intra_list_similarity:.4f}" if row.intra_list_similarity is not None else "n/a"
        logger.info(
            f"k={row.k:>3} recall={row.recall:.4f} precision={row.precision:.4f} "
            f"ndcg={row.ndcg:.4f} mrr={row.mrr:.4f} map={row.map:.4f} "
            f"coverage={row.coverage:.4f} novelty={row.novelty:.4f} intra_list_similarity={ils}"
        )

    log_path = Path(metrics_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **({} if is_full else {"eval_mode": args.eval_mode}),
                    "timestamp": timestamp,
                    "dataset": data_config["dataset"],
                    "model": model_name,
                    "split": args.split,
                    "k": row.k,
                    "n_users": row.n_users,
                    "recall": row.recall,
                    "precision": row.precision,
                    "ndcg": row.ndcg,
                    "mrr": row.mrr,
                    "map": row.map,
                    "coverage": row.coverage,
                    "novelty": row.novelty,
                    "intra_list_similarity": row.intra_list_similarity
                    if row.intra_list_similarity is not None
                    else "",
                }
            )
    logger.info(f"Appended {len(rows)} rows to {log_path}")


if __name__ == "__main__":
    main()
