"""Segment-level evaluation: slice each model's leave-one-out test accuracy by user
activity level, age bucket, and club membership status, instead of reporting one global
number that can hide very different experiences for different kinds of users. Reuses
the models already trained in Phase 5b part 1 (loo_train-fitted) unchanged -- no
retraining needed, since this is about *slicing* an existing evaluation, not a
different training condition (that's what run_coldstart_eval.py is for).

Usage:
    uv run python scripts/run_segment_eval.py \
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

import numpy as np
import polars as pl

from recsys_large_scale.data.loaders import load_hm_customers
from recsys_large_scale.data.preprocess import compute_item_counts
from recsys_large_scale.evaluation.harness import evaluate
from recsys_large_scale.evaluation.segments import bucket_ages, bucket_by_activity, map_customer_attribute
from recsys_large_scale.utils.config import load_config
from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)

SEGMENT_LOG_FIELDS = [
    "timestamp",
    "dataset",
    "model",
    "dimension",
    "segment",
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

MODELS = ["popularity_hm", "itemknn_hm", "als_hm", "bpr_hm"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--segment-log", default="artifacts/results/segment_log.csv")
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    eval_config = load_config(args.eval_config)
    processed_dir = Path(data_config["processed_dir"])
    raw_dir = Path(data_config["raw_dir"])

    train = pl.read_parquet(processed_dir / "loo_train.parquet")
    test = pl.read_parquet(processed_dir / "loo_test.parquet")
    id_maps = json.loads((processed_dir / "id_maps.json").read_text())
    n_users, n_items = id_maps["n_users"], id_maps["n_items"]
    user_id_to_idx = id_maps["user_id_to_idx"]

    item_counts = compute_item_counts(train, n_items)
    item_pop_prob = item_counts / item_counts.sum()

    logger.info("Building segment assignments (activity, age, membership)...")
    activity = bucket_by_activity(train, n_users)

    customers = load_hm_customers(raw_dir, data_config["customers_file"])
    ages = map_customer_attribute(customers, user_id_to_idx, n_users, "age")
    age_bucket = bucket_ages(ages)
    membership_raw = map_customer_attribute(customers, user_id_to_idx, n_users, "club_member_status")
    membership = np.array(["unknown" if v is None else v for v in membership_raw], dtype=object)

    dimensions = {"activity": activity, "age": age_bucket, "membership": membership}

    log_path = Path(args.segment_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEGMENT_LOG_FIELDS)
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

            for dim_name, labels in dimensions.items():
                labels_df = pl.DataFrame({"user_idx": np.arange(n_users), "segment": labels.astype(str)})
                test_with_segment = test.join(labels_df, on="user_idx")

                for segment_value in sorted(test_with_segment["segment"].unique().to_list()):
                    segment_eval_set = test_with_segment.filter(pl.col("segment") == segment_value).drop("segment")
                    if len(segment_eval_set) == 0:
                        continue

                    rows = evaluate(
                        model=model,
                        train=train,
                        eval_set=segment_eval_set,
                        n_users=n_users,
                        n_items=n_items,
                        item_pop_prob=item_pop_prob,
                        k_values=eval_config["k_values"],
                        batch_size=eval_config.get("batch_size", 2000),
                        item_embeddings=item_embeddings,
                    )
                    for row in rows:
                        logger.info(
                            f"{model_name} [{dim_name}={segment_value}, n={row.n_users:,}] "
                            f"k={row.k:>3} recall={row.recall:.4f} ndcg={row.ndcg:.4f}"
                        )
                        writer.writerow(
                            {
                                "timestamp": timestamp,
                                "dataset": data_config["dataset"],
                                "model": model_name,
                                "dimension": dim_name,
                                "segment": str(segment_value),
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

    logger.info(f"Wrote segment results to {log_path}")


if __name__ == "__main__":
    main()
