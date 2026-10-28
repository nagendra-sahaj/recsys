"""Build cold-start-fallback-wrapped variants of the global-split H&M models: for a
user with zero rows in `global_train`, route to a fallback instead of trusting the
base model's own (diagnosed-degenerate, see LEARNINGS.md Phase 5b part 2b) output.

Three fallbacks, all fit on global_train:
  - plain popularity (PopularityRecommender, already used elsewhere in this project)
  - age-bucket popularity (AgeBucketPopularityRecommender: most popular items among
    users in the same age bucket, falling back to global popularity for small/unknown
    buckets) -- a content-based fallback using customers.csv's age column
  - recency-weighted popularity (RecencyWeightedPopularityRecommender: interactions
    weighted by exponential recency decay instead of counted equally) -- a "what's
    trending right now" fallback, motivated by H&M's trend-driven retail structure

Wraps als_hm, itemknn_hm, and bpr_hm (the three models that showed the cold-start
collapse) with each fallback, saving 9 new artifacts. popularity_hm itself is not
wrapped -- it was already immune to cold-start (see part 2b), nothing to fix there.
No base-model retraining happens here; this only fits the two lightweight fallback
models and wraps already-trained artifacts, so it's fast.

Requires the global-split base models from README's Phase 5b RUNBOOK, part 2
(artifacts/models/global_hm/{popularity,itemknn,als,bpr}_hm.pkl) to already exist.

Usage:
    uv run python scripts/train_coldstart_fallback.py \
        --data-config configs/data_hm.yaml \
        --models-dir artifacts/models/global_hm
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from recsys_large_scale.data.loaders import load_hm_customers
from recsys_large_scale.evaluation.segments import bucket_ages, map_customer_attribute
from recsys_large_scale.models.age_bucket_popularity import AgeBucketPopularityRecommender
from recsys_large_scale.models.coldstart_fallback import ColdStartFallbackRecommender
from recsys_large_scale.models.popularity import PopularityRecommender
from recsys_large_scale.models.recency_popularity import RecencyWeightedPopularityRecommender
from recsys_large_scale.utils.config import load_config
from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)

BASE_MODELS = ["itemknn_hm", "als_hm", "bpr_hm"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--models-dir", default="artifacts/models/global_hm")
    parser.add_argument("--min-bucket-interactions", type=int, default=1000)
    parser.add_argument("--recency-half-life-days", type=float, default=7.0)
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    processed_dir = Path(data_config["processed_dir"])
    raw_dir = Path(data_config["raw_dir"])
    models_dir = Path(args.models_dir)

    train = pl.read_parquet(processed_dir / "global_train.parquet")
    id_maps = json.loads((processed_dir / "id_maps.json").read_text())
    n_users, n_items = id_maps["n_users"], id_maps["n_items"]
    user_id_to_idx = id_maps["user_id_to_idx"]

    warm_user_mask = np.zeros(n_users, dtype=bool)
    warm_user_mask[train["user_idx"].unique().to_numpy()] = True
    logger.info(f"{int(warm_user_mask.sum()):,}/{n_users:,} users have >=1 global_train interaction (warm)")

    logger.info("Fitting the plain popularity fallback on global_train...")
    popularity_fallback = PopularityRecommender()
    popularity_fallback.fit(train, n_users, n_items)

    logger.info("Fitting the age-bucket popularity fallback on global_train...")
    customers = load_hm_customers(raw_dir, data_config["customers_file"])
    ages = map_customer_attribute(customers, user_id_to_idx, n_users, "age")
    age_bucket = bucket_ages(ages)
    age_fallback = AgeBucketPopularityRecommender(min_bucket_interactions=args.min_bucket_interactions)
    age_fallback.fit(train, n_users, n_items, age_bucket)
    logger.info(f"age buckets with enough data for their own list: {sorted(age_fallback.bucket_scores.keys())}")

    logger.info(f"Fitting the recency-weighted popularity fallback on global_train (half_life_days={args.recency_half_life_days})...")
    recency_fallback = RecencyWeightedPopularityRecommender(half_life_days=args.recency_half_life_days)
    recency_fallback.fit(train, n_users, n_items)

    for base_name in BASE_MODELS:
        base_path = models_dir / f"{base_name}.pkl"
        if not base_path.exists():
            logger.warning(f"skipping {base_name}: {base_path} not found (train it first, see README)")
            continue
        with open(base_path, "rb") as f:
            base_model = pickle.load(f)

        for fallback_name, fallback_model in [
            ("pop", popularity_fallback),
            ("age", age_fallback),
            ("recency", recency_fallback),
        ]:
            wrapped = ColdStartFallbackRecommender(base_model, fallback_model, warm_user_mask)
            out_path = models_dir / f"{base_name}_coldfix_{fallback_name}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(wrapped, f)
            logger.info(f"Saved {out_path}")

    model_names = ["popularity_hm"]
    for base_name in BASE_MODELS:
        model_names += [base_name, f"{base_name}_coldfix_pop", f"{base_name}_coldfix_age", f"{base_name}_coldfix_recency"]
    logger.info(
        "Next: evaluate these alongside the plain base models, e.g.\n"
        "  uv run python scripts/run_coldstart_eval.py --data-config configs/data_hm.yaml "
        "--eval-config configs/eval_default.yaml --models-dir artifacts/models/global_hm "
        f"--models {','.join(model_names)}"
    )


if __name__ == "__main__":
    main()
