"""Fit a recommender model on the processed training split and save it.

Usage:
    uv run python scripts/train_model.py \
        --data-config configs/data_movielens25m.yaml \
        --model-config configs/model_popularity.yaml
"""

import argparse
import json
import pickle
from pathlib import Path

import polars as pl

from recsys_large_scale.models.als import ALSRecommender
from recsys_large_scale.models.bpr import BPRRecommender
from recsys_large_scale.models.itemknn import ItemKNNRecommender
from recsys_large_scale.models.popularity import PopularityRecommender
from recsys_large_scale.utils.config import load_config
from recsys_large_scale.utils.logging import get_logger
from recsys_large_scale.utils.timers import Timer

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "popularity": PopularityRecommender,
    "itemknn": ItemKNNRecommender,
    "als": ALSRecommender,
    "bpr": BPRRecommender,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument(
        "--train-file",
        default="loo_train.parquet",
        help="Which processed split to fit on -- e.g. global_train.parquet instead of "
        "the default loo_train.parquet, for models used in the cold-start comparison "
        "(see README's Phase 5b RUNBOOK, part 2), since global_time_split's holdout is "
        "the only split with genuinely unseen-at-train-time users to test against.",
    )
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    processed_dir = Path(data_config["processed_dir"])

    train = pl.read_parquet(processed_dir / args.train_file)
    id_maps = json.loads((processed_dir / "id_maps.json").read_text())
    n_users, n_items = id_maps["n_users"], id_maps["n_items"]

    model_type = model_config["model"]
    # "name" lets multiple configs of the same model type (e.g. an ALS hyperparameter
    # comparison) save/log under distinct labels instead of overwriting each other
    label = model_config.get("name", model_type)
    model = MODEL_REGISTRY[model_type](**model_config.get("params", {}))

    timer = Timer()
    with timer.measure("fit"):
        model.fit(train, n_users, n_items)
    logger.info(timer.report())

    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved fitted model to {out_path}")


if __name__ == "__main__":
    main()
