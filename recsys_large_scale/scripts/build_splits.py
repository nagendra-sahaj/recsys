"""Clean the raw MovieLens ratings, convert to implicit feedback, remap IDs, split, and
write parquet files to `processed_dir`.

Usage:
    uv run python scripts/build_splits.py --config configs/data_movielens25m.yaml
"""

import argparse
import json
from pathlib import Path

from recsys_large_scale.data.loaders import load_movielens1m_ratings, load_movielens_ratings
from recsys_large_scale.data.preprocess import (
    apply_id_maps,
    build_id_maps,
    filter_min_interactions,
    to_implicit_feedback,
)
from recsys_large_scale.data.splits import global_time_split, leave_one_out_split
from recsys_large_scale.utils.config import load_config
from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)

# Which raw-file reader a data config wants. Both return the identical
# (user_id, item_id, rating, timestamp) schema, so only the read step differs.
LOADERS = {
    "movielens_csv": load_movielens_ratings,  # ML-25M and other CSV-era releases
    "movielens_dat": load_movielens1m_ratings,  # ML-1M's "::"-separated ratings.dat
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path(config["raw_dir"])
    processed_dir = Path(config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    loader_name = config.get("loader", "movielens_csv")
    if loader_name not in LOADERS:
        raise ValueError(f"unknown loader {loader_name!r}; expected one of {sorted(LOADERS)}")
    raw = LOADERS[loader_name](raw_dir, config["ratings_file"])
    implicit_df = to_implicit_feedback(raw, config)
    filtered = filter_min_interactions(
        implicit_df,
        min_user=config["min_user_interactions"],
        min_item=config["min_item_interactions"],
    )

    id_maps = build_id_maps(filtered)
    indexed = apply_id_maps(filtered, id_maps)

    split_cfg = config["split"]
    if split_cfg["leave_one_out"]:
        loo = leave_one_out_split(indexed)
        loo.train.write_parquet(processed_dir / "loo_train.parquet")
        loo.val.write_parquet(processed_dir / "loo_val.parquet")
        loo.test.write_parquet(processed_dir / "loo_test.parquet")

    if split_cfg["global_time_split"]:
        gts = global_time_split(indexed, holdout_days=split_cfg["global_holdout_days"])
        gts.train.write_parquet(processed_dir / "global_train.parquet")
        gts.test.write_parquet(processed_dir / "global_test.parquet")

    id_maps_path = processed_dir / "id_maps.json"
    id_maps_path.write_text(
        json.dumps(
            {
                "n_users": id_maps.n_users,
                "n_items": id_maps.n_items,
                "user_id_to_idx": id_maps.user_id_to_idx,
                "item_id_to_idx": id_maps.item_id_to_idx,
            }
        )
    )

    logger.info(f"Done. n_users={id_maps.n_users:,} n_items={id_maps.n_items:,}")
    logger.info(f"Wrote processed splits + id_maps.json to {processed_dir}")


if __name__ == "__main__":
    main()
