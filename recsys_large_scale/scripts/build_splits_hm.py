"""Clean the raw H&M transactions, subsample to a recent date window, convert to
implicit feedback, remap IDs, split, and write parquet files to `processed_dir`. The
H&M-specific counterpart to scripts/build_splits.py -- shares filter_min_interactions/
build_id_maps/apply_id_maps and both split functions with the MovieLens pipeline, since
those are all dataset-agnostic (they only ever look at user_id/item_id/timestamp
columns); only loading, date-window subsampling, and the implicit-feedback conversion
are H&M-specific.

Usage:
    uv run python scripts/build_splits_hm.py --config configs/data_hm.yaml
"""

import argparse
import json
from pathlib import Path

from recsys_large_scale.data.loaders import load_hm_transactions
from recsys_large_scale.data.preprocess import (
    apply_id_maps,
    build_id_maps,
    filter_min_interactions,
    hm_to_implicit_feedback,
    subsample_by_date_window,
)
from recsys_large_scale.data.splits import global_time_split, leave_one_out_split
from recsys_large_scale.utils.config import load_config
from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path(config["raw_dir"])
    processed_dir = Path(config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = load_hm_transactions(raw_dir, config["transactions_file"])
    windowed = subsample_by_date_window(raw, config["date_window_days"])
    implicit_df = hm_to_implicit_feedback(windowed)
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
