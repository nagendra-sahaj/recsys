"""Export recsys_large_scale's processed interactions as a RecBole atomic file.

This is what makes the head-to-head airtight rather than approximate. Instead of letting
RecBole apply its own filters to its own copy of MovieLens, we hand it the exact
interaction set our pipeline produced -- already thresholded, already 5-core filtered,
already ID-remapped. Both systems then sort by timestamp per user and hold out the last
two interactions, so they are splitting identical data identically.

We export our three folds as three separate RecBole "benchmark" files
(`<name>.train.inter`, `<name>.valid.inter`, `<name>.test.inter`) rather than one
combined `.inter`. That is deliberate and important. With `benchmark_filename` set,
RecBole skips `_data_filtering()` and its own splitting entirely and uses the files
exactly as given (dataset.py:161 and dataset.py:1815), so the comparison controls the
*split* and not merely the data.

The alternative -- exporting one combined file and letting RecBole re-split it with
`eval_args.split: {'LS': 'valid_and_test'}` -- leaves a real confound. ML-1M users rate
in bursts, so many of a user's interactions share a timestamp; our splitter breaks those
ties by row order and RecBole by its own rule, which means the two systems can hold out
*different items for the same user*. Pre-splitting removes that variable rather than
hoping it is small.

Usage:
    uv run python scripts/export_to_recbole.py \
        --processed-dir ../recsys_large_scale/data/processed/movielens1m \
        --name ml1m-ours
"""

import argparse
from pathlib import Path

import pandas as pd

# our fold -> RecBole benchmark suffix. Order matters: RecBole concatenates the files and
# slices them back apart by cumulative length, so train must come first.
SPLIT_FILES = {
    "loo_train.parquet": "train",
    "loo_val.parquet": "valid",
    "loo_test.parquet": "test",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="recsys_large_scale's processed_dir for the dataset, e.g. "
        "../recsys_large_scale/data/processed/movielens1m",
    )
    parser.add_argument("--name", required=True, help="RecBole dataset name, e.g. ml1m-ours")
    parser.add_argument("--data-path", default="dataset", help="RecBole data_path root")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.data_path) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for filename, suffix in SPLIT_FILES.items():
        path = processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/build_splits.py in recsys_large_scale first."
            )
        frames[suffix] = pd.read_parquet(path, columns=["user_idx", "item_idx", "timestamp"])

    combined = pd.concat(frames.values(), ignore_index=True)
    n_dupes = combined.duplicated(subset=["user_idx", "item_idx"]).sum()
    if n_dupes:
        # Would mean the same (user, item) pair appears in more than one fold, which our
        # splits should make impossible -- worth failing loudly rather than exporting it.
        raise ValueError(f"{n_dupes:,} duplicate (user, item) pairs across the three folds")

    # RecBole atomic-file format: tab-separated, header cells carry the field type.
    # user_idx/item_idx are already contiguous 0-based ints from our build_id_maps, so
    # exporting them as tokens guarantees both systems see the identical item universe.
    for suffix, part in frames.items():
        part = part.copy()
        part.columns = ["user_id:token", "item_id:token", "timestamp:float"]
        out_path = out_dir / f"{args.name}.{suffix}.inter"
        part.to_csv(out_path, sep="\t", index=False)
        print(f"  {out_path.name}: {len(part):,} rows")

    print(
        f"\nWrote {len(combined):,} interactions across 3 folds "
        f"({combined['user_idx'].nunique():,} users, {combined['item_idx'].nunique():,} items) "
        f"to {out_dir}/"
    )
    print(
        "The config must set `benchmark_filename: [train, valid, test]` to consume these -- "
        "without it RecBole looks for a single combined .inter and re-splits it itself."
    )


if __name__ == "__main__":
    main()
