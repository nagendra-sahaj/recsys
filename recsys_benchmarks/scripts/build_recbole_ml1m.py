"""Build RecBole's `ml-1m` atomic file from the raw GroupLens release.

Why this exists: RecBole auto-downloads its preprocessed datasets from
`recbole.s3-accelerate.amazonaws.com`, and that bucket now returns HTTP 403. The
framework's own dataset mirror has rotted.

Routing around it costs almost nothing, because RecBole's `ml-1m.inter` is not a
derived artifact in any meaningful sense -- it is GroupLens's `ratings.dat` rewritten
into RecBole's tab-separated atomic format with typed column headers. No filtering is
baked into the file; all of that happens at config time via `val_interval` and
`user_inter_num_interval`/`item_inter_num_interval`, which configs/recbole_ml1m_full.yaml
already sets. So rebuilding from GroupLens gives RecBole exactly what it expected to
download, from the authoritative source rather than a third-party mirror.

This keeps B2's meaning intact: RecBole still applies *its own* preprocessing pipeline
to the raw ratings, independently of ours. (B3's `export_to_recbole.py` is the different
one -- that deliberately hands RecBole our already-filtered interactions.)

Usage:
    uv run python scripts/build_recbole_ml1m.py
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

GROUPLENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def read_ratings_dat(raw_text: str) -> pd.DataFrame:
    """ML-1M's ratings.dat is "::"-separated with no header. pandas separators are
    regex-capable, but "::" via the python engine is slow on a million rows, so split
    the lines directly instead."""
    rows = [line.split("::") for line in raw_text.splitlines() if line]
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    return df.astype({"user_id": "int64", "item_id": "int64", "rating": "float64", "timestamp": "int64"})


def load_raw_text(ratings_dat: Path | None) -> str:
    if ratings_dat and ratings_dat.exists():
        print(f"Using local {ratings_dat}")
        # ML-1M's .dat files are latin-1, not UTF-8
        return ratings_dat.read_text(encoding="latin-1")

    print(f"Downloading {GROUPLENS_URL} ...")
    with urllib.request.urlopen(GROUPLENS_URL) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.endswith("ratings.dat"))
        print(f"  extracting {name}")
        return zf.read(name).decode("latin-1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratings-dat",
        default="../recsys_large_scale/data/raw/movielens1m/ml-1m/ratings.dat",
        help="Reuse the raw file B1 already downloaded, if present; otherwise this "
        "script fetches ml-1m.zip from GroupLens itself.",
    )
    parser.add_argument("--data-path", default="dataset")
    parser.add_argument("--name", default="ml-1m")
    args = parser.parse_args()

    df = read_ratings_dat(load_raw_text(Path(args.ratings_dat) if args.ratings_dat else None))

    out_dir = Path(args.data_path) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.inter"

    # RecBole atomic-file header carries the type of each field. These four names must
    # match USER_ID_FIELD/ITEM_ID_FIELD/RATING_FIELD/TIME_FIELD and the `load_col` list
    # in the config, or RecBole silently loads nothing.
    df.columns = ["user_id:token", "item_id:token", "rating:float", "timestamp:float"]
    df.to_csv(out_path, sep="\t", index=False)

    print(
        f"\nWrote {len(df):,} interactions "
        f"({df['user_id:token'].nunique():,} users, {df['item_id:token'].nunique():,} items) "
        f"to {out_path}"
    )
    print("Expected for ML-1M: 1,000,209 interactions, 6,040 users, 3,706 items.")


if __name__ == "__main__":
    main()
