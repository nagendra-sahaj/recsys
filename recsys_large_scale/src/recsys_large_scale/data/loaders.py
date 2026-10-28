from pathlib import Path

import polars as pl

from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_file(raw_dir: str | Path, filename: str) -> Path:
    """Find `filename` under `raw_dir`, whether raw_dir is the folder containing it
    directly or a parent that still has the zip's original nested folder (e.g.
    data/raw/movielens25m/ vs data/raw/movielens25m/ml-25m/) -- avoids the setup
    breaking depending on exactly how a dataset's zip is laid out."""
    raw_dir = Path(raw_dir)
    direct = raw_dir / filename
    if direct.exists():
        return direct

    matches = list(raw_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} under {raw_dir} (checked directly and recursively). "
            f"Did you run scripts/download_data.py first?"
        )
    if len(matches) > 1:
        logger.info(f"Multiple {filename!r} found under {raw_dir}, using {matches[0]}")
    return matches[0]


def load_movielens_ratings(raw_dir: str | Path, ratings_file: str = "ratings.csv") -> pl.DataFrame:
    """Load the MovieLens ratings.csv as (user_id, item_id, rating, timestamp)."""
    path = _resolve_file(raw_dir, ratings_file)
    df = pl.read_csv(path)
    df = df.rename({"userId": "user_id", "movieId": "item_id"})
    logger.info(f"Loaded {len(df):,} raw rows from {path}")
    return df


def load_movielens1m_ratings(raw_dir: str | Path, ratings_file: str = "ratings.dat") -> pl.DataFrame:
    """Load MovieLens 1M's ratings.dat as (user_id, item_id, rating, timestamp).

    ML-1M predates the CSV-era MovieLens releases: it ships "::"-separated .dat files
    with no header row, so `load_movielens_ratings` can't read it. Polars separators are
    a single byte, so rather than fight the two-character delimiter we read each line
    whole (using \\x01, a byte that cannot occur in this file, as the "separator") and
    split it explicitly. Output schema is identical to the 25M loader's, so everything
    downstream -- to_implicit_feedback, filter_min_interactions, the splits -- is unchanged.
    """
    path = _resolve_file(raw_dir, ratings_file)
    df = (
        pl.read_csv(
            path,
            separator="\x01",
            has_header=False,
            new_columns=["raw"],
            quote_char=None,
            encoding="utf8-lossy",  # ML-1M's .dat files are latin-1, not UTF-8
        )
        .with_columns(pl.col("raw").str.split("::").alias("fields"))
        .select(
            pl.col("fields").list.get(0).cast(pl.Int64).alias("user_id"),
            pl.col("fields").list.get(1).cast(pl.Int64).alias("item_id"),
            pl.col("fields").list.get(2).cast(pl.Float64).alias("rating"),
            pl.col("fields").list.get(3).cast(pl.Int64).alias("timestamp"),
        )
    )
    logger.info(f"Loaded {len(df):,} raw rows from {path}")
    return df


def load_movielens_movies(raw_dir: str | Path, movies_file: str = "movies.csv") -> pl.DataFrame:
    """Load MovieLens movies.csv as (item_id, title, genres)."""
    path = _resolve_file(raw_dir, movies_file)
    df = pl.read_csv(path)
    df = df.rename({"movieId": "item_id"})
    return df


def load_hm_transactions(raw_dir: str | Path, transactions_file: str = "transactions_train.csv") -> pl.DataFrame:
    """Load H&M transactions_train.csv as (user_id, item_id, timestamp). `t_dat` is a
    day-granularity date string ("2019-06-15"); converted to epoch seconds at midnight
    so it's a plain int timestamp like MovieLens's, and every existing timestamp-based
    function (splits.py's rank/cutoff logic) works unchanged. Every row here is already
    a purchase (implicit positive) -- see preprocess.hm_to_implicit_feedback."""
    path = _resolve_file(raw_dir, transactions_file)
    df = pl.read_csv(path)
    df = df.rename({"customer_id": "user_id", "article_id": "item_id"})
    df = df.with_columns((pl.col("t_dat").str.to_date().cast(pl.Int32) * 86_400).alias("timestamp"))
    logger.info(f"Loaded {len(df):,} raw rows from {path}")
    return df.select(["user_id", "item_id", "timestamp"])


def load_hm_customers(raw_dir: str | Path, customers_file: str = "customers.csv") -> pl.DataFrame:
    """Load H&M customers.csv as (user_id, ...) -- age, club_member_status,
    fashion_news_frequency, etc., used for Phase 5's segment breakdowns and cold-start
    content baseline, not needed by the core data pipeline itself."""
    path = _resolve_file(raw_dir, customers_file)
    df = pl.read_csv(path)
    return df.rename({"customer_id": "user_id"})


def load_hm_articles(raw_dir: str | Path, articles_file: str = "articles.csv") -> pl.DataFrame:
    """Load H&M articles.csv as (item_id, ...) -- product_type, department, colour,
    section, etc., the H&M equivalent of MovieLens's genres for content-based features."""
    path = _resolve_file(raw_dir, articles_file)
    df = pl.read_csv(path)
    return df.rename({"article_id": "item_id"})
