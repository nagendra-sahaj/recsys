from dataclasses import dataclass

import numpy as np
import polars as pl

from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IdMaps:
    """Bidirectional mapping between raw dataset IDs and contiguous 0-indexed indices.
    Raw IDs are ints for MovieLens (userId/movieId) but H&M's customer_id is a 64-char
    hex string, so the key type varies by dataset -- only used as opaque dict keys/JSON
    round-trips, never arithmetic, so this isn't type-checked more strictly than that."""

    user_id_to_idx: dict[int | str, int]
    item_id_to_idx: dict[int | str, int]

    @property
    def n_users(self) -> int:
        return len(self.user_id_to_idx)

    @property
    def n_items(self) -> int:
        return len(self.item_id_to_idx)


def to_implicit_feedback(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Convert explicit ratings to implicit positive feedback per the config's
    `positive_feedback` block. Returns (user_id, item_id, timestamp, confidence)."""
    pf = config["positive_feedback"]
    if pf["mode"] == "threshold":
        out = df.filter(pl.col("rating") >= pf["min_rating"])
        out = out.with_columns(pl.lit(1.0).alias("confidence"))
    elif pf["mode"] == "all":
        alpha = pf["confidence_alpha"]
        out = df.with_columns((1 + alpha * pl.col("rating")).alias("confidence"))
    else:
        raise ValueError(f"unknown positive_feedback.mode: {pf['mode']!r}")

    before = len(df)
    after = len(out)
    logger.info(
        f"positive_feedback.mode={pf['mode']}: kept {after:,}/{before:,} rows "
        f"({100 * after / before:.1f}%)"
    )
    return out.select(["user_id", "item_id", "timestamp", "confidence"])


def hm_to_implicit_feedback(df: pl.DataFrame) -> pl.DataFrame:
    """H&M transactions are already implicit: every row is a purchase, so (unlike
    MovieLens's explicit star ratings) there's no threshold to apply -- every row is a
    positive with confidence=1.0. Exists purely to match to_implicit_feedback's output
    shape (user_id, item_id, timestamp, confidence) so both datasets feed the same
    downstream filter_min_interactions/build_id_maps/splits pipeline unchanged."""
    return df.with_columns(pl.lit(1.0).alias("confidence")).select(
        ["user_id", "item_id", "timestamp", "confidence"]
    )


def subsample_by_date_window(df: pl.DataFrame, window_days: int) -> pl.DataFrame:
    """Keep only the most recent `window_days` of the dataset's own timestamp range.
    H&M's full ~2-year, 31.8M-row transaction history is far more than this project
    needs to clear its >=1,000,000-row target on a laptop; taking a contiguous recent
    slice (rather than a random sample of rows) keeps the temporal splits downstream
    meaningful -- a random sample would scramble the "who bought what, in what order"
    structure that leave-one-out and the global time split both depend on."""
    max_ts = df["timestamp"].max()
    cutoff = max_ts - window_days * 86_400
    out = df.filter(pl.col("timestamp") >= cutoff)
    logger.info(f"date window ({window_days}d): kept {len(out):,}/{len(df):,} rows ({100 * len(out) / len(df):.1f}%)")
    return out


def filter_min_interactions(df: pl.DataFrame, min_user: int, min_item: int) -> pl.DataFrame:
    """Iteratively drop users/items below the interaction threshold, since removing
    one side can push counts on the other side below threshold too."""
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df.group_by("user_id").len()
        keep_users = user_counts.filter(pl.col("len") >= min_user)["user_id"]
        df = df.filter(pl.col("user_id").is_in(keep_users))

        item_counts = df.group_by("item_id").len()
        keep_items = item_counts.filter(pl.col("len") >= min_item)["item_id"]
        df = df.filter(pl.col("item_id").is_in(keep_items))

    logger.info(
        f"after min_user_interactions={min_user}, min_item_interactions={min_item}: "
        f"{len(df):,} rows, {df['user_id'].n_unique():,} users, {df['item_id'].n_unique():,} items"
    )
    return df


def build_id_maps(df: pl.DataFrame) -> IdMaps:
    user_ids = sorted(df["user_id"].unique().to_list())
    item_ids = sorted(df["item_id"].unique().to_list())
    return IdMaps(
        user_id_to_idx={uid: i for i, uid in enumerate(user_ids)},
        item_id_to_idx={iid: i for i, iid in enumerate(item_ids)},
    )


def compute_item_counts(df: pl.DataFrame, n_items: int) -> np.ndarray:
    """Interaction count per item index, as a dense (n_items,) float64 array --
    the shared building block for the popularity model and for the evaluation
    harness's novelty metric (both need "how popular is item i" over the same data)."""
    counts = np.zeros(n_items, dtype=np.float64)
    item_counts = df.group_by("item_idx").len()
    counts[item_counts["item_idx"].to_numpy()] = item_counts["len"].to_numpy()
    return counts


def compute_user_counts(df: pl.DataFrame, n_users: int) -> np.ndarray:
    """Interaction count per user index, as a dense (n_users,) float64 array -- the
    user-side mirror of compute_item_counts, used as a "how active is this user"
    feature for the Phase 3 reranker."""
    counts = np.zeros(n_users, dtype=np.float64)
    user_counts = df.group_by("user_idx").len()
    counts[user_counts["user_idx"].to_numpy()] = user_counts["len"].to_numpy()
    return counts


def apply_id_maps(df: pl.DataFrame, id_maps: IdMaps) -> pl.DataFrame:
    user_map_df = pl.DataFrame(
        {"user_id": list(id_maps.user_id_to_idx.keys()), "user_idx": list(id_maps.user_id_to_idx.values())}
    )
    item_map_df = pl.DataFrame(
        {"item_id": list(id_maps.item_id_to_idx.keys()), "item_idx": list(id_maps.item_id_to_idx.values())}
    )
    return df.join(user_map_df, on="user_id").join(item_map_df, on="item_id")
