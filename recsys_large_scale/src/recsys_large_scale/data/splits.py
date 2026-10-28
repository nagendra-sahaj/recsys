from dataclasses import dataclass

import polars as pl

from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)

SECONDS_PER_DAY = 86_400


@dataclass
class LeaveOneOutSplit:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


def leave_one_out_split(df: pl.DataFrame) -> LeaveOneOutSplit:
    """Per user, sort by timestamp: the most recent interaction -> test, the second
    most recent -> val, everything older -> train. Requires >=3 interactions/user
    (guaranteed upstream by min_user_interactions >= 5 in the default config)."""
    ranked = df.with_columns(
        pl.col("timestamp").rank(method="ordinal", descending=True).over("user_idx").alias("recency_rank")
    )
    test = ranked.filter(pl.col("recency_rank") == 1).drop("recency_rank")
    val = ranked.filter(pl.col("recency_rank") == 2).drop("recency_rank")
    train = ranked.filter(pl.col("recency_rank") > 2).drop("recency_rank")

    logger.info(
        f"leave-one-out split: train={len(train):,} val={len(val):,} test={len(test):,} rows"
    )
    _assert_no_leakage(train, val, test)
    return LeaveOneOutSplit(train=train, val=val, test=test)


@dataclass
class GlobalTimeSplit:
    train: pl.DataFrame
    test: pl.DataFrame
    cutoff_timestamp: int


def global_time_split(df: pl.DataFrame, holdout_days: int) -> GlobalTimeSplit:
    """A single global timestamp cutoff: the last `holdout_days` of calendar time (by
    the dataset's own max timestamp) is held out as test, regardless of which user it
    belongs to. Unlike leave_one_out_split, this can include genuinely cold-start users
    /items in test (interactions not seen at all in train) -- that's intentional, it's
    what makes this split a harder, more realistic check than per-user leave-one-out."""
    max_ts = df["timestamp"].max()
    cutoff = max_ts - holdout_days * SECONDS_PER_DAY

    train = df.filter(pl.col("timestamp") < cutoff)
    test = df.filter(pl.col("timestamp") >= cutoff)

    cold_users = test.filter(~pl.col("user_idx").is_in(train["user_idx"].to_list()))["user_idx"].n_unique()
    cold_items = test.filter(~pl.col("item_idx").is_in(train["item_idx"].to_list()))["item_idx"].n_unique()
    logger.info(
        f"global time split (cutoff={holdout_days}d before max timestamp): "
        f"train={len(train):,} test={len(test):,} rows; "
        f"test has {cold_users:,} users and {cold_items:,} items never seen in train"
    )
    return GlobalTimeSplit(train=train, test=test, cutoff_timestamp=cutoff)


def reduce_to_latest_per_user(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse a (possibly multi-row-per-user) split down to each user's single most
    recent row. global_time_split's test set can have multiple purchases per user in
    the holdout window, but evaluation.harness.evaluate() assumes exactly one
    ground-truth row per user -- the same assumption leave_one_out_split's own splits
    already satisfy natively. Used to evaluate cold-start accuracy against the global
    split (the only split with genuinely cold test users) through the same harness."""
    return df.sort("timestamp", descending=True).unique(subset=["user_idx"], keep="first")


def _assert_no_leakage(train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame) -> None:
    """Sanity check: for every user present in more than one split, later splits must
    have a strictly later (or equal, for ties) timestamp than train."""
    max_train_ts = train.group_by("user_idx").agg(pl.col("timestamp").max().alias("max_train_ts"))

    for name, split in [("val", val), ("test", test)]:
        joined = split.join(max_train_ts, on="user_idx", how="inner")
        violations = joined.filter(pl.col("timestamp") < pl.col("max_train_ts"))
        if len(violations) > 0:
            raise AssertionError(
                f"leakage detected: {len(violations)} rows in {name} have timestamp "
                f"earlier than that user's latest train timestamp"
            )
