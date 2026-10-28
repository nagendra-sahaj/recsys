import polars as pl
import pytest

from recsys_large_scale.data.splits import global_time_split, leave_one_out_split, reduce_to_latest_per_user

DAY = 86_400


def make_df(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    """rows: list of (user_idx, item_idx, timestamp)."""
    return pl.DataFrame(rows, schema=["user_idx", "item_idx", "timestamp"], orient="row")


def test_leave_one_out_picks_most_recent_as_test():
    df = make_df(
        [
            (0, 10, 1),
            (0, 11, 2),
            (0, 12, 3),
            (0, 13, 4),
            (0, 14, 5),
        ]
    )
    split = leave_one_out_split(df)

    assert split.test["item_idx"].to_list() == [14]
    assert split.test["timestamp"].to_list() == [5]
    assert split.val["item_idx"].to_list() == [13]
    assert sorted(split.train["item_idx"].to_list()) == [10, 11, 12]


def test_leave_one_out_is_independent_per_user():
    df = make_df(
        [
            (0, 10, 1),
            (0, 11, 2),
            (0, 12, 3),
            (1, 20, 100),
            (1, 21, 200),
            (1, 22, 300),
        ]
    )
    split = leave_one_out_split(df)

    test_by_user = dict(zip(split.test["user_idx"].to_list(), split.test["item_idx"].to_list()))
    assert test_by_user == {0: 12, 1: 22}


def test_leave_one_out_no_leakage_by_construction():
    """Every train timestamp for a user must be <= that user's val/test timestamps
    (the split function itself asserts this internally; this test just confirms it
    doesn't raise on a normal, non-adversarial input)."""
    df = make_df([(0, i, ts) for i, ts in enumerate(range(1, 11))])
    leave_one_out_split(df)  # should not raise


def test_global_time_split_cutoff_and_cold_start_counts():
    # user 0 interacts throughout; user 1 only appears in the holdout window (cold-start test user)
    df = make_df(
        [
            (0, 10, 0 * DAY),
            (0, 11, 5 * DAY),
            (0, 12, 9 * DAY),  # falls inside a 2-day holdout ending at day 10
            (1, 20, 10 * DAY),  # cold-start: user 1 never appears in train; also sets max_ts
        ]
    )
    result = global_time_split(df, holdout_days=2)

    assert result.cutoff_timestamp == 8 * DAY
    assert sorted(result.train["item_idx"].to_list()) == [10, 11]
    assert sorted(result.test["item_idx"].to_list()) == [12, 20]


def test_leave_one_out_requires_timestamp_column():
    df = pl.DataFrame({"user_idx": [0, 0], "item_idx": [10, 11]})
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        leave_one_out_split(df)


def test_reduce_to_latest_per_user_keeps_only_the_most_recent_row():
    df = make_df(
        [
            (0, 10, 1 * DAY),
            (0, 11, 3 * DAY),  # user 0's most recent
            (1, 20, 2 * DAY),  # user 1's only row
        ]
    )
    out = reduce_to_latest_per_user(df)

    assert len(out) == 2
    by_user = dict(zip(out["user_idx"].to_list(), out["item_idx"].to_list()))
    assert by_user == {0: 11, 1: 20}
