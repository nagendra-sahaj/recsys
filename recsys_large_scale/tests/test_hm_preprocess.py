import polars as pl

from recsys_large_scale.data.preprocess import hm_to_implicit_feedback, subsample_by_date_window


def test_hm_to_implicit_feedback_adds_constant_confidence():
    df = pl.DataFrame({"user_id": ["a", "b"], "item_id": [1, 2], "timestamp": [100, 200]})
    out = hm_to_implicit_feedback(df)

    assert out.columns == ["user_id", "item_id", "timestamp", "confidence"]
    assert out["confidence"].to_list() == [1.0, 1.0]


def test_subsample_by_date_window_keeps_only_recent_rows():
    day = 86_400
    df = pl.DataFrame(
        {
            "user_id": ["a", "a", "a"],
            "item_id": [1, 2, 3],
            # max timestamp is 10 days; a 3-day window should drop the day-0 row
            "timestamp": [0, 8 * day, 10 * day],
        }
    )
    out = subsample_by_date_window(df, window_days=3)

    assert out["timestamp"].to_list() == [8 * day, 10 * day]
