import numpy as np
import polars as pl
import pytest

from recsys_large_scale.models.recency_popularity import RecencyWeightedPopularityRecommender

DAY = 86_400


def test_recent_interactions_outweigh_older_ones_with_equal_counts():
    # item 0: 3 interactions, all 30 days old. item 1: 1 interaction, right at max_ts.
    train = pl.DataFrame(
        {
            "user_idx": [0, 1, 2, 3],
            "item_idx": [0, 0, 0, 1],
            "timestamp": [0, 0, 0, 30 * DAY],
        }
    )
    model = RecencyWeightedPopularityRecommender(half_life_days=7.0)
    model.fit(train, n_users=4, n_items=2)

    # item 0's 3 stale interactions decay to well under item 1's single fresh one
    assert model.item_scores[1] > model.item_scores[0]


def test_half_life_halves_weight_after_one_half_life():
    train = pl.DataFrame(
        {"user_idx": [0, 1], "item_idx": [0, 1], "timestamp": [0, 7 * DAY]}
    )
    model = RecencyWeightedPopularityRecommender(half_life_days=7.0)
    model.fit(train, n_users=2, n_items=2)

    # item 0 is exactly one half-life older than item 1 (the most recent interaction)
    assert model.item_scores[0] == pytest.approx(model.item_scores[1] * 0.5, rel=1e-6)


def test_score_batch_returns_same_row_for_every_user():
    train = pl.DataFrame({"user_idx": [0], "item_idx": [0], "timestamp": [0]})
    model = RecencyWeightedPopularityRecommender()
    model.fit(train, n_users=3, n_items=2)

    scores = model.score_batch(np.array([0, 1, 2]))
    np.testing.assert_array_equal(scores[0], scores[1])
    np.testing.assert_array_equal(scores[1], scores[2])
