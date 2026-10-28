import numpy as np
import polars as pl

from recsys_large_scale.models.age_bucket_popularity import AgeBucketPopularityRecommender
from recsys_large_scale.models.coldstart_fallback import ColdStartFallbackRecommender


class DummyModel:
    """Returns a fixed, obviously-wrong score matrix -- stands in for a degenerate
    base model's output on cold users."""

    def __init__(self, n_items: int, value: float):
        self.n_items = n_items
        self.value = value

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        return np.full((len(user_indices), self.n_items), self.value, dtype=np.float32)


def test_coldstart_fallback_routes_warm_and_cold_users_separately():
    n_items = 3
    base = DummyModel(n_items, value=1.0)
    fallback = DummyModel(n_items, value=9.0)
    warm_user_mask = np.array([True, False, True, False])  # users 1 and 3 are cold

    wrapper = ColdStartFallbackRecommender(base, fallback, warm_user_mask)
    scores = wrapper.score_batch(np.array([0, 1, 2, 3]))

    np.testing.assert_array_equal(scores[0], [1.0, 1.0, 1.0])  # warm -> base
    np.testing.assert_array_equal(scores[1], [9.0, 9.0, 9.0])  # cold -> fallback
    np.testing.assert_array_equal(scores[2], [1.0, 1.0, 1.0])  # warm -> base
    np.testing.assert_array_equal(scores[3], [9.0, 9.0, 9.0])  # cold -> fallback


def test_coldstart_fallback_exposes_base_models_item_embeddings():
    base = DummyModel(3, value=1.0)
    base.item_embeddings = np.eye(3)
    fallback = DummyModel(3, value=9.0)

    wrapper = ColdStartFallbackRecommender(base, fallback, np.array([True]))
    np.testing.assert_array_equal(wrapper.item_embeddings, np.eye(3))


def test_age_bucket_popularity_uses_bucket_specific_counts_when_enough_data():
    # 3 items, 2 age buckets. Bucket "young" users (0, 1) mostly buy item 0;
    # bucket "old" users (2, 3) mostly buy item 2.
    train = pl.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2, 3, 3],
            "item_idx": [0, 0, 0, 1, 2, 2, 2, 1],
        }
    )
    age_bucket = np.array(["young", "young", "old", "old"], dtype=object)

    model = AgeBucketPopularityRecommender(min_bucket_interactions=2)
    model.fit(train, n_users=4, n_items=3, age_bucket=age_bucket)

    scores = model.score_batch(np.array([0, 2]))  # one young user, one old user
    assert scores[0].argmax() == 0  # young bucket -> item 0 most popular
    assert scores[1].argmax() == 2  # old bucket -> item 2 most popular


def test_age_bucket_popularity_falls_back_to_global_for_small_bucket():
    train = pl.DataFrame({"user_idx": [0, 0, 0, 1], "item_idx": [0, 0, 0, 1]})
    age_bucket = np.array(["common", "rare"], dtype=object)

    # "rare" bucket has only 1 interaction -- below the threshold, so it should fall
    # back to the global list (dominated by item 0, from the "common" bucket)
    model = AgeBucketPopularityRecommender(min_bucket_interactions=2)
    model.fit(train, n_users=2, n_items=2, age_bucket=age_bucket)

    scores = model.score_batch(np.array([1]))  # the "rare"-bucket user
    assert scores[0].argmax() == 0  # falls back to the global-popular item, not item 1
