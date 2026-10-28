import numpy as np
import polars as pl

from recsys_large_scale.data.preprocess import compute_item_counts


class PopularityRecommender:
    """Non-personalized baseline: every user gets the same ranked list, ordered by how
    many training interactions each item has. This is the floor every other model in
    the roadmap needs to beat -- see LEARNINGS.md."""

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None:
        self.item_scores = compute_item_counts(train, n_items).astype(np.float32)
        self.n_items = n_items

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        return np.tile(self.item_scores, (len(user_indices), 1))
