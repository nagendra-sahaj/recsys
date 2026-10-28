import numpy as np
import polars as pl

SECONDS_PER_DAY = 86_400


class RecencyWeightedPopularityRecommender:
    """Popularity fallback that weights each training interaction by recency instead
    of counting every interaction equally: an interaction `days_ago` days before the
    train split's own most recent timestamp contributes weight
    `0.5 ** (days_ago / half_life_days)`, so a purchase from yesterday counts far more
    than one from three months back. Motivated by H&M's own retail structure -- what's
    "popular" in fashion is trend-driven and shifts continuously, so a cold user's
    first purchase plausibly skews toward what's selling *right now*, not what's been
    steadily popular over the whole training window (see
    models/age_bucket_popularity.py's similar reasoning for age, and LEARNINGS.md's
    Phase 5b cold-start entries). Otherwise a drop-in Recommender like
    models/popularity.py -- no extra fit-time arguments needed, since recency comes
    straight from `train`'s own timestamp column.
    """

    def __init__(self, half_life_days: float = 7.0):
        self.half_life_days = half_life_days

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None:
        self.n_items = n_items
        max_ts = train["timestamp"].max()
        days_ago = (max_ts - train["timestamp"]) / SECONDS_PER_DAY
        weight = (0.5 ** (days_ago / self.half_life_days)).to_numpy()

        item_idx = train["item_idx"].to_numpy()
        scores = np.zeros(n_items, dtype=np.float64)
        np.add.at(scores, item_idx, weight)
        self.item_scores = scores.astype(np.float32)

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        return np.tile(self.item_scores, (len(user_indices), 1))
