import numpy as np
import polars as pl

from recsys_large_scale.data.preprocess import compute_item_counts


class AgeBucketPopularityRecommender:
    """A slightly-smarter-than-generic popularity fallback: each user gets the most
    popular items among train interactions from users in the SAME age bucket, instead
    of one global list for everyone. Falls back to the global popularity list for a
    bucket with too few interactions to estimate reliably (`min_bucket_interactions`),
    or for a user whose age bucket is unknown -- avoiding a noisy per-bucket list from
    a small sample. Meant as a content-based cold-start fallback (see
    models/coldstart_fallback.py), not a general-purpose model in its own right --
    "most popular item in your age group" carries no personalization beyond age.

    `age_bucket` (n_users,) is a label array from evaluation.segments.bucket_ages,
    supplied at both fit and score time since this isn't part of the normal
    Recommender.fit(train, n_users, n_items) signature.
    """

    def __init__(self, min_bucket_interactions: int = 1000):
        self.min_bucket_interactions = min_bucket_interactions

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int, age_bucket: np.ndarray) -> None:
        self.n_items = n_items
        self.age_bucket = age_bucket
        self.global_scores = compute_item_counts(train, n_items).astype(np.float32)

        user_idx = train["user_idx"].to_numpy()
        item_idx = train["item_idx"].to_numpy()
        bucket_per_row = age_bucket[user_idx]

        self.bucket_scores: dict[str, np.ndarray] = {}
        for bucket in np.unique(age_bucket):
            mask = bucket_per_row == bucket
            if mask.sum() < self.min_bucket_interactions:
                continue  # too few interactions to trust a bucket-specific list -- keep the global fallback
            counts = np.zeros(n_items, dtype=np.float32)
            np.add.at(counts, item_idx[mask], 1.0)
            self.bucket_scores[bucket] = counts

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        buckets = self.age_bucket[user_indices]
        scores = np.tile(self.global_scores, (len(user_indices), 1))
        for bucket, bucket_score in self.bucket_scores.items():
            rows = np.where(buckets == bucket)[0]
            if len(rows) > 0:
                scores[rows] = bucket_score
        return scores
