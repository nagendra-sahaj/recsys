from typing import Protocol

import numpy as np
import polars as pl


class Recommender(Protocol):
    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None: ...

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        """Return a dense (len(user_indices), n_items) float32 array of scores, one
        row per requested user, over the full item catalog (unmasked -- the
        evaluation harness handles masking out train-seen items)."""
        ...
