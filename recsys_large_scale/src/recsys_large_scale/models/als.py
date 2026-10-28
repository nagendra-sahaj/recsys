import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


class ALSRecommender:
    """Matrix factorization via Alternating Least Squares on implicit feedback,
    confidence-weighted per Hu/Koren/Volinsky ("Collaborative Filtering for Implicit
    Feedback Datasets"): `implicit` internally scales whatever values we pass by
    `alpha`, then treats every observed (user, item) pair as confidence `1 + alpha *
    value` and everything else as confidence 1 with an assumed negative preference.

    Our data's `confidence` column is 1.0 for every positive interaction (a consequence
    of configs/data_movielens25m.yaml's threshold-based positive-feedback definition --
    ratings are kept-or-dropped, not graded), so here `alpha` acts as one global "how
    much more should observed interactions count than unobserved ones" weight, not a
    per-interaction graded signal. That's a standard, common way to run implicit ALS
    when there's no finer-grained preference strength to feed it.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        alpha: float = 20.0,
        iterations: int = 15,
        random_state: int = 42,
    ):
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            random_state=random_state,
        )

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None:
        rows = train["user_idx"].to_numpy()
        cols = train["item_idx"].to_numpy()
        data = train["confidence"].to_numpy().astype(np.float32)
        user_items = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        self.model.fit(user_items)
        self.item_embeddings = np.asarray(self.model.item_factors, dtype=np.float32)
        self.user_embeddings = np.asarray(self.model.user_factors, dtype=np.float32)

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        user_factors = np.asarray(self.model.user_factors[user_indices])
        item_factors = np.asarray(self.model.item_factors)
        return (user_factors @ item_factors.T).astype(np.float32)
