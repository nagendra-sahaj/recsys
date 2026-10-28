import numpy as np
import polars as pl
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix


class BPRRecommender:
    """Matrix factorization trained with Bayesian Personalized Ranking: for each
    observed (user, item) pair, sample an item the user hasn't interacted with and
    take an SGD step that pushes the observed item's score above the sampled one's --
    directly optimizing pairwise ranking order rather than reconstructing interaction
    counts/confidence the way ALS does. Replaces the sibling notebooks' from-scratch
    NumPy BPR (plain Python SGD loop) with `implicit`'s Cython/OpenMP implementation,
    which is necessary at this data scale -- the hand-rolled loop wouldn't finish in a
    reasonable time on 15M+ interactions.
    """

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 100,
        random_state: int = 42,
    ):
        self.model = BayesianPersonalizedRanking(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None:
        rows = train["user_idx"].to_numpy()
        cols = train["item_idx"].to_numpy()
        data = np.ones(len(train), dtype=np.float32)
        user_items = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        self.model.fit(user_items)
        # implicit's BPR appends one extra "bias" dimension to both factor matrices
        # (user_factors' last column is a constant 1.0 for every user; item_factors'
        # last column is that item's learned bias, so the dot product reduces to
        # taste_similarity + item_bias). score_batch below uses the raw factors as
        # implicit intends, but item_embeddings drops that column -- it's exposed for
        # the diversity/intra-list-similarity metric, which should compare taste
        # directions, not get diluted by a scalar bias dimension.
        self.item_embeddings = np.asarray(self.model.item_factors, dtype=np.float32)[:, :-1]

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        user_factors = np.asarray(self.model.user_factors[user_indices])
        item_factors = np.asarray(self.model.item_factors)
        return (user_factors @ item_factors.T).astype(np.float32)
