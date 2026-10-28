import numpy as np


class TwoTowerRecommender:
    """Wraps a pair of pre-trained embedding tables (one per user, one per item) from
    the Phase 6 two-tower neural model -- trained on Colab (see
    notebooks/two_tower_colab.ipynb), not locally, since it needs a GPU. This class
    itself does no training; it's just a thin adapter so the exported embeddings
    plug into this project's existing Recommender protocol (score_batch) and can be
    evaluated by the *exact same* scripts/evaluate_model.py every other model uses --
    the whole point of Phase 6 being "did DL earn its complexity" depends on a fair,
    identical evaluation path, not a separate one.

    Scoring is a plain dot product between the two towers' outputs, same convention
    as ALS/BPR -- the two-tower model's own nonlinearity (an MLP per tower) lives
    entirely upstream, inside the embeddings themselves, not in how they're compared.
    """

    def __init__(self, user_embeddings: np.ndarray, item_embeddings: np.ndarray):
        self.user_embeddings = user_embeddings.astype(np.float32)
        self.item_embeddings = item_embeddings.astype(np.float32)

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        return (self.user_embeddings[user_indices] @ self.item_embeddings.T).astype(np.float32)
