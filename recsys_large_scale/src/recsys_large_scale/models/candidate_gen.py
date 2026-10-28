import numpy as np

from recsys_large_scale.serving.ann_index import ANNIndex


class ANNCandidateGenerator:
    """The retrieval half of a two-stage recommender: wraps a base model's item
    embedding space with a FAISS index and, given a batch of users, returns their
    top-N candidate items by inner product -- without ever touching the full
    catalog's dense scores the way evaluation.harness's brute-force scoring does.
    A downstream reranker (models.reranker + features.engineer) re-scores just these
    candidates; see models.two_stage.TwoStageRecommender for how the two combine."""

    def __init__(self, item_embeddings: np.ndarray, user_embeddings: np.ndarray, nlist: int | None = None):
        self.item_embeddings = item_embeddings
        self.user_embeddings = user_embeddings
        self.index = ANNIndex(item_embeddings, nlist=nlist)

    def generate(self, user_indices: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (candidates, scores), both (len(user_indices), top_n) -- candidate
        item indices and their raw inner-product ANN scores, best-first. Not yet
        reranked, not yet masked for items the user has already interacted with (the
        caller/harness handles that, same as every other model here)."""
        query = self.user_embeddings[user_indices]
        return self.index.search(query, top_n)
