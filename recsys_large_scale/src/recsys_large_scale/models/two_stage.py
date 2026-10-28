import numpy as np

from recsys_large_scale.features.engineer import build_candidate_features
from recsys_large_scale.models.candidate_gen import ANNCandidateGenerator
from recsys_large_scale.models.reranker import LightGBMReranker


class TwoStageRecommender:
    """Two-stage recommender: FAISS ANN retrieval over an ALS embedding space
    surfaces `n_candidates` candidates per user, then a LightGBM reranker re-scores
    just those candidates using engineered features. Conforms to the same
    `score_batch(user_indices) -> (B, n_items)` interface as every other model here
    by scoring non-candidate items as -inf, so it plugs directly into the existing
    evaluation harness unchanged -- the harness's own top-K selection then naturally
    only ever picks from whatever the retrieval stage actually offered.

    Because the reranker can only ever surface what the retrieval stage found, this
    model's accuracy ceiling is capped by the ANN candidate set's own recall of the
    true top items -- see evaluation.metrics.ann_candidate_recall and LEARNINGS.md for
    the measured cost of that on this dataset.
    """

    def __init__(
        self,
        candidate_generator: ANNCandidateGenerator,
        itemknn_model,
        reranker: LightGBMReranker,
        item_pop_counts: np.ndarray,
        user_activity_counts: np.ndarray,
        user_genre_affinity: np.ndarray,
        item_genre_matrix: np.ndarray,
        n_items: int,
        n_candidates: int = 200,
    ):
        self.candidate_generator = candidate_generator
        self.itemknn_model = itemknn_model
        self.reranker = reranker
        self.item_pop_counts = item_pop_counts
        self.user_activity_counts = user_activity_counts
        self.user_genre_affinity = user_genre_affinity
        self.item_genre_matrix = item_genre_matrix
        self.n_items = n_items
        self.n_candidates = n_candidates
        # delegate to the retrieval stage's embedding space for the diversity metric --
        # the final recommendations are always drawn from these candidates anyway
        self.item_embeddings = candidate_generator.item_embeddings

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        b = len(user_indices)
        candidates, ann_scores = self.candidate_generator.generate(user_indices, self.n_candidates)

        itemknn_full = self.itemknn_model.score_batch(user_indices)  # (b, n_items)
        itemknn_scores = np.take_along_axis(itemknn_full, candidates, axis=1)  # (b, n_candidates)

        user_idx_flat = np.repeat(user_indices, self.n_candidates)
        item_idx_flat = candidates.ravel()

        X = build_candidate_features(
            user_idx=user_idx_flat,
            item_idx=item_idx_flat,
            ann_score=ann_scores.ravel(),
            item_pop_counts=self.item_pop_counts,
            user_activity_counts=self.user_activity_counts,
            itemknn_score=itemknn_scores.ravel(),
            user_genre_affinity=self.user_genre_affinity,
            item_genre_matrix=self.item_genre_matrix,
            # score_batch never injects candidates -- every row here is something the
            # retrieval stage genuinely found, so is_injected is always 0 at inference
            is_injected=np.zeros(len(user_idx_flat), dtype=np.float32),
        )
        rerank_scores = self.reranker.predict(X).reshape(b, self.n_candidates)

        scores = np.full((b, self.n_items), -np.inf, dtype=np.float32)
        np.put_along_axis(scores, candidates, rerank_scores.astype(np.float32), axis=1)
        return scores
