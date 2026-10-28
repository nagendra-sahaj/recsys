import numpy as np

from recsys_large_scale.features.engineer import build_candidate_features_hm, build_category_recency_features
from recsys_large_scale.models.candidate_gen import ANNCandidateGenerator
from recsys_large_scale.models.reranker import LightGBMReranker


class TwoStageRecommenderHM:
    """H&M counterpart to TwoStageRecommender (models/two_stage.py): same
    FAISS-retrieval + LightGBM-rerank architecture, but H&M's own feature set --
    category_affinity (product_group_name) standing in for MovieLens's genre_affinity,
    plus two recency features (days_since_category_purchase, has_category_history)
    that only make sense on H&M's real daily purchase timestamps. Kept as a separate
    class rather than a generalized/parametrized TwoStageRecommender so the
    already-working MovieLens two_stage.pkl artifact is never put at risk by a
    shared-code change (see LEARNINGS.md).
    """

    def __init__(
        self,
        candidate_generator: ANNCandidateGenerator,
        itemknn_model,
        reranker: LightGBMReranker,
        item_pop_counts: np.ndarray,
        user_activity_counts: np.ndarray,
        user_category_affinity: np.ndarray,
        item_category_matrix: np.ndarray,
        item_category_idx: np.ndarray,
        user_category_last_ts: np.ndarray,
        reference_timestamp: int,
        n_items: int,
        n_candidates: int = 200,
    ):
        self.candidate_generator = candidate_generator
        self.itemknn_model = itemknn_model
        self.reranker = reranker
        self.item_pop_counts = item_pop_counts
        self.user_activity_counts = user_activity_counts
        self.user_category_affinity = user_category_affinity
        self.item_category_matrix = item_category_matrix
        self.item_category_idx = item_category_idx
        self.user_category_last_ts = user_category_last_ts
        self.reference_timestamp = reference_timestamp
        self.n_items = n_items
        self.n_candidates = n_candidates
        self.item_embeddings = candidate_generator.item_embeddings

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        b = len(user_indices)
        candidates, ann_scores = self.candidate_generator.generate(user_indices, self.n_candidates)

        itemknn_full = self.itemknn_model.score_batch(user_indices)
        itemknn_scores = np.take_along_axis(itemknn_full, candidates, axis=1)

        user_idx_flat = np.repeat(user_indices, self.n_candidates)
        item_idx_flat = candidates.ravel()

        days_since, has_history = build_category_recency_features(
            user_idx=user_idx_flat,
            item_idx=item_idx_flat,
            item_category_idx=self.item_category_idx,
            user_category_last_ts=self.user_category_last_ts,
            reference_timestamp=self.reference_timestamp,
        )

        X = build_candidate_features_hm(
            user_idx=user_idx_flat,
            item_idx=item_idx_flat,
            ann_score=ann_scores.ravel(),
            item_pop_counts=self.item_pop_counts,
            user_activity_counts=self.user_activity_counts,
            itemknn_score=itemknn_scores.ravel(),
            user_category_affinity=self.user_category_affinity,
            item_category_matrix=self.item_category_matrix,
            # score_batch never injects candidates -- every row here is something the
            # retrieval stage genuinely found, so is_injected is always 0 at inference
            is_injected=np.zeros(len(user_idx_flat), dtype=np.float32),
            days_since_category_purchase=days_since,
            has_category_history=has_history,
        )
        rerank_scores = self.reranker.predict(X).reshape(b, self.n_candidates)

        scores = np.full((b, self.n_items), -np.inf, dtype=np.float32)
        np.put_along_axis(scores, candidates, rerank_scores.astype(np.float32), axis=1)
        return scores
