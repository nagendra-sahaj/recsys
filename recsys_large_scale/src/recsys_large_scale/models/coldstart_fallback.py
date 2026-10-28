import numpy as np


class ColdStartFallbackRecommender:
    """Wraps a base model with a fallback model for users the base model has no
    training history for. Per Phase 5b part 2b's cold-start finding (see
    LEARNINGS.md): ALS/item-item CF/BPR don't gracefully degrade for a user with zero
    training interactions -- their score_batch output for that user is a meaningless
    fixed list (a mechanical artifact of the model equations, unrelated to the user).
    This routes those users to a fallback model instead, per-user, while warm users
    still get the base model's real personalized scores.

    `warm_user_mask` (n_users,) bool must be computed from whatever `train` split the
    base model was actually fit on -- it is NOT recomputed here, since the wrapper has
    no way to know which split that was.
    """

    def __init__(self, base_model, fallback_model, warm_user_mask: np.ndarray):
        self.base_model = base_model
        self.fallback_model = fallback_model
        self.warm_user_mask = warm_user_mask
        # delegate to the base model's embedding space (if any) for the diversity
        # metric -- the fallback model (popularity/age-bucket popularity) has none
        self.item_embeddings = getattr(base_model, "item_embeddings", None)

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        base_scores = self.base_model.score_batch(user_indices)
        fallback_scores = self.fallback_model.score_batch(user_indices)
        warm = self.warm_user_mask[user_indices]
        return np.where(warm[:, None], base_scores, fallback_scores).astype(np.float32)
