import numpy as np

"""Ranking metrics for the single-relevant-item-per-user evaluation setting used by
`leave_one_out_split` (each user has exactly one held-out ground-truth item). All
functions here take a `hits` matrix: a (B, K) boolean array where hits[u, r] is True
iff the item recommended to user u at rank r (0-indexed) is that user's held-out item.
Build it with `hits_matrix()`.

Everything is written to also be correct (though not maximally informative) if this
evaluation setting is later extended to multiple relevant items per user -- see the
note on `average_precision_at_k`/`map_at_k`.
"""


def hits_matrix(topk: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """topk: (B, K) recommended item indices per user, best-first.
    ground_truth: (B,) the single held-out item index per user.
    Returns (B, K) bool."""
    return topk == ground_truth[:, None]


def recall_at_k(hits: np.ndarray) -> float:
    """Fraction of users whose held-out item appears anywhere in their top-K."""
    return float(hits.any(axis=1).mean())


def precision_at_k(hits: np.ndarray) -> float:
    """Mean fraction of each user's top-K slots that are a hit. With exactly one
    relevant item per user this is (0 or 1)/K, i.e. recall_at_k(hits) / K."""
    k = hits.shape[1]
    return float(hits.sum(axis=1).mean() / k)


def _first_hit_rank(hits: np.ndarray) -> np.ndarray:
    """1-indexed rank of each user's first hit, or 0 if there was no hit."""
    has_hit = hits.any(axis=1)
    first_pos = hits.argmax(axis=1)  # 0 when no True is present; has_hit gates that case
    return np.where(has_hit, first_pos + 1, 0)


def ndcg_at_k(hits: np.ndarray) -> float:
    """Binary-relevance NDCG@K. With one relevant item, IDCG is always 1 (the ideal
    ranking puts it at rank 1), so this reduces to DCG@K = 1/log2(rank+1) if hit."""
    rank = _first_hit_rank(hits)
    safe_rank = np.where(rank > 0, rank, 1)  # placeholder to avoid a log2(1)=0 divide-by-zero
    dcg = np.where(rank > 0, 1.0 / np.log2(safe_rank + 1), 0.0)
    return float(dcg.mean())


def mrr(hits: np.ndarray) -> float:
    """Mean Reciprocal Rank: average of 1/rank for the first hit, 0 if no hit."""
    rank = _first_hit_rank(hits)
    safe_rank = np.where(rank > 0, rank, 1)  # placeholder to avoid a divide-by-zero at rank=0
    rr = np.where(rank > 0, 1.0 / safe_rank, 0.0)
    return float(rr.mean())


def map_at_k(hits: np.ndarray) -> float:
    """Mean Average Precision@K. NOTE: with exactly one relevant item per user (the
    leave-one-out split used throughout Phases 1-4), average precision for a hit at
    rank r is exactly 1/r -- identical to reciprocal rank -- so MAP@K == MRR@K under
    this split and the two numbers will always agree. They diverge once a user can
    have multiple relevant held-out items (the Phase 5 H&M personalization setup),
    which is why both are implemented rather than dropping one as redundant now."""
    return mrr(hits)


def coverage_at_k(recommended_mask: np.ndarray) -> float:
    """recommended_mask: (n_items,) bool, True where that item appeared in ANY user's
    top-K over the whole evaluation run. Returns the fraction of the catalog ever
    recommended -- low coverage means the model keeps recommending the same items."""
    return float(recommended_mask.mean())


def novelty_at_k(topk: np.ndarray, item_pop_prob: np.ndarray) -> float:
    """Mean self-information (-log2 popularity) of recommended items, averaged over
    every recommendation slot (not deduplicated) across all users. Higher = the model
    is recommending less-popular items on average; a model that just reproduces the
    popularity baseline scores low here regardless of its accuracy metrics."""
    probs = np.clip(item_pop_prob[topk], 1e-12, None)
    return float((-np.log2(probs)).mean())


def top_popular_item_mask(item_pop_prob: np.ndarray, top_fraction: float = 0.01) -> np.ndarray:
    """Boolean (n_items,) mask marking the `top_fraction` most popular items by
    training interaction count (e.g. the top 1%) -- the reference set
    popularity_bias_at_k checks recommended items against. Rank-based (exactly
    round(top_fraction * n_items) items are marked), not a probability-value
    threshold, so ties/duplicate counts near the cutoff can't silently pull in far
    more or fewer items than intended."""
    n_items = len(item_pop_prob)
    n_popular = max(1, round(top_fraction * n_items))
    mask = np.zeros(n_items, dtype=bool)
    mask[np.argsort(-item_pop_prob)[:n_popular]] = True
    return mask


def popularity_bias_at_k(topk: np.ndarray, popular_mask: np.ndarray) -> float:
    """Mean fraction of each user's top-K recommendation slots filled by an item from
    `popular_mask` (see top_popular_item_mask) -- a direct, interpretable read on how
    much a model just reproduces the popularity baseline instead of personalizing.
    Complementary to novelty_at_k's information-theoretic version of the same idea:
    novelty penalizes popularity gradually (by how popular an item is), this instead
    asks what fraction of recommendations come from a fixed "clearly popular" set."""
    return float(popular_mask[topk].mean())


def ann_candidate_recall(exact_topk: np.ndarray, candidates: np.ndarray) -> float:
    """Diagnostic for a retrieval stage, not an accuracy metric against ground truth:
    exact_topk: (B, K) item indices from exact brute-force top-K scoring (e.g. a base
    model's own score_batch + argpartition, the same way evaluation.harness does it).
    candidates: (B, N) item indices an ANN/retrieval stage actually offered, any order.
    Returns the fraction of each user's *exact* top-K that's present anywhere in their
    candidate set, averaged across users -- i.e. how much of the "true" top-K a
    downstream reranker was even given a chance to select from. 1.0 means the
    retrieval stage's approximation lost nothing (relative to exact top-K at this K);
    lower values quantify a real accuracy ceiling the reranker cannot get past no
    matter how good it is, since it can only re-order what it was handed."""
    matches = exact_topk[:, :, None] == candidates[:, None, :]  # (B, K, N)
    hit_per_item = matches.any(axis=2)  # (B, K)
    return float(hit_per_item.mean())


def intra_list_similarity_at_k(topk_item_vectors: np.ndarray) -> float:
    """topk_item_vectors: (B, K, d) L2-normalized item embedding vectors, one row of K
    vectors per user's recommended list (in the order they were recommended -- order
    doesn't affect the result). Returns the mean pairwise cosine similarity *within*
    each user's list, averaged across users. LOWER means more diverse recommendation
    lists; a model that always recommends near-duplicates of the same theme scores
    high here regardless of its accuracy metrics. Requires K >= 2 (a 1-item list has no
    pairs); the caller is responsible for only calling this when that holds."""
    b, k, d = topk_item_vectors.shape
    if k < 2:
        raise ValueError(f"intra_list_similarity_at_k needs K >= 2, got K={k}")
    sim = topk_item_vectors @ topk_item_vectors.transpose(0, 2, 1)  # (B, K, K), diag == 1
    off_diagonal_sum = sim.sum(axis=(1, 2)) - np.trace(sim, axis1=1, axis2=2)
    per_user = off_diagonal_sum / (k * (k - 1))
    return float(per_user.mean())
