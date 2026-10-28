from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from recsys_large_scale.evaluation import metrics
from recsys_large_scale.models.base import Recommender
from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalRow:
    k: int
    n_users: int
    recall: float
    precision: float
    ndcg: float
    mrr: float
    map: float
    coverage: float
    novelty: float
    popularity_bias: float
    intra_list_similarity: float | None = None


def build_seen_matrix(train: pl.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    rows = train["user_idx"].to_numpy()
    cols = train["item_idx"].to_numpy()
    data = np.ones(len(train), dtype=bool)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def parse_eval_mode(eval_mode: str) -> int | None:
    """Translate an eval_mode string into a negative-sample count.

    "full" -> None (rank the held-out item against the entire catalog; what every number
    in this project has used so far). "uni<N>" -> N (rank it against only N uniformly
    sampled items), which is the protocol most of the published neural-recommender
    literature reports under. The naming deliberately matches RecBole's `eval_args.mode`
    so the two systems' settings line up one-to-one.
    """
    if eval_mode == "full":
        return None
    if eval_mode.startswith("uni"):
        suffix = eval_mode[3:]
        if suffix.isdigit() and int(suffix) > 0:
            return int(suffix)
    raise ValueError(f"unknown eval_mode {eval_mode!r}; expected 'full' or 'uni<N>' (e.g. 'uni100')")


def evaluate(
    model: Recommender,
    train: pl.DataFrame,
    eval_set: pl.DataFrame,
    n_users: int,
    n_items: int,
    item_pop_prob: np.ndarray,
    k_values: list[int] = (5, 10, 20),
    batch_size: int = 2000,
    item_embeddings: np.ndarray | None = None,
    eval_mode: str = "full",
    seed: int = 42,
) -> list[EvalRow]:
    """Generalizes the sibling notebooks' `evaluate(score_fn, ...)` pattern: score the
    full catalog for a batch of users, mask out items they've already interacted with
    in `train`, take top-K, and compute metrics against `eval_set` (one row per user =
    that user's single held-out ground-truth item, e.g. loo_val.parquet/loo_test.parquet).

    Batched over users (not one Python call per user) so this stays fast at hundreds of
    thousands of users, and computes all `k_values` from a single top-(max K) pass
    rather than re-scoring per K.

    `item_embeddings`, if given (an (n_items, d) array -- e.g. an ALS/BPR model's
    learned item factors), additionally computes intra-list similarity@K (see
    `metrics.intra_list_similarity_at_k`). Left as None for models with no embedding
    space (popularity, item-item CF), in which case `EvalRow.intra_list_similarity`
    stays None rather than being silently computed as 0 or omitted.

    `eval_mode` selects the ranking protocol. The default "full" is full-catalog ranking
    and leaves behaviour exactly as it has always been. "uni<N>" instead restricts each
    user's candidates to N uniformly sampled items plus their held-out item -- the
    sampled-negative protocol used by much of the published literature, provided here so
    the two protocols can be measured against each other rather than argued about.
    """
    n_negatives = parse_eval_mode(eval_mode)
    if n_negatives is not None:
        logger.info(
            f"eval_mode={eval_mode}: ranking against {n_negatives} sampled negatives + the "
            f"held-out item, NOT the full {n_items:,}-item catalog. Accuracy metrics are not "
            f"comparable to full-catalog runs, and coverage/novelty/popularity_bias/ILS are "
            f"computed over a random candidate pool, so they are not meaningful here."
        )

    seen = build_seen_matrix(train, n_users, n_items)

    gt_users = eval_set["user_idx"].to_numpy()
    gt_items = eval_set["item_idx"].to_numpy()
    if len(np.unique(gt_users)) != len(gt_users):
        raise ValueError(
            "evaluate() assumes exactly one ground-truth row per user; "
            "eval_set has duplicate user_idx values"
        )
    user_to_gt = dict(zip(gt_users.tolist(), gt_items.tolist()))
    eval_users = np.array(sorted(user_to_gt.keys()))
    n_eval_users = len(eval_users)
    k_values = sorted(k_values)
    k_max = min(max(k_values), n_items)

    item_unit_vectors = _l2_normalize_rows(item_embeddings.astype(np.float32)) if item_embeddings is not None else None
    popular_mask = metrics.top_popular_item_mask(item_pop_prob)

    sums = {k: {"recall": 0.0, "precision": 0.0, "ndcg": 0.0, "mrr": 0.0, "map": 0.0} for k in k_values}
    recommended_mask = {k: np.zeros(n_items, dtype=bool) for k in k_values}
    novelty_sum = {k: 0.0 for k in k_values}
    novelty_count = {k: 0 for k in k_values}
    popularity_bias_sum = {k: 0.0 for k in k_values}
    ils_sum = {k: 0.0 for k in k_values}
    ils_count = {k: 0 for k in k_values}

    n_batches = (n_eval_users + batch_size - 1) // batch_size
    for b_idx, start in enumerate(range(0, n_eval_users, batch_size)):
        batch_users = eval_users[start : start + batch_size]
        b = len(batch_users)

        scores = model.score_batch(batch_users).astype(np.float32, copy=False)
        batch_seen = seen[batch_users].toarray()
        scores = np.where(batch_seen, -np.inf, scores)

        ground_truth = np.fromiter((user_to_gt[u] for u in batch_users), dtype=np.int64, count=b)

        if n_negatives is not None:
            # Sampled-negative protocol: everything outside each user's candidate pool is
            # pushed to -inf so it can never enter the top-K. Negatives are drawn with
            # replacement from the whole catalog; duplicates and draws that land on an
            # already-seen item collapse out against the seen-mask above, so the effective
            # pool is N+1 minus a fraction of a percent. Seeded per batch (not per call) so
            # a given user's negatives don't depend on the batch_size in use.
            rng = np.random.default_rng([seed, start])
            sampled = rng.integers(0, n_items, size=(b, n_negatives))
            candidates = np.zeros((b, n_items), dtype=bool)
            candidates[np.repeat(np.arange(b), n_negatives), sampled.ravel()] = True
            candidates[np.arange(b), ground_truth] = True
            scores = np.where(candidates, scores, -np.inf)

        part = np.argpartition(-scores, k_max - 1, axis=1)[:, :k_max]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        topk_max = np.take_along_axis(part, order, axis=1)

        hits_max = metrics.hits_matrix(topk_max, ground_truth)

        for k in k_values:
            k_slice = min(k, k_max)
            hits_k = hits_max[:, :k_slice]
            topk_k = topk_max[:, :k_slice]

            sums[k]["recall"] += metrics.recall_at_k(hits_k) * b
            sums[k]["precision"] += metrics.precision_at_k(hits_k) * b
            sums[k]["ndcg"] += metrics.ndcg_at_k(hits_k) * b
            sums[k]["mrr"] += metrics.mrr(hits_k) * b
            sums[k]["map"] += metrics.map_at_k(hits_k) * b

            recommended_mask[k][topk_k.ravel()] = True
            novelty_sum[k] += metrics.novelty_at_k(topk_k, item_pop_prob) * topk_k.size
            novelty_count[k] += topk_k.size
            popularity_bias_sum[k] += metrics.popularity_bias_at_k(topk_k, popular_mask) * b

            if item_unit_vectors is not None and k_slice >= 2:
                topk_vectors = item_unit_vectors[topk_k]  # (b, k_slice, d)
                ils_sum[k] += metrics.intra_list_similarity_at_k(topk_vectors) * b
                ils_count[k] += b

        if (b_idx + 1) % 20 == 0 or (b_idx + 1) == n_batches:
            logger.info(f"evaluated batch {b_idx + 1}/{n_batches} ({start + b:,}/{n_eval_users:,} users)")

    rows = []
    for k in k_values:
        rows.append(
            EvalRow(
                k=k,
                n_users=n_eval_users,
                recall=sums[k]["recall"] / n_eval_users,
                precision=sums[k]["precision"] / n_eval_users,
                ndcg=sums[k]["ndcg"] / n_eval_users,
                mrr=sums[k]["mrr"] / n_eval_users,
                map=sums[k]["map"] / n_eval_users,
                coverage=metrics.coverage_at_k(recommended_mask[k]),
                novelty=novelty_sum[k] / novelty_count[k],
                popularity_bias=popularity_bias_sum[k] / n_eval_users,
                intra_list_similarity=(ils_sum[k] / ils_count[k]) if ils_count[k] > 0 else None,
            )
        )
    return rows
