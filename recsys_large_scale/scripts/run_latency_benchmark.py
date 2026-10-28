"""In-process latency benchmark: how long does it take to produce a ranked top-K list
for one user at a time? No HTTP server involved -- we're not serving this to anyone, so
a plain per-call `time.perf_counter()` loop is the honest way to measure it, without
web-framework/serialization overhead muddying the number we actually care about.

Three paths, each timed over --n-requests users (one call per user, matching how a real
online request would arrive):

  1. brute_force_exact   -- score every item with the base ALS model, argpartition top-K
                             (models/als.py's score_batch)
  2. faiss_ann_retrieval -- two-stage's retrieval stage alone: FAISS ANN search over the
                             same ALS item embeddings, top-K already sorted by FAISS
                             (models/candidate_gen.py)
  3. two_stage_full      -- faiss_ann_retrieval's n_candidates, rescored by the LightGBM
                             reranker, then sorted down to top-K

(1) vs (2) isolates pure retrieval speed (exact vs. approximate). (3) minus (2) is an
approximate reranker overhead (feature building + LightGBM inference on n_candidates
rows). Path 3 deliberately doesn't call TwoStageRecommender.score_batch -- that method
scatters scores into a full (1, n_items) array filled with -inf so it can plug into the
evaluation harness's batched top-K convention, which is wasted work no real serving path
would do; here we sort just the n_candidates rerank scores directly.

Usage:
    uv run python scripts/run_latency_benchmark.py \
        --data-config configs/data_movielens25m.yaml \
        --als-model artifacts/models/als_alpha1.pkl \
        --two-stage-model artifacts/models/two_stage.pkl
"""

import argparse
import csv
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from recsys_large_scale.features.engineer import build_candidate_features
from recsys_large_scale.utils.config import load_config
from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)

LATENCY_LOG_FIELDS = ["timestamp", "dataset", "path", "n_requests", "k", "mean_ms", "p50_ms", "p95_ms", "p99_ms"]


def time_per_request(fn, users: np.ndarray, warmup: int) -> np.ndarray:
    """Calls fn(user_idx) once per user, timing every call after the first `warmup`
    ones (warmup absorbs one-off costs like thread-pool spin-up and page faults on
    first access to the embedding arrays, which a real long-running service would
    also only pay once)."""
    for u in users[:warmup]:
        fn(u)
    latencies_ms = np.empty(len(users) - warmup, dtype=np.float64)
    for i, u in enumerate(users[warmup:]):
        start = time.perf_counter()
        fn(u)
        latencies_ms[i] = (time.perf_counter() - start) * 1000
    return latencies_ms


def summarize(label: str, latencies_ms: np.ndarray) -> dict:
    return {
        "path": label,
        "mean_ms": float(np.mean(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--als-model", default="artifacts/models/als_alpha1.pkl")
    parser.add_argument("--two-stage-model", default="artifacts/models/two_stage.pkl")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n-requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-log", default="artifacts/results/latency_log.csv")
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    processed_dir = Path(data_config["processed_dir"])
    id_maps = json.loads((processed_dir / "id_maps.json").read_text())
    n_users = id_maps["n_users"]

    with open(args.als_model, "rb") as f:
        als_model = pickle.load(f)
    with open(args.two_stage_model, "rb") as f:
        two_stage_model = pickle.load(f)
    candidate_gen = two_stage_model.candidate_generator
    itemknn_model = two_stage_model.itemknn_model
    reranker = two_stage_model.reranker
    n_candidates = two_stage_model.n_candidates

    rng = np.random.default_rng(args.seed)
    total_users_needed = args.n_requests + args.warmup
    users = rng.choice(n_users, size=total_users_needed, replace=False)

    k = args.k

    def brute_force_exact(u: int) -> np.ndarray:
        scores = als_model.score_batch(np.array([u]))[0]
        part = np.argpartition(-scores, k - 1)[:k]
        return part[np.argsort(-scores[part])]

    def faiss_ann_retrieval(u: int) -> np.ndarray:
        candidates, _ = candidate_gen.generate(np.array([u]), k)
        return candidates[0]

    def two_stage_full(u: int) -> np.ndarray:
        u_arr = np.array([u])
        candidates, ann_scores = candidate_gen.generate(u_arr, n_candidates)
        itemknn_full = itemknn_model.score_batch(u_arr)
        itemknn_scores = np.take_along_axis(itemknn_full, candidates, axis=1)
        X = build_candidate_features(
            user_idx=np.repeat(u_arr, n_candidates),
            item_idx=candidates.ravel(),
            ann_score=ann_scores.ravel(),
            item_pop_counts=two_stage_model.item_pop_counts,
            user_activity_counts=two_stage_model.user_activity_counts,
            itemknn_score=itemknn_scores.ravel(),
            user_genre_affinity=two_stage_model.user_genre_affinity,
            item_genre_matrix=two_stage_model.item_genre_matrix,
            is_injected=np.zeros(n_candidates, dtype=np.float32),
        )
        rerank_scores = reranker.predict(X)
        order = np.argsort(-rerank_scores)[:k]
        return candidates[0, order]

    paths = {
        "brute_force_exact": brute_force_exact,
        "faiss_ann_retrieval": faiss_ann_retrieval,
        "two_stage_full": two_stage_full,
    }

    results = []
    for label, fn in paths.items():
        logger.info(f"Timing {label} over {args.n_requests} requests (k={k}, warmup={args.warmup})...")
        latencies_ms = time_per_request(fn, users, args.warmup)
        summary = summarize(label, latencies_ms)
        results.append(summary)
        logger.info(
            f"{label}: mean={summary['mean_ms']:.3f}ms p50={summary['p50_ms']:.3f}ms "
            f"p95={summary['p95_ms']:.3f}ms p99={summary['p99_ms']:.3f}ms"
        )

    faiss_p50 = next(r["p50_ms"] for r in results if r["path"] == "faiss_ann_retrieval")
    two_stage_p50 = next(r["p50_ms"] for r in results if r["path"] == "two_stage_full")
    logger.info(
        f"Approximate reranker overhead (two_stage_full p50 - faiss_ann_retrieval p50): "
        f"{two_stage_p50 - faiss_p50:.3f}ms -- feature building + LightGBM inference on "
        f"{n_candidates} candidates"
    )

    log_path = Path(args.latency_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LATENCY_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "dataset": data_config["dataset"],
                    "path": r["path"],
                    "n_requests": args.n_requests,
                    "k": k,
                    "mean_ms": r["mean_ms"],
                    "p50_ms": r["p50_ms"],
                    "p95_ms": r["p95_ms"],
                    "p99_ms": r["p99_ms"],
                }
            )
    logger.info(f"Appended {len(results)} rows to {log_path}")


if __name__ == "__main__":
    main()
