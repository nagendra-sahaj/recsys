"""Build and train the two-stage recommender: FAISS candidate generation (over a base
ALS model's embeddings) + a LightGBM reranker over engineered candidate features.

Leakage note: the reranker's training labels come from the *validation* split
(loo_val.parquet), never test -- each val user's held-out item becomes a positive
label among that user's ANN-retrieved candidates (injected into the candidate set if
the retrieval stage didn't surface it on its own). All features (item popularity, user
activity, genre affinity, item-item CF score) are computed from the *train* split only,
via models that were themselves fit on train -- so the reranker never sees anything
from val used as a feature input, only as its label. The two-stage pipeline is then
evaluated on test via the ordinary scripts/evaluate_model.py, exactly like every other
model, since TwoStageRecommender conforms to the same score_batch(...) interface.

Usage:
    uv run python scripts/run_two_stage_pipeline.py \
        --data-config configs/data_movielens25m.yaml \
        --pipeline-config configs/model_two_stage.yaml
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from recsys_large_scale.data.loaders import load_movielens_movies
from recsys_large_scale.data.preprocess import compute_item_counts, compute_user_counts
from recsys_large_scale.evaluation.metrics import ann_candidate_recall
from recsys_large_scale.features.engineer import build_candidate_features, build_item_genre_matrix, build_user_genre_affinity
from recsys_large_scale.models.candidate_gen import ANNCandidateGenerator
from recsys_large_scale.models.reranker import LightGBMReranker
from recsys_large_scale.models.two_stage import TwoStageRecommender
from recsys_large_scale.utils.config import load_config
from recsys_large_scale.utils.logging import get_logger
from recsys_large_scale.utils.timers import Timer

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    pipeline_config = load_config(args.pipeline_config)
    rng = np.random.default_rng(pipeline_config["random_seed"])

    processed_dir = Path(data_config["processed_dir"])
    train = pl.read_parquet(processed_dir / "loo_train.parquet")
    val = pl.read_parquet(processed_dir / "loo_val.parquet")
    id_maps = json.loads((processed_dir / "id_maps.json").read_text())
    n_users, n_items = id_maps["n_users"], id_maps["n_items"]
    # JSON round-trips dict keys as strings; movies.csv's item_id column is int
    item_id_to_idx = {int(k): v for k, v in id_maps["item_id_to_idx"].items()}

    with open(pipeline_config["base_als_model"], "rb") as f:
        als_model = pickle.load(f)
    with open(pipeline_config["itemknn_model"], "rb") as f:
        itemknn_model = pickle.load(f)

    logger.info("Building content features (genres) and popularity/activity counts from train...")
    movies = load_movielens_movies(data_config["raw_dir"], data_config["movies_file"])
    item_genre_matrix, genre_names = build_item_genre_matrix(movies, item_id_to_idx, n_items)
    user_genre_affinity = build_user_genre_affinity(train, item_genre_matrix, n_users)
    item_pop_counts = compute_item_counts(train, n_items)
    user_activity_counts = compute_user_counts(train, n_users)
    logger.info(f"{len(genre_names)} genres found: {genre_names}")

    n_candidates = pipeline_config["n_candidates"]
    ann_cfg = pipeline_config["ann_index"]
    candidate_gen = ANNCandidateGenerator(
        item_embeddings=als_model.item_embeddings,
        user_embeddings=als_model.user_embeddings,
        nlist=ann_cfg["nlist"],
    )
    candidate_gen.index.nprobe = ann_cfg["nprobe"]

    logger.info("Building reranker training set from the validation split...")
    val_user_to_item = dict(zip(val["user_idx"].to_list(), val["item_idx"].to_list()))
    all_val_users = np.array(sorted(val_user_to_item.keys()))
    max_training_users = pipeline_config["max_training_users"]
    if len(all_val_users) > max_training_users:
        training_users = rng.choice(all_val_users, size=max_training_users, replace=False)
        training_users.sort()
    else:
        training_users = all_val_users

    candidates, ann_scores = candidate_gen.generate(training_users, n_candidates)
    ground_truth = np.array([val_user_to_item[u] for u in training_users])

    # inject the true held-out item into the candidate set for users the ANN stage
    # missed, so the reranker has a positive example to learn from for every query
    missed = ~(candidates == ground_truth[:, None]).any(axis=1)
    n_missed = int(missed.sum())
    is_injected = np.zeros_like(candidates, dtype=np.float32)
    if n_missed > 0:
        candidates[missed, -1] = ground_truth[missed]
        ann_scores[missed, -1] = np.einsum(
            "ij,ij->i", als_model.user_embeddings[training_users[missed]], als_model.item_embeddings[ground_truth[missed]]
        )
        # flag the injected slot so the reranker can learn NOT to read its
        # (necessarily low) ann_score the same way as an organically-retrieved
        # candidate's -- see the is_injected feature's docstring in features/engineer.py
        is_injected[missed, -1] = 1.0
    logger.info(
        f"{n_missed:,}/{len(training_users):,} training users' held-out item was outside the "
        f"top-{n_candidates} ANN candidates and had to be injected"
    )

    labels = (candidates == ground_truth[:, None]).astype(np.float32)
    itemknn_full = itemknn_model.score_batch(training_users)
    itemknn_scores = np.take_along_axis(itemknn_full, candidates, axis=1)

    X = build_candidate_features(
        user_idx=np.repeat(training_users, n_candidates),
        item_idx=candidates.ravel(),
        ann_score=ann_scores.ravel(),
        item_pop_counts=item_pop_counts,
        user_activity_counts=user_activity_counts,
        itemknn_score=itemknn_scores.ravel(),
        user_genre_affinity=user_genre_affinity,
        item_genre_matrix=item_genre_matrix,
        is_injected=is_injected.ravel(),
    )
    y = labels.ravel()
    group_sizes = np.full(len(training_users), n_candidates)

    logger.info(f"Training reranker on {len(y):,} (user, candidate) rows across {len(training_users):,} queries...")
    reranker = LightGBMReranker(**pipeline_config["reranker"])
    timer = Timer()
    with timer.measure("reranker fit"):
        reranker.fit(X, y, group_sizes)
    logger.info(timer.report())

    logger.info("Running the ANN-vs-exact candidate recall diagnostic...")
    n_check = min(pipeline_config["ann_recall_check_users"], n_users)
    check_users = rng.choice(n_users, size=n_check, replace=False)
    check_users.sort()
    k_check = 20
    exact_scores = als_model.score_batch(check_users)
    exact_part = np.argpartition(-exact_scores, k_check - 1, axis=1)[:, :k_check]
    exact_order = np.argsort(-np.take_along_axis(exact_scores, exact_part, axis=1), axis=1)
    exact_topk = np.take_along_axis(exact_part, exact_order, axis=1)
    check_candidates, _ = candidate_gen.generate(check_users, n_candidates)
    recall = ann_candidate_recall(exact_topk, check_candidates)
    logger.info(
        f"ANN candidate recall: {recall:.4f} of each user's exact top-{k_check} (by the base ALS "
        f"model's own scoring) is present within the {n_candidates} ANN-retrieved candidates "
        f"(nlist={ann_cfg['nlist']}, nprobe={ann_cfg['nprobe']}). This is the two-stage pipeline's "
        f"accuracy ceiling from retrieval alone, independent of the reranker."
    )

    model = TwoStageRecommender(
        candidate_generator=candidate_gen,
        itemknn_model=itemknn_model,
        reranker=reranker,
        item_pop_counts=item_pop_counts,
        user_activity_counts=user_activity_counts,
        user_genre_affinity=user_genre_affinity,
        item_genre_matrix=item_genre_matrix,
        n_items=n_items,
        n_candidates=n_candidates,
    )

    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pipeline_config['name']}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved two-stage model to {out_path}")
    logger.info(
        "Next: evaluate it like any other model, e.g.\n"
        f"  uv run python scripts/evaluate_model.py --data-config {args.data_config} "
        f"--eval-config configs/eval_default.yaml --model-artifact {out_path} --split val"
    )


if __name__ == "__main__":
    main()
