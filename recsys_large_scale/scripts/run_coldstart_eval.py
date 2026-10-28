"""Cold-start evaluation: how much does each model's accuracy degrade for users it has
literally never seen anything about?

Only global_time_split's test set can contain such cases -- leave_one_out_split's test
set can't, by construction (every LOO test user needs >=1 train row just to be eligible
for that split at all). For the comparison to mean anything, the models evaluated here
must themselves have been trained on global_train, not loo_train -- see README's
Phase 5b RUNBOOK, part 2 for the exact training commands (same model configs as
part 1, just with --train-file global_train.parquet --artifacts-dir artifacts/models/global_hm).

Usage:
    uv run python scripts/run_coldstart_eval.py \
        --data-config configs/data_hm.yaml \
        --eval-config configs/eval_default.yaml \
        --models-dir artifacts/models/global_hm
"""

import argparse
import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from recsys_large_scale.data.preprocess import compute_item_counts
from recsys_large_scale.data.splits import reduce_to_latest_per_user
from recsys_large_scale.evaluation.harness import evaluate
from recsys_large_scale.evaluation.segments import cold_user_mask
from recsys_large_scale.utils.config import load_config
from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)

COLDSTART_LOG_FIELDS = [
    "timestamp",
    "dataset",
    "model",
    "segment",
    "k",
    "n_users",
    "recall",
    "precision",
    "ndcg",
    "mrr",
    "map",
    "coverage",
    "novelty",
    "intra_list_similarity",
]

# als_alpha1_hm skipped: Phase 5b part 1 already established alpha=20 wins on H&M, no
# need to retrain and re-compare the loser here too.
DEFAULT_MODELS = ["popularity_hm", "itemknn_hm", "als_hm", "bpr_hm"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--models-dir", default="artifacts/models/global_hm")
    parser.add_argument("--coldstart-log", default="artifacts/results/coldstart_log.csv")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model artifact names (without .pkl) to evaluate, e.g. to "
        "add the cold-start-fallback-wrapped variants from train_coldstart_fallback.py "
        "alongside the plain base models.",
    )
    args = parser.parse_args()
    models = args.models.split(",")

    data_config = load_config(args.data_config)
    eval_config = load_config(args.eval_config)
    processed_dir = Path(data_config["processed_dir"])

    train = pl.read_parquet(processed_dir / "global_train.parquet")
    raw_test = pl.read_parquet(processed_dir / "global_test.parquet")
    eval_set = reduce_to_latest_per_user(raw_test)
    id_maps = json.loads((processed_dir / "id_maps.json").read_text())
    n_users, n_items = id_maps["n_users"], id_maps["n_items"]

    item_counts = compute_item_counts(train, n_items)
    item_pop_prob = item_counts / item_counts.sum()

    eval_users = eval_set["user_idx"].to_numpy()
    cold = cold_user_mask(eval_users, train)
    logger.info(
        f"global test: {len(eval_set):,} users after reducing to one row/user "
        f"({int(cold.sum()):,} cold / {int((~cold).sum()):,} warm)"
    )

    segments = {
        "all": eval_set,
        "cold": eval_set.filter(pl.Series(cold)),
        "warm": eval_set.filter(pl.Series(~cold)),
    }

    log_path = Path(args.coldstart_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLDSTART_LOG_FIELDS)
        if write_header:
            writer.writeheader()

        for model_name in models:
            model_path = Path(args.models_dir) / f"{model_name}.pkl"
            if not model_path.exists():
                logger.warning(f"skipping {model_name}: {model_path} not found (train it first)")
                continue
            with open(model_path, "rb") as mf:
                model = pickle.load(mf)
            item_embeddings = getattr(model, "item_embeddings", None)

            for segment_name, segment_eval_set in segments.items():
                if len(segment_eval_set) == 0:
                    logger.warning(f"skipping {model_name}/{segment_name}: no users in this segment")
                    continue
                rows = evaluate(
                    model=model,
                    train=train,
                    eval_set=segment_eval_set,
                    n_users=n_users,
                    n_items=n_items,
                    item_pop_prob=item_pop_prob,
                    k_values=eval_config["k_values"],
                    batch_size=eval_config.get("batch_size", 2000),
                    item_embeddings=item_embeddings,
                )
                for row in rows:
                    logger.info(
                        f"{model_name} [{segment_name}, n={row.n_users:,}] k={row.k:>3} "
                        f"recall={row.recall:.4f} ndcg={row.ndcg:.4f} mrr={row.mrr:.4f}"
                    )
                    writer.writerow(
                        {
                            "timestamp": timestamp,
                            "dataset": data_config["dataset"],
                            "model": model_name,
                            "segment": segment_name,
                            "k": row.k,
                            "n_users": row.n_users,
                            "recall": row.recall,
                            "precision": row.precision,
                            "ndcg": row.ndcg,
                            "mrr": row.mrr,
                            "map": row.map,
                            "coverage": row.coverage,
                            "novelty": row.novelty,
                            "intra_list_similarity": row.intra_list_similarity
                            if row.intra_list_similarity is not None
                            else "",
                        }
                    )

    logger.info(f"Wrote cold-start results to {log_path}")


if __name__ == "__main__":
    main()
