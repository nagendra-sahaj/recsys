"""Wrap the embeddings exported from notebooks/two_tower_colab.ipynb into a pickled
TwoTowerRecommender, so it can be evaluated by the exact same scripts/evaluate_model.py
every other model in this project uses.

Usage (after training on Colab and downloading user_embeddings.npy/item_embeddings.npy
into, e.g., artifacts/colab/):
    uv run python scripts/build_two_tower_model.py \
        --user-embeddings artifacts/colab/user_embeddings.npy \
        --item-embeddings artifacts/colab/item_embeddings.npy \
        --data-config configs/data_movielens25m.yaml
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from recsys_large_scale.models.two_tower import TwoTowerRecommender
from recsys_large_scale.utils.config import load_config
from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-embeddings", required=True)
    parser.add_argument("--item-embeddings", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--name", default="two_tower")
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    processed_dir = Path(data_config["processed_dir"])
    id_maps = json.loads((processed_dir / "id_maps.json").read_text())
    n_users, n_items = id_maps["n_users"], id_maps["n_items"]

    user_embeddings = np.load(args.user_embeddings)
    item_embeddings = np.load(args.item_embeddings)
    if user_embeddings.shape[0] != n_users:
        raise ValueError(f"user_embeddings has {user_embeddings.shape[0]} rows, expected n_users={n_users}")
    if item_embeddings.shape[0] != n_items:
        raise ValueError(f"item_embeddings has {item_embeddings.shape[0]} rows, expected n_items={n_items}")
    logger.info(
        f"Loaded embeddings: user {user_embeddings.shape}, item {item_embeddings.shape} "
        f"(embedding_dim={user_embeddings.shape[1]})"
    )

    model = TwoTowerRecommender(user_embeddings, item_embeddings)

    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved {out_path}")
    logger.info(
        "Next: evaluate it like any other model, e.g.\n"
        f"  uv run python scripts/evaluate_model.py --data-config {args.data_config} "
        f"--eval-config configs/eval_default.yaml --model-artifact {out_path} --split val"
    )


if __name__ == "__main__":
    main()
