"""Download and extract the MovieLens 25M dataset.

Usage:
    uv run python scripts/download_data.py --config configs/data_movielens25m.yaml
"""

import argparse

from recsys_large_scale.data.download import download_and_extract
from recsys_large_scale.utils.config import load_config
from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    extracted_dir = download_and_extract(config["download_url"], config["raw_dir"])
    logger.info(f"Data ready at: {extracted_dir}")
    logger.info(
        "Note: raw_dir in the config should point at this extracted folder "
        "(the default configs already do, since ml-25m.zip has one top-level folder)."
    )


if __name__ == "__main__":
    main()
