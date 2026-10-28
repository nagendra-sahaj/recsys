"""Diagnose why RecBole's Pop and our popularity baseline disagree.

Popularity is the control for this whole comparison: it is the same trivial algorithm in
both codebases, so on identical data it must produce identical results. It does not.
On our own exported split, test Recall@10 reads:

    ours                                   0.0441
    RecBole Pop                            0.0358   (-19%)
    RecBole Pop, neg sampling disabled     0.0275   (-38%)

Two hypotheses have already been falsified by measurement, not argument:

  * Split tie-breaking -- ruled out: feeding RecBole our exact pre-split folds via
    `benchmark_filename` made the gap wider, not narrower.
  * Negative-sample pollution of `item_cnt` -- ruled out: disabling negative sampling
    made Pop worse still, when pure positive counting should have matched us exactly.

And one strong constraint: under `uni100` on the same split the two agree to within 1.4%
(0.4534 vs 0.4470). Agreement under sampled negatives but disagreement under full-catalog
ranking means the two popularity *rankings* differ somewhere that only matters when you
rank the whole catalog -- a coarse 101-item comparison cannot see it.

So: stop theorising and read RecBole's actual learned counts out of its checkpoint, map
them back to our item ids, and diff the two rankings directly.

Usage:
    uv run python scripts/diagnose_pop.py \
        --config configs/recbole_ml1m_ours_full.yaml \
        --train ../recsys_large_scale/data/processed/movielens1m/loo_train.parquet
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recbole_ml1m_ours_full.yaml")
    parser.add_argument(
        "--train", default="../recsys_large_scale/data/processed/movielens1m/loo_train.parquet"
    )
    parser.add_argument("--checkpoint", default=None, help="defaults to newest saved/Pop-*.pth")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    import torch
    from recbole.config import Config
    from recbole.data import create_dataset

    # --- RecBole side -------------------------------------------------------------
    config = Config(model="Pop", config_file_list=[args.config])
    dataset = create_dataset(config)
    # field2id_token maps RecBole's internal contiguous id -> the original token string.
    # Our exported tokens are stringified item_idx values, so this is the bridge back.
    item_tokens = dataset.field2id_token[config["ITEM_ID_FIELD"]]

    checkpoint_path = args.checkpoint or max(glob.glob("saved/Pop-*.pth"), key=os.path.getmtime)
    print(f"RecBole checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    item_cnt = ckpt["other_parameter"]["item_cnt"].squeeze(-1).cpu().numpy().astype(np.int64)

    recbole = pd.DataFrame({"internal_id": np.arange(len(item_cnt)), "recbole_count": item_cnt})
    recbole["token"] = [item_tokens[i] for i in recbole.internal_id]
    recbole = recbole[recbole.token != "[PAD]"]
    recbole["item_idx"] = recbole.token.astype(np.int64)

    # --- our side -----------------------------------------------------------------
    train = pd.read_parquet(args.train, columns=["item_idx"])
    ours = train.value_counts("item_idx").rename("our_count").reset_index()

    merged = recbole.merge(ours, on="item_idx", how="outer").fillna({"recbole_count": 0, "our_count": 0})

    # --- the comparisons that matter ----------------------------------------------
    print(f"\nitems: recbole={len(recbole):,} ours={len(ours):,} merged={len(merged):,}")
    print(f"total counts: recbole={int(merged.recbole_count.sum()):,} ours={int(merged.our_count.sum()):,}")
    print(f"  (our train rows = {len(train):,}; a RecBole total that is a clean multiple of")
    print("   this means counts accumulated once per training epoch, which is harmless;")
    print("   an inflated non-multiple means extra items were counted, which is not.)")

    exact = (merged.recbole_count == merged.our_count).mean()
    corr = np.corrcoef(merged.recbole_count, merged.our_count)[0, 1]
    print(f"\nitems whose counts match exactly: {exact:.1%}")
    print(f"correlation between the two count vectors: {corr:.6f}")

    print(f"\n--- top {args.top} by each system's own count ---")
    r_top = merged.nlargest(args.top, "recbole_count")[["item_idx", "recbole_count"]].reset_index(drop=True)
    o_top = merged.nlargest(args.top, "our_count")[["item_idx", "our_count"]].reset_index(drop=True)
    side = pd.concat(
        [r_top.rename(columns={"item_idx": "recbole_item"}), o_top.rename(columns={"item_idx": "our_item"})],
        axis=1,
    )
    print(side.to_string(index=False))

    overlap = len(set(r_top.item_idx) & set(o_top.item_idx))
    print(f"\ntop-{args.top} overlap: {overlap}/{args.top}")
    if overlap < args.top:
        print("  -> the two popularity rankings genuinely differ at the head of the catalog,")
        print("     which is exactly the part that determines Recall@10.")
    else:
        print("  -> same head of the ranking; the discrepancy is in evaluation, not the model.")


if __name__ == "__main__":
    main()
