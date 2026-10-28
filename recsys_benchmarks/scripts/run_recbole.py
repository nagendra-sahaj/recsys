"""Run one or more RecBole models under a single config and log results in the same
CSV schema recsys_large_scale uses, so the two sides can be joined directly.

RecBole reports both a validation and a test result. We log BOTH, because every existing
number in recsys_large_scale's metrics_log.csv is on the val fold (the user's
second-most-recent item) -- so RecBole's `best_valid_result` is the row that actually
lines up with ours, and its `test_result` is the one that lines up with published work.
Getting those two mixed up would be an easy way to draw a confidently wrong conclusion.

NOTE ON `run_single_model` BELOW: it reimplements RecBole's own `run_recbole()` rather
than calling it. That is not gratuitous -- `recbole/quick_start/quick_start.py` does
`from ray import tune` at module scope, and it is the ONLY file in RecBole that touches
ray (verified by grep over the installed package). RecBole pins `ray<=2.6.3`, a 2023
release that calls `np.bool8`, removed in NumPy 2.0 -- so importing RecBole's convenience
wrapper drags a dead dependency onto the critical path for a tuner we never use. Going
one level down to the public API it wraps (Config / create_dataset / data_preparation /
get_model / get_trainer, none of which import ray) removes the problem rather than
pinning around it.

Usage:
    uv run python scripts/run_recbole.py \
        --config configs/recbole_ml1m_full.yaml \
        --models Pop,ItemKNN,BPR
"""

import argparse
import csv
import traceback
from datetime import datetime, timezone
from pathlib import Path

LOG_FIELDS = [
    "timestamp",
    "source",
    "dataset",
    "model",
    "split",
    "eval_mode",
    "k",
    "recall",
    "precision",
    "ndcg",
    "mrr",
    "map",
    "hit",
    "itemcoverage",
    "averagepopularity",
    "shannonentropy",
]


def patch_collector_for_non_tensor_keys() -> None:
    """Work around an upstream RecBole bug that breaks the beyond-accuracy metrics.

    `Collector.data_collect()` stores `data.num_items`/`data.num_users` as plain ints and
    `data.count_items`/`data.count_users` as `collections.Counter`s. `get_data_struct()`
    then loops over every registered key calling `.cpu()` on it, which only works for
    tensors -- so the first validation epoch dies with
    `AttributeError: 'int' object has no attribute 'cpu'`.

    Those keys are registered only when a metric asks for them: ItemCoverage needs
    `data.num_items`, AveragePopularity needs `data.count_items` (ShannonEntropy is fine,
    it only needs tensors). So the bug fires for exactly the metrics we want in order to
    compare catalog coverage against our own `coverage_at_k`.

    The patch is upstream's own loop with a type guard added -- strictly narrower
    behaviour, nothing else changed. The alternative is dropping ItemCoverage and
    AveragePopularity from the configs, which costs us the coverage comparison.
    """
    import copy

    import torch
    from recbole.evaluator.collector import Collector

    def get_data_struct(self):
        for key in self.data_struct._data_dict:
            value = self.data_struct._data_dict[key]
            if torch.is_tensor(value):  # <-- the fix; upstream calls .cpu() unguarded
                self.data_struct._data_dict[key] = value.cpu()
        returned_struct = copy.deepcopy(self.data_struct)
        for key in ["rec.topk", "rec.meanrank", "rec.score", "rec.items", "data.label"]:
            if key in self.data_struct:
                del self.data_struct[key]
        return returned_struct

    Collector.get_data_struct = get_data_struct


def patch_torch_load_for_recbole_checkpoints() -> None:
    """Allow RecBole to reload its own checkpoints under PyTorch >= 2.6.

    After training, `trainer.evaluate(..., load_best_model=True)` reloads the best epoch's
    checkpoint via `torch.load(checkpoint_file, map_location=...)`. PyTorch 2.6 flipped
    that function's `weights_only` default from False to True, and RecBole's checkpoints
    contain more than tensors (its `Config` object among them), so the restricted
    unpickler rejects them: "Weights only load failed ... Unsupported operand 149".

    Unlike the other fixes here, this one is PyTorch moving forward rather than RecBole
    lagging -- and the security tightening behind it is sound. It is safe to relax in this
    specific case because the file is not foreign: `run_single_model` wrote it to ./saved/
    seconds earlier in this same process. We are not loading an untrusted artifact.
    """
    import torch

    if getattr(torch.load, "_recbole_patched", False):
        return

    original_load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load._recbole_patched = True
    torch.load = load


def patch_scipy_dok_update() -> None:
    """Restore `dok_matrix._update`, which RecBole's graph models call and SciPy removed.

    LightGCN, NGCF, NCL, SpectralCF and GCMC all build their adjacency matrix by
    assembling a {(row, col): 1} dict and calling `A._update(data_dict)` -- a private
    SciPy method that skipped per-key validation for speed. SciPy removed it, so those
    models die with `AttributeError: 'dok_matrix' object has no attribute '_update'`.
    Modern SciPy has a *public* `update()` doing the same job with validation; it used to
    raise NotImplementedError, which is precisely why RecBole used the private one.

    Aliasing to the public method is the whole fix. The obvious-looking alternative --
    `dict.update(self, data)`, since dok_matrix subclasses dict -- would be silently
    WRONG: modern `_dok_base` keeps its data in `self._dict`, not in the dict base, so
    that version writes to the wrong place and yields an empty adjacency matrix. The
    models would then train on no graph at all and report plausible-looking numbers,
    which is a far worse failure than the crash.
    """
    from scipy.sparse import dok_matrix

    if hasattr(dok_matrix, "_update"):
        return

    def _update(self, data):
        self.update(data)

    dok_matrix._update = _update


def patch_recbole_pop_counting() -> None:
    """Fix RecBole's Pop model, which does not actually count popularity.

    `Pop.calculate_loss` accumulates with:

        self.item_cnt[item, :] = self.item_cnt[item, :] + 1

    `item` is a batch of item ids containing many duplicates. Indexed assignment in
    PyTorch gathers the right-hand side and then scatters it, so duplicate indices do not
    accumulate -- last write wins. An item appearing 50 times in a batch is credited +1,
    not +50. The result counts *batches in which an item appeared*, which saturates at
    (batches per epoch x epochs) for anything popular.

    Measured on ML-1M: 562,308 train rows / 4096 batch size = 138 batches per epoch, best
    checkpoint at epoch 2 -> ceiling 276. Every one of RecBole's top-20 items came out at
    exactly 275, against true counts ranging 2803 down to 1734 (scripts/diagnose_pop.py).
    The head of the ranking -- precisely what Recall@10 depends on -- collapses into one
    saturated tie broken by index order.

    `torch.bincount` accumulates duplicates correctly. Opt-in via --fix-pop, so the
    default reproduces stock RecBole: what a paper using this baseline actually gets.
    """
    import torch
    from recbole.model.general_recommender.pop import Pop

    def calculate_loss(self, interaction):
        item = interaction[self.ITEM_ID]
        counts = torch.bincount(item, minlength=self.item_cnt.shape[0])
        self.item_cnt += counts.unsqueeze(-1).to(self.item_cnt.dtype)
        self.max_cnt = torch.max(self.item_cnt, dim=0)[0]
        return torch.nn.Parameter(torch.zeros(1)).to(self.device)

    Pop.calculate_loss = calculate_loss


def run_single_model(model_name: str, config_file: str, config_dict: dict | None = None) -> dict:
    """RecBole's run_recbole(), minus the ray import. Mirrors the upstream sequence in
    recbole/quick_start/quick_start.py so results are identical to the stock runner.

    One deliberate omission: upstream calls `get_flops()` purely to log a FLOPs estimate.
    It is informational, depends on `thop`, and is a known failure point for some model
    architectures -- there is no reason to let it break an evaluation sweep.
    """
    from logging import getLogger

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_logger, init_seed

    config = Config(model=model_name, config_file_list=[config_file], config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )

    return {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def rows_from_result(result: dict) -> dict[int, dict[str, float]]:
    """RecBole returns a flat {'recall@10': 0.1, 'ndcg@10': 0.05, ...} mapping.
    Regroup it by K so each K becomes one CSV row, matching our metrics_log layout."""
    by_k: dict[int, dict[str, float]] = {}
    for key, value in result.items():
        if "@" not in key:
            continue  # e.g. 'auc' -- not a top-K metric, no K to group under
        metric, _, k_str = key.partition("@")
        by_k.setdefault(int(k_str), {})[metric.lower()] = float(value)
    return by_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", required=True, help="comma-separated, e.g. Pop,ItemKNN,BPR")
    parser.add_argument("--log", default="results/recbole_log.csv")
    parser.add_argument(
        "--config-dict",
        default=None,
        help="Inline YAML of per-run config overrides, applied on top of --config. Needed "
        "because some settings are only correct for some models -- e.g. "
        "'train_neg_sample_args: ~' is right for Pop (whose item counts are otherwise "
        "polluted by sampled negatives) but would break BPR, which needs pairwise negatives.",
    )
    parser.add_argument(
        "--fix-pop",
        action="store_true",
        help="Correct RecBole's Pop model, whose item counts saturate because duplicate "
        "indices don't accumulate under indexed assignment (see patch_recbole_pop_counting). "
        "Off by default so the baseline reproduces stock RecBole.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Override the `dataset` column written to the log, so a variant run (e.g. Pop "
        "with negative sampling disabled) doesn't overwrite the baseline row.",
    )
    args = parser.parse_args()

    import yaml

    config_dict = yaml.safe_load(args.config_dict) if args.config_dict else None
    if config_dict is not None and not isinstance(config_dict, dict):
        raise SystemExit(f"--config-dict must parse to a mapping, got {type(config_dict).__name__}")

    patch_collector_for_non_tensor_keys()
    patch_torch_load_for_recbole_checkpoints()
    patch_scipy_dok_update()
    if args.fix_pop:
        patch_recbole_pop_counting()

    config_path = Path(args.config)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()

    failures: list[tuple[str, str]] = []
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for model in models:
            print(f"\n{'=' * 70}\n{model} ({config_path.name})\n{'=' * 70}")
            try:
                result = run_single_model(model, str(config_path), config_dict)
            except Exception as exc:  # noqa: BLE001 - one bad model shouldn't kill the sweep
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                failures.append((model, f"{type(exc).__name__}: {exc}"))
                continue

            timestamp = datetime.now(timezone.utc).isoformat()
            for split, payload in [
                ("val", result.get("best_valid_result") or {}),
                ("test", result.get("test_result") or {}),
            ]:
                for k, metrics in sorted(rows_from_result(payload).items()):
                    writer.writerow(
                        {
                            "timestamp": timestamp,
                            "source": "recbole",
                            "dataset": args.tag or config_path.stem,
                            "model": model,
                            "split": split,
                            "eval_mode": "uni100" if "uni100" in config_path.stem else "full",
                            "k": k,
                            **metrics,
                        }
                    )
                    f.flush()  # keep partial results if a later model crashes the process
            print(f"  logged -> {log_path}")

    if failures:
        print(f"\n{len(failures)}/{len(models)} model(s) failed:")
        for model, reason in failures:
            print(f"  - {model}: {reason}")


if __name__ == "__main__":
    main()
