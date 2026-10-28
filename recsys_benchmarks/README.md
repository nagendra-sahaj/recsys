# recsys_benchmarks — comparing `recsys_large_scale` against an outside reference

`recsys_large_scale` measures itself against itself: popularity → item-item CF → ALS →
BPR → two-stage → two-tower, all through one shared harness. That answers "which of our
models is best" and says nothing about "are any of them actually good."

This project answers the second question by running **RecBole** — the framework most
modern recommender-systems papers are built on — on the same data under the same
protocol, so the two can be compared directly.

## The problem this exists to avoid

It is tempting to look up a published MovieLens number and put it next to ours. That
comparison is invalid, for four reasons that stack:

1. **Dataset.** We built on ML-25M; the literature standard is ML-1M. Published
   full-ranking leave-one-out results on ML-25M barely exist.
2. **Preprocessing.** Ours is `rating >= 3.5` + 5-core. Published work commonly uses
   `rating >= 4` and/or 10-core, which changes the user/item universe being ranked over.
3. **Protocol.** Our harness ranks the held-out item against the **entire catalog**.
   Much of the neural-recommender literature ranks it against **~100 sampled negatives**,
   which inflates the same model's apparent accuracy enormously.
4. **Our own split quirk.** We report on the *val* fold (second-most-recent item) while
   the most-recent item sits in `loo_test`, excluded from training. Standard leave-one-out
   trains on everything but the test item.

So instead of importing numbers, we reproduce a reference implementation locally and
control all four variables. Two independent codebases, identical interactions, identical
split, identical ranking protocol — then any remaining difference is actually the models.

## Why RecBole, and why its own Python

RecBole ([paper](https://arxiv.org/pdf/2011.01731), v1.2.1) ships 90+ models and, crucially,
its evaluation settings map one-to-one onto ours:

| RecBole setting | Our equivalent |
|---|---|
| `eval_args.group_by: user` | per-user split |
| `eval_args.order: TO` | timestamp ordering in `splits.leave_one_out_split` |
| `eval_args.split: {'LS': 'valid_and_test'}` | last item → test, 2nd-last → val |
| `eval_args.mode: full` | full-catalog ranking in `harness.evaluate` |
| `eval_args.mode: uni100` | our new `--eval-mode uni100` |

**The catch:** RecBole depends on `ray>=1.13.0,<=2.6.3`, and ray 2.6.3's newest macOS
arm64 wheel is `cp311` — there is no cp312 or later. RecBole therefore cannot share
`recsys_large_scale`'s Python 3.14 environment, which is the main reason this is a
separate project rather than another folder over there.

## Setup

```bash
cd recsys_benchmarks
uv sync
```

`ray` resolves cleanly on Python 3.11 (confirmed on an M-series Mac, Oct 2026).

### The ray problem, and why `run_recbole.py` sidesteps it

RecBole pins `ray<=2.6.3`. That release is from 2023, and the ecosystem has moved past
it twice over: it imports `pkg_resources` (which setuptools 81 began removing) and calls
`np.bool8` (removed in NumPy 2.0). Both surface as import-time crashes on a modern
install, and neither will ever be fixed upstream — ray 2.6.3 is frozen.

Pinning around them (`setuptools<81`, `numpy<2`) works but drags the whole environment
backwards to satisfy a dependency we never use. RecBole only needs ray for its
hyperparameter tuner, and `recbole/quick_start/quick_start.py` is the **only** file in
the package that imports it:

```bash
grep -rl "import ray\|from ray" .venv/lib/python3.11/site-packages/recbole/
# -> recbole/quick_start/quick_start.py
```

So `scripts/run_recbole.py` doesn't call `run_recbole()`. It reimplements that function
against the public API one level down — `Config`, `create_dataset`, `data_preparation`,
`get_model`, `get_trainer` — none of which touch ray. The sequence mirrors upstream
exactly, so results are identical to the stock runner; the only deliberate omission is
the `get_flops()` call, which exists purely to log a FLOPs estimate and is a known
failure point for some architectures.

The `setuptools>=68,<81` pin stays in `pyproject.toml` as cheap insurance in case
something else in the tree reaches for `pkg_resources`, but with ray off the critical
path it should no longer be load-bearing.

### Why numpy and pandas are capped

Removing ray was necessary but not sufficient — RecBole itself is NumPy-1-only.
`recbole/config/configurator.py` has a `compatibility_settings()` method that restores
the NumPy 1.20-deprecated aliases:

```python
np.bool = np.bool_;  np.int = np.int_;  np.float = np.float_
np.complex = np.complex_;  np.object = np.object_;  np.str = np.str_
```

`np.float_`, `np.complex_` and `np.unicode_` were removed outright in NumPy 2.0, and
this method is called unconditionally from `Config.__init__` — so every model fails
before any data is read. A method named `compatibility_settings` being the thing that
breaks compatibility is a decent miniature of the whole problem.

RecBole declares only lower bounds (`numpy>=1.17.2`, `pandas>=1.3`), so a fresh resolve
in 2026 pulls numpy 2.4 and pandas 3.0 against a framework released in Feb 2025. The
`numpy>=1.26,<2` and `pandas>=2.0,<3` caps hold the environment at the era RecBole was
actually written against. Expect `uv sync` to backtrack torch and scipy accordingly;
that is intended, not a problem — RecBole only needs `torch>=1.10`.

### A note for the write-up

Getting the field's standard framework to *start* took four fixes: a missing
`setuptools`, a `pkg_resources` removal, a dead ray dependency, and a NumPy 2 break — none
of them related to recommendation, all of them before a single model trained. The
reproducibility papers we cite ask whether published *results* reproduce. This is the
more mundane layer underneath: whether the code still runs at all, three years on.

## Where each phase runs

The work is split across two projects because of the Python version constraint above,
so the phase numbering spans both READMEs:

| Phase | What | Where it runs |
|---|---|---|
| B0 | `uv sync` — install RecBole | here |
| B1 | our five models on ML-1M, both protocols | `../recsys_large_scale` (commands below) |
| B2 | RecBole on its own ml-1m | here |
| B3 | RecBole on our exported split, then join | here |
| B4 | the protocol study (both sides) | both |
| B5 | scale up to ML-25M | here |

B1 and B2 are independent — either order works. B3 is the first step that needs B1's
output, since it feeds RecBole the interaction set our pipeline produced.

## B1 — our five models on ML-1M

Runs in the *other* project, because that's where our pipeline and its Python 3.14
environment live. The commands are repeated here so this file reads as one continuous
runbook; the fuller explanation of each step is under "Phase B1 RUNBOOK — MovieLens 1M"
at the bottom of `../recsys_large_scale/README.md`.

ML-1M is ~6,040 users, so every command below is seconds to a couple of minutes.

```bash
cd ../recsys_large_scale

# 1. download + build splits (ML-1M ships "::"-separated .dat files, hence loader: movielens_dat)
uv run python scripts/download_data.py --config configs/data_movielens1m.yaml
uv run python scripts/build_splits.py  --config configs/data_movielens1m.yaml

# 2. train
for m in popularity itemknn als als_alpha1 bpr; do
  uv run python scripts/train_model.py \
    --data-config configs/data_movielens1m.yaml \
    --model-config configs/model_${m}_ml1m.yaml
done

# 3. evaluate on BOTH folds, full-catalog -> artifacts/results/metrics_log.csv
for m in popularity itemknn als als_alpha1 bpr; do
  for s in val test; do
    uv run python scripts/evaluate_model.py \
      --data-config configs/data_movielens1m.yaml \
      --eval-config configs/eval_default.yaml \
      --model-artifact artifacts/models/${m}_ml1m.pkl --split $s
  done
done

# 4. evaluate again under sampled negatives -> artifacts/results/protocol_log.csv
for m in popularity itemknn als als_alpha1 bpr; do
  uv run python scripts/evaluate_model.py \
    --data-config configs/data_movielens1m.yaml \
    --eval-config configs/eval_default.yaml \
    --model-artifact artifacts/models/${m}_ml1m.pkl --split val --eval-mode uni100
done

cd ../recsys_benchmarks
```

Step 3 covers both folds because B3 compares test-to-test (see the note under B2 on
RecBole's validation-fold selection bias). Step 4 is the our-side half of B4: identical
models, identical data, only the ranking protocol changed — expect the numbers to jump
sharply, and note that coverage/novelty/popularity_bias/ILS are meaningless under that
mode, since they describe a random candidate pool rather than the model.

## B2 — RecBole on its own ml-1m

This is the control: does our RecBole configuration produce sane numbers at all?

**First, build the dataset.** RecBole would normally auto-download it, but its mirror
(`recbole.s3-accelerate.amazonaws.com`) now returns HTTP 403 — the hosted copy has
rotted. `build_recbole_ml1m.py` reconstructs the identical atomic file from GroupLens,
the authoritative source. Nothing is lost in doing so: RecBole's `ml-1m.inter` is just
`ratings.dat` in tab-separated form with typed headers, and all filtering happens at
config time, not in the file.

```bash
uv run python scripts/build_recbole_ml1m.py
```

Expect `1,000,209 interactions, 6,040 users, 3,706 items` — the script prints those
reference figures so a mismatch is obvious immediately. It reuses the raw file B1
downloaded if present, and fetches `ml-1m.zip` itself otherwise.

**Then run the models.**

```bash
uv run python scripts/run_recbole.py \
  --config configs/recbole_ml1m_full.yaml \
  --models Pop,ItemKNN,BPR --fix-pop
```

Then widen to neural models once the classical three work:

```bash
uv run python scripts/run_recbole.py \
  --config configs/recbole_ml1m_full.yaml \
  --models NeuMF,LightGCN,MultiVAE
```

All six names are confirmed present in RecBole 1.2.1's general-recommender module, as are
`EASE`, `SLIMElastic`, `ADMMSLIM`, `RecVAE`, `NGCF`, `DMF`, `SimpleX` and `Random` if you
want to widen further. To re-list them in any future version:

```bash
uv run python -c "
import recbole.model.general_recommender as g
print(sorted(n for n in dir(g) if n[0].isupper()))"
```

`run_recbole.py` catches a failure per model and keeps going, printing a summary of what
failed at the end, so a bad name or an out-of-memory model costs you one skipped entry
rather than the whole sweep.

Results land in `results/recbole_log.csv`, with **both** `val` and `test` rows.

**Use the test rows for the head-to-head.** RecBole early-stops on the validation fold,
so `best_valid_result` is a maximum selected over training epochs using that same fold —
optimistically biased. The effect is visible and large: BPR scores 0.0923 recall@10 on
val but 0.0759 on test (−18%), while ItemKNN, which has no epochs to select over, drops
only 7.5%. Our models train for fixed iterations with no early stopping, so comparing
val-to-val would credit RecBole with a purely methodological advantage. Both sides' test
numbers are clean.

## B3 — The airtight head-to-head

Hand RecBole our exact interactions instead of its own copy of MovieLens:

```bash
uv run python scripts/export_to_recbole.py \
  --processed-dir ../recsys_large_scale/data/processed/movielens1m \
  --name ml1m-ours

uv run python scripts/run_recbole.py \
  --config configs/recbole_ml1m_ours_full.yaml \
  --models Pop,ItemKNN,BPR,NeuMF,MultiVAE,LightGCN --fix-pop
```

The export writes our three folds as three separate files
(`ml1m-ours.{train,valid,test}.inter`), and the config consumes them via
`benchmark_filename`. RecBole then skips both its own filtering and its own splitting and
uses the folds exactly as given, so the comparison controls the *split*, not merely the
data — both systems hold out identical items for identical users.

Then join both sides:

```bash
uv run python scripts/build_comparison.py \
  --ours-dataset movielens1m \
  --recbole-dataset recbole_ml1m_ours_full
```

This writes `results/comparison.csv` and a grouped bar chart, and prints which models
exist on only one side.

### Why every Pop run needs `--fix-pop`

**RecBole's `Pop` model is broken and must not be used as-is.** `Pop.calculate_loss`
accumulates with `self.item_cnt[item, :] = self.item_cnt[item, :] + 1`, where `item` is a
batch of ids full of duplicates. PyTorch indexed assignment gathers then scatters, so
duplicates do not accumulate — an item appearing 50 times in a batch is credited +1, not
+50. The result counts *batches an item appeared in*, saturating every popular item at
(batches per epoch × epochs). On ML-1M all of RecBole's top-20 items came out at exactly
275, against true counts of 2803 down to 1734.

`--fix-pop` swaps the loop for `torch.bincount`, which accumulates correctly. It is
opt-in rather than always-on so the default still reproduces stock RecBole — what a paper
using this baseline actually gets. Pair it with `train_neg_sample_args: ~`, since `Pop`
is `InputType.POINTWISE` and otherwise counts sampled negatives as real interactions:

```bash
uv run python scripts/run_recbole.py \
  --config configs/recbole_ml1m_ours_full.yaml \
  --models Pop --fix-pop \
  --config-dict 'train_neg_sample_args: ~' \
  --tag recbole_ml1m_ours_full_popcorrect
```

With both applied, RecBole's Pop reproduces ours to four decimal places on the validation
fold (0.0449 recall / 0.0209 NDCG, both systems). The full investigation — including two
hypotheses falsified along the way — is the "Phase B3" entry in `LEARNINGS.md`.

`--tag` writes to a separate `dataset` value so a variant never overwrites a baseline row.
`--config-dict` applies per-run overrides, needed because a setting can be right for one
model and wrong for another (disabling negative sampling suits Pop but would break BPR,
which needs pairwise negatives). `scripts/diagnose_pop.py` reads RecBole's learned counts
out of its checkpoint and diffs the ranking against ours item by item.

**The checks that decide whether a comparison means anything:**

- **Our `popularity` vs. RecBole's corrected `Pop` must come out nearly identical.** Same
  trivial algorithm, same data. If they diverge, one of the two harnesses has a bug and
  nothing else can be trusted until it is explained. (Expect a residual of ~0.0003 on the
  *test* fold only: RecBole masks train ∪ valid there while we mask train only, so it
  masks one extra item and scores marginally higher. On val the masks match exactly.)
- **RecBole-on-its-own-ml-1m (B2) vs. RecBole-on-our-export (B3) should be close.** A
  large gap means our *preprocessing* rather than our *models* drives the difference —
  itself a finding worth writing down.

A caveat that *used* to apply here and no longer does: when RecBole did its own splitting,
timestamp ties were a real confound — ML-1M users rate in bursts, our splitter breaks ties
by row order and RecBole by its own rule, so the two could hold out different items for
the same user. Pre-splitting via `benchmark_filename` removes that entirely; the folds are
now bit-for-bit identical. (It was also the first hypothesis for the Pop discrepancy, and
measurement falsified it — pre-splitting made that gap *wider*, which is what redirected
the investigation to RecBole's counting loop.)

## B4 — The protocol study

Run the same models under sampled-negative ranking on both sides.

On **our split** (the one that compares directly to our `--eval-mode uni100` numbers,
since both systems then hold out identical items):

```bash
uv run python scripts/run_recbole.py \
  --config configs/recbole_ml1m_ours_uni100.yaml \
  --models Pop,ItemKNN,BPR,NeuMF,MultiVAE,LightGCN --fix-pop
```

Optionally also on **RecBole's own ml-1m**, which measures the protocol effect entirely
within RecBole, independent of anything of ours:

```bash
uv run python scripts/run_recbole.py \
  --config configs/recbole_ml1m_uni100.yaml \
  --models Pop,ItemKNN,BPR --fix-pop
```

The our-side half is step 4 of the Phase B1 RUNBOOK in `../recsys_large_scale/README.md`
(`--eval-mode uni100`). Expect both sides to jump dramatically for *identical models on
identical data*. That gap, measured rather than asserted, is the answer to why published
numbers so often look several times better than ours.

## B5 — Scale up to ML-25M

```bash
uv run python scripts/export_to_recbole.py \
  --processed-dir ../recsys_large_scale/data/processed/movielens25m \
  --name ml25m-ours

uv run python scripts/run_recbole.py \
  --config configs/recbole_ml25m_ours_full.yaml \
  --models Pop,ItemKNN,BPR --fix-pop
```

Start with the classical three, then decide whether the neural models are affordable
based on how long that took. For reference: Pop + ItemKNN + BPR took **3.5 hours** on an
M-series CPU, so LightGCN (graph convolution over a 161,572 × 23,965 adjacency each
epoch) is likely impractical. Dropping it is a documented outcome, not a failure.

```bash
uv run python scripts/run_recbole.py \
  --config configs/recbole_ml25m_ours_full.yaml \
  --models NeuMF,MultiVAE,LightGCN
```

Two more runs are needed before the ML-25M comparison is valid.

**A corrected Pop**, since the stock baseline is broken (see above):

```bash
uv run python scripts/run_recbole.py \
  --config configs/recbole_ml25m_ours_full.yaml \
  --models Pop --fix-pop \
  --config-dict 'train_neg_sample_args: ~' \
  --tag recbole_ml25m_ours_full_popcorrect
```

**Our own models on the test fold.** Phases 0–6 only ever evaluated ML-25M on `val`,
because test-to-test only became the right comparison once B2 exposed RecBole's
early-stopping bias. This is evaluation only — no retraining:

```bash
cd ../recsys_large_scale
for m in popularity itemknn als als_alpha1 bpr two_stage two_tower; do
  uv run python scripts/evaluate_model.py \
    --data-config configs/data_movielens25m.yaml \
    --eval-config configs/eval_default.yaml \
    --model-artifact artifacts/models/${m}.pkl --split test
done
cd ../recsys_benchmarks
```

15.5M interactions on CPU: expect this to be slow, and expect some neural models to be
impractical. Dropping them here is a documented outcome, not a failure — the classical
models are what the comparison rests on.

## Context worth reading

- Dacrema, Cremonesi & Jannach, *[Are We Really Making Much Progress?](https://arxiv.org/pdf/1907.06902)*
  (RecSys'19) — reproduced 18 published neural recommenders; only 7 reproduced, and most
  of those were matched or beaten by well-tuned classical baselines. Directly echoes this
  project's own Phase 6 finding, where the two-tower model needed three attempts to reach
  parity with item-item CF.
- Ji, Sun et al., *[Our Model Achieves Excellent Performance on MovieLens: What Does it
  Mean?](https://arxiv.org/abs/2307.09985)* (TOIS 2024) — on what strong MovieLens numbers
  do and don't imply about real-world recommendation quality.

---

## B6 — Comparing against published numbers directly

B1–B5 compare us against RecBole, an implementation we run ourselves. That is rigorous
but answers a narrower question than "how do we compare to published results". B6 closes
that gap by matching the setup of a specific, highly-cited paper exactly.

**The paper:** Neural Collaborative Filtering, He et al., WWW'17
([arXiv 1708.05031](https://arxiv.org/pdf/1708.05031)) — the origin of the
sampled-negative protocol most ML-1M results use. From its §4.1 and Table 1:

| | NCF paper | our `data_movielens1m.yaml` |
|---|---|---|
| interactions | 1,000,209 | 574,376 |
| items / users | 3,706 / 6,040 | 3,125 / 6,034 |
| rating threshold | **none** — every rating is a positive | `>= 3.5` |
| k-core | none (ML-1M's own >=20/user) | 5-core, iterative |
| held out per user | 1 test + 1 validation | 1 test + 1 validation |
| ranking protocol | **100 sampled negatives**, cut at 10 | full catalog |

The protocol already matches our `--eval-mode uni100` exactly (100 negatives plus the
true item). Only the preprocessing differed, which `configs/data_movielens1m_ncf.yaml`
now reproduces.

Run from `../recsys_large_scale`:

```bash
cd ../recsys_large_scale

uv run python scripts/build_splits.py --config configs/data_movielens1m_ncf.yaml
```

**Check the printed shape before going further: 6,040 users / 3,706 items / 1,000,209
interactions.** Anything else means the comparison is invalid.

```bash
for m in popularity itemknn als als_alpha1 bpr; do
  uv run python scripts/train_model.py \
    --data-config configs/data_movielens1m_ncf.yaml \
    --model-config configs/model_${m}_ncf.yaml
done

for m in popularity itemknn als als_alpha1 bpr; do
  uv run python scripts/evaluate_model.py \
    --data-config configs/data_movielens1m_ncf.yaml \
    --eval-config configs/eval_default.yaml \
    --model-artifact artifacts/models/${m}_ncf.pkl --split test --eval-mode uni100
done

cd ../recsys_benchmarks
```

Note `--split test`: the paper holds out each user's *latest* interaction as test, which
is our `loo_test`. (Its validation item is randomly sampled rather than second-most-recent
— a minor difference in what leaves the training set.)

### What to compare against

From the paper itself (Table 2 is exact; the BPR/eALS figures are derived from its stated
"relative improvement over eALS and BPR is 4.5% and 4.9%" and read off Figure 4, so treat
them as approximate):

| model | HR@10 | NDCG@10 | source |
|---|---|---|---|
| NeuMF (64 factors, pre-trained) | **0.730** | **0.447** | Table 2, exact |
| NeuMF (64 factors, no pre-training) | 0.705 | 0.426 | Table 2, exact |
| NeuMF (16 factors) | 0.707 | 0.426 | Table 2, exact |
| eALS | ~0.70 | ~0.43 | derived, approximate |
| BPR | ~0.70 | ~0.43 | derived, approximate |
| ItemKNN | ~0.62 | ~0.40 | Figure 4, approximate |

One caveat worth carrying into any conclusion: Dacrema et al. observed that NCF's released
code selects the number of epochs using the *test* set, so these published numbers are
optimistically biased. A close result on our part is therefore better than it looks.
