# Recommender Systems at Scale

Advanced recommender-systems project: multiple modeling approaches at ≥1,000,000 rows,
proper accuracy/ranking metrics, latency measurement, and personalization.

Primary dataset: [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) (25M
ratings). Secondary dataset (Phase 5, personalization): the Kaggle
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
dataset.

**How this project works**: everything runs by hand — env setup, data download, training,
evaluation, serving — following the RUNBOOK steps below. Code (`src/`, `scripts/`,
`configs/`) is provided for each phase; the "Phase N RUNBOOK" section states exactly what
to run and what to look at in the output. Each phase ends with a "What to check" step
listing the numbers that phase produces, so a run can be verified against the recorded
ones rather than merely completing without error.

Every model has a write-up in [`LEARNINGS.md`](LEARNINGS.md): what worked, what didn't,
and why. Read that file alongside the runbooks — it carries the reasoning behind the
configuration choices the configs encode, and several source files point back to it.

## Project layout

```
configs/      one YAML per dataset/model/eval — no hardcoded constants in code
data/         gitignored; raw/ and processed/ populated by the scripts below
src/recsys_large_scale/
    data/       download, load, preprocess, split
    models/     popularity, item-item CF, ALS, BPR, candidate gen, reranker
    features/   reranker feature engineering
    evaluation/ metrics + evaluation harness + segment breakdowns
    serving/    FAISS index, FastAPI app, latency benchmark
    utils/      logging, timers, config loading
scripts/      thin CLIs that call into src/ — this is what you actually run
notebooks/    exploratory/teaching notebooks only; pipeline logic lives in src/
artifacts/    gitignored; trained models, indices, metrics_log.csv
tests/        pytest — mainly the metrics/splits correctness checks
```

## One-time environment setup

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv` if you don't have it).

```bash
cd recsys_large_scale
uv sync --group dev
```

This creates `.venv/` and installs everything (pandas, numpy, scipy, scikit-learn,
implicit, faiss-cpu, lightgbm, polars, fastapi, pytest, etc.) per `pyproject.toml`. It
may take a few minutes the first time (implicit/faiss/lightgbm are compiled packages).

**LightGBM note** (needed from Phase 3 onward, not Phase 0): it needs the OpenMP runtime,
which isn't bundled on macOS:
```bash
brew install libomp
```
If you hit an import error on `lightgbm` later, this is almost always why.

Sanity-check the environment:
```bash
uv run python -c "import pandas, numpy, scipy, sklearn, implicit, faiss, lightgbm, polars; print('ok')"
```

---

## Phase 0 RUNBOOK — Setup + Data

**What this phase does**: downloads MovieLens 25M, converts explicit ratings to
implicit positive feedback, filters sparse users/items, remaps IDs to contiguous
indices, and produces two kinds of train/val/test splits (per-user leave-one-out, and a
global timestamp holdout) as parquet files — with a test suite proving there's no
train/test leakage.

1. **Download the dataset** (~250MB zip, downloaded via plain HTTP — no login needed):
   ```bash
   uv run python scripts/download_data.py --config configs/data_movielens25m.yaml
   ```
   This extracts into `data/raw/movielens25m/`. Safe to re-run — it skips the download
   if that folder already has content.

2. **Build the processed splits**:
   ```bash
   uv run python scripts/build_splits.py --config configs/data_movielens25m.yaml
   ```
   Watch the log output — it prints, at each stage: raw row count, how many rows survive
   the `positive_feedback` filter, how many users/items survive the
   `min_user_interactions`/`min_item_interactions` filter, the final `n_users`/`n_items`,
   and the row counts of each split (`loo_train`/`loo_val`/`loo_test`,
   `global_train`/`global_test`), plus how many test-split users/items were never seen
   in train (the global split's cold-start count).

3. **Run the correctness tests** (these check the split logic itself — no leakage, most
   recent interaction correctly held out, etc. — on small synthetic data, not the full
   dataset):
   ```bash
   uv run pytest tests/ -v
   ```
   All tests should pass. If one fails, that's a real bug in the split logic — worth
   flagging before building anything on top of it.

4. **What to check**: the log output from step 2 (row counts, n_users, n_items,
   cold-start counts) against the confirmed scale recorded below, and a clean pytest
   result from step 3. `LEARNINGS.md` opens with the dataset's actual scale and shape.

### Confirmed scale (run 2026-09-17)

- Raw: 25,000,095 ratings.
- After `positive_feedback.mode=threshold` (rating ≥ 3.5): 15,630,129 rows (62.5%).
- After `min_user_interactions=5`, `min_item_interactions=5`: 15,584,360 rows,
  **161,572 users, 23,965 items**.
- Leave-one-out split: train=15,261,216 / val=161,572 / test=161,572.
- Global time split (14-day holdout): train=15,555,128 / test=29,232, of which 192 test
  users and 6 test items are never seen in train (genuine cold-start cases).
- All 5 split-correctness tests pass.

This comfortably clears the ≥1,000,000-row target (15.6M interactions) while staying a
manageable in-memory size for the Mac-local modeling phases.

**Note**: `download_data.py`'s zip extraction can leave `ratings.csv`/`movies.csv` one
level deeper than `raw_dir` (e.g. `data/raw/movielens25m/ml-25m/ratings.csv`). The data
loaders now search recursively under `raw_dir` for the expected filename, so either
`raw_dir: data/raw/movielens25m` or the more specific
`data/raw/movielens25m/ml-25m` (as currently set in `configs/data_movielens25m.yaml`)
works — no manual config edits needed on a fresh run.

---

---

## Phase 1 RUNBOOK — Baselines + Evaluation Harness

**What this phase does**: two models (a non-personalized popularity baseline and
item-item collaborative filtering) plus a generalized evaluation harness that scores
every user's full catalog, masks out items they've already seen, and computes seven
metrics per K: Recall, Precision, NDCG, MRR, MAP, Catalog Coverage, and Novelty. This
generalizes the sibling notebooks' `evaluate(score_fn, ...)` pattern — same idea (mask
seen items, top-K via argpartition), batched for speed and extended with more metrics.

Note on MAP vs. MRR: with this dataset's leave-one-out split (exactly one held-out item
per user), MAP@K and MRR@K are mathematically identical — see the docstring in
`evaluation/metrics.py`. They'll diverge once evaluation moves to a setting with
multiple relevant items per user (planned for Phase 5's H&M personalization work).

1. **Run the correctness tests first** (metrics computed against hand-worked toy
   examples, plus a harness test that specifically proves masking works — it constructs
   a case where, if masking silently failed, the metrics would be provably wrong):
   ```bash
   uv run pytest tests/ -v
   ```

2. **Train the popularity baseline**:
   ```bash
   uv run python scripts/train_model.py \
     --data-config configs/data_movielens25m.yaml \
     --model-config configs/model_popularity.yaml
   ```
   Instant — this is just counting interactions per item. Saves
   `artifacts/models/popularity.pkl`.

3. **Train item-item CF**:
   ```bash
   uv run python scripts/train_model.py \
     --data-config configs/data_movielens25m.yaml \
     --model-config configs/model_itemknn.yaml
   ```
   This computes a ~24,000 x 24,000 item-item cosine similarity matrix, then truncates
   it to each item's top-200 neighbors — expect this to take a few minutes (the
   truncation step logs progress; the log will tell you the similarity matrix's
   nonzero count before/after truncation). Saves `artifacts/models/itemknn.pkl`.

4. **Evaluate both, on the validation split** (test split stays untouched until you're
   done comparing/tuning):
   ```bash
   uv run python scripts/evaluate_model.py \
     --data-config configs/data_movielens25m.yaml \
     --eval-config configs/eval_default.yaml \
     --model-artifact artifacts/models/popularity.pkl --split val

   uv run python scripts/evaluate_model.py \
     --data-config configs/data_movielens25m.yaml \
     --eval-config configs/eval_default.yaml \
     --model-artifact artifacts/models/itemknn.pkl --split val
   ```
   Each prints Recall/Precision/NDCG/MRR/MAP/Coverage/Novelty at K=5,10,20 and appends a
   row per K to `artifacts/results/metrics_log.csv`.

5. **What to check**: the printed metrics for both models, also written to
   `artifacts/results/metrics_log.csv`. Popularity's numbers are the floor; item-item
   CF's are the first personalized result to set against it. The first two `LEARNINGS.md`
   entries record both, with an explanation of *why* CF moves the numbers the way it does
   (or doesn't).

---

---

## Phase 2 RUNBOOK — Scaled Matrix Factorization (ALS + BPR)

**What this phase does**: two latent-factor models via `implicit` (Cython/OpenMP,
native on Apple Silicon) — ALS (confidence-weighted, per Hu/Koren/Volinsky) and BPR
(pairwise ranking loss) — replacing the sibling notebooks' from-scratch NumPy BPR,
which wouldn't finish in reasonable time at this scale. Also adds a new metric,
**intra-list similarity@K** (mean pairwise cosine similarity among each user's
recommended list, using the model's own learned item embeddings — lower means more
diverse lists); it's only computed for models that have an embedding space, so
popularity/item-item CF rows keep `intra_list_similarity` blank.

There's also a small hyperparameter comparison built in: `configs/model_als.yaml`
(alpha=20, i.e. observed interactions count for a lot) vs.
`configs/model_als_alpha1.yaml` (alpha=1, i.e. barely more than an unobserved pair) —
same everything else. This is a direct, interpretable ablation of what `alpha` is
actually doing in our setup (see the code comment in `models/als.py` for why alpha acts
as one global weight here rather than a per-interaction graded signal).

1. **Run the tests** (adds correctness checks for intra-list-similarity, including that
   it's correctly left `None` when a model has no embeddings):
   ```bash
   uv run pytest tests/ -v
   ```

2. **Train ALS** (default, alpha=20):
   ```bash
   uv run python scripts/train_model.py \
     --data-config configs/data_movielens25m.yaml \
     --model-config configs/model_als.yaml
   ```
   Expect this to take on the order of a minute or two at 15.6M interactions (factors=64,
   15 iterations, multi-threaded).

3. **Train the ALS alpha=1 comparison variant**:
   ```bash
   uv run python scripts/train_model.py \
     --data-config configs/data_movielens25m.yaml \
     --model-config configs/model_als_alpha1.yaml
   ```

4. **Train BPR**:
   ```bash
   uv run python scripts/train_model.py \
     --data-config configs/data_movielens25m.yaml \
     --model-config configs/model_bpr.yaml
   ```
   BPR here runs 100 SGD epochs over 15.6M interactions — this will likely take longer
   than ALS (the log will show a per-epoch progress bar with a running training AUC, so
   you can see it's actually progressing rather than stalled). If it's taking too long,
   it's safe to Ctrl+C, lower `iterations` in `configs/model_bpr.yaml`, and re-run.

5. **Evaluate all three on the validation split**:
   ```bash
   uv run python scripts/evaluate_model.py \
     --data-config configs/data_movielens25m.yaml --eval-config configs/eval_default.yaml \
     --model-artifact artifacts/models/als.pkl --split val

   uv run python scripts/evaluate_model.py \
     --data-config configs/data_movielens25m.yaml --eval-config configs/eval_default.yaml \
     --model-artifact artifacts/models/als_alpha1.pkl --split val

   uv run python scripts/evaluate_model.py \
     --data-config configs/data_movielens25m.yaml --eval-config configs/eval_default.yaml \
     --model-artifact artifacts/models/bpr.pkl --split val
   ```
   Each appends rows (now including `intra_list_similarity`) to
   `artifacts/results/metrics_log.csv`.

6. **What to check**: the printed metrics (or the CSV) for all three runs.
   `LEARNINGS.md` has entries for ALS and BPR against the Phase 1 baselines, plus a
   short note on what the alpha=20 vs. alpha=1 comparison actually showed.

---

---

## Phase 3 RUNBOOK — Two-Stage Ranking (Candidate Generation + Reranking)

**What this phase does**: a FAISS ANN index (`IndexIVFFlat`, deliberately approximate)
retrieves `n_candidates` (200) candidate items per user from the ALS α=1 model's
embedding space (Phase 2 found α=1 generalizes best — see `LEARNINGS.md`), then a
LightGBM `lambdarank` reranker re-scores just those candidates using 5 engineered
features (the ANN score itself, item popularity, user activity, item-item CF's score
for that pair, and a genre-affinity score from `movies.csv`). The reranker is trained
on the **validation** split (never test — see the leakage note in
`scripts/run_two_stage_pipeline.py`'s docstring), and the whole thing evaluates through
the exact same `scripts/evaluate_model.py` every other model uses, since
`TwoStageRecommender` implements the same `score_batch` interface.

**Known environment gotcha, already fixed in the code but worth knowing about**: this
phase is the first to load `faiss` and `lightgbm` in the same process, and on macOS
that combination is unstable in more than one way — each of faiss / scikit-learn /
lightgbm ships its own separate physical copy of the OpenMP runtime (faiss and
scikit-learn bundle a private `libomp.dylib` inside their wheels; lightgbm links
Homebrew's). Confirmed experimentally, there are two distinct failure modes depending
on what triggers it:
- LightGBM's multi-threaded `fit()` **segfaults** (exit code 139) once `faiss` is also
  loaded — fixed by `models/reranker.py` pinning `num_threads=1`.
- Importing `faiss` before `lightgbm`, then unpickling a fitted LightGBM model, also
  **segfaults** — even with `num_threads=1` already set. And the fix for *that* (import
  `lightgbm` first) trades it for a **deadlock** (0% CPU, process never returns) the
  first time FAISS then runs a multi-threaded search.

The combination actually verified stable end-to-end — unpickling the real
`two_stage.pkl` and calling `score_batch` repeatedly — is: **`lightgbm` imported before
`faiss`, and FAISS forced single-threaded.** Both are handled automatically, not
something you need to do yourself:
- `src/recsys_large_scale/__init__.py` imports `lightgbm` first, so it's loaded before
  `faiss` no matter which submodule or script touches this package first.
- `serving/ann_index.py` calls `faiss.omp_set_num_threads(1)` at import time (a cheap
  tradeoff — this project's ~24K-item catalog is far too small for FAISS's own speed to
  ever be the bottleneck).

If you ever see exit code 139 (segfault) or a process that just sits at ~0% CPU forever
around code that touches both libraries, suspect this class of issue first, not a new
bug — and don't reorder or remove that top-of-package `import lightgbm` (an
auto-formatter alphabetizing imports is exactly the kind of change that could silently
reintroduce this).

1. **Run the tests** (feature-engineering correctness + the ANN-vs-exact recall
   diagnostic's own logic):
   ```bash
   uv run pytest tests/ -v
   ```

2. **Train the two-stage pipeline**:
   ```bash
   uv run python scripts/run_two_stage_pipeline.py \
     --data-config configs/data_movielens25m.yaml \
     --pipeline-config configs/model_two_stage.yaml
   ```
   Watch the log for two specific numbers before it finishes:
   - how many of the 20,000 training (validation) users had their held-out item
     outside the top-200 ANN candidates and had to be injected — a direct read on how
     much the retrieval stage alone would have missed;
   - the **ANN candidate recall** diagnostic (a 5,000-user sample): what fraction of
     each user's *exact* top-20 (by the base ALS model's own full-catalog scoring) is
     present within the 200 ANN-retrieved candidates. This number is the two-stage
     pipeline's accuracy ceiling from retrieval alone — the reranker cannot recover
     whatever this step already lost.

   LightGBM training is single-threaded (see the gotcha above), so this may take
   longer than Phase 2's models felt like they should, proportionate to
   `max_training_users × n_candidates` rows (20,000 × 200 = 4M rows by default).

3. **Evaluate it exactly like every other model**:
   ```bash
   uv run python scripts/evaluate_model.py \
     --data-config configs/data_movielens25m.yaml \
     --eval-config configs/eval_default.yaml \
     --model-artifact artifacts/models/two_stage.pkl --split val
   ```

4. **What to check**: the ANN candidate recall number, the injection count from
   step 2, and the evaluate_model.py output from step 3. The `LEARNINGS.md` entry
   compares the two-stage pipeline against ALS/BPR — including whether the retrieval
   ceiling from step 2 actually explains the accuracy shortfall, rather than just
   asserting it does.

### Update: `is_injected` feature (fixes a real regression found in the first run)

The first real run of this pipeline (see `LEARNINGS.md`) came back *worse* than every
prior model, including plain popularity. Diagnosis: 61.7% of training queries needed
their true item injected into the candidate set (ANN missed it), and an injected item's
`ann_score` is *necessarily* low — yet its label is positive, teaching the reranker a
majority-share contradiction for its own strongest feature. Fixed by adding a 6th
feature, `is_injected` (always 0 at inference — score_batch never injects, only real
retrieval candidates reach it), so the reranker can learn to treat injected and organic
rows differently instead of conflating them. If you already ran the pipeline before
this fix landed, **re-run steps 2 and 3 above** — `run_two_stage_pipeline.py` will
retrain and overwrite `artifacts/models/two_stage.pkl` with the new 6-feature reranker.

---

---

## Phase 4 RUNBOOK — Latency Benchmark (Retrieval, not Serving)

**What this phase does, and why it's scoped down from the original plan**: the plan
originally paired a FastAPI serving app with a brute-force-vs-FAISS latency comparison.
Those are actually two separate concerns — FastAPI is about serving *mechanics* (HTTP,
serialization, routing overhead), which only matters if we cared about exposing this as
a real service. We don't (nothing here is being hosted for anyone), so it's dropped.
What's left is the actually-interesting question: how much does approximate retrieval
(FAISS) speed things up over exact scoring, and how much latency does the reranker
stage add on top — answered with a plain in-process timing loop, no server involved.

`scripts/run_latency_benchmark.py` times three ways of producing a ranked top-K list,
one user "request" at a time (a loop calling the scoring path once per sampled user —
this simulates one-request-at-a-time arrival without needing an actual HTTP layer):

1. **`brute_force_exact`** — the base ALS model scores every item in the catalog,
   `argpartition` picks the top-K.
2. **`faiss_ann_retrieval`** — the two-stage pipeline's retrieval stage alone: FAISS
   ANN search over the same ALS item embeddings, top-K already sorted by FAISS.
3. **`two_stage_full`** — retrieval's 200 candidates, rescored by the LightGBM
   reranker, sorted down to top-K.

(1) vs (2) isolates pure retrieval speed, exact vs. approximate. (3) minus (2) is an
approximate reranker overhead — feature building plus LightGBM inference on 200 rows.
Each path is timed after a warmup period (absorbs one-off costs like first-touch page
faults on the embedding arrays), and reports mean/p50/p95/p99 in milliseconds.

**Honest expectation going in**: this catalog is only ~24,000 items, so brute-force
full-catalog scoring is likely already sub-millisecond — the interesting story is
probably more about p50-vs-p99 tail behavior and the reranker's added cost than about
FAISS being dramatically faster. That's a legitimate finding, not a failed experiment,
if that's what the numbers show.

1. **Run it**:
   ```bash
   uv run python scripts/run_latency_benchmark.py \
     --data-config configs/data_movielens25m.yaml \
     --als-model artifacts/models/als_alpha1.pkl \
     --two-stage-model artifacts/models/two_stage.pkl
   ```
   Defaults: k=20, 500 timed requests, 50 warmup requests. Appends one row per path to
   `artifacts/results/latency_log.csv`. Takes seconds, not minutes — this is a read-only
   benchmark against already-trained models, no training involved.

2. **What to check**: the printed mean/p50/p95/p99 for all three paths (or the CSV).
   The `LEARNINGS.md` entry covers what the retrieval-vs-rerank latency tradeoff
   actually looks like at this scale, and whether FAISS's approximate speedup was even
   necessary given the catalog size.

---

---

---

## Phase 5a RUNBOOK — H&M Data Pipeline

**Scope note**: Phase 5 (personalization) is a big phase — data pipeline, cold-start
experiments, segment breakdowns, popularity-bias analysis, and a recency-aware reranker
feature. Splitting it the same way Phase 0 kicked off the MovieLens work: get the data
pipeline running and confirm real row/user/item counts first (5a, this section), then
build the modeling/analysis pieces against the dataset's actual shape rather than
assumed numbers (5b, next).

**What this phase does**: downloads three CSVs from the Kaggle H&M competition
(transactions, customers, articles — deliberately *not* the ~36GB product-image
archive, which we don't need), subsamples to a recent date window, converts to implicit
feedback (every purchase is already a positive — no rating threshold like MovieLens
needed), remaps IDs, and reuses the exact same `filter_min_interactions` /
`build_id_maps` / `leave_one_out_split` / `global_time_split` logic already proven on
MovieLens (those functions only ever look at generic `user_id`/`item_id`/`timestamp`
columns, so nothing dataset-specific had to change there).

1. **Set up Kaggle credentials** (one-time, if you don't already have these):
   - Create a Kaggle account if needed, go to the
     [competition page](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
     and click **Join Competition** / accept the rules — the API returns a 403 until you
     do this, even with a valid token.
   - Account → Settings → **Create New API Token**, which downloads `kaggle.json`. Place
     it at `~/.kaggle/kaggle.json` and lock down its permissions:
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
     chmod 600 ~/.kaggle/kaggle.json
     ```

2. **Download just the three CSVs** (not the whole competition bundle — `-f` requests a
   single file at a time, each arrives pre-zipped):
   ```bash
   mkdir -p data/raw/hm
   uv run kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f transactions_train.csv -p data/raw/hm
   uv run kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f customers.csv -p data/raw/hm
   uv run kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f articles.csv -p data/raw/hm
   cd data/raw/hm && for f in *.zip; do unzip -o "$f" && rm "$f"; done && cd -
   ```
   Expect this to be noticeably bigger and slower than MovieLens's 250MB —
   `transactions_train.csv` alone is on the order of a few GB (31.8M rows, full ~2-year
   history; we subsample it down in the next step, but the download itself is the full
   file — Kaggle doesn't support partial-file downloads).

3. **Build the processed splits**:
   ```bash
   uv run python scripts/build_splits_hm.py --config configs/data_hm.yaml
   ```
   Watch the log for: how many rows survive the 90-day date window, how many survive
   the min-interactions filter, final `n_users`/`n_items`, split row counts, and the
   global split's cold-start user/item counts — same shape of output as Phase 0's
   MovieLens run, so directly comparable.

4. **Run the tests** (H&M-specific: the implicit-feedback conversion and date-window
   subsampling logic; everything else — splits, id maps — is already covered by Phase
   0's tests since that logic is unchanged):
   ```bash
   uv run pytest tests/ -v
   ```

5. **What to check**: the log output from step 3 (row counts, n_users, n_items,
   cold-start counts) and the pytest result, against the "Confirmed scale" note under
   Phase 5b part 1. If the 90-day window produces something wildly off from a useful
   scale (too small to be interesting, or too large to be fast on your machine),
   `date_window_days` in `configs/data_hm.yaml` is the one knob to adjust before
   re-running step 3. Phase 5b builds on these splits: training models on H&M,
   cold-start experiments, segment breakdowns, popularity-bias analysis, and the
   recency-aware reranker feature.

---

---

---

## Phase 5b RUNBOOK, part 1 — Baselines + MF on H&M

### Confirmed scale (run 2026-09-22)

- Raw: 31,788,324 transactions.
- After the 90-day date window: 3,904,391 rows (12.3%).
- After `min_user_interactions=3`, `min_item_interactions=5`: 3,653,414 rows,
  **378,631 users, 27,779 items**.
- Leave-one-out split: train=2,896,152 / val=378,631 / test=378,631.
- Global time split (7-day holdout): train=3,407,640 / test=245,774, of which 10,501
  test users and 303 test items are never seen in train — genuine cold-start cases,
  and a much larger cold-start fraction than MovieLens's global split (192 users / 6
  items) had, which makes sense: a 90-day retail window turns over its user base much
  faster than 25M movie ratings accumulated over years.

Similar catalog size to MovieLens (27,779 vs. 23,965 items) but over 2x the users
(378,631 vs. 161,572) and far sparser per-user history (train rows ÷ users ≈ 7.7 vs.
MovieLens's ≈ 94) — expect every model here to have a harder time than it did on
MovieLens, purely from data sparsity, before any modeling-choice differences.

**What this step does**: trains the same four models from Phases 1-2 — popularity,
item-item CF, ALS (both alpha=20 and alpha=1), BPR — on H&M, via the *exact same*
`train_model.py`/`evaluate_model.py` scripts, unchanged. Nothing model-specific had to
be rewritten because every model's `fit`/`score_batch` only ever look at generic
`user_idx`/`item_idx`/`confidence` columns — the same reuse `data/splits.py` already
demonstrated with H&M's loading/preprocessing.

1. **Train all four**:
   ```bash
   uv run python scripts/train_model.py --data-config configs/data_hm.yaml --model-config configs/model_popularity_hm.yaml
   uv run python scripts/train_model.py --data-config configs/data_hm.yaml --model-config configs/model_itemknn_hm.yaml
   uv run python scripts/train_model.py --data-config configs/data_hm.yaml --model-config configs/model_als_hm.yaml
   uv run python scripts/train_model.py --data-config configs/data_hm.yaml --model-config configs/model_als_alpha1_hm.yaml
   uv run python scripts/train_model.py --data-config configs/data_hm.yaml --model-config configs/model_bpr_hm.yaml
   ```
   The `_hm` suffix in each config's `name` keeps these artifacts (`artifacts/models/*_hm.pkl`)
   separate from the MovieLens ones already in that folder. Given H&M's sparser
   per-user history and fewer train rows than MovieLens (2.9M vs. 15.3M), expect these
   to run faster than their MovieLens counterparts did — item-item CF's similarity
   matrix is the one step that scales with catalog size (27,779 items) more than row
   count, so that one may still take a couple of minutes.

2. **Evaluate all five on the validation split**:
   ```bash
   for m in popularity_hm itemknn_hm als_hm als_alpha1_hm bpr_hm; do
     uv run python scripts/evaluate_model.py \
       --data-config configs/data_hm.yaml --eval-config configs/eval_default.yaml \
       --model-artifact artifacts/models/$m.pkl --split val
   done
   ```
   Appends rows to the *same* `artifacts/results/metrics_log.csv` as MovieLens's runs
   — the `dataset` column (`hm` vs. `movielens25m`) keeps them apart, so nothing gets
   overwritten or confused.

3. **What to check**: the printed metrics (or the CSV rows where `dataset=hm`).
   `LEARNINGS.md` compares these against their MovieLens counterparts — specifically
   whether the alpha=20-vs-alpha=1 ALS finding transfers to a sparser, purchase-based
   signal, and how much the sparsity itself (not the algorithm) explains the accuracy
   gap vs. MovieLens. These trained models (especially ALS and item-item CF) are also
   what the next parts of Phase 5b build on: cold-start experiments, segment
   breakdowns, popularity-bias analysis, and the recency-aware two-stage reranker.

---

---

---

## Phase 5b RUNBOOK, part 2 — Cold-start + Segment Breakdowns

**What this step does**: two separate analyses, both slicing accuracy by *who* the
user is instead of just reporting one global number.

**Cold-start** (`scripts/run_coldstart_eval.py`) answers: how much does accuracy
degrade for a user the model has never seen anything about? Only
`global_time_split`'s test set can contain such cases (leave-one-out's test set can't,
by construction — every LOO test user needs at least one train row just to be eligible
for that split). So this part needs its own models, trained on `global_train.parquet`
instead of `loo_train.parquet` (same configs from part 1, just pointed at a different
train file via the new `--train-file` flag). The eval script then buckets
`global_test`'s users into **cold** (zero rows in `global_train` — genuinely never
seen) and **warm** (has train history), and reports Recall/NDCG/MRR/etc. for each
bucket separately, plus "all" as a reference point.

Worth predicting before running it, so the result is a real check rather than just a
number: popularity's list is the same for every user regardless of history, so its
cold-vs-warm gap should be close to zero. Item-item CF and ALS/BPR *do* depend on the
user's row in the training matrix — a cold user's row there is entirely empty, so
their cold-segment accuracy should collapse toward the popularity floor or below it.

1. **Train the four models on the global split** (itemknn's similarity matrix doesn't
   depend on the split's user count, so it'll take about as long as part 1 did; ALS/BPR
   fit on `global_train`'s 3.4M rows, similar scale to part 1's 2.9M):
   ```bash
   uv run python scripts/train_model.py --data-config configs/data_hm.yaml --model-config configs/model_popularity_hm.yaml --train-file global_train.parquet --artifacts-dir artifacts/models/global_hm
   uv run python scripts/train_model.py --data-config configs/data_hm.yaml --model-config configs/model_itemknn_hm.yaml --train-file global_train.parquet --artifacts-dir artifacts/models/global_hm
   uv run python scripts/train_model.py --data-config configs/data_hm.yaml --model-config configs/model_als_hm.yaml --train-file global_train.parquet --artifacts-dir artifacts/models/global_hm
   uv run python scripts/train_model.py --data-config configs/data_hm.yaml --model-config configs/model_bpr_hm.yaml --train-file global_train.parquet --artifacts-dir artifacts/models/global_hm
   ```
   (`als_alpha1_hm` is deliberately skipped — part 1 already settled that alpha=20 wins
   on H&M, no need to re-run the loser here too.)

2. **Run the cold-start evaluation**:
   ```bash
   uv run python scripts/run_coldstart_eval.py \
     --data-config configs/data_hm.yaml \
     --eval-config configs/eval_default.yaml \
     --models-dir artifacts/models/global_hm
   ```
   Watch the log's first line for the cold/warm split size (expect roughly 10,501 cold
   out of the global test set, per Phase 5a's confirmed numbers), then each model's
   all/cold/warm rows. Appends to `artifacts/results/coldstart_log.csv`.

**Segment breakdowns** (`scripts/run_segment_eval.py`) slice the leave-one-out test
set's accuracy three ways: **activity** (train-interaction-count terciles — low/medium/
high, computed from the actual distribution, not a guessed threshold), **age** (bucketed
from `customers.csv`, with an explicit "unknown" bucket rather than dropping missing
ages), and **membership** (`club_member_status` — ACTIVE/PRE-CREATE/LEFT CLUB/unknown).
This one reuses the *already-trained* part-1 models (`artifacts/models/*_hm.pkl`)
unchanged — no retraining needed, since it's slicing an existing evaluation, not
changing the training condition.

3. **Run the segment evaluation**:
   ```bash
   uv run python scripts/run_segment_eval.py \
     --data-config configs/data_hm.yaml \
     --eval-config configs/eval_default.yaml \
     --models-dir artifacts/models
   ```
   Appends to `artifacts/results/segment_log.csv` — one row per (model, dimension,
   segment value, k).

4. **What to check**: both scripts' log output (or the two CSVs). The `LEARNINGS.md`
   entry covers what personalization actually buys across different kinds of
   users/situations — including whether item-item CF's Phase 5b-part-1 win holds up
   uniformly, or is concentrated in one segment.

---

---

---

## Phase 5b RUNBOOK, part 3 — Popularity-Bias Analysis

**What this step does**: a new metric, `popularity_bias` (added to
`evaluation/metrics.py`/`harness.py` as a regular `EvalRow` field, right alongside
Recall/NDCG/novelty/coverage — every future `evaluate()` call reports it automatically,
no separate pass needed). It's the fraction of each user's top-K recommendation slots
filled by an item from the catalog's top-1% most popular items (by train interaction
count). Where `novelty` answers "how popular are the recommended items, on an
information-theoretic scale," `popularity_bias` answers the more direct question: "how
often does this model just hand back a top-seller?" `scripts/run_popularity_bias_eval.py`
runs it for all four Phase 5b part-1 models on H&M and reports Recall/NDCG alongside it
in one table, so the accuracy-vs-personalization tradeoff doesn't need cross-referencing
two separate result files.

1. **Run it** (reuses the already-trained `artifacts/models/*_hm.pkl` models, no
   retraining):
   ```bash
   uv run python scripts/run_popularity_bias_eval.py \
     --data-config configs/data_hm.yaml \
     --eval-config configs/eval_default.yaml \
     --models-dir artifacts/models
   ```
   Appends to `artifacts/results/popularity_bias_log.csv`.

2. **What to check**: the log output (or the CSV). Worth predicting before
   looking: popularity's own popularity_bias should be close to 1.0 by construction (a
   large chunk of any reasonable top-K *is* the top-1% set), and Phase 5b part 1 already
   showed BPR trades accuracy for higher coverage/novelty than ALS — this metric should
   make that same tradeoff even more directly visible, and shows whether item-item CF's
   surprising accuracy win comes *with* or *despite* leaning on popular items.

---

---

---

## Phase 5b RUNBOOK, part 2c — Fixing Cold-start with a Fallback

**What this step does**: part 2b showed ALS/item-item CF/BPR silently collapse to a
meaningless fixed list for a user with zero `global_train` history — this step
actually fixes that, instead of just documenting it. `ColdStartFallbackRecommender`
wraps any base model with a fallback model: warm users (have training history) still
get the base model's real scores; cold users get routed to the fallback instead. Three
fallbacks, none of which depend on the user's own purchase history (since a cold user
has none):
- **`pop`** — plain popularity (the same model class used throughout this project).
- **`age`** — `AgeBucketPopularityRecommender`: most popular items among users in the
  *same age bucket* (from `customers.csv`), falling back to global popularity for a
  bucket with too little data to trust its own list.
- **`recency`** — `RecencyWeightedPopularityRecommender`: popularity weighted by
  exponential recency decay (`half_life_days`, default 7) instead of counting every
  train interaction equally — a "what's trending right now" list, motivated by H&M's
  trend-driven retail structure (a cold user's first purchase plausibly skews toward
  what's selling *this week*, not what's been steadily popular over the whole 90-day
  window).

None of these retrain the base model — this only fits three lightweight fallback
models and wraps the already-trained `global_hm` artifacts from part 2, so it's fast.

1. **Build the wrapped models** (requires part 2's `artifacts/models/global_hm/*.pkl`
   to already exist):
   ```bash
   uv run python scripts/train_coldstart_fallback.py \
     --data-config configs/data_hm.yaml \
     --models-dir artifacts/models/global_hm
   ```
   Saves 9 new artifacts: `{itemknn,als,bpr}_hm_coldfix_{pop,age,recency}.pkl`. Watch
   the log for the warm-user count (should roughly match part 2b's 51,805) and which
   age buckets had enough data to get their own popularity list rather than falling
   back to global. Add `--recency-half-life-days <N>` to try a different decay speed
   than the default 7-day half-life.

2. **Evaluate the plain and fixed versions side by side** (the script prints the exact
   command with all 10 model names at the end of step 1 — copy it from there, or use
   this):
   ```bash
   uv run python scripts/run_coldstart_eval.py \
     --data-config configs/data_hm.yaml \
     --eval-config configs/eval_default.yaml \
     --models-dir artifacts/models/global_hm \
     --models popularity_hm,itemknn_hm,itemknn_hm_coldfix_pop,itemknn_hm_coldfix_age,itemknn_hm_coldfix_recency,als_hm,als_hm_coldfix_pop,als_hm_coldfix_age,als_hm_coldfix_recency,bpr_hm,bpr_hm_coldfix_pop,bpr_hm_coldfix_age,bpr_hm_coldfix_recency
   ```
   (Step 1's log output prints this exact command with the right model list — copy it
   from there if you changed which base models exist.) Appends to the same
   `artifacts/results/coldstart_log.csv` from part 2b, so the before/after comparison
   is a direct row-by-row read of that one file (filter to `segment=cold` to see the
   fix's effect in isolation, or `segment=all` for the net effect including warm
   users, who should be completely unaffected).

3. **What to check**: the log output (or the CSV). `pop` already confirmed the
   fix works (cold-segment recall jumps from ~0 to plain popularity's level); `age`
   and `recency` are both trying to beat plain popularity by using a bit more context
   about the cold user — worth comparing them directly against each other, not just
   against the "broken" baseline.

---

---

---

## Phase 5b RUNBOOK, part 4 — Two-Stage Ranking + Recency Feature on H&M

**What this step does**: builds an H&M two-stage pipeline from scratch (FAISS
candidate generation over `als_hm`'s embeddings + a LightGBM reranker) — this didn't
exist yet; Phase 3 only built it for MovieLens. Same architecture, same
leakage-avoidance approach (reranker trained on `loo_val`, never test), same
injected-positive + `is_injected` handling as the MovieLens version, but with H&M's
own feature set: `category_affinity` (from `articles.csv`'s `product_group_name`, ~19
broad categories) stands in for MovieLens's `genre_affinity`, plus **two new recency
features**:
- `days_since_category_purchase` — how many days ago this user last bought something
  in this candidate item's category, as of the train split's own latest timestamp.
- `has_category_history` — 1.0 if that purchase exists, 0.0 if it doesn't (paired
  with a fixed sentinel value for "never," the same `is_injected`-style pairing used
  to stop a large sentinel from being misread as just an extreme case of "a long time
  ago" instead of a different situation entirely).

Motivation: H&M's genre-affinity equivalent (`category_affinity`) captures lifetime
taste, but can't distinguish "bought a coat yesterday" from "bought one 18 months
ago" — retail purchasing has a time dimension MovieLens's ratings don't carry the same
way. Kept as an entirely separate script/model class from the MovieLens pipeline
(`scripts/run_two_stage_pipeline_hm.py`, `models/two_stage_hm.py`) so the
already-working `two_stage.pkl` is never put at risk by a shared-code change.

1. **Run the tests** (new feature-engineering functions: category matrix, per-user
   category-recency lookup, the sentinel-pairing logic):
   ```bash
   uv run pytest tests/ -v
   ```

2. **Train the H&M two-stage pipeline** (requires `als_hm.pkl`/`itemknn_hm.pkl` from
   Phase 5b part 1 to already exist):
   ```bash
   uv run python scripts/run_two_stage_pipeline_hm.py \
     --data-config configs/data_hm.yaml \
     --pipeline-config configs/model_two_stage_hm.yaml
   ```
   Watch the log for: how many categories were found, how many training users needed
   their held-out item injected (compare against Phase 3's MovieLens run — H&M's
   sparser per-user history may push this higher or lower), and the ANN candidate
   recall diagnostic (the retrieval-stage accuracy ceiling, same interpretation as
   Phase 3).

3. **Evaluate it like any other model**:
   ```bash
   uv run python scripts/evaluate_model.py \
     --data-config configs/data_hm.yaml \
     --eval-config configs/eval_default.yaml \
     --model-artifact artifacts/models/two_stage_hm.pkl --split val
   ```

4. **Check the reranker's feature importance** (optional but useful — confirms
   whether the recency features actually get used, rather than just assuming it from
   the accuracy number):
   ```bash
   uv run python -c "
   import pickle
   from recsys_large_scale.features.engineer import FEATURE_NAMES_HM
   model = pickle.load(open('artifacts/models/two_stage_hm.pkl', 'rb'))
   for name, imp in sorted(zip(FEATURE_NAMES_HM, model.reranker.model.feature_importances_), key=lambda x: -x[1]):
       print(f'{name:32} {imp}')
   "
   ```

5. **What to check**: the category count, injection count, ANN recall,
   evaluate_model.py output, and the feature importance ranking. `LEARNINGS.md` sets
   these against `als_hm`/`itemknn_hm` (does two-stage beat single-stage on H&M the way
   it did, partially, on MovieLens?) and writes up what the recency feature specifically
   bought.

---

---

---

## Phase 6 RUNBOOK — Two-Tower Neural Retrieval (Colab, Optional Stretch)

**What this phase does**: the one genuinely optional piece of the roadmap — a neural
two-tower retrieval model, trained on **MovieLens 25M** (chosen over H&M: MovieLens
already has every baseline measured on it — popularity, item-item CF, ALS, BPR,
two-stage — making it the cleanest answer to "did DL earn its extra complexity
here?"). A user tower and an item tower (ID embedding → small MLP → final vector,
dim=64 to match ALS's own `factors=64` for a fair capacity comparison) are trained
with an in-batch sampled softmax loss on GPU, then scored by a plain dot product —
same scoring convention as ALS/BPR.

**Why Colab, and how the handoff works**: this is the one piece of the project that's
actually GPU-shaped (a full batch-size × batch-size score matrix every step). Colab
trains the model and exports two numpy arrays (`user_embeddings.npy`,
`item_embeddings.npy`) — nothing else happens there. Evaluation then happens **locally,
through the exact same `evaluate_model.py` every other model used** — not a separate
Colab-side metrics implementation — since Phase 6's entire point is a fair comparison
against Phases 2–3's numbers.

1. **Upload two files to Colab**: your local `data/processed/movielens25m/loo_train.parquet`
   and `id_maps.json` (via Colab's file-browser upload, or mount Drive if you'd rather
   keep them there). Then **Runtime → Change runtime type → GPU**.

2. **Open and run `notebooks/two_tower_colab.ipynb` top to bottom** (upload it to Colab,
   or open it from Drive/GitHub). It trains for 5 epochs by default (`N_EPOCHS` near
   the top of the training cell — raise it if the per-epoch loss is still dropping
   noticeably), then downloads `user_embeddings.npy` and `item_embeddings.npy` to your
   machine via the browser.

3. **Back on your local machine**, wrap the embeddings into a picklable model and
   evaluate it exactly like every other model:
   ```bash
   uv run python scripts/build_two_tower_model.py \
     --user-embeddings <path to downloaded user_embeddings.npy> \
     --item-embeddings <path to downloaded item_embeddings.npy> \
     --data-config configs/data_movielens25m.yaml

   uv run python scripts/evaluate_model.py \
     --data-config configs/data_movielens25m.yaml \
     --eval-config configs/eval_default.yaml \
     --model-artifact artifacts/models/two_tower.pkl --split val
   ```

4. **What to check**: the per-epoch training loss trend from Colab (is it still
   decreasing, or flattened?) and the `evaluate_model.py` output. The closing
   `LEARNINGS.md` entry sets these directly against
   `als.pkl`/`als_alpha1.pkl`/`itemknn.pkl`/`bpr.pkl`/`two_stage.pkl`'s existing numbers
   and asks whether the added complexity (a GPU, a new framework, a less interpretable
   model) was worth it here — including the known, undone limitation flagged in the
   notebook (in-batch negative sampling isn't popularity-corrected, a plausible source
   of its own bias, comparable to BPR's).

---

## Phase B1 RUNBOOK — MovieLens 1M (for the external benchmark comparison)

Everything above measures this project against itself. Phase B measures it against an
outside reference implementation — see `../recsys_benchmarks/README.md` for the other
half. This section covers only the part that runs *here*.

**Why ML-1M at all**, when we already have ML-25M: ML-1M is the dataset the published
literature actually reports on. ML-25M is barely benchmarked, so comparing our ML-25M
numbers to anyone else's means comparing across datasets, preprocessing *and* protocol
at once. Running our own pipeline on ML-1M removes two of those three variables.

`configs/data_movielens1m.yaml` is deliberately identical to the 25M config apart from
paths and the loader — same `min_rating: 3.5`, same 5-core filters — so the only
difference between our two MovieLens results is dataset size. (ML-1M ratings are whole
stars, so a 3.5 threshold keeps exactly the 4s and 5s, which also happens to match the
convention most published ML-1M work uses.)

1. **Download and build the splits.** ML-1M ships `::`-separated `.dat` files rather
   than CSVs, which is why the config carries a `loader: movielens_dat` key — the new
   `load_movielens1m_ratings` handles that format, everything downstream is unchanged.
   ```bash
   uv run python scripts/download_data.py --config configs/data_movielens1m.yaml
   uv run python scripts/build_splits.py --config configs/data_movielens1m.yaml
   ```
   Expect roughly 6,040 users and ~3,700 items — three orders of magnitude smaller than
   ML-25M, so every command below finishes in seconds to a couple of minutes.

2. **Train the five models.**
   ```bash
   for m in popularity itemknn als als_alpha1 bpr; do
     uv run python scripts/train_model.py \
       --data-config configs/data_movielens1m.yaml \
       --model-config configs/model_${m}_ml1m.yaml
   done
   ```

3. **Evaluate them, full-catalog (our standard protocol), on BOTH folds.** Results append
   to `artifacts/results/metrics_log.csv` tagged `dataset=movielens1m`.
   ```bash
   for m in popularity itemknn als als_alpha1 bpr; do
     for s in val test; do
       uv run python scripts/evaluate_model.py \
         --data-config configs/data_movielens1m.yaml \
         --eval-config configs/eval_default.yaml \
         --model-artifact artifacts/models/${m}_ml1m.pkl --split $s
     done
   done
   ```
   **Why both folds**, when every earlier phase reported val only: RecBole early-stops on
   the validation fold, so its `best_valid_result` is a maximum selected over training
   epochs *using that same fold* — optimistically biased. Our models train for a fixed
   number of iterations with no early stopping, so our val numbers have no such bias.
   Comparing val-to-val would hand RecBole a purely methodological advantage. Both sides'
   test numbers are clean, so the head-to-head in B3 uses test-to-test.

   Expect test to come out somewhat below val on both sides — the test item is each
   user's most recent interaction, so it sits further past the training window and is
   genuinely harder. RecBole shows exactly this (ItemKNN 0.0872 val → 0.0807 test). If
   our models show the same pattern, that's another sign the two setups agree.

4. **Evaluate them again under the sampled-negative protocol.** This is the new
   `--eval-mode` flag: instead of ranking the held-out item against all ~3,700 items,
   it ranks against 100 random ones plus the true item — the protocol much of the
   published neural-recommender literature reports under. Results go to a *separate*
   `artifacts/results/protocol_log.csv`, deliberately not mixed into `metrics_log.csv`.
   ```bash
   for m in popularity itemknn als als_alpha1 bpr; do
     uv run python scripts/evaluate_model.py \
       --data-config configs/data_movielens1m.yaml \
       --eval-config configs/eval_default.yaml \
       --model-artifact artifacts/models/${m}_ml1m.pkl --split val --eval-mode uni100
   done
   ```
   Expect these numbers to be dramatically higher than step 3's for the *same models on
   the same data*. That gap is the point: it is the size of the measurement artifact you
   would be importing if you compared our numbers to a published table without checking
   its protocol first. Note that coverage/novelty/popularity_bias/ILS are meaningless
   under this mode (they describe a random candidate pool, not the model) — the harness
   logs a warning saying so.

5. **What to check**: both logs. Next step is `../recsys_benchmarks/` for the RecBole
   side.
