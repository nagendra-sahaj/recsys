# LEARNINGS — external benchmark comparison

Companion to [`../recsys_large_scale/LEARNINGS.md`](../recsys_large_scale/LEARNINGS.md),
which covers Phases 0–6: everything we built, and how each model compared to the one
before it. That document is a closed loop — it measures our models against each other.

This file covers Phase B, which asks the question that loop can't answer: **measured
against an outside implementation, are any of those numbers good?** It lives here rather
than there because the work runs in a separate project with its own Python environment
(see [README.md](README.md) for why), and because its findings are about *evaluation
methodology* as much as about models.

Same format as the main document: core idea, shortcomings grounded in mechanism, metrics
against the previous best, and an interpretation of why the numbers moved.


## Phase B1 — Our own models on MovieLens 1M, and the protocol study

B1 puts our five models on ML-1M through our own pipeline — the same dataset RecBole is
measured on in B2 below — and additionally re-runs them under the sampled-negative
protocol most of the literature reports.

(Phases are numbered by the plan, not by execution order: B2 was actually run first,
which is why this entry forward-references its results.)

### The preprocessing check nobody expects to pass

| | users | items | interactions |
|---|---|---|---|
| ours (`data_movielens1m.yaml`) | 6,034 | 3,125 | 574,376 |
| RecBole (`val_interval` + `*_inter_num_interval`) | 6,034 | 3,125 | 574,376 |

Exact agreement on all three. Two independently written implementations of "keep
rating >= 3.5, then iteratively drop users/items below 5 interactions" converged on the
identical dataset. That matters more than it looks: it removes preprocessing as an
explanation for any difference between the two systems, which is usually the first thing
to suspect when cross-framework numbers disagree.

(The iterative part is load-bearing. `filter_min_interactions` loops until the row count
stops changing, because dropping users can push item counts below threshold and vice
versa. A single-pass filter would not have matched.)

### Results (ours, ML-1M, full-catalog leave-one-out, K=10)

| model | Recall val | Recall test | NDCG test | Coverage | Novelty |
|---|---|---|---|---|---|
| als_alpha1 | 0.0968 | **0.0852** | **0.0416** | 0.247 | 9.27 |
| als | 0.0863 | 0.0774 | 0.0391 | 0.482 | 9.60 |
| bpr | 0.0762 | 0.0694 | 0.0344 | 0.762 | 10.94 |
| itemknn | 0.0631 | 0.0603 | 0.0302 | 0.110 | 8.44 |
| popularity | 0.0449 | 0.0441 | 0.0224 | 0.037 | 8.07 |

**The model ordering is identical to ML-25M's** (als_alpha1 > als > bpr > itemknn >
popularity), on a dataset 25× smaller. The alpha=1-beats-alpha=20 result also holds here,
consistent with the density story from Phase 5b: ML-1M is denser still than ML-25M, and
alpha=1 keeps winning, while the much sparser H&M data preferred alpha=20.

### Confirming we have no selection bias

B2 below finds RecBole's neural models drop 11–18% from val to test because they early-stop on
the validation fold. Our models train for fixed iterations, so they should drop only by
the genuine difficulty gap. They do:

| model | val→test Δ |
|---|---|
| popularity | −1.8% |
| itemknn | −4.4% |
| bpr | −8.9% |
| als | −10.3% |
| als_alpha1 | −12.0% |

All within or near the −7.5% band ItemKNN showed on RecBole's side, and nothing
approaching BPR's −17.8% there. The prediction held, which is the evidence that the
test-to-test comparison in B3 is the right one.

### The protocol study: the same models, measured two ways

Our models evaluated on the identical val fold, changing only the candidate set — full
catalog (3,125 items) versus 100 uniformly sampled negatives plus the true item:

| model | Recall@10 full | Recall@10 uni100 | ratio |
|---|---|---|---|
| popularity | 0.0449 | 0.4534 | **10.1×** |
| itemknn | 0.0631 | 0.5734 | 9.1× |
| als | 0.0863 | 0.6954 | 8.1× |
| bpr | 0.0762 | 0.5948 | 7.8× |
| als_alpha1 | 0.0968 | 0.6725 | 6.9× |

**Two findings here, and the second is the more important one.**

First, the magnitude. Changing nothing but the candidate set moves every model by 7–10×.
No modelling decision in this entire project — not ALS vs popularity, not three attempts
at a two-tower network — moved any metric remotely that far. The protocol dominates.

And our sampled-negative numbers land directly on published ones: papers using the
1-positive-vs-99-negatives protocol report ML-1M HR@10 of roughly 0.52 for ItemKNN and
0.54 for MF-BPR; ours are 0.5734 and 0.5948. So the apparent 6× shortfall against the
literature was never a quality gap. Measured the way those papers measure,
our models are competitive with them. That is the question this whole phase existed to
answer.

Second, and worse for anyone comparing across papers: **the protocol does not merely
inflate scores, it reorders models.** Under full-catalog ranking `als_alpha1` beats `als`
(0.0968 vs 0.0863). Under uni100 that flips — `als` wins (0.6954 vs 0.6725). Two
hyperparameter settings of one algorithm on one dataset, and which is "better" depends on
the evaluation protocol. Inflation alone would be survivable, since a reader could mentally
discount it; rank inversion is not, because it means a sampled-negative leaderboard cannot
be trusted even for *relative* conclusions.

Note also that popularity shows the largest inflation (10.1×). With only 100 random
competitors, recommending globally popular items is a far better strategy than it is
against the full catalog — the sampled protocol systematically flatters exactly the
baseline that personalization is supposed to beat.

Coverage, novelty, popularity-bias and ILS are not reported for the uni100 runs: computed
over a random 101-item pool they describe the sampler, not the model. The harness logs a
warning and writes these runs to a separate `protocol_log.csv` rather than mixing
protocols in `metrics_log.csv`.

### Open questions going into B3

Comparing our test-fold numbers against B2's is suggestive but not yet valid, because B2
let RecBole do its own splitting:

- **Our `als_alpha1` (0.0416 NDCG@10) would top RecBole's table**, above ItemKNN's 0.0403
  and every neural model — and RecBole ships no iALS to check it against. Provisional.
- **Our ItemKNN (0.0302) is 25% below RecBole's (0.0403).** RecBole runs `k=100,
  shrink=0.0`; our config uses `top_k_neighbors: 200`, and the similarity normalization
  likely differs too. A real finding about our implementation, not a bridge defect.
- **Popularity — the control — disagrees: ours 0.0441 vs RecBole's 0.0388 recall@10.**
  Same trivial algorithm, same dataset shape, 13% apart. Leading hypothesis is split
  tie-breaking: ML-1M users rate in bursts, so many interactions share a timestamp, and
  the two systems break those ties differently, holding out different items for the same
  user. B3 tests this by handing RecBole our three folds directly via
  `benchmark_filename`, which bypasses its splitter entirely. If the control closes,
  everything downstream is trustworthy; if not, there is a real difference in scoring or
  masking to find first.

---

## Phase B2 — An outside reference: RecBole on MovieLens 1M

**Why this phase exists.** Every number in the main LEARNINGS document compares our
models to *our other models*. That answers "which of ours is best" and says nothing about
whether any of them is good. Phase B runs [RecBole](https://arxiv.org/pdf/2011.01731) — the
framework most modern recommender-systems papers are built on — as an independent
implementation to measure against. B2 is RecBole measuring RecBole on ML-1M; B3 puts our
models on identical data.

ML-1M rather than ML-25M because ML-1M is what the literature actually reports on. Our
ML-25M numbers can't be compared to anything published without varying dataset,
preprocessing *and* protocol simultaneously.

### Results (RecBole 1.2.1, ML-1M, rating>=3.5, 5-core, full-catalog leave-one-out)

Test fold, K=10:

| model | Recall@10 | NDCG@10 | Coverage@10 | AvgPopularity@10 |
|---|---|---|---|---|
| ItemKNN | 0.0807 | **0.0403** | 0.342 | 1302 |
| MultiVAE | 0.0822 | 0.0402 | 0.495 | 1154 |
| BPR | 0.0759 | 0.0385 | 0.427 | 1231 |
| LightGCN | 0.0719 | 0.0367 | 0.276 | 1376 |
| NeuMF | 0.0737 | 0.0366 | 0.633 | 907 |
| Pop | 0.0388 | 0.0181 | 0.037 | 1761 |

**ItemKNN — co-occurrence similarity, nothing learned — tops the table.** It ties
MultiVAE (0.0403 vs 0.0402 NDCG; a 0.0001 gap on a single seed is a tie, not a win) and
clearly beats LightGCN (SIGIR'20) and NeuMF (WWW'17). This reproduces
[Dacrema et al.'s RecSys'19 finding](https://arxiv.org/pdf/1907.06902) in our own run
rather than citing it: well-tuned classical baselines match or beat published neural
architectures. It also echoes this project's own Phase 6 result, where the two-tower
model needed three attempts and two bug fixes to reach parity with item-item CF.

### The finding that nearly cost us the conclusion: validation-fold selection bias

The original plan compared our val-fold numbers against RecBole's. Reading the val and
test columns side by side showed why that would have been wrong:

| model | val→test Δ (Recall@10) | epochs selected on val? |
|---|---|---|
| Pop | **+0.5%** | no training at all |
| ItemKNN | −7.5% | no epochs to select over |
| LightGCN | −11.5% | yes |
| MultiVAE | −11.6% | yes |
| NeuMF | −17.5% | yes |
| BPR | −17.8% | yes |

That gradient decomposes cleanly. Pop is flat because nothing is fitted. ItemKNN's −7.5%
is the genuine difficulty gap — the test item is each user's *most recent* interaction,
so it sits further past the training window than the val item. Everything beyond that
−7.5% is early-stopping bias: RecBole selects the best epoch by NDCG@10 **on the
validation fold**, so `best_valid_result` is a maximum chosen using the fold it reports.

The consequence is not cosmetic. **On the val fold, MultiVAE, BPR and NeuMF all appear
to beat ItemKNN. On the test fold, ItemKNN is first.** The val ranking is an artifact of
model selection. Our own models train for fixed iterations with no early stopping, so our
val numbers carry no such bias — comparing val-to-val would have credited RecBole's
neural models with an advantage that is purely methodological, and the resulting
"neural models beat our classical ones" conclusion would have been undetectably wrong.

B3 therefore compares **test-to-test**, where both sides are clean.

### How these compare to published ML-1M numbers

There is no clean published full-ranking leave-one-out table on ML-1M to check against —
which is itself the point of the exercise. The closest available references:

- A [2025 study on LLM dataset memorization](https://arxiv.org/html/2505.10212v1)
  reports full-ranking ML-1M NDCG@10 of 0.0337 (ItemKNN), 0.0356 (BPRMF) and 0.0311
  (LightGCN). Ours are 0.0403 / 0.0385 / 0.0367 — the same magnitude and nearly the same
  ordering. Their HR@10 (0.18–0.21) is far above ours (~0.08) because they hold out 20%
  of each user's interactions; with many held-out positives a hit is much easier than
  with our single one. A structural difference, not a quality difference.
- [GRAFT-Rec](https://link.springer.com/article/10.1007/s13278-026-01618-z) reports
  ML-1M NDCG@10 of 0.0313 (ItemKNN), 0.0184 (BPR-MF), 0.0198 (LightGCN) under a temporal
  split — again the same magnitude, and again **ItemKNN on top**.
- Sampled-negative papers (1 positive vs 99 negatives) report ItemKNN HR@10 ≈ 0.52 and
  MF-BPR ≈ 0.54 on the same dataset — roughly **6× our numbers for the same algorithms**.

That last line is the whole reason Phase B exists. Three sources, the same models, the
same dataset, numbers spanning an order of magnitude — the protocol dominates. Phase B4
measures that gap directly on both sides rather than citing it.

### What it took to run the framework at all

RecBole 1.2.1 (Feb 2025) needed five fixes before a single model trained, none of them
related to recommendation:

1. `setuptools` absent — `uv` doesn't seed virtualenvs with it.
2. `pkg_resources` removed in setuptools 81, which ray 2.6.3 imports. Capped at `<81`.
3. `ray` on the critical path via `recbole.quick_start` (the only file importing it),
   breaking on `np.bool8`. Bypassed by calling the public API one level down.
4. NumPy 2 — RecBole's `compatibility_settings()` executes `np.float = np.float_`
   unconditionally in `Config.__init__`; `np.float_` was removed in NumPy 2.0. A method
   named "compatibility settings" is what breaks compatibility. Capped `numpy<2`.
5. Three genuine upstream bugs: the evaluation collector calls `.cpu()` on every
   registered key including plain ints (fires whenever ItemCoverage or AveragePopularity
   is requested — i.e. the released framework cannot compute its own beyond-accuracy
   metrics); `torch.load` defaults changed in PyTorch 2.6 so RecBole can't reload its own
   checkpoints; and the graph models call `dok_matrix._update`, a private SciPy method
   since removed, breaking LightGCN/NGCF/NCL/SpectralCF/GCMC.

Plus RecBole's dataset mirror now returns HTTP 403, so ml-1m had to be rebuilt from
GroupLens.

Worth recording because the reproducibility literature asks whether published *results*
reproduce. This is the layer underneath: whether the code still runs, 18 months on. One
of those fixes also had a silent-failure trap — the obvious patch for the SciPy issue
(`dict.update(self, data)`, since `dok_matrix` subclasses `dict`) writes to the wrong
place in modern SciPy, yielding an **empty adjacency matrix**. LightGCN would have
trained on no graph and reported plausible numbers. Verified `nnz=3` vs `nnz=0` on a toy
matrix before trusting it.

---

## Phase B3 — The head-to-head, and a broken baseline in RecBole

B3 removes every remaining confound: `export_to_recbole.py` writes our three folds as
separate files and `benchmark_filename` makes RecBole consume them as-is, skipping both
its own filtering and its own splitting (`dataset.py:161`, `dataset.py:1815`). Both
systems hold out identical items for identical users.

### The control failed, and the failure was the finding

Popularity gates this comparison: the same trivial algorithm in both codebases, which on
identical data must agree. It did not, and tightening the setup made it worse:

| Pop, test Recall@10 | |
|---|---|
| ours | 0.0441 |
| RecBole, its own split (B2) | 0.0388 (−12%) |
| RecBole, our split | 0.0358 (−19%) |
| RecBole, our split, negative sampling disabled | 0.0275 (−38%) |

Two hypotheses were falsified by measurement rather than argument. **Split tie-breaking**
— ML-1M users rate in bursts, so shared timestamps could send the two systems to
different held-out items; feeding RecBole our exact pre-split folds *widened* the gap.
**Negative-sample pollution** — `Pop` is `InputType.POINTWISE` with negative sampling on
by default, so `interaction[ITEM_ID]` carries sampled negatives too; disabling it made
Pop *worse*, when pure positive counting should have reproduced our number exactly.

Masking was checked directly and also ruled out: RecBole's `used_ids` accumulate across
phases (`sampler.py:244`), so test masks train ∪ valid while we mask train only — but for
the **validation** fold the masks are identical and the gap persisted there too.

The decisive constraint came from B4: under `uni100` on the same split the two agreed to
within 1.4%, while under full-catalog ranking they differed by 20%. Agreement against 100
random competitors but disagreement against the whole catalog means the two popularity
*rankings* differ at the head — exactly what Recall@10 depends on, and exactly what a
101-item comparison is too coarse to see.

### Root cause: duplicate indices don't accumulate

`scripts/diagnose_pop.py` reads RecBole's learned `item_cnt` out of its checkpoint, maps
internal ids back to ours, and diffs the rankings. **Every one of RecBole's top-20 items
had count exactly 275**, against true counts running 2803 down to 1734. Overall
correlation 0.828, but only 0.1% of items matched exactly.

`Pop.calculate_loss` accumulates with:

```python
self.item_cnt[item, :] = self.item_cnt[item, :] + 1
```

`item` is a batch of ids full of duplicates. PyTorch indexed assignment gathers the
right-hand side then scatters it, so duplicates do not accumulate — last write wins. An
item appearing 50 times in a batch is credited **+1, not +50**. What RecBole computes is
the number of *batches in which an item appeared*, which saturates at
(batches per epoch × epochs) for anything popular.

The arithmetic confirms it exactly: 562,308 train rows / 4096 batch size = 138 batches
per epoch, best checkpoint at epoch 2, ceiling 276 — and the popular items all pin at 275.
Their true ordering is destroyed and the head of the ranking becomes an arbitrary
index-order tie-break among saturated items.

This retrospectively explains every observation: ours scored higher because we had real
popularity ordering; disabling negative sampling merely reshuffled an already-arbitrary
tie-break; and `uni100` agreed because against 100 random competitors any
saturated-popular item serves about as well.

### Confirmation

`--fix-pop` replaces the counting loop with `torch.bincount` (which accumulates
duplicates), run together with `train_neg_sample_args: ~`:

| | val recall | val ndcg | test recall | test ndcg |
|---|---|---|---|---|
| ours | 0.0449 | 0.0209 | 0.0441 | 0.0224 |
| RecBole, corrected | **0.0449** | **0.0209** | 0.0444 | 0.0226 |

Val matches to four decimal places. The +0.0003 test residual is itself explained by the
masking difference above — RecBole masks one extra item (the validation item) when
scoring test, so it should score marginally *higher*, which is the observed direction.
Control passed, residual accounted for, bridge validated.

**Why this matters beyond our project.** Popularity is the floor every other model is
measured against. RecBole's `Pop` understates it by ~20% on ML-1M, and the error is not a
constant you could correct for — it depends on batch size and epoch count, so it varies
per experiment. Every paper using this baseline reports an inflated margin over
popularity. We only caught it because we had an independent implementation of the same
trivial algorithm to disagree with, which is the argument for building a comparison this
way instead of reading numbers out of published tables. Note also the natural assumption
was backwards: the established framework was wrong and the thing under test was right.

Scope: the bug is confined to `Pop`'s counting loop. ItemKNN, BPR and the neural models
build their own data structures and are unaffected.

### The head-to-head

Test fold, K=10, identical data and identical splits:

| model | system | Recall@10 | NDCG@10 | Coverage |
|---|---|---|---|---|
| **ALS (α=1)** | **ours** | **0.0852** | **0.0416** | 0.247 |
| MultiVAE | RecBole | 0.0795 | 0.0395 | 0.497 |
| ItemKNN | RecBole | 0.0784 | 0.0391 | 0.343 |
| ALS (α=20) | ours | 0.0774 | 0.0391 | 0.482 |
| BPR | RecBole | 0.0777 | 0.0389 | 0.455 |
| LightGCN | RecBole | 0.0716 | 0.0356 | 0.339 |
| NeuMF | RecBole | 0.0723 | 0.0355 | 0.481 |
| BPR | ours | 0.0694 | 0.0344 | 0.762 |
| ItemKNN | ours | 0.0603 | 0.0302 | 0.110 |
| Popularity | ours | 0.0441 | 0.0224 | 0.037 |

**Our ALS (α=1) is the strongest model on the board**, ahead of MultiVAE, LightGCN,
NeuMF and RecBole's ItemKNN — and RecBole ships no iALS to contest it, which is part of
why this comparison was worth building. The margin over the best neural model is 5%
on NDCG@10; on a single seed with no variance estimate that is suggestive, not decisive,
but "competitive with the best RecBole offers" is well supported.

The result also echoes B2 and the Dacrema et al. finding from the other direction: the
best models here are a 2008 matrix factorization and a 2018 VAE, with the graph network
(LightGCN, SIGIR'20) and the neural CF model (NeuMF, WWW'17) at the bottom of the
learned-model group.

**One gap is ours to own: our ItemKNN (0.0302) is 23% below RecBole's (0.0391).** Not a
bridge defect — the data and split are identical and the Pop control now passes. RecBole
runs `k=100` with shrinkage; our config uses `top_k_neighbors: 200` with no shrinkage
term, and the similarity normalization likely differs too. A genuine implementation
difference worth chasing, and the clearest actionable improvement this comparison surfaced
for our own code.

## Phase B4 — The protocol study, both sides

B1 measured the full-catalog-vs-sampled-negatives gap on our models. B4 repeats it inside
RecBole, so the finding rests on two independent codebases rather than one.

Recall@10, validation fold, changing only the candidate set:

| system | model | full catalog | uni100 | ratio |
|---|---|---|---|---|
| RecBole | Pop | 0.0386 | 0.4579 | **11.9×** |
| RecBole | ItemKNN | 0.0872 | 0.6535 | 7.5× |
| RecBole | BPR | 0.0923 | 0.6898 | 7.5× |
| ours | popularity | 0.0449 | 0.4534 | **10.1×** |
| ours | itemknn | 0.0631 | 0.5734 | 9.1× |
| ours | bpr | 0.0762 | 0.5948 | 7.8× |

Two independent implementations, same conclusion: **7–12× inflation, and popularity
inflates most of all.** Against 100 random competitors, recommending globally popular
items is a far better strategy than against a 3,125-item catalog — so the protocol that
dominates the literature systematically flatters precisely the baseline that
personalization is supposed to beat. Every "our model beats popularity by X%" claim
measured this way is reporting a smaller X than full-catalog ranking would give it.

Combined with the rank inversion documented in B1 (`als_alpha1` beats `als` under full
ranking and loses under uni100), the conclusion is stronger than "sampled numbers are
inflated": **a sampled-negative leaderboard is unreliable even for relative comparisons.**

A useful side effect: under uni100 on the identical split, our popularity and RecBole's
Pop agree to within 1.4% (0.4534 vs 0.4470, explicable by different random negatives).
That became the key diagnostic constraint in B3 below.

---

## Phase B5 — Does any of it hold at 25× scale?

B1–B4 ran on ML-1M (574,376 interactions). B5 repeats the head-to-head on ML-25M
(15,261,216 interactions, 161,572 users, 23,965 items) through the same pre-split
`benchmark_filename` pipeline, to test whether the conclusions were artefacts of a small
dataset.

**Cost:** Pop + ItemKNN + BPR took **3.5 hours** on an M-series CPU, against minutes on
ML-1M. The neural models were deliberately not run at this scale — see the end of this
entry.

### The control passes again

| Pop, K=10 | ours | RecBole (corrected) |
|---|---|---|
| val | 0.0491 / 0.0247 | **0.0491 / 0.0247** |
| test | 0.0472 / 0.0237 | 0.0479 / 0.0242 |

Exact agreement on the validation fold, and the same small test residual as ML-1M, from
the same cause (RecBole masks train ∪ valid for test, we mask train only — one extra
masked item, so it scores marginally higher). The bridge is validated at both scales.

Worth recording that the stock, unfixed `Pop` came in at 0.0420/0.0202 val — a 14%
understatement, against 17% on ML-1M. A prediction made before this run was that the bug
would be *much* worse at scale, since saturation is bounded by batches-per-epoch and
ML-25M has 1,863 batches against ML-1M's 138. It was not: the saturation ceiling rises
roughly in step with the true counts, so the distortion stays proportional instead of
compounding.

### The head-to-head reverses

Test fold, K=10, identical data and splits:

| rank | model | system | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| 1 | **BPR** | **RecBole** | 0.0882 | **0.0441** |
| 2 | ALS (α=1) | ours | 0.0860 | 0.0430 |
| 3 | ItemKNN | RecBole | 0.0796 | 0.0402 |
| 3= | ALS (α=20) | ours | 0.0821 | 0.0402 |
| 5 | itemknn | ours | 0.0708 | 0.0360 |
| 6 | bpr | ours | 0.0701 | 0.0349 |
| 7 | two_tower | ours | 0.0710 | 0.0348 |
| 8 | two_stage | ours | 0.0681 | 0.0332 |
| 9 | popularity | ours | 0.0472 | 0.0237 |

**Our ALS tops the board on ML-1M and does not at ML-25M.** RecBole's BPR takes first
place. This is the result B5 existed to find, and it partly walks back the ML-1M
conclusion: "our best model is competitive with the best RecBole offers" survives, "our
best model wins" does not.

### Two implementation gaps in our own code, both now confirmed at two scales

**BPR, and the gap widens with scale:**

| | ours | RecBole | gap |
|---|---|---|---|
| ML-1M | 0.0344 | 0.0389 | +13% |
| ML-25M | 0.0349 | 0.0441 | +26% |

Our BPR barely moved going from 574K to 15.3M interactions (0.0344 → 0.0349) while
RecBole's gained substantially (0.0389 → 0.0441). The most likely cause is under-training
rather than a modelling difference: `configs/model_bpr.yaml` fixes `iterations: 100`
regardless of dataset size, whereas RecBole runs Adam with early stopping for up to 100
epochs and halts when validation plateaus. At 25× the data, a fixed iteration count that
sufficed for ML-1M plausibly does not. Testable cheaply by raising `iterations` and
watching whether the gap closes.

**ItemKNN, consistent at both scales:** ours 0.0302 vs 0.0391 on ML-1M (−23%), 0.0360 vs
0.0402 on ML-25M (−10%). RecBole runs `k=100` with a shrinkage term; our config uses
`top_k_neighbors: 200` with none, and the similarity normalization likely differs too.

These two are arguably the most useful output of the whole phase. Phases 0–6 could show
which of our models beat our other models; only an outside implementation of the *same*
algorithms could show that two of them were leaving accuracy on the table.

### What held at scale

- **Our own models' ordering is unchanged** across a 25× size difference: als_alpha1 >
  als > itemknn ≈ bpr > two_tower > two_stage > popularity, and alpha=1 still beats
  alpha=20.
- **RecBole's `Pop` bug** reproduces, at similar proportional severity.
- **The protocol findings** from B1/B4 are dataset-independent by construction and need no
  re-validation.
- **ItemKNN remains competitive with learned models** — on ML-25M RecBole's ItemKNN ties
  our ALS (α=20) and beats four of our seven models, consistent with B2 and with
  Dacrema et al.

### Neural models at this scale: a documented omission

NeuMF, LightGCN and MultiVAE were not run as part of the main B5 sweep. With three
classical models costing 3.5 hours, LightGCN — graph convolution over a 161,572 × 23,965
adjacency every epoch, for up to 100 epochs — is a different order of magnitude, and the
marginal value had dropped: the question "does anything in RecBole beat our best at
scale?" was already answered *yes* by BPR, so a neural model confirming it refines a
conclusion rather than changing one. All three have ML-1M numbers, which is where the
comparison against published work actually lives.

MultiVAE (the strongest neural model on ML-1M) is being run separately; this entry will
be extended when it completes.

---

## Phase B6 — Against published numbers, not a framework we run

B1–B5 compare our models against RecBole: an independent implementation, but one we
execute ourselves. That answers *"is my code as good as a reference implementation's?"*
and not *"do my numbers match what the literature reports?"* B6 closes that gap by
reproducing the setup of a specific published paper exactly, so our numbers can sit beside
its printed table.

**The paper:** *Neural Collaborative Filtering*, He et al., WWW'17
([arXiv 1708.05031](https://arxiv.org/pdf/1708.05031)) — the origin of the
sampled-negative protocol that dominates MovieLens reporting, and therefore the natural
reference point for the inflation measured in B4.

### Matching the setup

Read off the paper's §4.1 and Table 1 rather than a secondary summary:

| | NCF paper | our usual ML-1M config | `data_movielens1m_ncf.yaml` |
|---|---|---|---|
| rating threshold | **none** — every rating a positive | `>= 3.5` | none |
| k-core filtering | none (ML-1M's own >=20/user) | 5-core, iterative | none |
| interactions | 1,000,209 | 574,376 | **1,000,209** |
| users / items | 6,040 / 3,706 | 6,034 / 3,125 | **6,040 / 3,706** |
| held out per user | 1 test + 1 validation | 1 test + 1 validation | 1 test + 1 validation |
| candidate set | 100 sampled negatives | full catalog | 100 sampled negatives |

The gate was `build_splits.py` reproducing the paper's Table 1 shape before anything else
ran; it printed 1,000,209 / 6,040 / 3,706 and `kept 1,000,209/1,000,209 rows (100.0%)`,
confirming no threshold was applied. Evaluated with `--split test --eval-mode uni100`.

Note that `uni100` already matched the paper's protocol before this phase — the paper
"randomly samples 100 items that are not interacted by the user, ranking the test item
among the 100 items". Only the *preprocessing* differed, which is what this config fixes.
The 3,125-item catalog our usual filtering produces is a materially easier ranking problem
than the paper's 3,706, which is why B4's numbers could not be set beside published ones.

### Results

Test fold, K=10. HR@10 is the paper's term for what our harness logs as recall; with one
held-out item per user they are the same quantity.

| model | HR@10 | NDCG@10 | source |
|---|---|---|---|
| NeuMF, 64 factors, pre-trained | 0.730 | 0.447 | published, exact (Table 2) |
| NeuMF, 64 factors, no pre-training | 0.705 | 0.426 | published, exact (Table 2) |
| eALS / BPR | ~0.70 | ~0.43 | published, approximate |
| **our ALS (α=20)** | **0.6940** | **0.4119** | |
| **our ALS (α=1)** | 0.6775 | **0.4171** | |
| our BPR | 0.6053 | 0.3540 | |
| ItemKNN | ~0.62 | ~0.40 | published, approximate |
| our ItemKNN | 0.5717 | 0.3213 | |
| our popularity | 0.4618 | 0.2567 | |

**Our ALS reproduces the paper's own matrix-factorization baselines.** eALS is the closest
algorithmic relative to our ALS — both are implicit-feedback alternating least squares —
and we land inside its reported range. Against the paper's headline model we are within
3% on HR@10 and 2% on NDCG@10 of NeuMF without pre-training, and within 7% of its best
pre-trained configuration.

Note the α ordering flips by metric here: α=20 wins on HR@10 (0.6940 vs 0.6775) while
α=1 wins on NDCG@10 (0.4171 vs 0.4119). α=20 gets the item into the top 10 more often;
α=1 places it higher when it does. On a single seed, treat that as a tie rather than a
finding.

### Caveats, both of which cut in our favour

- Only the NeuMF rows are exact. The eALS, BPR and ItemKNN figures are derived from the
  paper's stated "4.5% and 4.9%" relative improvements and read off its Figure 4, since
  the baselines are not tabulated. They are labelled approximate wherever they appear.
- Dacrema et al. observed that NCF's released code selects the number of training epochs
  using the **test** set, so the published figures are optimistically biased. Matching
  them closely is better than it looks.
- Our validation item is each user's second-most-recent interaction; the paper samples
  one at random. A small difference in which interaction leaves the training set.
- Single seed, no variance estimate, as everywhere else in this project.

### What B6 adds that B1–B5 could not

**Independent corroboration of the two implementation gaps.** Our BPR (0.3540 NDCG@10)
and ItemKNN (0.3213) fall clearly below the paper's figures for the same algorithms
(~0.43 and ~0.40), while our ALS matches its MF baselines. These are the same two models
RecBole flagged in B3 and B5 — now confirmed against a reference with no connection to
RecBole whatsoever. Two unrelated sources, same two models, same direction.

That is a materially stronger basis for acting on those two gaps than either source alone
would give, and it is the concrete answer to the question that started this phase: the
headline model did not need improving — it was already at published-baseline level — but
two others did.
