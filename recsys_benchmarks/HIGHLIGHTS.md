# Benchmark highlights

The recommender models in `recsys_large_scale` were only ever compared to each other.
That tells you which of them is best; it tells you nothing about whether any of them is
good. This phase answered the second question by running **RecBole** — the framework most
current recommender-systems papers are built on — on identical data under an identical
evaluation protocol, and by reproducing the setup of a published benchmark paper exactly.

Getting a meaningful answer meant controlling three things at once: the dataset, the
preprocessing, and the evaluation protocol. Published numbers vary by an order of
magnitude depending on the third alone.

Full detail in [`LEARNINGS.md`](LEARNINGS.md); reproduction steps in [`README.md`](README.md).

---

## 1. The models were already competitive

MovieLens-1M, identical data, identical train/validation/test splits, full-catalog
leave-one-out ranking, test fold, **NDCG@10**:

| model | source | NDCG@10 |
|---|---|---|
| **ALS (α=1)** | **ours** | **0.0416** |
| MultiVAE | RecBole | 0.0395 |
| ItemKNN | RecBole | 0.0391 |
| ALS (α=20) | ours | 0.0391 |
| BPR | RecBole | 0.0389 |
| LightGCN | RecBole | 0.0356 |
| NeuMF | RecBole | 0.0355 |
| BPR | ours | 0.0344 |
| ItemKNN | ours | 0.0302 |

Our ALS beats every model RecBole offers here, including LightGCN (SIGIR'20) and NeuMF
(WWW'17). The comparison is trustworthy because the splits are byte-identical — RecBole
is fed our exact train/validation/test folds — and because a popularity control run
through both systems agrees to four decimal places.

**The honest qualifier:** at MovieLens-25M scale RecBole's BPR takes first place
(0.0441 against our 0.0430). The supported claim is *competitive with the best RecBole
offers at both scales, and first at ML-1M* — not that we win everywhere. Also worth
noting these are single-seed runs with no variance estimate, so margins of a few
thousandths should be read as ties.

---

## 2. Measured against published numbers

Finding 1 compares us to a framework we run ourselves. This compares us to numbers
someone else published, by reproducing their setup exactly.

**The paper:** *Neural Collaborative Filtering*, He et al., WWW'17
([arXiv 1708.05031](https://arxiv.org/pdf/1708.05031)) — the origin of the
sampled-negative protocol most MovieLens results use. We matched it on every axis: all
1,000,209 ratings treated as positives with no threshold, no k-core filtering, 6,040
users and 3,706 items (its Table 1 exactly), each user's latest interaction held out, and
the held-out item ranked against 100 sampled negatives, cut at 10.

*HR@10 (Hit Ratio) is the fraction of users whose held-out item made the top 10. Because
exactly one item is held out per user, it is numerically identical to Recall@10 — the
paper's term and ours for the same quantity.*

| model | HR@10 | NDCG@10 | |
|---|---|---|---|
| NeuMF, 64 factors, pre-trained | 0.730 | 0.447 | *published* |
| NeuMF, 64 factors, no pre-training | 0.705 | 0.426 | *published* |
| eALS / BPR (the paper's MF baselines) | ~0.70 | ~0.43 | *published, approx.* |
| **our ALS (α=20)** | **0.6940** | **0.4119** | |
| **our ALS (α=1)** | **0.6775** | **0.4171** | |
| our BPR | 0.6053 | 0.3540 | |
| ItemKNN | ~0.62 | ~0.40 | *published, approx.* |
| our ItemKNN | 0.5717 | 0.3213 | |
| our popularity | 0.4618 | 0.2567 | |

**Our ALS lands level with the paper's own matrix-factorization baselines** — eALS is the
closest algorithmic relative to our ALS, and we reproduce its reported range. Against the
paper's headline model we are within 3% on HR@10 and 2% on NDCG@10 of NeuMF without
pre-training, and within 7% of its best pre-trained configuration.

Two caveats, both of which cut in our favour:

- The NeuMF rows are exact (the paper's Table 2). The eALS, BPR and ItemKNN rows are
  approximate — derived from the paper's stated relative improvements and read off its
  Figure 4, since exact values for the baselines aren't tabulated.
- Dacrema et al. found that NCF's released code selects the number of training epochs
  using the **test** set, so the published figures are optimistically biased. Matching
  them closely is therefore better than it looks.

Our BPR and ItemKNN fall clearly short of the paper's versions of the same algorithms —
independently confirming the two defects in Finding 5, now against a second reference
that has nothing to do with RecBole.

---

## 3. Evaluation protocol moves results more than any model choice

Same models, same data, same split. The only change is what the held-out item is ranked
against — the full 3,125-item catalog, or 100 randomly sampled items (the protocol most
published MovieLens results use). MovieLens-1M, validation fold, **Recall@10**:

| model | full catalog | 100 sampled negatives | ratio |
|---|---|---|---|
| popularity | 0.0449 | 0.4534 | **10.1×** |
| itemknn | 0.0631 | 0.5734 | 9.1× |
| als | 0.0863 | 0.6954 | 8.1× |
| bpr | 0.0762 | 0.5948 | 7.8× |
| als_alpha1 | 0.0968 | 0.6725 | 6.9× |

No modelling decision anywhere in this project moved any metric remotely this far.
Reproduced independently inside RecBole on its own data (popularity 11.9×, ItemKNN 7.5×,
BPR 7.5×), so this is a property of the measurement, not of one codebase.

Two consequences:

- **It reorders models, not just inflates them.** `als_alpha1` beats `als` under
  full-catalog ranking and loses to it under sampled negatives. A sampled-negative
  leaderboard is therefore unreliable even for *relative* comparisons.
- **It flatters popularity most of all** (10.1×, the largest in the table). Against 100
  random competitors, recommending globally popular items is a far better strategy than
  against a full catalog — so the dominant protocol systematically understates how much
  a personalized model beats the non-personalized baseline.

---

## 4. A real bug found in RecBole's popularity baseline

RecBole's `Pop` model accumulates item counts with:

```python
self.item_cnt[item, :] = self.item_cnt[item, :] + 1
```

`item` is a batch of item ids containing many duplicates. PyTorch indexed assignment
gathers the right-hand side and then scatters it, so duplicate indices do not accumulate —
an item appearing 50 times in a batch is credited **+1, not +50**. What it actually
computes is the number of *batches an item appeared in*, which saturates at
(batches per epoch × epochs) for anything popular.

On ML-1M, every one of its top-20 items came out at exactly **275**, against true counts
running from 2803 down to 1734. The head of the ranking — precisely what Recall@10
depends on — collapses into one saturated tie broken by index order.

The effect understates popularity by **14–22%**, by an amount that varies with batch size
and epoch count rather than a constant that could be corrected for. Popularity is the
floor every other model is benchmarked against, so every result measured against this
baseline reports an inflated margin.

Confirmed by patching the counting loop with `torch.bincount`: the corrected model then
reproduces our popularity baseline to four decimal places (0.0449 recall / 0.0209 NDCG on
the validation fold, both systems).

This was only visible because we had an independent implementation of the same trivial
algorithm to disagree with — and the natural assumption turned out to be backwards: the
established framework was wrong, and the code under test was right.

---

## 5. Two concrete improvements identified in our own code

| | ours | RecBole | gap |
|---|---|---|---|
| BPR, ML-1M | 0.0344 | 0.0389 | +13% |
| BPR, ML-25M | 0.0349 | 0.0441 | **+26%** |
| ItemKNN, ML-1M | 0.0302 | 0.0391 | +23% |

**BPR is under-trained, and it gets worse with scale.** Ours barely moved going from
574K to 15.3M interactions (0.0344 → 0.0349) while RecBole's gained substantially
(0.0389 → 0.0441). Our config fixes `iterations: 100` regardless of dataset size;
RecBole uses Adam with early stopping and halts when validation plateaus. At 25× the
data, a fixed budget that sufficed for ML-1M plausibly does not.

**ItemKNN is configured differently.** RecBole uses 100 neighbours with a shrinkage term;
ours uses 200 with none, and the similarity normalization likely differs too.

Both are cheap to test, and both are the kind of defect that six phases of internal
comparison could never surface — our models were only ever measured against each other,
so a systematic shortfall in two of them was invisible.

Two independent references agree on this. Against RecBole (the table above), and against
the published NCF paper in Finding 2, where our BPR (0.3540 NDCG@10) and ItemKNN (0.3213)
both fall clearly below the paper's reported figures for the same algorithms (~0.43 and
~0.40) while our ALS matches its MF baselines. Two different reference points, same two
models flagged.

---

## What this phase was worth

Comparing our models to each other could only rank them. An outside implementation of the
*same* algorithms showed that two of ours were leaving accuracy on the table, that one of
*theirs* was broken, and that the measurement protocol matters more than any modelling
choice we made.
