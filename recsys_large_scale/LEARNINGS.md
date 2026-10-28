# Learnings

Running log of what was built, what its shortcomings turned out to be, and how it
compared to whatever came before — one entry per model/phase, filled in once real
numbers exist (see the [README](README.md) RUNBOOK for how each phase is run).

Template per entry:

```
## <Model / Phase name>

**Core idea**: ...

**Known shortcomings**: ...

**Metrics vs. previous best**:
| metric | previous best | this model | delta |
|---|---|---|---|

**Interpretation**: why the numbers moved the way they did, tied to what changed
mechanically between this model and the last one.
```

---

## Metrics glossary

Every metric here is computed in `evaluation/metrics.py` (unit-tested against
hand-worked examples in `tests/test_metrics.py`) and combined across all evaluation
users by `evaluation/harness.py`. Context that applies to all of them: each user in
our validation/test set has exactly **one held-out item** (their most recent
interaction — see `data/splits.py`'s leave-one-out split), and a model produces a
ranked list of K recommended items per user, with items the user already interacted
with in training masked out first. "A hit" below always means "the held-out item
appears somewhere in that user's top-K list."

Toy setup used in the worked examples below: 4 users, K=3, a catalog of 5 items
(0–4). Each user's top-3 recommended items and held-out item:

| user | top-3 recommended | held-out item | hit? | rank of hit |
|---|---|---|---|---|
| A | [0, 3, 4] | 0 | yes | 1 |
| B | [1, 4, 2] | 2 | yes | 3 |
| C | [0, 1, 3] | 4 | no | — |
| D | [1, 2, 3] | 1 | yes | 1 |

(This is the exact fixture used in `tests/test_metrics.py::toy_hits`.)

### Recall@K
Out of all evaluated users, what fraction had their held-out item show up *anywhere*
in their top-K list? Answers "does the model find the right thing at all, within K
tries?" — doesn't care where in the list it landed.

*Example*: A, B, D hit; C misses → Recall@3 = 3/4 = **0.75**.

### Precision@K
Out of the K recommended slots, what fraction were actually relevant? Since each user
here has exactly one relevant item, a hit contributes 1/K and a miss contributes 0.

*Example*: A → 1/3, B → 1/3, C → 0, D → 1/3. Precision@3 = (1/3+1/3+0+1/3)/4 = **0.25**.
(With one relevant item per user, Precision@K is always just Recall@K ÷ K — it doesn't
carry extra information over Recall in *this* setup, but becomes distinct once a user
can have multiple relevant items, e.g. the personalization phase.)

### NDCG@K (Normalized Discounted Cumulative Gain)
Like Recall, but rewards finding the item *near the top* of the list more than near
the bottom. A hit at rank r scores `1 / log2(r + 1)`; a miss scores 0. (With one
relevant item, the "ideal" ranking always puts it at rank 1, so the "normalized" part
is just dividing by 1 — the score above already is the normalized one.)

*Example*: A hits at rank 1 → 1/log2(2) = **1.0** (best possible). B hits at rank 3 →
1/log2(4) = **0.5**. C misses → 0. D hits at rank 1 → 1.0.
NDCG@3 = (1.0 + 0.5 + 0 + 1.0)/4 = **0.625**.

### MRR (Mean Reciprocal Rank)
Average of `1/rank` for whichever rank the hit landed at (0 if no hit). Similar spirit
to NDCG (rewards early hits) but decays faster/differently — `1/rank` vs. NDCG's
`1/log2(rank+1)`.

*Example*: A → 1/1 = 1.0, B → 1/3 = 0.333, C → 0, D → 1/1 = 1.0.
MRR@3 = (1.0 + 0.333 + 0 + 1.0)/4 = **0.583**.

### MAP@K (Mean Average Precision)
The general definition: for each user, walk down their top-K list and every time you
hit a relevant item, compute precision-so-far (hits found ÷ position); average those
per-hit precisions for that user (this matters when a user has *multiple* relevant
items); then average across users.

**In our current setup** (exactly one relevant item per user), a hit at rank r has
exactly one "precision-so-far" value to average: `1/r` — which is identical to
Reciprocal Rank. **So MAP@K == MRR@K here, and always will, until evaluation moves to
a setting with multiple relevant items per user** (planned for the H&M personalization
phase). That's not a bug — `map_at_k()` is implemented generally, it just happens to
collapse to the same formula under this specific split. Numerically: MAP@3 = **0.583**,
same as MRR above.

### Coverage@K
Across *every* user's top-K list combined, what fraction of the entire catalog ever
gets recommended to *somebody*? A population-level metric, not a per-user one.

*Example*: catalog of 5 items; across A/B/C/D's lists, items {0,1,2,3,4} all appear at
least once → Coverage@3 = 5/5 = **1.0** (the toy example is too small to show a
low-coverage case — see the real numbers below, where popularity's Coverage@10 is
under 1% of a 23,965-item catalog).

Low coverage = the model keeps recommending the same handful of items to everyone
(the popularity baseline's defining flaw). High coverage = different users get
genuinely different slices of the catalog.

### Novelty@K
How "unsurprising" are the recommended items, on average? Measured in **bits of
self-information**: for an item with popularity probability `p` (its share of all
training interactions), self-information = `-log2(p)`. Popular items have high `p` →
low self-information → low novelty contribution. Rare items have low `p` → high
self-information → high novelty contribution. Novelty@K averages this over *every*
recommended slot across all users (not deduplicated — an item recommended to 1,000
users counts 1,000 times).

*Worked example* (from `tests/test_metrics.py::test_novelty_at_k`): 3 items with
popularity probabilities `[0.5, 0.25, 0.25]` → self-information
`[-log2(0.5), -log2(0.25), -log2(0.25)] = [1, 2, 2]` bits. If item 0 is recommended to
user A alongside item 1, and to user B alongside item 2 (K=2), the 4 recommended slots
are items `[0, 1, 0, 2]` → self-informations `[1, 2, 1, 2]` → **Novelty@2 = 1.5 bits**.

A model that only ever recommends blockbusters scores low here regardless of how
accurate it is; a model that surfaces long-tail items scores higher.

### Intra-list similarity@K (a diversity metric)
How similar are the items *within one user's own list* to each other, on average?
Needs the model's own learned item embeddings (an (n_items, d) vector per item, e.g.
ALS/BPR's factor matrices) — **only computed for models that have an embedding
space**, so popularity and item-item CF leave this metric blank in `metrics_log.csv`.

For each user: take every pair of items in their top-K list, compute cosine similarity
between their embedding vectors, average all those pairwise similarities. Then average
that per-user number across all users. 1.0 = every item in the list points in exactly
the same learned "taste direction" (maximally repetitive — e.g. 10 near-duplicate
sequels); 0.0 = the items are unrelated in the model's own learned space (maximally
diverse).

*Worked example* (from `tests/test_metrics.py::test_intra_list_similarity_at_k_mixed_list`):
one user's top-3 list has items with (toy, 2-dimensional) embeddings
`[1,0]`, `[0,1]`, `[1,0]`. Pairwise cosine similarities: (item1,item2)=0,
(item1,item3)=1, (item2,item3)=0. Averaging all 6 ordered pairs (each unordered pair
counted twice, matching the diagonal-excluded (K×K) similarity matrix the code
actually computes): (0+1+0 + 0+1+0)/6 = **0.333**.

Lower = a genuinely more varied list; a model can have great accuracy metrics while
still recommending 10 near-identical items to every user, and this is the metric that
would catch that.

---

## Popularity baseline

**Core idea**: rank every item by its raw training-interaction count and recommend that
same global ranking to every user (minus items they've already interacted with).

**Known shortcomings**:
- Zero personalization by construction, and the actual run confirms just how severe
  that is: Coverage@10 is 0.78% of the 23,965-item catalog (~188 distinct items) across
  161,572 evaluation users. The small amount of coverage that *does* exist is a masking
  artifact, not personalization — a user who's already interacted with the top hits
  gets pushed down to the next-most-popular items, so slightly different users see
  slightly different lists purely because of what they've already seen, not because the
  model knows anything about their taste.
- Lowest novelty of anything we'll build here (8.05–8.43 bits across K=5/10/20) — it
  always leads with the most-interacted-with items, definitionally the opposite of novel.
- Absolute accuracy looks small in isolation (Recall@10 ≈ 4.9%, NDCG@10 ≈ 2.5%), which
  is expected at this catalog size (23,965 items) with exactly one held-out ground-truth
  item per user — worth keeping as the reference scale for later models rather than
  comparing to some assumed "good" number.

**Metrics vs. previous best**: n/a — first model, this *is* the floor.

**Interpretation**: exactly the behavior expected of a model with no per-user signal —
weak on accuracy, ranking quality, coverage, and novelty alike, but nearly free to
compute. Every later model needs to beat this to justify its added complexity.

---

## Item-item collaborative filtering

**Core idea**: cosine similarity between items based on shared users, truncated to each
item's top-200 nearest neighbors. A user's score for a candidate item is the sum of
similarities to items they've already interacted with — the first model in the roadmap
with genuine per-user signal.

**Known shortcomings**:
- Still fully cold-start: a user or item with no training interactions has no row/column
  in the similarity computation and can't be scored at all. This doesn't surface in
  these numbers only because Phase 0 already filtered to users/items with ≥5
  interactions.
- Similarity is co-occurrence-only, with no notion of latent taste dimensions — two
  items with an identical audience but no *shared raters* (small or fragmented user
  overlap) get zero similarity even though a smarter model might infer they're related.
  This is exactly the gap a latent-factor model (ALS/BPR, Phase 2) is built to close.
- The top-200-neighbor truncation is a real tradeoff, not a free lunch: it's what keeps
  fitting tractable at 24K items, but it clips any item whose true useful neighborhood
  is larger or more diffuse than 200.

**Metrics vs. previous best** (K=10, validation split):

| metric | popularity | item-item CF | delta |
|---|---|---|---|
| Recall@10 | 0.0491 | 0.0797 | +62% |
| Precision@10 | 0.00491 | 0.00797 | +62% |
| NDCG@10 | 0.0247 | 0.0414 | +67% |
| MRR@10 / MAP@10 | 0.0174 | 0.0298 | +72% |
| Coverage@10 | 0.78% | 9.61% | 12.3x |
| Novelty@10 (bits) | 8.23 | 8.98 | +0.75 bits |

(Same direction and similar relative magnitude at K=5 and K=20 too — see
`artifacts/results/metrics_log.csv`.)

**Interpretation**: every metric moved the same direction, which is the expected story
for the first personalized model against a non-personalized floor. The moment a model
can tell users apart at all — even via the crude proxy of "items co-rated with what you
liked" — it wins on accuracy/ranking (Recall/NDCG/MRR/MAP all up 60–70%) *and* on
coverage (12x more of the catalog surfaces, since different users now pull different
neighborhoods) *and*, more subtly, on novelty. Cosine similarity isn't explicitly
optimizing for novelty, but it's no longer anchored to one global popularity ranking, so
the "same top-10 for everyone" concentration effect that suppressed popularity's novelty
score goes away as a side effect of personalization. That all four metric families moved
together here is not something to expect by default — Phase 2's ALS/BPR are more likely
to show an accuracy-vs-diversity tradeoff — but for a first personalized model beating a
non-personalized floor, this is the expected direction.

---

## ALS matrix factorization

**Core idea**: learn a 64-dimensional embedding vector for every user and every item
such that their dot product approximates a confidence-weighted implicit-feedback
signal (Hu/Koren/Volinsky), solved via Alternating Least Squares — fix item vectors and
solve for user vectors in closed form, then fix user vectors and solve for item
vectors, repeat. The first model with a genuine learned "taste space," rather than a
co-occurrence table.

**Known shortcomings**:
- Cold-start persists exactly as before — a user/item needs a learned embedding to be
  scored at all.
- The confidence weight `alpha` is a real, non-obvious hyperparameter, not a "set it
  and forget it" constant — see the ablation below, where it turns out *not* to be a
  "higher is better" knob.
- A pure dot-product score has no natural interpretation as a probability or rating —
  it's only meaningful as a relative ranking (same caveat applies to BPR below).

**Metrics vs. previous best** (K=10, validation split; canonical `alpha=20` config vs.
item-item CF):

| metric | item-item CF | ALS (α=20) | delta |
|---|---|---|---|
| Recall@10 | 0.0797 | 0.0966 | +21% |
| NDCG@10 | 0.0414 | 0.0470 | +14% |
| MRR@10 / MAP@10 | 0.0298 | 0.0322 | +8% |
| Coverage@10 | 9.61% | 15.6% | +62% (relative) |
| Novelty@10 (bits) | 8.98 | 9.93 | +0.95 bits |
| Intra-list similarity@10 | n/a (no embeddings) | 0.498 | first model to report this |

**Interpretation**: same direction as popularity→CF — a genuine latent-factor model
beats a co-occurrence-only one on every accuracy/ranking metric, plausibly because the
learned embedding space captures *transitive* taste relationships item-item cosine
similarity structurally cannot see (two items with no shared raters but similar
taste-space neighbors). Intra-list similarity of 0.498 has no prior baseline to compare
against yet — it becomes the reference point for the alpha ablation right below, and
for BPR after that.

### Hyperparameter ablation: alpha=20 vs. alpha=1

Everything else held constant (factors=64, regularization=0.01, iterations=15) — only
the confidence weight on *observed* interactions changes. Recall `alpha`'s role: under
Hu/Koren/Volinsky, confidence = `1 + alpha * value`, so with our binary confidence data
(`value = 1.0` for every observed positive), alpha=20 means observed pairs get
confidence 21 vs. an unobserved pair's confidence 1 (a 21x contrast); alpha=1 means
confidence 2 vs. 1 (a 2x contrast).

| metric | ALS (α=20) | ALS (α=1) | delta |
|---|---|---|---|
| Recall@10 | 0.0966 | **0.1044** | +8% (α=1 wins) |
| NDCG@10 | 0.0470 | **0.0534** | +14% (α=1 wins) |
| MRR@10 / MAP@10 | 0.0322 | **0.0381** | +18% (α=1 wins) |
| Coverage@10 | **15.6%** | 5.6% | -64% (α=20 wins) |
| Novelty@10 (bits) | **9.93** | 9.62 | -0.32 bits (α=20 wins) |
| Intra-list similarity@10 | 0.498 | **0.325** | -35% (α=1 is more diverse per-list) |

**Interpretation**: a genuine tradeoff, not a case where one config strictly wins.
Higher alpha pushes the optimization to reconstruct observed interactions with much
higher confidence relative to the (uniform) negative signal from everything
unobserved. That stronger contrast appears to pull each user's learned vector toward
faithfully reflecting *all* of their actual history, including less-mainstream
watches — which plausibly explains the higher coverage and novelty at alpha=20 (the
model is more willing to surface less-popular items it saw a specific user watch). But
that same aggressive fit to training data comes at a real cost: alpha=1's gentler
weighting generalizes *better* to the held-out next item (Recall/NDCG/MRR all higher).
Put simply, alpha trades "faithfully reconstruct this user's history" against
"correctly predict what they'll do next," and 20 overshoots that tradeoff on this
dataset. This is a two-point ablation, not a tuned grid search — the honest conclusion
is "alpha needs validation-based tuning, don't just pick a textbook value," not "alpha=1
is the right answer."

---

## Bayesian Personalized Ranking (BPR)

**Core idea**: repeatedly sample a user's observed item and a random unobserved item,
and take an SGD step that pushes the observed item's predicted score above the sampled
one's — directly optimizing pairwise ranking order, via `implicit`'s Cython/OpenMP
implementation (replacing the sibling notebooks' from-scratch NumPy BPR, which
wouldn't finish in reasonable time at this scale).

**Known shortcomings**:
- Same cold-start limits as ALS.
- Trained here with `implicit`'s default random-negative sampling and a fixed learning
  rate (0.01) for 100 epochs — no learning-rate schedule and no hyperparameter search.
  SGD-based training is known to be more sensitive to both than ALS's alternating
  closed-form solve, and the results below are consistent with that.
- `implicit`'s BPR appends an extra bias dimension to its factor matrices internally
  (confirmed by inspecting the fitted arrays directly: every user's last learned factor
  is exactly 1.0, only the item side varies per item) — handled correctly here
  (`models/bpr.py` strips it before exposing `item_embeddings` for the diversity
  metric), but worth knowing if you ever inspect the raw `implicit` model object
  yourself.

**Metrics vs. previous best** (K=10, validation split):

| metric | item-item CF | ALS (α=20) | ALS (α=1) | BPR |
|---|---|---|---|---|
| Recall@10 | 0.0797 | 0.0966 | **0.1044** | 0.0812 |
| NDCG@10 | 0.0414 | 0.0470 | **0.0534** | 0.0403 |
| MRR@10 / MAP@10 | 0.0298 | 0.0322 | **0.0381** | 0.0281 |
| Coverage@10 | 9.61% | 15.6% | 5.6% | **53.0%** |
| Novelty@10 (bits) | 8.98 | 9.93 | 9.62 | **12.36** |
| Intra-list similarity@10 | n/a | 0.498 | 0.325 | 0.533 |

**Interpretation**: the first clear case in this roadmap where the "next" model does
*not* uniformly win. BPR's NDCG and MRR/MAP land slightly *below* even plain item-item
CF, and its Recall is only marginally above CF's (+1.9%) — well below both ALS variants
on every accuracy/ranking metric. At the same time, BPR has by far the widest
population-level reach (53% catalog coverage — 3.4x ALS's best) and the highest novelty
(12.36 bits) of anything built so far. Oddly, though, its intra-list similarity (0.533)
is the *highest* — i.e. least diverse — of the three embedding models: BPR spreads
very broadly across users and leans toward less-popular items, while each individual
user's own list is actually slightly more internally homogeneous than ALS's. That
combination is a good reminder that coverage (spread across the whole user population),
novelty (popularity of individual items), and intra-list similarity (homogeneity within
one list) are genuinely independent axes — none of them is a proxy for the others.

Two mechanism-level explanations, both about *how BPR was trained* rather than BPR
being a worse idea in principle:
1. **Pairwise ranking with uniform random negative sampling carries no
   confidence/frequency signal.** Every observed interaction is treated identically
   regardless of how many other users liked that item too, so BPR has far less inherent
   pull toward re-recommending popular items than ALS's confidence-weighted objective —
   which is exactly why coverage and novelty jump. The flip side is less signal-driven
   reason to rank any *specific* held-out "next" item highly.
2. **SGD convergence here is an open, unverified question.** We ran `implicit`'s
   defaults (learning rate 0.01, 100 epochs) with no convergence check (e.g. plotting
   training AUC across epochs) and no learning-rate tuning. ALS's alternating solve
   tends to be robust with minimal tuning; BPR's SGD is known to be considerably more
   sensitive to both epoch count and learning rate. It's entirely plausible BPR is
   undertrained relative to its real ceiling here, not fundamentally worse than ALS on
   this data.

**This run doesn't resolve which explanation dominates** — a natural follow-up (not
required for the roadmap to keep moving) would be re-running BPR with more epochs or a
tuned learning rate and checking whether Recall/NDCG close the gap with ALS while
coverage/novelty stay high. Either way, this is exactly the kind of result the
project's comparative-learning goal exists to surface: a more sophisticated-sounding
model that doesn't automatically win, with a concrete, mechanism-grounded (if not yet
fully verified) explanation — rather than assuming "newer = better" or waving the result
away as noise.

---

## Two-stage ranking (FAISS candidate generation + LightGBM reranking)

**Core idea**: retrieve `n_candidates=200` candidates per user via approximate
nearest-neighbor search (FAISS `IndexIVFFlat`, nlist=100/nprobe=10) over the ALS α=1
model's embedding space, then re-score just those candidates with a LightGBM
`lambdarank` model trained on 5 engineered features (the ANN score itself, item
popularity, user activity, item-item CF's score for that pair, and genre affinity).
Reranker training labels come from the validation split (never test), with the base
model's ANN retrieval score reused as the strongest input feature.

**Known shortcomings** — this run surfaced a severe, diagnosable one, not a vague
caveat:
- **Retrieval ceiling (moderate, expected)**: the ANN-vs-exact recall diagnostic found
  only **79.17%** of each user's exact top-20 items (by the base ALS model's own
  full-catalog scoring) are present within the 200 approximate candidates — roughly
  21% of what a reranker could ideally select from is structurally unreachable, purely
  from using an approximate index. This is exactly the cost the plan expected and asked
  to be measured, not a surprise.
- **Training-label injection bias (severe — the dominant factor)**: **12,339 of 20,000
  training queries (61.7%) had their true held-out item missed by ANN retrieval
  entirely**, and had to be injected into the candidate set to give the reranker a
  positive example at all. But an injected item's `ann_score` feature is *necessarily*
  low — if it had scored well enough to be a genuine top-200 neighbor, ANN would have
  found it without help. So for well over half the reranker's training examples, the
  label says "this is the relevant item" while its strongest feature says "this looks
  like a bad candidate" — a majority-share contradiction of the pattern the other ~38%
  of (non-injected) examples teach. This is the most likely explanation for why the
  reranker's final ranking ends up *worse* than just using the raw ANN score directly.

**Metrics vs. previous best** (K=10, validation split):

| metric | als_alpha1 (retrieval base) | itemknn | popularity | two_stage |
|---|---|---|---|---|
| Recall@10 | **0.1044** | 0.0797 | 0.0491 | 0.0453 |
| NDCG@10 | **0.0534** | 0.0414 | 0.0247 | 0.0206 |
| MRR@10 / MAP@10 | **0.0381** | 0.0298 | 0.0174 | 0.0133 |
| Coverage@10 | 5.56% | 9.61% | 0.78% | **30.36%** |
| Novelty@10 (bits) | 9.62 | 8.98 | 8.23 | **11.25** |
| Intra-list similarity@10 | 0.325 | n/a | n/a | 0.347 |

**Interpretation**: this is the first model in the roadmap that regresses below *every*
prior model on accuracy/ranking, including the plain popularity floor — a genuinely bad
outcome for a more sophisticated architecture, worth taking at face value rather than
explained away. The two diagnostics above make it possible to actually attribute *why*
rather than just observe *that*: the ~21% retrieval-ceiling loss is real but far too
small on its own to explain a >55% relative drop below the raw ALS scores the pipeline
is built on top of. The 61.7% injection rate is the far more convincing explanation —
the reranker was trained on a majority-contradictory signal for its own strongest
feature. Coverage (30%, nearly double BPR's) and novelty (11.25 bits, second only to
BPR) both jumped, consistent with a model that's effectively learned to *discount* the
(corrupted) `ann_score` feature in favor of popularity/itemknn/genre signals that stay
consistent across both injected and organic rows — spreading recommendations more
broadly and toward less-popular items, but at real accuracy cost.

**This looked like a real, fixable methodology bug, not evidence that two-stage ranking
doesn't work here** — confirmed below.

### Fix: `is_injected` feature

**In plain words, what actually went wrong and what the fix does:**

To train the reranker, each user needed one positive example ("a movie they really
watched next") among a pile of negatives ("movies they didn't"). The candidate pool
came from the ANN search — the fast approximate lookup that finds the ~200 movies most
similar to a user's taste. For 12,339 of 20,000 users (62%), the movie they *actually*
watched next wasn't in that user's top-200 ANN results at all — recommendation is hard,
and the approximate search often misses. Since the reranker still needed a positive
example for every user, the code force-inserted the true movie into the candidate list
anyway (replacing the last slot).

That backfired because every candidate carries an `ann_score` — the retrieval model's
own confidence that this item fits the user. A normally-picked candidate has a
*decent* score, because it made the top-200. A force-inserted one, by definition, did
**not** make the top-200 — that's exactly why it had to be forced in — so its score is
low. The reranker was therefore shown thousands of examples saying "low ann_score...
and it's correct" (the injected ones) right alongside thousands saying "high
ann_score... and it's correct" (the normal ones) — a flat contradiction. Since the
contradictory case was the *majority* (62%), the reranker learned that `ann_score`
barely means anything, and threw away its single most useful signal. That's why the
first version of `two_stage` performed worse than even the plain popularity list.

**The fix**: add one extra flag to each candidate, `is_injected` (1 if force-inserted,
0 if ANN found it normally), so the reranker can tell the two situations apart instead
of conflating them — "a low ann_score is a bad sign normally, but ignore that rule for
the injected ones." At actual prediction time nothing is ever force-inserted (we only
ever show the reranker what it genuinely retrieved), so `is_injected` is always 0
there — it only mattered as an "exception" tag during training. That one flag is why
accuracy nearly doubled: the reranker went from a corrupted understanding of its best
feature back to a correct one.

**More precisely**: added a 6th candidate feature, `is_injected` (1.0 for a
force-inserted positive during training, always 0.0 at inference since `score_batch`
never injects anything — see `features/engineer.py`), so the reranker can learn to
treat injected and organically-retrieved rows differently instead of reading
`ann_score` the same contradictory way for both. Retrained with no other change.

**Metrics before vs. after the fix, and against the base model** (K=10, validation):

| metric | two_stage (broken) | two_stage (fixed) | delta (fix) | als_alpha1 (base) | itemknn |
|---|---|---|---|---|---|
| Recall@10 | 0.0453 | **0.0875** | +93% | 0.1044 | 0.0797 |
| NDCG@10 | 0.0206 | **0.0446** | +117% | 0.0534 | 0.0414 |
| MRR@10 / MAP@10 | 0.0133 | **0.0318** | +140% | 0.0381 | 0.0298 |
| Coverage@10 | 30.36% | 16.39% | -46% | 5.56% | 9.61% |
| Novelty@10 (bits) | 11.25 | 10.60 | -5.8% | 9.62 | 8.98 |
| Intra-list similarity@10 | 0.347 | 0.388 | +12% | 0.325 | n/a |

**Interpretation**: the fix recovered nearly all of the lost accuracy (Recall/NDCG/MRR
all up 93-140%), which is strong confirmation the diagnosis was right — the reranker
had genuinely been corrupted by the unflagged injected examples, not just "randomly
noisy." With that corrected, the picture is now a coherent, explainable two-stage
tradeoff rather than a broken one: two_stage beats item-item CF on every accuracy
metric (Recall/NDCG/MRR all higher than itemknn's), lands consistently **~16-17% below
its own base ALS model** on those same metrics (Recall: 0.0875/0.1044 = 83.8% of base;
NDCG: 83.5%; MRR: 83.4% — remarkably consistent across all three), while still keeping
a real coverage/novelty advantage over the base model (3x the coverage, +1 bit of
novelty) rather than the previous run's coverage-via-brokenness. That ~83-84% retention
figure lines up closely with the 79.17% ANN candidate recall ceiling measured earlier
(a slightly stricter bar — full top-20 overlap vs. just needing the one true item
present) — consistent with the remaining gap being mostly the retrieval stage's
approximation cost, not a leftover training artifact. Coverage/novelty didn't fully
return to the base model's low numbers, either, which makes sense: the reranker still
legitimately blends in popularity/itemknn/genre signal alongside `ann_score`, so some
divergence from pure ALS ranking is expected and desired, not a bug.

The two experiments not tried (excluding injected rows entirely from training; dropping
`ann_score` as a feature) remain open if there's appetite to see whether they'd close
the remaining ~16-17% gap further — flagged here rather than assumed unnecessary.

---

## Latency: retrieval vs. reranking (Phase 4)

**Core idea**: not a new model — a timing study of the two-stage pipeline's pieces,
answering "where does the time actually go?" rather than "how accurate is it?". Three
paths, each timed per-request (one call per sampled user, `scripts/run_latency_benchmark.py`):
`brute_force_exact` (ALS scores the full ~24K-item catalog, `argpartition` top-K),
`faiss_ann_retrieval` (the two-stage pipeline's retrieval stage alone — FAISS `IndexIVFFlat`
search, `nlist=100`/`nprobe=10`), and `two_stage_full` (retrieval's 200 candidates,
rescored by the LightGBM reranker, sorted to top-K). Deliberately scoped down from the
original plan's FastAPI-serving angle — this project isn't hosting anything for real
users, so a plain in-process timing loop answers the actual question (retrieval vs.
rerank cost) without web-framework overhead in the way; see the Phase 4 RUNBOOK in
`README.md` for the reasoning.

**Known shortcomings going in**: the catalog (~24,000 items) is small enough that exact
brute-force scoring was expected to already be sub-millisecond, which would make FAISS's
approximate speedup a methodology exercise more than a real win — worth checking against
real numbers rather than assuming.

**Latency, K=20, 500 timed requests (2026-09-22)**:
| path | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| brute_force_exact | 0.180ms | 0.186ms | 0.243ms | 0.252ms |
| faiss_ann_retrieval | 0.0095ms | 0.0085ms | 0.0153ms | 0.0211ms |
| two_stage_full | 0.645ms | 0.642ms | 0.734ms | 0.972ms |

**Interpretation**: the shortcoming predicted above held up — brute-force was already a
fifth of a millisecond, so FAISS's ~19x speedup at retrieval (0.0095ms vs. 0.180ms,
because IVF only searches ~10 of 100 clusters, i.e. ~10% of the catalog, instead of
scanning all of it) is real but practically inconsequential in absolute terms. The
number that actually matters: reranking dominates total cost. `two_stage_full`
(0.645ms) is ~68x slower than retrieval alone, and that gap is entirely the reranker
stage — building the 6-feature matrix for 200 candidates plus LightGBM inference on
them. It's also where essentially all of the tail-latency risk lives: p99 for retrieval
alone barely moves off its mean (0.021ms vs. 0.0095ms), while `two_stage_full`'s p99
(0.972ms) is 51% higher than its own mean — the reranker's per-request cost is far less
predictable than either pure-retrieval path. Bottom line: at this catalog size, the
two-stage architecture's latency story isn't "FAISS makes retrieval fast" (it was
already fast) — it's "the reranker is the entire latency budget," which is the
opposite of the accuracy story (Phase 3), where retrieval's ANN approximation, not the
reranker, was the accuracy-limiting stage. Speed and accuracy bottleneck in different
stages of the same pipeline.

---

## Phase 5b, part 1 — Baselines + MF on H&M

**Core idea**: not a new model — the same four Phase 1-2 models (popularity,
item-item CF, ALS at alpha=20 and alpha=1, BPR), retrained unchanged on a second,
structurally different dataset (H&M purchases, 90-day window: 378,631 users, 27,779
items, 3,653,414 rows) to see which MovieLens findings are general properties of the
models and which were specific to movie-rating data. The two datasets differ in a way
that matters mechanically, not just in size: H&M has ~2.3x MovieLens's users over a
similar-size catalog, but **far less signal per user** (≈7.7 train interactions/user
vs. MovieLens's ≈94) and a correspondingly sparser overall matrix (~0.035% of the
user-item grid is a positive, vs. MovieLens's ~0.4% — over 10x sparser).

**Known shortcomings going in**: every model was expected to score lower here purely
from sparsity, independent of which algorithm is "better" — ALS/BPR need enough
per-user signal to estimate a stable latent vector, and item-item CF needs enough
co-purchase overlap to estimate reliable similarities; H&M's 90-day window stresses
both, just in different ways (few interactions per user, but a large user base feeding
each item's co-occurrence counts).

**Metrics, H&M, K=10, validation**:
| metric | popularity_hm | itemknn_hm | als_hm (α=20) | als_alpha1_hm (α=1) | bpr_hm |
|---|---|---|---|---|---|
| Recall@10 | 0.0093 | **0.0958** | 0.0843 | 0.0513 | 0.0455 |
| NDCG@10 | 0.0046 | **0.0623** | 0.0496 | 0.0320 | 0.0269 |
| MRR@10 / MAP@10 | 0.0032 | **0.0519** | 0.0391 | 0.0261 | 0.0213 |
| Coverage@10 | 0.05% | **98.33%** | 10.99% | 2.52% | 45.47% |
| Novelty@10 (bits) | 9.81 | **15.00** | 11.63 | 11.05 | 13.99 |
| Intra-list similarity@10 | n/a | n/a | 0.680 | 0.476 | 0.654 |

**Same models, MovieLens vs. H&M, Recall@10** (context for how much of the gap is
sparsity vs. algorithm):
| model | MovieLens | H&M | H&M as % of ML |
|---|---|---|---|
| popularity | 0.0491 | 0.0093 | 19% |
| itemknn | 0.0797 | 0.0958 | 120% |
| als (best alpha) | 0.1044 (α=1) | 0.0843 (α=20) | 81% |
| bpr | 0.0812 | 0.0455 | 56% |

**Interpretation**: two MovieLens findings reversed on H&M, and both reversals trace
back to the sparsity difference, not noise:

1. **Item-item CF beat ALS here** (Recall@10 0.0958 vs. 0.0843), the opposite of
   MovieLens (0.0797 vs. 0.1044) — in fact itemknn is the *only* model whose H&M
   accuracy exceeds its MovieLens accuracy at all (120%). Mechanically: ALS has to
   jointly learn a 64-dimensional latent vector per user from an average of just 7.7
   observed interactions — not much signal to fit 64 free parameters against, and
   factorization models are known to need denser data to avoid poorly-estimated
   factors. Item-item CF never estimates anything per-user; it only needs enough
   *per-item* co-purchase overlap, and with 378,631 users feeding statistics into just
   27,779 items (≈131 interactions/item on average), the item side of this dataset is
   actually reasonably dense even though the user side is sparse. Non-parametric,
   co-occurrence-based similarity tolerates sparse users better than a jointly-learned
   embedding space does.

2. **Alpha=20 beat alpha=1 here** (Recall@10 0.0843 vs. 0.0513), the opposite of
   MovieLens, where alpha=1 clearly won. This directly answers the question posed when
   these H&M configs were written: the "alpha=1 generalizes better" finding does
   **not** transfer — it was a property of MovieLens's relatively denser signal, not a
   universal rule about implicit-feedback confidence weighting. Mechanically, `alpha`
   controls how much more an observed interaction counts than everything else ALS
   treats as an implicit negative; with H&M's matrix over 10x sparser than
   MovieLens's, the tiny number of true positives needs to be weighted *heavily* to
   avoid being drowned out by the overwhelming mass of unobserved (user, item) pairs.
   On MovieLens's denser matrix, that same heavy weighting instead pushed ALS toward
   overfitting popular co-occurrence patterns — which is exactly what Phase 2's
   `LEARNINGS.md` entry found. Same knob, opposite optimal value, because the
   underlying data density is what alpha is actually compensating for.

3. **BPR reproduced its known shortcoming**: it underperforms ALS on every accuracy
   metric here too (Recall@10 0.0455 vs. als_hm's 0.0843) while carrying much higher
   coverage (45.47% vs. 10.99%) and novelty (13.99 vs. 11.63) — the same
   accuracy-for-exposure-diversity tradeoff documented for MovieLens's BPR entry,
   holding up on a structurally different dataset.

4. **Popularity is still the floor**, and by a wider margin proportionally than on
   MovieLens (H&M's popularity Recall@10 is only 19% of MovieLens's, the steepest drop
   of any model) — plausible explanation: H&M's catalog turns over seasonally, so a
   90-day popularity signal is a less stable predictor of what any given user wants
   next than movie popularity accumulated over years, on top of the sparsity effect
   every model already faces.

Item-item CF's near-total coverage (98.33%) and top novelty (15.00 bits) alongside its
*best* accuracy is a genuinely different picture than MovieLens, where accuracy and
diversity mostly traded off against each other. Worth treating as a hypothesis to keep
an eye on through the rest of Phase 5, not yet a settled conclusion — cold-start and
segment-level breakdowns (next) will show whether itemknn's win holds up uniformly or
is concentrated in some slice of users/items.

---

## Phase 5b, part 2a — Segment Breakdowns on H&M

**Core idea**: slice each of the four Phase 5b-part-1 models' leave-one-out test
accuracy by *who* the user is — activity-level tercile (low/medium/high train
interaction count), age bucket, and club membership status (from `customers.csv`) —
instead of trusting one global number to represent every user's experience.
`scripts/run_segment_eval.py` reuses the already-trained models unchanged; only the
evaluation set is filtered per segment.

**Known shortcomings/expectations going in**: the natural guess was that more-active
users would be *easier* to predict (more training signal per user) and that
age/membership would matter less, since neither is a strong behavioral signal by
itself.

**Recall@10 by activity tercile**:
| activity | n_users | popularity_hm | itemknn_hm | als_hm | bpr_hm |
|---|---|---|---|---|---|
| low | 152,175 | 0.0105 | **0.1004** | 0.0999 | 0.0582 |
| medium | 102,671 | 0.0093 | **0.0907** | 0.0780 | 0.0430 |
| high | 123,785 | 0.0069 | **0.0694** | 0.0502 | 0.0231 |

**Recall@10 by age bucket** (all models, narrower spread than activity):
| age | n_users | popularity_hm | itemknn_hm | als_hm | bpr_hm |
|---|---|---|---|---|---|
| <25 | 103,631 | 0.0117 | 0.0839 | 0.0780 | 0.0422 |
| 25-34 | 124,017 | 0.0088 | 0.0931 | 0.0791 | 0.0426 |
| 35-44 | 41,997 | 0.0070 | 0.0867 | 0.0771 | 0.0434 |
| 45-54 | 67,481 | 0.0083 | 0.0847 | 0.0753 | 0.0421 |
| 55+ | 39,694 | 0.0061 | 0.0861 | 0.0770 | 0.0433 |
| unknown | 1,811 | 0.0099 | 0.0922 | 0.0883 | 0.0574 |

**Recall@10 by club membership status**:
| membership | n_users | popularity_hm | itemknn_hm | als_hm | bpr_hm |
|---|---|---|---|---|---|
| ACTIVE | 372,124 | 0.0089 | 0.0865 | 0.0767 | 0.0419 |
| PRE-CREATE | 5,957 | 0.0138 | **0.1526** | 0.1410 | 0.0849 |
| unknown | 501 | 0.0160 | 0.1138 | 0.0898 | 0.0519 |
| LEFT CLUB | 49 | 0.0000 | 0.1429 | 0.0612 | 0.1020 |

**Interpretation**:

1. **The activity-level guess was backwards**: recall gets *worse*, not better, as
   users get more active — true for every single model, popularity included (high:
   0.0069 vs. low: 0.0105). More training signal didn't make these users easier to
   predict. The likely mechanism: leave-one-out recall asks "did we guess the *one*
   specific next item," and a low-activity user (near the 3-interaction floor) tends
   to have a narrow, repetitive taste signature (repurchasing wardrobe basics is
   common in fashion retail) that's easy to extrapolate one step further, while a
   high-activity user's purchase history spans a wider range of categories, making
   "the next specific item out of 27,779" a genuinely harder guess even with more data
   to work from. Worth remembering for Phase 6 or any future work here: more
   interactions per user isn't automatically an accuracy advantage for this kind of
   next-item task.

2. **Item-item CF's part-1 win is not an aggregation artifact** — it beats ALS in
   *every single segment* tested (all 3 activity buckets, all 6 age buckets, and 3 of 4
   membership groups, the sole exception being the 49-user LEFT CLUB group, too small
   to read anything into). This is a much stronger result than the global number alone
   would justify: item-item CF's advantage on this dataset is uniform, not
   concentrated in some slice that happened to dominate the aggregate.

3. **Age barely matters**: every model's recall stays within a narrow band across all
   six age buckets (itemknn: 0.084-0.093, roughly an 11% relative spread — much
   tighter than activity's 45% spread from low to high). Age alone isn't doing much
   work here, at least for the models tested.

4. **PRE-CREATE customers are the easiest segment by far** (itemknn 0.1526 vs. ACTIVE's
   0.0865, a 76% relative lift, same ordering for every model) — plausibly a narrower,
   more predictable purchase pattern for accounts created through a lighter-weight
   flow (e.g. guest-checkout-derived records) than full active members' broader
   shopping histories. At 5,957 users this is a real sample, not noise, unlike the
   49-user LEFT CLUB group (0% popularity recall there is a 0-of-49 fluke, not a
   finding — flagged rather than reported as if it were reliable).

---

## Phase 5b, part 2b — Cold-start on H&M

**Core idea**: `scripts/run_coldstart_eval.py`, run against the same four models
retrained on `global_train.parquet` instead of `loo_train.parquet` — the only split
with genuinely cold test users (10,501 of them, zero rows in train), since
leave-one-out's test set can never contain a user the model hasn't seen at all.

**Prediction going in**: popularity's list doesn't depend on user history at all, so
its cold-vs-warm gap should be near zero; itemknn/ALS/BPR *do* depend on the user's row
in the training matrix, so their cold-segment accuracy should collapse.

**Recall@10, all / cold / warm**:
| model | all (n=62,306) | cold (n=10,501) | warm (n=51,805) | cold vs. warm |
|---|---|---|---|---|
| popularity_hm | 0.0090 | **0.0119** | 0.0084 | **+41%** |
| itemknn_hm | 0.0145 | 0.00019 | 0.0174 | **-99%** |
| als_hm | 0.0159 | 0.00019 | 0.0190 | **-99%** |
| bpr_hm | 0.0070 | 0.0000 | 0.0085 | **-100%** |

**Interpretation**: the prediction held up, and more precisely than expected —
itemknn's and als's cold-segment recall aren't just both near-zero, they're **bit-for-bit
identical** (0.00019045805161413198 for both), and so is their cold-segment coverage@10
(0.00035998..., i.e. exactly 10 distinct items ever recommended across all 10,501 cold
users, for *both* models). That's not a coincidence — it's the two models hitting the
same mechanical failure mode from two different angles:

- **ALS**: a cold user's factor vector comes from the closed-form update
  `x_u = (Yᵀ C^u Y + λI)⁻¹ Yᵀ C^u p(u)`. With zero observed interactions, every item's
  confidence is the baseline (`c=1`) and every preference is `p=0`, so the right-hand
  side is exactly the zero vector — `x_u = 0`. A zero user vector dotted with any item
  vector is 0, so **every item scores identically (0)** for a cold user.
- **Item-item CF**: a cold user's row in the user-item matrix is all zeros (they're
  absent from train), so `score = user_row @ similarity_matrix` is the zero vector too
  — the exact same all-items-tied outcome, for an unrelated mechanical reason.

Both then fall through to `numpy.argpartition`'s tie-breaking, which is deterministic
and index-order-based — so both models silently hand *every* cold user the identical
fixed list of ~10 items, drawn from wherever the tie-break lands, with no relationship
to popularity, content, or anything the user actually did. That fixed list happened to
match 2 of 10,501 users' true next purchase (hence the shared 0.00019 recall) — that
"hit" is luck, not signal.

**BPR's cold-segment collapse is total (0/10,501)**, and its cold coverage@10 is also
exactly ~10 items — so it's *also* handing every cold user one fixed list, just a
different one than ALS/itemknn's, and one that happened to match nobody. The likely
mechanism ties back to BPR's own bias column (documented in the Phase 2 entry: every
user's factor vector has a constant `1.0` appended as a bias term). BPR's SGD training
only touches factors for users it actually samples pairwise triples from — a cold user
is never sampled, so their non-bias dimensions stay at whatever they were initialized
to. If that initialization is uniform (e.g. zero) across users, a cold user's score
reduces to just the item's own learned bias term — a fixed, popularity-*adjacent* but
not identical ranking, which is consistent with getting a fixed list that nonetheless
missed every single ground-truth item. Flagged as the most likely mechanism given
BPR's known bias-column property, not something directly re-verified by inspecting raw
factor arrays here.

**Popularity's cold segment actually did *better* than warm** (+41%), the one genuine
surprise. A plausible retail-specific explanation: cold users are, by definition, brand
new to the training window — their first purchase plausibly skews toward
currently-trending/best-selling items (a common new-customer pattern), while warm
(established) users may already own the popular staples and are shopping further into
their own niche preferences, which a global popularity list is less likely to guess.
Worth treating as a hypothesis, not a proven mechanism, unlike the ALS/itemknn
tie-breaking result above which follows directly from the model equations.

**Bottom line for this project's cold-start story**: popularity isn't just "the floor"
here the way it was in Phase 5b part 1 — for a genuinely new user, it's the only model
of the four that produces a *meaningful* ranking at all. ALS, item-item CF, and BPR
don't gracefully degrade for cold users; they silently collapse to an arbitrary,
mechanically-determined fixed list with no real relationship to that user, which is a
much sharper and more concrete shortcoming than "personalized models struggle with
cold-start" as a vague statement — a production system would need an explicit
popularity (or content-based) fallback specifically gated on "does this user have any
training history," not just trust these models to degrade gracefully on their own.

---

## Phase 5b, part 3 — Popularity-Bias Analysis on H&M

**Core idea**: a new metric, `popularity_bias` — the fraction of each user's top-K
recommendation slots filled by an item from the catalog's top-1% most popular items
(by train interaction count) — now computed by `evaluate()` for every model
automatically (`evaluation/metrics.py`'s `top_popular_item_mask`/`popularity_bias_at_k`,
wired into `EvalRow`). Unlike `novelty` (an information-theoretic average), this is a
direct, interpretable number: "what fraction of this model's recommendations are just
top-sellers?" Run via `scripts/run_popularity_bias_eval.py` on the four Phase 5b
part-1 models, reusing the already-trained artifacts.

**Question this was built to answer**: Phase 5b part 1 found item-item CF beats ALS on
H&M accuracy while having *higher* coverage/novelty too — an unusual combination, since
MovieLens's models mostly traded accuracy off against diversity. Was itemknn's win real
personalization, or was it secretly leaning on popular items in a way novelty/coverage
alone didn't make obvious?

**Popularity-bias, K=10**:
| model | recall@10 | popularity_bias@10 | coverage@10 |
|---|---|---|---|
| popularity_hm | 0.0090 | **1.0000** | 0.0005 |
| als_hm | 0.0777 | 0.4390 | 0.1099 |
| itemknn_hm | **0.0876** | 0.0922 | 0.9833 |
| bpr_hm | 0.0426 | 0.0344 | 0.4547 |

**Interpretation**: the question is answered cleanly — **itemknn's win is real
personalization, not disguised popularity**. It has the best accuracy of the four *and*
one of the lowest popularity-bias scores (9.2% of its recommendations are top-1% items,
vs. ALS's 43.9% — ALS leans on popular items **4.8x more** than itemknn does, despite
being less accurate). Popularity itself sits at exactly 1.0000 by construction (its
list *is* the popular-items list), which is a useful sanity check that the metric
measures what it claims to.

This also sharpens the BPR finding from part 1: BPR has the *lowest* popularity bias of
any personalized model (3.4%, even lower than itemknn's), which now reads less like
"BPR is more personalized" and more like "BPR actively avoids popular items past the
point of usefulness" — it sacrifices more accuracy (Recall@10 0.0426, roughly half of
itemknn's) than its modest coverage/novelty gain over itemknn would justify. ALS sits
at the opposite extreme: it's the one model here that's genuinely leaning on popularity
as a crutch (44% of its list), which is consistent with matrix factorization's known
tendency to gravitate toward frequently-observed (i.e. popular) items through its loss
function, and helps explain *why* ALS underperforms itemknn on this sparse dataset
beyond the factor-estimation argument from part 1 — it's not just estimating worse
personalized signal, a meaningful chunk of its list isn't really personalized at all.

Coverage and popularity_bias move together as expected (low bias ↔ high coverage: 9.2%/
98.3% for itemknn, 43.9%/11.0% for ALS) — not a new independent finding, but a useful
consistency check that the two metrics are measuring compatible things from different
angles.

---

## Phase 5b, part 2c — Fixing Cold-start: Fallback Comparison

**Core idea**: part 2b diagnosed the cold-start collapse; this actually fixes it.
`ColdStartFallbackRecommender` (`models/coldstart_fallback.py`) wraps itemknn_hm,
als_hm, and bpr_hm with a fallback for cold users, tested against three fallback
strategies fit on `global_train`: plain popularity (`pop`), age-bucket popularity
(`age` — most popular items among users in the same age bucket), and recency-weighted
popularity (`recency` — interactions weighted by exponential decay, half-life 7 days,
instead of counted equally — a "what's trending right now" list).

**Cold-segment Recall by fallback** (identical across all three base models — once a
user is cold, only the fallback determines their score, not which base model wraps
it, which is itself a useful confirmation that the wrapper routes correctly):
| fallback | Recall@5 | Recall@10 | Recall@20 | Novelty@10 |
|---|---|---|---|---|
| (broken, no fallback) | ~0.0002 | ~0.0002 | ~0.0002 | 16.5-16.8 |
| pop | 0.0055 | 0.0119 | 0.0190 | 9.79 |
| age | **0.0082** | 0.0126 | 0.0228 | 9.98 |
| recency | 0.0070 | **0.0139** | **0.0351** | **10.61** |

**Net effect on the blended "all" segment, Recall@10, per base model**:
| base model | broken | +pop | +age | +recency |
|---|---|---|---|---|
| itemknn | 0.01453 | 0.01650 | 0.01661 | **0.01684** (+15.9%) |
| als | 0.01586 | 0.01783 | 0.01794 | **0.01817** (+14.6%) |
| bpr | 0.00703 | 0.00904 | 0.00915 | **0.00937** (+33.3%) |

**Interpretation**: recency wins overall, but not uniformly — the ranking flips with
K. At K=5, age is actually best (0.0082 vs. recency's 0.0070, a 15% edge); by K=10
recency has caught up and passed it (+11% over age); by K=20 recency pulls decisively
ahead (0.0351 vs. age's 0.0228, +54%, and +85% over plain popularity — nearly double).
A plausible read: age-bucket popularity is a stable, concentrated prior ("this
demographic's enduring favorites"), which is a good bet for the *very top* guess but
draws from a narrower effective set; recency's trending list is broader and shifts
day-to-day, less likely to nail the single best guess but covering more of the real
near-term signal once the list is allowed to grow — consistent with novelty being
highest for recency too (10.61 vs. age's 9.98 vs. pop's 9.79): it's not just more
accurate at K=20, it's doing so with a *less* generic list, not a more generic one.

Worth being precise about what's *not* happening here: none of these three fallbacks
personalize a cold user beyond a shared bucket. `recency` hands the exact same list to
every single cold user regardless of age, membership, or anything else about them
(confirmed by its coverage numbers matching the same "~K distinct items total, shared
across the whole 10,501-user cold segment" pattern as `pop`/`age` — just a different K
items). That a *completely non-personalized* "what's trending" signal beats a
demographically-bucketed one past the top few ranks is itself informative: for a
genuinely brand-new H&M customer, what's selling across the whole store right now
carries more predictive power than what their age group has historically favored.

**Bottom line for the cold-start story**: the collapse from part 2b is fixed, not
just documented — every base model's cold-segment recall goes from a meaningless
~0.0002 to a real, competitive-with-plain-popularity-or-better number, and the net
accuracy gain shows up in the blended metric too, largest for BPR (+33%) since its
cold-start failure was the most total to begin with. `recency` is the best default of
the three tried; `age` remains a reasonable alternative specifically when the product
surface only needs a handful of slots (K=5) rather than a fuller list.

---

## Phase 5b, part 4 — Two-Stage Ranking + Recency Feature on H&M

**Core idea**: builds an H&M two-stage pipeline from scratch (it didn't exist before —
Phase 3 only built one for MovieLens): FAISS retrieval over `als_hm`'s embeddings +
a LightGBM reranker, with H&M's own feature set (`category_affinity` from
`articles.csv`'s `product_group_name`, in place of genre_affinity) plus two new
recency features — `days_since_category_purchase` and `has_category_history` — using
H&M's real daily purchase timestamps (see `features/engineer.py`'s
`build_category_recency_features` and the README's Phase 5b part 4 RUNBOOK for the
full mechanism).

**Metrics, K=10, validation, 378,631 users**:
| model | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| popularity_hm | 0.0093 | 0.0046 | 0.0032 |
| bpr_hm | 0.0455 | 0.0269 | 0.0213 |
| als_alpha1_hm | 0.0513 | 0.0320 | 0.0261 |
| als_hm | 0.0843 | 0.0496 | 0.0391 |
| itemknn_hm (best single-stage) | 0.0958 | 0.0623 | 0.0519 |
| **two_stage_hm** | **0.1117** | **0.0709** | **0.0584** |

**Reranker feature importance** (LightGBM split count):
| feature | splits |
|---|---|
| user_activity_log | 697 |
| ann_score | 645 |
| itemknn_score | 605 |
| item_popularity_log | 441 |
| days_since_category_purchase | 288 |
| category_affinity | 224 |
| is_injected | 100 |
| has_category_history | **0** |

**Training diagnostics**: 12,965/20,000 (64.8%) of the reranker's training (val) users
needed their held-out item injected into the candidate set; ANN candidate recall
(vs. als_hm's own exact top-20) was **0.9106**.

**Interpretation**, three things worth unpacking:

1. **`has_category_history` at exactly 0 is a real result, not a bug** — and a useful
   contrast with why `is_injected` *did* need its own flag back in Phase 3. An
   injected row's `ann_score` overlaps with genuinely-low organic scores (no clean
   threshold separates them), so a flag was necessary to resolve real ambiguity. Here,
   the "never purchased" sentinel (9999 days) is nothing like a real value (max
   possible is ~90, the date window's size) — one LightGBM split (`days_since > N`)
   perfectly separates "never" from "any real value" with zero overlap, so
   `has_category_history` carries no information `days_since_category_purchase`
   doesn't already encode on its own. The lesson generalizes: an explicit binary flag
   for a "special case" only earns its keep when the special case would otherwise be
   genuinely confusable with real values along that feature's own ordering — an
   extreme, cleanly-separable sentinel makes the flag redundant for a tree model,
   which can always carve out an exact threshold on its own. `days_since_category_purchase`
   itself *is* meaningfully used (288 splits, more than `category_affinity`'s 224) —
   the recency signal made it in, just not via the flag.

2. **Two-stage beats every single-stage model outright here (+16.6% Recall@10 over
   itemknn_hm, +32.5% over als_hm)** — the opposite of MovieLens, where two-stage
   landed ~83-84% of its own base ALS model's accuracy, capped by a 79.17% ANN
   recall ceiling. The 91.06% ANN candidate recall here is a real part of the answer
   (a much less lossy retrieval stage leaves the reranker far less room to lose
   accuracy to approximation), but it's not the whole story — reconciling it with the
   64.8% injection rate is worth being precise about, since the two numbers look like
   they contradict each other:
   - **Injection rate (64.8% missed)** asks: is each val user's *true next purchase*
     inside als_hm's top-200 ANN candidates?
   - **ANN candidate recall (91.06%)** asks: does the *approximate* FAISS index
     reproduce what als_hm's own *exact*, brute-force top-20 would have been?
   
   These can both be true at once because als_hm's own exact ranking usually isn't
   the true answer either — als_hm's own Recall@20 is only 0.1075 (its exact top-20
   contains the true item barely 11% of the time). So FAISS is doing its approximation
   job well (91% — it faithfully reproduces als_hm's *own* preferences), it's just
   that als_hm's preferences usually aren't correct in the first place — most of the
   64.8% injection rate reflects als_hm's own accuracy ceiling, not FAISS's
   approximation loss on top of it.
   
   Given that, two_stage_hm's retrieval-stage ceiling (candidates reachable via the
   200-item pool) is roughly bounded by that same ~35% "not-missed" figure from
   training — yet the full pipeline still reaches Recall@10=0.1117 across *all* users,
   noticeably higher than als_hm's own Recall@10 (0.0843) computed with **no**
   retrieval bottleneck at all (full 27,779-item catalog, exact scoring). The
   reranker more than makes up for losing ~65% of users to the retrieval cutoff by
   being dramatically better, among the users it *can* still reach, at placing the
   right item near the top of a ~200-item shortlist than als_hm's own dot-product
   ranking is at placing it near the top of the entire catalog. Concretely: it isn't
   working from als_hm's ranking alone — `itemknn_score` (H&M's best single-stage
   model) is its 3rd-most-used feature, so the reranker gets to blend als_hm's
   retrieval-stage candidates with itemknn's independently-strong signal plus category
   recency, something no single-stage model has access to. MovieLens's two-stage
   pipeline didn't get this same lift because MovieLens's ALS was already the
   strongest single-stage model there (unlike H&M, where itemknn wins) and its
   retrieval ceiling was tighter (79% vs. 91%) — less room for the reranker to
   recover, less complementary signal to blend in even when it could.

3. **`user_activity_log` is the single most-used feature (697 splits, ahead of even
   `ann_score`)** — the reranker leans on "how active is this user overall" more than
   on the retrieval stage's own confidence score, consistent with Phase 5b part 1's
   finding that H&M accuracy varies sharply (and counter-intuitively) by activity
   level.

---

## Phase 6 — Two-Tower Neural Retrieval, attempt 1: embedding collapse (MovieLens)

**Core idea**: a neural two-tower model (user/item ID embedding → small MLP per
tower, dim=64 to match ALS's own `factors=64`), scored by a plain dot product like
ALS/BPR, trained with an in-batch sampled softmax loss on Colab GPU (see
`notebooks/two_tower_colab.ipynb`), then evaluated locally through the exact same
`evaluate_model.py` every other model uses. The optional DL stretch from the plan —
does nonlinearity (what ALS structurally can't do) earn its complexity here?

**First real run — a genuine failure, worth documenting precisely rather than just
silently fixing and re-running**:

| metric (K=10) | two_tower (attempt 1) | popularity (floor) | als_alpha1 (best MF) |
|---|---|---|---|
| Recall@10 | 0.0208 | 0.0491 | 0.1044 |
| NDCG@10 | 0.0097 | 0.0247 | 0.0534 |
| Intra-list similarity@10 | **0.9997** | n/a | 0.4981 |

Two-tower landed **below plain popularity** — worse than the simplest possible
baseline — and intra-list similarity near 1.0 (every recommended item for a given
user is nearly identical to every other one in that same list) was the tell.

**Diagnosis** (confirmed directly by inspecting the exported embeddings, not just
inferred from the accuracy number): mean pairwise cosine similarity among 500 random
item embeddings was **0.997** (near-parallel — should be close to 0 for embeddings
that actually encode distinct item content), while item embedding *norms* varied
meaningfully (mean 35.2, std 1.1). This is **embedding/representation collapse**, a
known failure mode of dot-product-scored contrastive training: with no normalization,
the network can lower the in-batch softmax loss just by learning per-item
*magnitude* (an easy, nearly content-free shortcut — in the degenerate limit, every
item points in the same direction $d$, so $u\cdot y_i \approx \|y_i\|(u\cdot d)$,
meaning the ranking of items becomes almost the same for every user, just rescaled by
a user-specific constant) instead of learning real *directional* structure that would
actually differentiate items for different users. Five epochs wasn't enough signal to
escape that cheap local solution.

**Fix applied for attempt 2**: L2-normalize both towers' outputs before scoring
(removing magnitude as a free parameter entirely, so direction must carry all the
signal), divide by a temperature (0.05), raise training to 20 epochs, export the
embeddings already normalized.

**Attempt 2 vs. every classical model, K=10, validation**:
| model | Recall@10 | NDCG@10 | Coverage@10 | ILS@10 |
|---|---|---|---|---|
| two_tower (attempt 1, collapsed) | 0.0208 | 0.0097 | 0.1267 | 0.9997 |
| **two_tower (attempt 2, fixed)** | **0.0385** | **0.0180** | **0.5995** | **0.8796** |
| popularity (floor) | 0.0491 | 0.0247 | 0.0078 | n/a |
| itemknn | 0.0797 | 0.0414 | 0.0961 | n/a |
| bpr | 0.0812 | 0.0403 | 0.5297 | 0.5333 |
| two_stage | 0.0875 | 0.0446 | 0.1639 | 0.3875 |
| als_alpha1 (best overall) | 0.1044 | 0.0534 | 0.0556 | 0.3248 |

**Interpretation**: the fix was real and worked as diagnosed — accuracy nearly
doubled and intra-list similarity dropped by more than half, confirming the
collapse hypothesis rather than just papering over the symptom. But the model
**still underperforms every classical baseline, including plain popularity** — the
simplest possible non-personalized model beats a GPU-trained neural network here by
28%. Coverage (0.60, higher than every other model) paired with still-elevated ILS
(0.88, worse than ALS's 0.32 or two-stage's 0.39) is consistent with a model that has
stopped collapsing but still hasn't learned confident, well-separated item
representations — it's spreading recommendations widely more because its rankings
are still noisy than because it's found genuinely diverse good matches.

**Attempt 3**: three further, specifically-targeted fixes — bigger batches (4096→16384,
more/harder in-batch negatives per step), a **logQ correction** (subtracting each
item's training-set log-frequency from its column's logits, so a popular item stops
being penalized just for showing up often as a negative — Yi et al. 2019's standard
fix for exactly this bias), and more epochs (20→40) to compensate for fewer gradient
steps per epoch at the larger batch size.

**Attempt 3 vs. every classical model, K=10, validation**:
| model | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 | ILS@10 |
|---|---|---|---|---|---|
| popularity (floor) | 0.0491 | 0.0247 | 0.0078 | 8.23 | n/a |
| itemknn | 0.0797 | 0.0414 | 0.0961 | 8.98 | n/a |
| **two_tower (attempt 3)** | **0.0781** | **0.0382** | **0.1271** | **9.48** | **0.8530** |
| bpr | 0.0812 | 0.0403 | 0.5297 | 12.36 | 0.5333 |
| two_stage | 0.0875 | 0.0446 | 0.1639 | 10.60 | 0.3875 |
| als_alpha1 (best overall) | 0.1044 | 0.0534 | 0.0556 | 9.62 | 0.3248 |

**Interpretation**: attempt 3 roughly doubled recall again (0.0385→0.0781) and is now
**directly competitive with itemknn and BPR** (within 2-4%, essentially tied given
measurement noise) — a dramatic turnaround from "below plain popularity." Two other
numbers moved in a way that confirms *why*, not just *that*, it improved: novelty
dropped sharply (14.47→9.48, now close to ALS's 9.62) and coverage dropped back down
to an ALS-like range (0.5995→0.1271) — both consistent with a model that stopped
hedging across a wide, low-confidence set of items (attempt 2's behavior) and started
making concentrated, confident, more accurate predictions instead, exactly what the
logQ correction and larger batches were meant to produce. `intra_list_similarity`
continued its downward trend (0.88→0.85) but remains well above ALS/two-stage's
0.32-0.39 — the model still hasn't learned as cleanly separated an embedding space as
the classical methods, even though its ranking accuracy has caught up.

**Where this leaves Phase 6's question**: after three attempts and two real,
correctly-diagnosed bugs fixed, the two-tower model reached rough parity with
itemknn/BPR but still falls short of `two_stage` and `als_alpha1` (the project's best
model throughout). ALS gets to 0.1044 via 15 rounds of **exact** alternating solves in
minutes on a laptop CPU, with one hyperparameter fully ablated; matching even its
*weaker* competitors with a neural model took a GPU, a full training pipeline, and
real debugging across embedding collapse and sampling bias. That's a legitimate,
informative answer to "did DL earn its complexity here" even without a fourth
attempt: not "DL can't work," but "DL needed substantially more engineering investment
to reach parity with methods that were simpler, faster, and cheaper to get right the
first time" — and even at parity, it hasn't yet surpassed the best classical result.

---

## Phase B — External benchmark comparison (continued elsewhere)

Phases 0–6 above compare our models to each other. Phase B compares them to RecBole, the
framework most recommender-systems papers are built on, to find out whether these numbers
are any good by outside standards.

That work runs in a separate project (`../recsys_benchmarks/`) because RecBole cannot
share this project's Python 3.14 environment, and its write-ups live in
[`../recsys_benchmarks/LEARNINGS.md`](../recsys_benchmarks/LEARNINGS.md).

Headline findings so far, so this document isn't missing them:

- On ML-1M under identical full-catalog leave-one-out evaluation, **ItemKNN beats
  LightGCN and NeuMF and ties MultiVAE** — reproducing Dacrema et al.'s RecSys'19
  result in our own run, and echoing this project's Phase 6 finding that the two-tower
  model needed three attempts to reach parity with item-item CF.
- Head-to-head on identical data and splits, **our ALS (α=1) tops the board on ML-1M**
  (NDCG@10 0.0416) ahead of MultiVAE, RecBole's ItemKNN, LightGCN and NeuMF — but **not
  at ML-25M**, where RecBole's BPR takes first (0.0441 vs our 0.0430). "Competitive with
  the best RecBole offers" holds at both scales; "wins" does not.
- The comparison surfaced **two real gaps in our own implementations**, each confirmed at
  two dataset sizes: our BPR trails RecBole's by 13% on ML-1M and 26% on ML-25M (the
  widening points at `iterations: 100` being a fixed budget regardless of dataset size,
  against RecBole's early stopping), and our ItemKNN trails at both scales (RecBole uses
  `k=100` with shrinkage; ours uses 200 neighbours with none). Phases 0–6 could only show
  which of our models beat our others; an outside implementation of the *same* algorithms
  showed two of them were leaving accuracy on the table.
- Measured against **published numbers** under the Neural Collaborative Filtering paper's
  exact setup (He et al., WWW'17 — same 1,000,209 interactions, 6,040 users, 3,706 items,
  100 sampled negatives), **our ALS reproduces that paper's own MF baselines** (HR@10
  0.6940 / NDCG@10 0.4119 against eALS's reported ~0.70 / ~0.43) and sits within 3% and 2%
  of its headline NeuMF result without pre-training. Its BPR and ItemKNN figures confirm
  the same two implementation gaps RecBole flagged, from an unrelated reference.
- **We found a real bug in RecBole's `Pop` baseline**, which does not accumulate
  duplicate indices and so counts batch-presence rather than interactions, saturating
  every popular item at the same value. It understates popularity by ~20% on ML-1M, by
  an amount that varies with batch size and epoch count. Popularity is the floor every
  other model is measured against, so every paper using this baseline reports an inflated
  margin over it. Caught only because we had an independent implementation of the same
  trivial algorithm to disagree with.
- **Evaluation protocol dominates model choice**, measured on our own models: switching
  from full-catalog ranking to 100 sampled negatives moves Recall@10 by **7–10×** on
  identical data — far more than any modelling decision in Phases 0–6 moved anything. Our
  sampled-protocol numbers land on top of published ones (itemknn 0.5734 / bpr 0.5948 vs
  ~0.52 / ~0.54 reported), so the apparent gap to the literature was never a quality gap.
- **The protocol also reorders models**, not just inflates them: `als_alpha1` beats `als`
  under full-catalog ranking and loses to it under sampled negatives. A sampled-negative
  leaderboard can't be trusted even for relative conclusions.
- Our ML-1M model ordering is **identical to ML-25M's** despite a 25× size difference,
  and the alpha=1-beats-alpha=20 result holds, consistent with Phase 5b's density story.
- Comparing on the validation fold would have inverted the ranking, because RecBole
  early-stops on that fold. Our models use fixed iterations and carry no such bias.
