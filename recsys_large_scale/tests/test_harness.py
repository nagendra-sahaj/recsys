import numpy as np
import polars as pl
import pytest

from recsys_large_scale.evaluation.harness import evaluate, parse_eval_mode


class FakeModel:
    """Deterministic stand-in for a real Recommender: scores every item by its own
    index, so item 5 always looks "best" and item 0 always looks "worst" regardless of
    which user is asked -- lets the test control exactly what the harness should mask
    out and what should surface in top-K."""

    def __init__(self, n_items: int):
        self.n_items = n_items

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        row = np.arange(self.n_items, dtype=np.float32)
        return np.tile(row, (len(user_indices), 1))


def test_evaluate_masks_seen_items_before_ranking():
    """User 0 has already interacted with items 5, 4, 3 -- the three highest-scoring
    items under FakeModel. If masking didn't work, top-2 would be [5, 4] and the
    held-out ground-truth item (2) would never be found -> recall would be 0. If
    masking works, top-2 becomes [2, 1] and recall is 1."""
    n_items = 6
    train = pl.DataFrame(
        {"user_idx": [0, 0, 0], "item_idx": [5, 4, 3], "timestamp": [1, 2, 3]}
    )
    eval_set = pl.DataFrame({"user_idx": [0], "item_idx": [2]})
    item_pop_prob = np.full(n_items, 1 / n_items)

    rows = evaluate(
        model=FakeModel(n_items),
        train=train,
        eval_set=eval_set,
        n_users=1,
        n_items=n_items,
        item_pop_prob=item_pop_prob,
        k_values=[2],
        batch_size=10,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.k == 2
    assert row.n_users == 1
    assert row.recall == pytest.approx(1.0)
    assert row.ndcg == pytest.approx(1.0)
    assert row.mrr == pytest.approx(1.0)
    assert row.map == pytest.approx(1.0)
    assert row.precision == pytest.approx(0.5)
    # top-2 recommended items are {2, 1} out of 6 -> coverage 2/6
    assert row.coverage == pytest.approx(2 / 6)


def test_evaluate_handles_multiple_k_values_in_one_pass():
    n_items = 6
    train = pl.DataFrame({"user_idx": [0], "item_idx": [5], "timestamp": [1]})
    eval_set = pl.DataFrame({"user_idx": [0], "item_idx": [3]})
    item_pop_prob = np.full(n_items, 1 / n_items)

    rows = evaluate(
        model=FakeModel(n_items),
        train=train,
        eval_set=eval_set,
        n_users=1,
        n_items=n_items,
        item_pop_prob=item_pop_prob,
        k_values=[1, 3],
        batch_size=10,
    )
    rows_by_k = {r.k: r for r in rows}

    # unmasked order is [5,4,3,2,1,0]; item 5 is masked (seen) -> order becomes [4,3,2,1,0]
    # ground truth 3 is at rank 2, so it's outside top-1 but inside top-3
    assert rows_by_k[1].recall == pytest.approx(0.0)
    assert rows_by_k[3].recall == pytest.approx(1.0)
    assert rows_by_k[3].mrr == pytest.approx(0.5)  # hit at rank 2


def test_evaluate_computes_intra_list_similarity_when_embeddings_given():
    n_items = 6
    train = pl.DataFrame({"user_idx": [0, 0, 0], "item_idx": [5, 4, 3], "timestamp": [1, 2, 3]})
    eval_set = pl.DataFrame({"user_idx": [0], "item_idx": [2]})
    item_pop_prob = np.full(n_items, 1 / n_items)

    # top-2 for user 0 is [2, 1] (see test_evaluate_masks_seen_items_before_ranking);
    # give those two items orthogonal embeddings so the expected similarity is exactly 0
    item_embeddings = np.zeros((n_items, 2), dtype=np.float32)
    item_embeddings[2] = [1.0, 0.0]
    item_embeddings[1] = [0.0, 1.0]

    rows = evaluate(
        model=FakeModel(n_items),
        train=train,
        eval_set=eval_set,
        n_users=1,
        n_items=n_items,
        item_pop_prob=item_pop_prob,
        k_values=[2],
        batch_size=10,
        item_embeddings=item_embeddings,
    )
    assert rows[0].intra_list_similarity == pytest.approx(0.0, abs=1e-6)


def test_evaluate_leaves_intra_list_similarity_none_without_embeddings():
    n_items = 6
    train = pl.DataFrame({"user_idx": [0], "item_idx": [5], "timestamp": [1]})
    eval_set = pl.DataFrame({"user_idx": [0], "item_idx": [3]})
    item_pop_prob = np.full(n_items, 1 / n_items)

    rows = evaluate(
        model=FakeModel(n_items),
        train=train,
        eval_set=eval_set,
        n_users=1,
        n_items=n_items,
        item_pop_prob=item_pop_prob,
        k_values=[2],
    )
    assert rows[0].intra_list_similarity is None


def test_parse_eval_mode():
    assert parse_eval_mode("full") is None
    assert parse_eval_mode("uni100") == 100
    assert parse_eval_mode("uni1") == 1
    for bad in ["uni", "uni0", "uni-5", "uniN", "pop100", "sampled", ""]:
        with pytest.raises(ValueError, match="unknown eval_mode"):
            parse_eval_mode(bad)


def test_evaluate_defaults_to_full_catalog_ranking():
    """The protocol switch must not disturb anything already measured: passing
    eval_mode='full' explicitly has to reproduce the default call exactly."""
    n_items = 6
    train = pl.DataFrame({"user_idx": [0, 0, 0], "item_idx": [5, 4, 3], "timestamp": [1, 2, 3]})
    eval_set = pl.DataFrame({"user_idx": [0], "item_idx": [2]})
    item_pop_prob = np.full(n_items, 1 / n_items)
    kwargs = dict(
        model=FakeModel(n_items),
        train=train,
        eval_set=eval_set,
        n_users=1,
        n_items=n_items,
        item_pop_prob=item_pop_prob,
        k_values=[2],
    )

    assert evaluate(**kwargs) == evaluate(**kwargs, eval_mode="full")


def test_evaluate_sampled_negatives_restricts_the_candidate_pool():
    """With a 500-item catalog and uni4, a user's held-out item competes against only
    ~4 random others instead of 499. FakeModel scores purely by item index, so under
    full ranking a low-indexed ground-truth item is hopeless, while under uni4 it wins
    whenever its handful of sampled rivals happen to be lower-indexed still. The point
    of the test is that the two protocols give different answers for an identical model
    and identical data -- which is the entire finding the protocol study is after."""
    n_items = 500
    n_users = 200
    # every user has one seen item and one held-out item, all in the low index range
    train = pl.DataFrame(
        {"user_idx": list(range(n_users)), "item_idx": [0] * n_users, "timestamp": [1] * n_users}
    )
    eval_set = pl.DataFrame({"user_idx": list(range(n_users)), "item_idx": [1] * n_users})
    item_pop_prob = np.full(n_items, 1 / n_items)
    kwargs = dict(
        model=FakeModel(n_items),
        train=train,
        eval_set=eval_set,
        n_users=n_users,
        n_items=n_items,
        item_pop_prob=item_pop_prob,
        k_values=[10],
    )

    full = evaluate(**kwargs)[0]
    sampled = evaluate(**kwargs, eval_mode="uni4")[0]

    # item 1 is the 2nd-worst item in a 500-item catalog: never in a full-ranking top-10
    assert full.recall == pytest.approx(0.0)
    # against only ~4 random rivals it reaches top-10 every time, since top-10 of a
    # 5-item candidate pool is the whole pool
    assert sampled.recall == pytest.approx(1.0)


def test_evaluate_sampled_negatives_are_reproducible():
    n_items = 300
    n_users = 50
    train = pl.DataFrame(
        {"user_idx": list(range(n_users)), "item_idx": [0] * n_users, "timestamp": [1] * n_users}
    )
    eval_set = pl.DataFrame({"user_idx": list(range(n_users)), "item_idx": [7] * n_users})
    item_pop_prob = np.full(n_items, 1 / n_items)
    kwargs = dict(
        model=FakeModel(n_items),
        train=train,
        eval_set=eval_set,
        n_users=n_users,
        n_items=n_items,
        item_pop_prob=item_pop_prob,
        k_values=[5],
        eval_mode="uni20",
    )

    assert evaluate(**kwargs, seed=42) == evaluate(**kwargs, seed=42)
    assert evaluate(**kwargs, seed=42) != evaluate(**kwargs, seed=7)


def test_evaluate_rejects_duplicate_users_in_eval_set():
    n_items = 4
    train = pl.DataFrame({"user_idx": [0], "item_idx": [0], "timestamp": [1]})
    eval_set = pl.DataFrame({"user_idx": [0, 0], "item_idx": [1, 2]})
    item_pop_prob = np.full(n_items, 1 / n_items)

    with pytest.raises(ValueError, match="duplicate user_idx"):
        evaluate(
            model=FakeModel(n_items),
            train=train,
            eval_set=eval_set,
            n_users=1,
            n_items=n_items,
            item_pop_prob=item_pop_prob,
            k_values=[2],
        )
