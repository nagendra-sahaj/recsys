import numpy as np
import pytest

from recsys_large_scale.evaluation import metrics


@pytest.fixture
def toy_hits():
    """4 users, K=3, n_items=5. Hand-worked-out cases:
    A: ground truth at topk[0]      -> hit at rank 1
    B: ground truth at topk[2]      -> hit at rank 3
    C: ground truth not in topk[:3] -> miss
    D: ground truth at topk[0]      -> hit at rank 1
    """
    topk = np.array(
        [
            [0, 3, 4],  # A, ground truth = 0
            [1, 4, 2],  # B, ground truth = 2
            [0, 1, 3],  # C, ground truth = 4 (not present)
            [1, 2, 3],  # D, ground truth = 1
        ]
    )
    ground_truth = np.array([0, 2, 4, 1])
    return metrics.hits_matrix(topk, ground_truth), topk


def test_hits_matrix(toy_hits):
    hits, _ = toy_hits
    expected = np.array(
        [
            [True, False, False],
            [False, False, True],
            [False, False, False],
            [True, False, False],
        ]
    )
    np.testing.assert_array_equal(hits, expected)


def test_recall_at_k(toy_hits):
    hits, _ = toy_hits
    # A, B, D hit; C misses -> 3/4
    assert metrics.recall_at_k(hits) == pytest.approx(0.75)


def test_precision_at_k(toy_hits):
    hits, _ = toy_hits
    # each hit contributes 1/3, misses contribute 0: (1/3 + 1/3 + 0 + 1/3) / 4
    assert metrics.precision_at_k(hits) == pytest.approx((1 / 3 * 3) / 4)


def test_ndcg_at_k(toy_hits):
    hits, _ = toy_hits
    # rank-1 hits (A, D) -> 1/log2(2) = 1.0 each; rank-3 hit (B) -> 1/log2(4) = 0.5; C -> 0
    expected = (1.0 + 0.5 + 0.0 + 1.0) / 4
    assert metrics.ndcg_at_k(hits) == pytest.approx(expected)


def test_mrr(toy_hits):
    hits, _ = toy_hits
    # A, D -> 1/1; B -> 1/3; C -> 0
    expected = (1.0 + 1 / 3 + 0.0 + 1.0) / 4
    assert metrics.mrr(hits) == pytest.approx(expected)


def test_map_at_k_equals_mrr_for_single_relevant_item(toy_hits):
    hits, _ = toy_hits
    assert metrics.map_at_k(hits) == pytest.approx(metrics.mrr(hits))


def test_coverage_at_k():
    mask = np.array([True, True, False, True, False, False])
    assert metrics.coverage_at_k(mask) == pytest.approx(3 / 6)


def test_coverage_at_k_full_catalog_recommended():
    mask = np.ones(5, dtype=bool)
    assert metrics.coverage_at_k(mask) == pytest.approx(1.0)


def test_novelty_at_k():
    # item 0 has popularity 0.5 (self-info 1 bit), items 1 and 2 have 0.25 (2 bits each)
    item_pop_prob = np.array([0.5, 0.25, 0.25])
    topk = np.array([[0, 1], [0, 2]])
    # self-infos: [1, 2, 1, 2] -> mean 1.5
    assert metrics.novelty_at_k(topk, item_pop_prob) == pytest.approx(1.5)


def test_novelty_at_k_prefers_less_popular_items():
    item_pop_prob = np.array([0.9, 0.05, 0.05])
    popular_topk = np.array([[0, 0]])
    unpopular_topk = np.array([[1, 2]])
    assert metrics.novelty_at_k(unpopular_topk, item_pop_prob) > metrics.novelty_at_k(
        popular_topk, item_pop_prob
    )


def test_top_popular_item_mask_marks_exactly_the_top_fraction():
    # 10 items, popularity strictly decreasing by index -> top 20% is items 0, 1
    item_pop_prob = np.linspace(1.0, 0.1, 10)
    mask = metrics.top_popular_item_mask(item_pop_prob, top_fraction=0.2)

    assert mask.sum() == 2
    assert mask[0] and mask[1]
    assert not mask[2:].any()


def test_top_popular_item_mask_always_marks_at_least_one_item():
    item_pop_prob = np.array([0.5, 0.3, 0.2])
    mask = metrics.top_popular_item_mask(item_pop_prob, top_fraction=0.01)
    assert mask.sum() == 1
    assert mask[0]  # the single most popular item


def test_popularity_bias_at_k():
    popular_mask = np.array([True, True, False, False, False])
    topk = np.array(
        [
            [0, 1, 2],  # 2/3 popular
            [2, 3, 4],  # 0/3 popular
        ]
    )
    assert metrics.popularity_bias_at_k(topk, popular_mask) == pytest.approx(1 / 3)


def test_intra_list_similarity_at_k_mixed_list():
    # one user, K=3: items 0 and 2 are identical unit vectors (similarity 1), item 1 is
    # orthogonal to both (similarity 0). Off-diagonal sum = 2*(1 + 0 + 0) = 2, over 6 pairs.
    vectors = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
    assert metrics.intra_list_similarity_at_k(vectors) == pytest.approx(2 / 6)


def test_intra_list_similarity_at_k_identical_items_is_maximally_similar():
    vectors = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    assert metrics.intra_list_similarity_at_k(vectors) == pytest.approx(1.0)


def test_intra_list_similarity_at_k_orthogonal_items_is_zero():
    vectors = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    assert metrics.intra_list_similarity_at_k(vectors) == pytest.approx(0.0)


def test_intra_list_similarity_at_k_rejects_k_less_than_2():
    vectors = np.array([[[1.0, 0.0]]])
    with pytest.raises(ValueError, match="K >= 2"):
        metrics.intra_list_similarity_at_k(vectors)


def test_ann_candidate_recall_partial_and_full_match():
    exact_topk = np.array([[1, 2, 3], [4, 5, 6]])
    candidates = np.array([[3, 1, 9, 10], [4, 5, 6, 7]])
    # user 0: 2 of exact top-3 (1, 3) are present in candidates; user 1: all 3 are
    expected = (2 / 3 + 3 / 3) / 2
    assert metrics.ann_candidate_recall(exact_topk, candidates) == pytest.approx(expected)


def test_ann_candidate_recall_no_overlap_is_zero():
    exact_topk = np.array([[1, 2, 3]])
    candidates = np.array([[4, 5, 6, 7]])
    assert metrics.ann_candidate_recall(exact_topk, candidates) == pytest.approx(0.0)


def test_ann_candidate_recall_full_containment_is_one():
    exact_topk = np.array([[1, 2, 3]])
    candidates = np.array([[3, 2, 1, 9]])
    assert metrics.ann_candidate_recall(exact_topk, candidates) == pytest.approx(1.0)
