import numpy as np
import polars as pl

from recsys_large_scale.features import engineer


def test_build_item_genre_matrix_basic():
    movies = pl.DataFrame(
        {"item_id": [10, 20, 30], "genres": ["Action|Comedy", "Comedy", "(no genres listed)"]}
    )
    item_id_to_idx = {10: 0, 20: 1, 30: 2}
    matrix, genre_names = engineer.build_item_genre_matrix(movies, item_id_to_idx, n_items=3)

    assert genre_names == ["Action", "Comedy"]
    np.testing.assert_array_equal(matrix, np.array([[1, 1], [0, 1], [0, 0]], dtype=np.float32))


def test_build_item_genre_matrix_excludes_unmapped_items_from_vocabulary():
    # item_id 99 has no id_maps entry (e.g. filtered out during Phase 0 cleaning) --
    # its genre ("Comedy") should not even show up as a column, since nothing mapped
    # will ever set it
    movies = pl.DataFrame({"item_id": [10, 99], "genres": ["Action", "Comedy"]})
    item_id_to_idx = {10: 0}
    matrix, genre_names = engineer.build_item_genre_matrix(movies, item_id_to_idx, n_items=1)

    assert genre_names == ["Action"]
    assert matrix.shape == (1, 1)
    np.testing.assert_array_equal(matrix, np.array([[1.0]], dtype=np.float32))


def test_build_user_genre_affinity_accumulates_across_interactions():
    # item_genre_matrix: item 0 = Action, item 1 = Comedy, item 2 = Action+Comedy
    item_genre_matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    train = pl.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "item_idx": [0, 2, 1],  # user 0 watched items 0 and 2; user 1 watched item 1
        }
    )
    affinity = engineer.build_user_genre_affinity(train, item_genre_matrix, n_users=2)

    # user 0: Action (item0) + Action+Comedy (item2) -> Action=2, Comedy=1
    np.testing.assert_array_equal(affinity[0], np.array([2.0, 1.0]))
    # user 1: Comedy (item1) only -> Action=0, Comedy=1
    np.testing.assert_array_equal(affinity[1], np.array([0.0, 1.0]))


def test_build_candidate_features_shape_and_values():
    item_genre_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    user_genre_affinity = np.array([[3.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    item_pop_counts = np.array([10.0, 90.0])
    user_activity_counts = np.array([4.0, 24.0])

    user_idx = np.array([0, 1])
    item_idx = np.array([0, 1])
    ann_score = np.array([0.5, 0.9])
    itemknn_score = np.array([0.1, 0.2])
    is_injected = np.array([0.0, 1.0])

    X = engineer.build_candidate_features(
        user_idx, item_idx, ann_score, item_pop_counts, user_activity_counts, itemknn_score,
        user_genre_affinity, item_genre_matrix, is_injected,
    )

    assert X.shape == (2, len(engineer.FEATURE_NAMES))
    # row 0: ann_score, log1p(10), log1p(4), itemknn_score, genre_affinity=3*1+0*0=3, is_injected=0
    expected_row0 = [0.5, np.log1p(10.0), np.log1p(4.0), 0.1, 3.0, 0.0]
    np.testing.assert_allclose(X[0], expected_row0, rtol=1e-6)
    # row 1: genre_affinity = 0*0 + 5*1 = 5, is_injected=1
    expected_row1 = [0.9, np.log1p(90.0), np.log1p(24.0), 0.2, 5.0, 1.0]
    np.testing.assert_allclose(X[1], expected_row1, rtol=1e-6)


def test_build_item_category_matrix_one_category_per_item():
    articles = pl.DataFrame(
        {"item_id": [10, 20, 30], "product_group_name": ["Shoes", "Accessories", None]}
    )
    item_id_to_idx = {10: 0, 20: 1, 30: 2}
    matrix, item_category_idx, categories = engineer.build_item_category_matrix(
        articles, item_id_to_idx, n_items=3
    )

    assert categories == ["Accessories", "Shoes"]
    np.testing.assert_array_equal(matrix, np.array([[0, 1], [1, 0], [0, 0]], dtype=np.float32))
    np.testing.assert_array_equal(item_category_idx, [1, 0, -1])  # item 2 has a null category


def test_build_user_category_last_purchase_takes_the_max_timestamp_per_category():
    # item 0 -> category 0, item 1 -> category 1, item 2 has no category (-1)
    item_category_idx = np.array([0, 1, -1])
    train = pl.DataFrame(
        {
            "user_idx": [0, 0, 0, 1],
            "item_idx": [0, 0, 1, 2],
            "timestamp": [100, 300, 200, 999],  # user 0 bought category 0 at t=100 and t=300
        }
    )
    last_ts = engineer.build_user_category_last_purchase(train, item_category_idx, n_users=2, n_categories=2)

    assert last_ts[0, 0] == 300  # most recent of the two category-0 purchases
    assert last_ts[0, 1] == 200
    assert last_ts[1, 0] == -1  # user 1 never bought category 0
    assert last_ts[1, 1] == -1  # user 1's only purchase (item 2) has no category, excluded


def test_build_category_recency_features_never_purchased_sentinel():
    item_category_idx = np.array([0, 1])  # item 0 -> category 0, item 1 -> category 1
    user_category_last_ts = np.array([[500, -1]])  # user 0: bought category 0 at t=500, never category 1

    days_since, has_history = engineer.build_category_recency_features(
        user_idx=np.array([0, 0]),
        item_idx=np.array([0, 1]),
        item_category_idx=item_category_idx,
        user_category_last_ts=user_category_last_ts,
        reference_timestamp=500 + 3 * engineer.SECONDS_PER_DAY,
    )

    np.testing.assert_allclose(days_since, [3.0, engineer.NEVER_PURCHASED_DAYS])
    np.testing.assert_array_equal(has_history, [1.0, 0.0])


def test_build_candidate_features_hm_shape_and_values():
    item_category_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    user_category_affinity = np.array([[2.0, 0.0]], dtype=np.float32)
    item_pop_counts = np.array([10.0, 90.0])
    user_activity_counts = np.array([4.0])

    X = engineer.build_candidate_features_hm(
        user_idx=np.array([0, 0]),
        item_idx=np.array([0, 1]),
        ann_score=np.array([0.5, 0.9]),
        item_pop_counts=item_pop_counts,
        user_activity_counts=user_activity_counts,
        itemknn_score=np.array([0.1, 0.2]),
        user_category_affinity=user_category_affinity,
        item_category_matrix=item_category_matrix,
        is_injected=np.array([0.0, 1.0]),
        days_since_category_purchase=np.array([3.0, engineer.NEVER_PURCHASED_DAYS]),
        has_category_history=np.array([1.0, 0.0]),
    )

    assert X.shape == (2, len(engineer.FEATURE_NAMES_HM))
    expected_row0 = [0.5, np.log1p(10.0), np.log1p(4.0), 0.1, 2.0, 0.0, 3.0, 1.0]
    np.testing.assert_allclose(X[0], expected_row0, rtol=1e-6)
    expected_row1 = [0.9, np.log1p(90.0), np.log1p(4.0), 0.2, 0.0, 1.0, engineer.NEVER_PURCHASED_DAYS, 0.0]
    np.testing.assert_allclose(X[1], expected_row1, rtol=1e-6)
