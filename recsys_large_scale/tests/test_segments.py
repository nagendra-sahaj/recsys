import numpy as np
import polars as pl

from recsys_large_scale.evaluation.segments import (
    bucket_ages,
    bucket_by_activity,
    cold_user_mask,
    map_customer_attribute,
)


def test_cold_user_mask_flags_users_absent_from_train():
    train = pl.DataFrame({"user_idx": [0, 0, 1], "item_idx": [10, 11, 12]})
    eval_users = np.array([0, 1, 2])  # user 2 never appears in train

    mask = cold_user_mask(eval_users, train)

    np.testing.assert_array_equal(mask, [False, False, True])


def test_bucket_by_activity_splits_into_terciles():
    # 9 users with counts 1..9 -> terciles at q1=3.67, q2=6.33: {1,2,3}=low, {4,5,6}=medium, {7,8,9}=high
    rows = []
    for user_idx, count in enumerate(range(1, 10)):
        rows.extend([(user_idx, item_idx) for item_idx in range(count)])
    train = pl.DataFrame(rows, schema=["user_idx", "item_idx"], orient="row")

    buckets = bucket_by_activity(train, n_users=9)

    assert list(buckets[:3]) == ["low", "low", "low"]
    assert list(buckets[3:6]) == ["medium", "medium", "medium"]
    assert list(buckets[6:9]) == ["high", "high", "high"]


def test_map_customer_attribute_aligns_by_user_id_and_defaults_to_none():
    customers = pl.DataFrame({"user_id": ["abc", "def"], "age": [30, 45]})
    user_id_to_idx = {"abc": 0, "def": 1}

    ages = map_customer_attribute(customers, user_id_to_idx, n_users=3, column="age")

    assert ages[0] == 30
    assert ages[1] == 45
    assert ages[2] is None  # user_idx 2 has no matching customers.csv row


def test_bucket_ages_labels_edges_and_none_as_unknown():
    ages = np.array([20, 25, 34, 60, None], dtype=object)

    labels = bucket_ages(ages, edges=(25, 35, 45, 55))

    assert list(labels) == ["<25", "25-34", "25-34", "55+", "unknown"]
