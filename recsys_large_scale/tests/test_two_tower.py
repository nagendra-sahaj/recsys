import numpy as np

from recsys_large_scale.models.two_tower import TwoTowerRecommender


def test_score_batch_is_dot_product_of_the_two_towers():
    user_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    item_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    model = TwoTowerRecommender(user_embeddings, item_embeddings)

    scores = model.score_batch(np.array([0, 1]))

    expected = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(scores, expected)


def test_item_embeddings_exposed_for_diversity_metric():
    user_embeddings = np.zeros((2, 2), dtype=np.float32)
    item_embeddings = np.eye(2, dtype=np.float32)
    model = TwoTowerRecommender(user_embeddings, item_embeddings)
    np.testing.assert_array_equal(model.item_embeddings, item_embeddings)
