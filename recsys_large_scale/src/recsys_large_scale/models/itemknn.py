import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from recsys_large_scale.utils.logging import get_logger

logger = get_logger(__name__)


class ItemKNNRecommender:
    """Item-item collaborative filtering: cosine similarity between items based on
    shared users, truncated to each item's top-K nearest neighbors. A user's score for
    a candidate item is the sum of similarities to items the user has already
    interacted with -- the same mechanism as the sibling projects' notebooks (cosine
    similarity on a sparse user-item matrix), scaled to this catalog size with a
    neighbor cap so the similarity matrix stays sparse and scoring stays fast."""

    def __init__(self, top_k_neighbors: int = 200):
        self.top_k_neighbors = top_k_neighbors

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None:
        rows = train["user_idx"].to_numpy()
        cols = train["item_idx"].to_numpy()
        data = np.ones(len(train), dtype=np.float32)
        self.user_item = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        self.n_items = n_items

        item_user_norm = normalize(self.user_item.T.tocsr(), norm="l2", axis=1)
        logger.info(f"Computing item-item cosine similarity for {n_items:,} items...")
        sim = (item_user_norm @ item_user_norm.T).tocsr()
        sim.setdiag(0)  # drop self-similarity (always 1.0 after L2 normalization)
        sim.eliminate_zeros()

        logger.info(
            f"Truncating similarity matrix to top-{self.top_k_neighbors} neighbors/item "
            f"(before truncation: {sim.nnz:,} nonzeros)"
        )
        self.similarity = _truncate_top_k(sim, self.top_k_neighbors)
        logger.info(f"After truncation: {self.similarity.nnz:,} nonzeros")

    def score_batch(self, user_indices: np.ndarray) -> np.ndarray:
        batch = self.user_item[user_indices]
        scores = batch @ self.similarity
        return np.asarray(scores.todense(), dtype=np.float32)


def _truncate_top_k(sim: csr_matrix, k: int) -> csr_matrix:
    """Keep only the k largest-similarity neighbors per row (item). A plain Python
    loop over rows, but argpartition per row is O(row_nnz) so this stays tractable
    even at tens of thousands of items -- expect it to take on the order of a minute
    at MovieLens-25M scale, not the milliseconds the rest of fit() takes."""
    sim = sim.tocsr()
    data_parts, idx_parts, indptr = [], [], [0]
    for row in range(sim.shape[0]):
        start, end = sim.indptr[row], sim.indptr[row + 1]
        row_data = sim.data[start:end]
        row_idx = sim.indices[start:end]
        if len(row_data) > k:
            top_pos = np.argpartition(-row_data, k)[:k]
            row_data = row_data[top_pos]
            row_idx = row_idx[top_pos]
        data_parts.append(row_data)
        idx_parts.append(row_idx)
        indptr.append(indptr[-1] + len(row_data))

    data = np.concatenate(data_parts) if data_parts else np.array([], dtype=sim.dtype)
    indices = np.concatenate(idx_parts) if idx_parts else np.array([], dtype=np.int32)
    return csr_matrix((data, indices, indptr), shape=sim.shape)
