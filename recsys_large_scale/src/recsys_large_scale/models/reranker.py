import numpy as np
from lightgbm import LGBMRanker


class LightGBMReranker:
    """LambdaMART reranker over engineered candidate features: takes the N candidates
    the retrieval stage surfaced for a user and re-scores them using a gradient-
    boosted model trained on relevance labels (1 = the held-out ground-truth item was
    among that user's candidates, 0 = every other candidate) -- see
    scripts/run_two_stage_pipeline.py for how the training set is built."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
        num_threads: int = 1,
    ):
        # num_threads=1 is not a style choice -- lightgbm's multi-threaded fit()
        # reliably segfaults (confirmed via smoke test, exit code 139) in a process
        # that has also loaded faiss. Each of faiss/scikit-learn/lightgbm ships its own
        # separate physical copy of the OpenMP runtime on macOS (faiss and scikit-learn
        # bundle a private libomp.dylib inside their wheels; lightgbm links Homebrew's),
        # and lightgbm's own multi-threaded execution is what crashes once more than
        # one of those copies is loaded in the same process -- regardless of faiss's
        # own thread count. faiss itself is left at its default (multi-threaded)
        # setting since that combination is safe and faiss's own workload here (a
        # ~24K-item index) is cheap enough that single-threading lightgbm is the
        # right place to pay this cost, not faiss.
        self.model = LGBMRanker(
            objective="lambdarank",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            num_threads=num_threads,
            verbosity=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, group_sizes: np.ndarray) -> None:
        self.model.fit(X, y, group=group_sizes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
