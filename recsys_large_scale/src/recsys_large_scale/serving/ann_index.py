import faiss
import numpy as np

# See recsys_large_scale/__init__.py: faiss must load after lightgbm, and even then a
# multi-threaded FAISS can deadlock against lightgbm's thread pool. Forcing FAISS
# single-threaded is the combination verified stable -- a cheap tradeoff, since this
# project's item catalog (tens of thousands of items) is far too small for FAISS's own
# speed to ever be the bottleneck.
faiss.omp_set_num_threads(1)


class ANNIndex:
    """Wraps a FAISS index over item embeddings for (approximate) nearest-neighbor
    retrieval by inner product -- deliberately raw inner product, not cosine, so a
    search here matches the same `user_vec @ item_vec` scoring every model in this
    project uses (see models/als.py's score_batch).

    `nlist=None` builds an exact `IndexFlatIP` (brute-force, but done inside FAISS).
    `nlist=<int>` builds an approximate `IndexIVFFlat` with that many clusters --
    faster at large catalog sizes, at the cost of no longer being guaranteed to find
    the true top-N (see evaluation.metrics.ann_candidate_recall for quantifying that
    cost, and LEARNINGS.md for the actual measured number on this dataset).
    """

    def __init__(self, item_embeddings: np.ndarray, nlist: int | None = None, nprobe: int = 10):
        embeddings = np.ascontiguousarray(item_embeddings, dtype=np.float32)
        self.d = embeddings.shape[1]
        self.nlist = nlist
        self.nprobe = nprobe

        if nlist is None:
            self.index = faiss.IndexFlatIP(self.d)
        else:
            quantizer = faiss.IndexFlatIP(self.d)
            self.index = faiss.IndexIVFFlat(quantizer, self.d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.nprobe = nprobe
        self.index.add(embeddings)

    def search(self, query_vectors: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (indices, scores), both (len(query_vectors), top_n)."""
        query = np.ascontiguousarray(query_vectors, dtype=np.float32)
        scores, indices = self.index.search(query, top_n)
        return indices, scores

    # A raw FAISS index (a SWIG-wrapped C++ object) isn't picklable by default;
    # serialize_index/deserialize_index round-trip it through bytes, which pickle
    # handles fine as part of this object's state -- lets ANNIndex (and anything that
    # holds one, e.g. TwoStageRecommender) be pickled the same plain way every other
    # model in this project is.
    def __getstate__(self) -> dict:
        return {"serialized": faiss.serialize_index(self.index), "d": self.d, "nlist": self.nlist, "nprobe": self.nprobe}

    def __setstate__(self, state: dict) -> None:
        self.index = faiss.deserialize_index(state["serialized"])
        self.d = state["d"]
        self.nlist = state["nlist"]
        self.nprobe = state["nprobe"]
