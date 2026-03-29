from __future__ import annotations

import logging
from typing import Optional

from thesis_platform.algorithms.math_utils import cosine_similarity

logger = logging.getLogger(__name__)

# Try to import FAISS for accelerated KNN
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. KNN will use brute-force search.")


class FAISSIndex:
    """FAISS-backed KNN index for fast similarity search."""

    def __init__(self, dimension: int, metric: str = "cosine"):
        """Initialize FAISS index.

        Args:
            dimension: Vector dimension
            metric: "cosine" or "euclidean"
        """
        self.dimension = dimension
        self.metric = metric
        self.index: Optional[faiss.Index] = None
        self._id_to_idx: dict[int, int] = {}
        self._idx_to_id: dict[int, int] = {}

    def build(self, vectors: list[list[float]], ids: list[int]) -> None:
        """Build the FAISS index from vectors.

        Args:
            vectors: List of vectors
            ids: Unique IDs corresponding to each vector
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not available")

        import numpy as np

        # Convert to numpy array
        xb = np.array(vectors, dtype='float32')
        d = xb.shape[1]

        if self.metric == "cosine":
            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(xb, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            xb = xb / norms
            # Use inner product index (after normalization, ip = cosine similarity)
            self.index = faiss.IndexFlatIP(d)
        elif self.metric == "euclidean":
            self.index = faiss.IndexFlatL2(d)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        self.index.add(xb)

        # Build id mappings
        self._id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}
        self._idx_to_id = {idx: id_ for idx, id_ in enumerate(ids)}

    def search(self, query_vector: list[float], top_k: int) -> list[int]:
        """Search for top-k most similar vectors.

        Args:
            query_vector: Query vector
            top_k: Number of results to return

        Returns:
            List of original IDs of top-k most similar vectors
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        import numpy as np

        xq = np.array([query_vector], dtype='float32')

        if self.metric == "cosine":
            norm = np.linalg.norm(xq)
            if norm > 0:
                xq = xq / norm

        _, I = self.index.search(xq, top_k)

        # Convert indices back to original IDs
        return [self._idx_to_id[idx] for idx in I[0] if idx < len(self._idx_to_id)]


def cosine_top_k(query_vector: list[float], corpus_vectors: list[list[float]], top_k: int) -> list[int]:
    """Return the indices of the most similar corpus vectors.

    Uses FAISS if available for better performance on large corpora.
    Falls back to brute-force cosine similarity otherwise.
    """

    if len(corpus_vectors) == 0:
        return []

    # Use FAISS if available and corpus is large enough
    if FAISS_AVAILABLE and len(corpus_vectors) > 100:
        try:
            dimension = len(corpus_vectors[0])
            faiss_index = FAISSIndex(dimension=dimension, metric="cosine")
            ids = list(range(len(corpus_vectors)))
            faiss_index.build(corpus_vectors, ids)
            result = faiss_index.search(query_vector, top_k)
            return result[:top_k]
        except Exception as e:
            logger.warning(f"FAISS search failed, falling back to brute force: {e}")

    # Fallback to brute-force
    similarities = [(idx, cosine_similarity(vector, query_vector)) for idx, vector in enumerate(corpus_vectors)]
    similarities.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in similarities[:top_k]]


def euclidean_top_k(query_vector: list[float], corpus_vectors: list[list[float]], top_k: int) -> list[int]:
    """Return the indices of the closest corpus vectors using Euclidean distance.

    Uses FAISS if available for better performance on large corpora.
    """

    if len(corpus_vectors) == 0:
        return []

    if FAISS_AVAILABLE and len(corpus_vectors) > 100:
        try:
            dimension = len(corpus_vectors[0])
            faiss_index = FAISSIndex(dimension=dimension, metric="euclidean")
            ids = list(range(len(corpus_vectors)))
            faiss_index.build(corpus_vectors, ids)
            result = faiss_index.search(query_vector, top_k)
            return result[:top_k]
        except Exception as e:
            logger.warning(f"FAISS search failed, falling back to brute force: {e}")

    # Fallback to brute-force
    from thesis_platform.algorithms.math_utils import l2_norm, subtract

    distances = [(idx, l2_norm(subtract(vector, query_vector))) for idx, vector in enumerate(corpus_vectors)]
    distances.sort(key=lambda item: item[1])
    return [idx for idx, _ in distances[:top_k]]
