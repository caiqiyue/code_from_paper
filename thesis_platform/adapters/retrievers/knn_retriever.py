from __future__ import annotations

from thesis_platform.algorithms.retrievers.knn_core import cosine_top_k
from thesis_platform.core.schemas import PairedSample


class KNNRetriever:
    """Retriever that uses embedding similarity to fetch real anchor samples."""

    def __init__(self, config, repo_root):
        """Store the top-k retrieval width."""

        del repo_root
        self.top_k = int(config.get("top_k", 3))

    def retrieve(self, bad_samples, client_ctx):
        """Retrieve the most similar local samples for each bad sample."""

        corpus = client_ctx.train_samples or client_ctx.all_samples
        if not bad_samples:
            return []
        corpus_vectors = client_ctx.embedder.embed_texts([sample.text for sample in corpus]) if corpus else []
        pairs: list[PairedSample] = []
        for idx, bad_sample in enumerate(bad_samples):
            if corpus:
                query_vector = client_ctx.embedder.embed_texts([bad_sample.text])[0]
                indices = cosine_top_k(query_vector, corpus_vectors, self.top_k)
                real_samples = [corpus[index] for index in indices]
            else:
                real_samples = []
            pairs.append(
                PairedSample(
                    pair_id=f"{client_ctx.client_id}_pair_{idx}",
                    client_id=client_ctx.client_id,
                    round_id=bad_sample.round_id,
                    bad_sample=bad_sample,
                    real_samples=real_samples,
                )
            )
        return pairs
