from __future__ import annotations

import random

from thesis_platform.core.schemas import PairedSample


class RandomRetriever:
    """Retriever that uses random selection to fetch anchor samples for ablation experiments."""

    def __init__(self, config, repo_root):
        """Initialize random retriever with optional seed and top_k."""

        del repo_root
        self.top_k = int(config.get("top_k", 3))
        self.seed = int(config.get("seed", 42))

    def retrieve(self, bad_samples, client_ctx):
        """Retrieve random local samples for each bad sample."""

        corpus = client_ctx.train_samples or client_ctx.all_samples
        pairs: list[PairedSample] = []
        if not bad_samples:
            return pairs

        # ClientContext is per-client state; the round is carried by samples.
        round_id = int(bad_samples[0].round_id or 0)
        rng = random.Random(self.seed + round_id)

        for idx, bad_sample in enumerate(bad_samples):
            if corpus:
                # Randomly sample without replacement if corpus is large enough
                num_to_sample = min(self.top_k, len(corpus))
                real_samples = rng.sample(corpus, num_to_sample)
            else:
                real_samples = []
            pairs.append(
                PairedSample(
                    pair_id=f"{client_ctx.client_id}_random_pair_{idx}",
                    client_id=client_ctx.client_id,
                    round_id=bad_sample.round_id,
                    bad_sample=bad_sample,
                    real_samples=real_samples,
                )
            )
        return pairs
