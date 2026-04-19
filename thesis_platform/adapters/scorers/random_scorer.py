from __future__ import annotations

import random

from thesis_platform.core.schemas import ScoredSample


class RandomScorer:
    """Deterministic random baseline for Stage A bad-sample selection."""

    def __init__(self, config, repo_root):
        del repo_root
        self.seed = int(config.get("seed", config.get("random_fallback_seed", 42)))
        self.score_direction = str(config.get("score_direction", "larger_is_worse"))

    def score(self, samples, client_ctx):
        del client_ctx
        rng = random.Random(self.seed)
        return [
            ScoredSample.from_sample(
                sample,
                client_id=sample.client_id,
                score=rng.random(),
                score_name="random",
                score_direction=self.score_direction,
                meta={"baseline": "random"},
            )
            for sample in samples
        ]
