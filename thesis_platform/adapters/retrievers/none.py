from __future__ import annotations

from thesis_platform.core.schemas import PairedSample


class NoRetriever:
    """No-op retriever used by the pure scorer smoke configuration."""

    def __init__(self, config, repo_root):
        """Keep a uniform adapter constructor signature."""

        del config, repo_root

    def retrieve(self, bad_samples, client_ctx):
        """Wrap bad samples into empty pairs without retrieving anchors."""

        return [
            PairedSample(
                pair_id=f"{client_ctx.client_id}_noop_pair_{idx}",
                client_id=client_ctx.client_id,
                round_id=bad_sample.round_id,
                bad_sample=bad_sample,
                real_samples=[],
            )
            for idx, bad_sample in enumerate(bad_samples)
        ]
