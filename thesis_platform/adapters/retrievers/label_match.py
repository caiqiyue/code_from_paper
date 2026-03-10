from __future__ import annotations

from thesis_platform.core.schemas import PairedSample


class LabelMatchRetriever:
    """Simple label-based retriever for labeled classification-style data."""

    def __init__(self, config, repo_root):
        """Keep a uniform adapter constructor signature."""

        del config, repo_root

    def retrieve(self, bad_samples, client_ctx):
        """Retrieve samples sharing the same label as each bad sample."""

        pairs: list[PairedSample] = []
        for idx, bad_sample in enumerate(bad_samples):
            matches = [sample for sample in client_ctx.train_samples if sample.label == bad_sample.label and sample.label is not None]
            pairs.append(
                PairedSample(
                    pair_id=f"{client_ctx.client_id}_label_pair_{idx}",
                    client_id=client_ctx.client_id,
                    round_id=bad_sample.round_id,
                    bad_sample=bad_sample,
                    real_samples=matches[:3],
                )
            )
        return pairs
