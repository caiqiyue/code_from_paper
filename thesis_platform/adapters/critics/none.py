from __future__ import annotations


class NoCritic:
    """No-op critic used when the pipeline should stop before critique generation."""

    def __init__(self, config, repo_root):
        """Keep a uniform adapter constructor signature."""

        del config, repo_root

    def critique(self, paired_samples, client_ctx):
        """Return no critique items."""

        del paired_samples, client_ctx
        return []
