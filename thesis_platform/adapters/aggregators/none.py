from __future__ import annotations


class NoAggregator:
    """No-op aggregator used when prompt updates are disabled."""

    def __init__(self, config, repo_root):
        """Keep a uniform adapter constructor signature."""

        del config, repo_root

    def aggregate(self, client_critiques, server_ctx):
        """Return no prompt update."""

        del client_critiques, server_ctx
        return None
