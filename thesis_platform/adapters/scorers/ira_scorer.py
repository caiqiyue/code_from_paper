from __future__ import annotations


class IRAScorer:
    """Reserved scorer adapter for the future IRA implementation."""

    def __init__(self, config, repo_root):
        """Accept config now so the registry shape stays stable."""

        del config, repo_root

    def score(self, samples, client_ctx):
        """Fail explicitly because IRA is outside the MVP execution path."""

        del samples, client_ctx
        raise RuntimeError("ira is registered but not enabled in the MVP.")
