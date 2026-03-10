from __future__ import annotations


class IRAScorer:
    def __init__(self, config, repo_root):
        del config, repo_root

    def score(self, samples, client_ctx):
        del samples, client_ctx
        raise RuntimeError("ira is registered but not enabled in the MVP.")
