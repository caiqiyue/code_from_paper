from __future__ import annotations


class NoCritic:
    def __init__(self, config, repo_root):
        del config, repo_root

    def critique(self, paired_samples, client_ctx):
        del paired_samples, client_ctx
        return []
