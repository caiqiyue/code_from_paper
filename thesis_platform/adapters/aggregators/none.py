from __future__ import annotations


class NoAggregator:
    def __init__(self, config, repo_root):
        del config, repo_root

    def aggregate(self, client_critiques, server_ctx):
        del client_critiques, server_ctx
        return None
