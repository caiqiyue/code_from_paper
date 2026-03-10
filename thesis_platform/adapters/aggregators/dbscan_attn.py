from __future__ import annotations


class DBSCANAttnAggregator:
    def __init__(self, config, repo_root):
        del config, repo_root

    def aggregate(self, client_critiques, server_ctx):
        del client_critiques, server_ctx
        raise RuntimeError("dbscan_attn is registered but not enabled in the MVP.")
