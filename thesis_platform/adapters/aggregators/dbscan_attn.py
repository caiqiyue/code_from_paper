from __future__ import annotations


class DBSCANAttnAggregator:
    """Reserved adapter for the future DBSCAN-Attn server aggregator."""

    def __init__(self, config, repo_root):
        """Accept config now so the registry and config layout stay stable."""

        del config, repo_root

    def aggregate(self, client_critiques, server_ctx):
        """Fail explicitly because DBSCAN-Attn is not part of the MVP."""

        del client_critiques, server_ctx
        raise RuntimeError("dbscan_attn is registered but not enabled in the MVP.")
