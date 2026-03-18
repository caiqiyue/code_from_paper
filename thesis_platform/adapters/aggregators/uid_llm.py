from __future__ import annotations

from thesis_platform.algorithms.aggregators.dbscan_core import aggregate_uid_critiques


class UIDLLMAggregator:
    """Research-mode UID aggregator that summarizes rules with the server LLM."""

    def __init__(self, config, repo_root):
        """Store aggregation settings."""

        del repo_root
        self.max_rules = int(config.get("max_rules", 5))

    def aggregate(self, client_critiques, server_ctx):
        """Aggregate critique rules through the server-side LLM."""

        if server_ctx.text_backend is None:
            raise ValueError("uid_llm requires a server text backend.")
        return aggregate_uid_critiques(
            client_critiques,
            round_id=len(server_ctx.prompt_history) - 1,
            max_rules=self.max_rules,
            text_backend=server_ctx.text_backend,
            base_prompt=server_ctx.base_prompt,
        )
