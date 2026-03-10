from __future__ import annotations

from thesis_platform.algorithms.aggregators.summarization_core import summarize_critiques


class SummarizationAggregator:
    def __init__(self, config, repo_root):
        del repo_root
        self.max_rules = int(config.get("max_rules", 5))

    def aggregate(self, client_critiques, server_ctx):
        return summarize_critiques(
            client_critiques,
            round_id=len(server_ctx.prompt_history) - 1,
            mode="summarization",
            max_rules=self.max_rules,
        )
