from __future__ import annotations

from thesis_platform.algorithms.aggregators.summarization_core import summarize_critiques


class SummarizationAggregator:
    """Frequency-based critique summarizer used as a simple server aggregator."""

    def __init__(self, config, repo_root):
        """Store the maximum number of rules to keep after aggregation."""

        del repo_root
        self.max_rules = int(config.get("max_rules", 5))

    def aggregate(self, client_critiques, server_ctx):
        """Aggregate critique rules with the summarization strategy."""

        return summarize_critiques(
            client_critiques,
            round_id=len(server_ctx.prompt_history) - 1,
            mode="summarization",
            max_rules=self.max_rules,
        )
