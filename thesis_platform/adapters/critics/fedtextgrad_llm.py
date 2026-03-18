from __future__ import annotations

from thesis_platform.algorithms.critics.fedtextgrad_core import build_textual_gradient_critique


class FedTextGradLLMCritic:
    """Research-mode textual-gradient critic backed by a local small language model."""

    def __init__(self, config, repo_root):
        """Store critique compression and redaction settings."""

        del repo_root
        self.max_rules = int(config.get("compress_to_n_rules", 3))
        self.redact_enable = bool(config.get("redact_enable", True))

    def critique(self, paired_samples, client_ctx):
        """Generate critique rules for each retrieved bad/real pair."""

        if client_ctx.text_backend is None:
            raise ValueError("fedtextgrad_llm requires a client text backend.")
        return [
            build_textual_gradient_critique(
                pair,
                text_backend=client_ctx.text_backend,
                max_rules=self.max_rules,
                redact_enable=self.redact_enable,
            )
            for pair in paired_samples
        ]
