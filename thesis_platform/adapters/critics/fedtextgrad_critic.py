from __future__ import annotations

from thesis_platform.algorithms.critics.contrastive_critic_core import build_critique
from thesis_platform.algorithms.critics.fedtextgrad_core import build_textual_gradient_critique


class FedTextGradCritic:
    """Contrastive critic inspired by FedTextGrad textual feedback design.

    Supports two engines:
    - "heuristic" (default): rule-based contrastive analysis, no LLM needed
    - "model": uses the text_backend from client_ctx for LLM-based critique
    """

    def __init__(self, config, repo_root):
        """Store critique compression, redaction, and engine settings."""

        del repo_root
        self.max_rules = int(config.get("compress_to_n_rules", 2))
        self.redact_enable = bool(config.get("redact_enable", True))
        self.engine = str(config.get("engine", "heuristic"))

    def critique(self, paired_samples, client_ctx):
        """Generate critique rules for each retrieved bad/real pair."""

        if self.engine == "model" and client_ctx.text_backend is not None:
            return [
                build_textual_gradient_critique(
                    pair,
                    text_backend=client_ctx.text_backend,
                    max_rules=self.max_rules,
                    redact_enable=self.redact_enable,
                )
                for pair in paired_samples
            ]
        # Fallback to heuristic when engine=heuristic or no text_backend available
        return [
            build_critique(pair, max_rules=self.max_rules, redact_enable=self.redact_enable)
            for pair in paired_samples
        ]
