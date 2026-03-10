from __future__ import annotations

from thesis_platform.algorithms.critics.contrastive_critic_core import build_critique


class FedTextGradCritic:
    def __init__(self, config, repo_root):
        del repo_root
        self.max_rules = int(config.get("compress_to_n_rules", 2))
        self.redact_enable = bool(config.get("redact_enable", True))

    def critique(self, paired_samples, client_ctx):
        del client_ctx
        return [
            build_critique(pair, max_rules=self.max_rules, redact_enable=self.redact_enable)
            for pair in paired_samples
        ]
