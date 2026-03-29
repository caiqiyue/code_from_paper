from __future__ import annotations

from thesis_platform.algorithms.critics.fedtextgrad_core import build_textual_gradient_critique
from thesis_platform.models.backends import build_text_backend


class FedTextGradQwenCritic:
    """FedTextGrad critic that uses a dedicated Qwen model as text backend.

    This critic builds its own Qwen backend independent of the global
    client/server backends, allowing it to use Qwen for critique generation
    while other components use different models.
    """

    def __init__(self, config, repo_root):
        """Initialize with Qwen model configuration.

        Args:
            config: Critic configuration containing:
                - model_name_or_path: Path to Qwen model (default: thesis_platform/open_model/qwen_2_0_5b_instruct)
                - compress_to_n_rules: Max number of critique rules
                - redact_enable: Whether to redact sensitive info
            repo_root: Root directory of the repository
        """
        from pathlib import Path

        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.max_rules = int(config.get("compress_to_n_rules", 2))
        self.redact_enable = bool(config.get("redact_enable", True))

        # Build Qwen-specific backend
        model_path = config.get(
            "model_name_or_path",
            "thesis_platform/open_model/qwen_2_0_5b_instruct",
        )
        backend_cfg = {
            "engine": "transformers",
            "model_name_or_path": model_path,
            "device": "auto",
            "dtype": "auto",
            "temperature": 0.7,
            "max_new_tokens": 196,
            "use_chat_template": True,
        }
        self._text_backend = build_text_backend(
            {**backend_cfg, "role": "critic"},
            repo_root=str(self.repo_root),
        )

    def critique(self, paired_samples, client_ctx):
        """Generate critique rules using Qwen model for each paired sample."""

        del client_ctx  # Uses self._text_backend instead of client_ctx.text_backend
        return [
            build_textual_gradient_critique(
                pair,
                text_backend=self._text_backend,
                max_rules=self.max_rules,
                redact_enable=self.redact_enable,
            )
            for pair in paired_samples
        ]
