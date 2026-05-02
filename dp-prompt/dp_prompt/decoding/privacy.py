from __future__ import annotations

from typing import Any


def build_privacy_controls_summary(privacy_cfg: dict[str, Any]) -> dict[str, Any]:
    logits_cfg = privacy_cfg.get("logits_clipping", {})
    return {
        "temperature": privacy_cfg.get("temperature", 1.0),
        "logits_clipping": {
            "enabled": bool(logits_cfg.get("enabled", False)),
            "lower_bound": logits_cfg.get("lower_bound"),
            "upper_bound": logits_cfg.get("upper_bound"),
        },
        "max_generated_tokens": privacy_cfg.get("max_generated_tokens", 128),
        "stop_sequences": list(privacy_cfg.get("stop_sequences", [])),
        "privacy_report_mode": privacy_cfg.get("report_privacy_summary", True),
        "paper_faithfulness_note": (
            "Round 1 reproduces the paper-style decoding controls and experiment loop "
            "with a local open-source backend."
        ),
    }
