from __future__ import annotations

import re

NUMBER_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def redact_rule_text(text: str) -> str:
    """Apply a light-weight rule-level redaction pass."""

    text = NUMBER_RE.sub("<NUMBER>", text)
    text = ENTITY_RE.sub("<ENTITY>", text)
    return text
