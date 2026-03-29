from __future__ import annotations

import re

NUMBER_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}|[A-Z]{2,})\b")
SAFE_CAPITALIZED_WORDS = {
    "add",
    "adjust",
    "align",
    "avoid",
    "clarify",
    "content",
    "focus",
    "format",
    "improve",
    "include",
    "keep",
    "make",
    "output",
    "prompt",
    "reduce",
    "remove",
    "replace",
    "response",
    "retain",
    "rewrite",
    "sample",
    "structure",
    "text",
    "tone",
    "tighten",
    "use",
    "wording",
}


def _replace_entity(match: re.Match[str]) -> str:
    word = match.group(0)
    if word.lower() in SAFE_CAPITALIZED_WORDS:
        return word
    return "<ENTITY>"


def redact_rule_text(text: str) -> str:
    """Apply a light-weight rule-level redaction pass."""

    text = NUMBER_RE.sub("<NUMBER>", text)
    text = ENTITY_RE.sub(_replace_entity, text)
    return text
