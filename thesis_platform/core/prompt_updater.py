from __future__ import annotations

from thesis_platform.core.schemas import PromptUpdate


RULES_HEADER = "### Aggregated Rules"


def apply_prompt_update(current_prompt: str, update: PromptUpdate) -> str:
    rules_block = RULES_HEADER + "\n" + "\n".join(f"- {rule}" for rule in update.rules)
    if RULES_HEADER in current_prompt:
        prefix = current_prompt.split(RULES_HEADER, maxsplit=1)[0].rstrip()
        return f"{prefix}\n\n{rules_block}".strip()
    return f"{current_prompt.strip()}\n\n{rules_block}".strip()
