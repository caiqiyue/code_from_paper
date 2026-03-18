from __future__ import annotations

from thesis_platform.core.schemas import PromptUpdate

BASE_HEADER = "### Base Instruction"
RULES_HEADER = "### Current Round Guidance"
MEMORY_HEADER = "### Memory Summary"


def _extract_base_prompt(current_prompt: str) -> str:
    """Recover the immutable base instruction from a formatted prompt."""

    if BASE_HEADER in current_prompt:
        remainder = current_prompt.split(BASE_HEADER, maxsplit=1)[1]
        for stop_header in (RULES_HEADER, MEMORY_HEADER):
            if stop_header in remainder:
                remainder = remainder.split(stop_header, maxsplit=1)[0]
        return remainder.strip()
    for stop_header in (RULES_HEADER, MEMORY_HEADER):
        if stop_header in current_prompt:
            return current_prompt.split(stop_header, maxsplit=1)[0].strip()
    return current_prompt.strip()


def apply_prompt_update(current_prompt: str, update: PromptUpdate) -> str:
    """Rebuild the prompt using base instruction, current guidance, and memory."""

    base_prompt = update.meta.get("base_prompt") or _extract_base_prompt(current_prompt)
    current_rules = update.rules
    memory_rules = update.meta.get("memory_rules", [])

    sections = [
        BASE_HEADER,
        str(base_prompt).strip(),
        "",
        RULES_HEADER,
        "\n".join(f"- {rule}" for rule in current_rules).strip(),
    ]
    if memory_rules:
        sections.extend(
            [
                "",
                MEMORY_HEADER,
                "\n".join(f"- {rule}" for rule in memory_rules).strip(),
            ]
        )
    return "\n".join(part for part in sections if part is not None).strip()
