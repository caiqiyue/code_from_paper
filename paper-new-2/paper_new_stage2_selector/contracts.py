from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BootstrapPromptRecord:
    prompt_index: int
    prompt_text: str
    seed_texts: list[str]


@dataclass(slots=True)
class GeneratedSampleRecord:
    record_index: int
    prompt_index: int
    prompt_text: str
    seed_texts: list[str]
    raw_text: str
    baseline_text: str
    consistency_score: float = 0.0
    template_penalty: float = 0.0
    duplicate_penalty: float = 0.0
    final_score: float = 0.0
    rejected_reason: str = ""


@dataclass(slots=True)
class Stage2SelectionResult:
    selected_records: list[GeneratedSampleRecord] = field(default_factory=list)
    rejected_records: list[GeneratedSampleRecord] = field(default_factory=list)
    raw_clean_count: int = 0
    target_count: int = 0
