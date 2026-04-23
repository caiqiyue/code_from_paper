from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CandidateRecord:
    """One Stage 1 candidate together with its selector-side scores."""

    index: int
    text: str
    vector: list[float]
    private_support: float
    genericity_penalty: float
    redundancy_penalty: float
    accept_score: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SelectorDecision:
    """The final Stage 1 selector decision."""

    selected_indices: list[int]
    hard_negative_indices: list[int]
    accept_scores: list[float]
    base_scores: list[float]
    hard_negative_reason: dict[int, str]
    candidate_records: list[CandidateRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_records"] = [asdict(record) for record in self.candidate_records]
        return payload


@dataclass(slots=True)
class BoundaryState:
    """Structured boundary summary derived from boundary negatives."""

    reject_score_ceiling: float
    reject_score_floor: float
    negative_centroid: list[float]
    negative_pattern_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeneratorContract:
    """Explicit contract for the only legal Stage 1 candidate generator."""

    backend: str
    source: str
    initial_prompt: str
    candidate_count: int
    generated_per_round: int
    exemplars_per_prompt: int
    max_new_tokens: int
    llm_backend: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateGeneratorHandle:
    """Generator wrapper that keeps the contract and resolved dependencies together."""

    generator: Any
    text_backend: Any
    contract: dict[str, Any]
    repo_root: Path
    resource_root: Path
