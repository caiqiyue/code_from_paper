from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Sample:
    """Unified sample object used throughout the whole platform pipeline."""

    sample_id: str
    client_id: str
    round_id: int
    source: str
    dataset_name: str
    task_type: str
    text: str
    instruction: str | None = None
    response: str | None = None
    label: str | int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoredSample(Sample):
    """Sample enriched with a badness score returned by a scorer adapter."""

    score: float = 0.0
    score_name: str = ""
    score_direction: str = "larger_is_worse"

    @classmethod
    def from_sample(
        cls,
        sample: Sample,
        *,
        client_id: str,
        score: float,
        score_name: str,
        score_direction: str = "larger_is_worse",
        meta: dict[str, Any] | None = None,
    ) -> "ScoredSample":
        """Build a scored sample while preserving the original sample payload."""

        merged_meta = dict(sample.meta)
        if meta:
            merged_meta.update(meta)
        return cls(
            sample_id=sample.sample_id,
            client_id=client_id,
            round_id=sample.round_id,
            source=sample.source,
            dataset_name=sample.dataset_name,
            task_type=sample.task_type,
            text=sample.text,
            instruction=sample.instruction,
            response=sample.response,
            label=sample.label,
            meta=merged_meta,
            score=score,
            score_name=score_name,
            score_direction=score_direction,
        )


@dataclass(slots=True)
class PairedSample:
    """A bad sample bundled with retrieved real anchor samples."""

    pair_id: str
    client_id: str
    round_id: int
    bad_sample: Sample
    real_samples: list[Sample]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Critique:
    """Structured textual feedback generated from a bad/real sample comparison."""

    critique_id: str
    client_id: str
    round_id: int
    bad_sample_id: str
    real_sample_ids: list[str]
    rules: list[str]
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptUpdate:
    """Aggregated prompt-level update produced by a server-side aggregator."""

    update_id: str
    round_id: int
    rules: list[str]
    summary: str
    prompt_text: str
    meta: dict[str, Any] = field(default_factory=dict)
