from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thesis_platform.core.schemas import Sample


@dataclass(slots=True)
class ClientPartition:
    client_id: str
    samples: list[Sample]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SampledRound:
    round_id: int
    client_ids: list[str]


@dataclass(slots=True)
class ExperimentRuntime:
    config_path: Path
    config: dict[str, Any]
    output_root: Path
    dataset_name: str
    runner_mode: str
    seed: int


@dataclass(slots=True)
class RunArtifacts:
    synthetic_texts: list[str]
    stage_summary: dict[str, Any]
    eval_summary: dict[str, Any] | None = None
