from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thesis_platform.core.schemas import Sample


@dataclass(slots=True)
class RoundContext:
    round_id: int
    prompt_text: str
    public_seed_samples: list[Sample]
    config: dict[str, Any]
    output_dir: Path


@dataclass(slots=True)
class ClientContext:
    client_id: str
    train_samples: list[Sample]
    validation_samples: list[Sample]
    all_samples: list[Sample]
    embedder: Any
    config: dict[str, Any]


@dataclass(slots=True)
class ServerContext:
    experiment_id: str
    prompt_text: str
    prompt_history: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None


@dataclass(slots=True)
class EvalContext:
    dataset_name: str
    output_dir: Path
    config: dict[str, Any]
