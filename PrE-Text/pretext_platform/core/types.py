from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelPaths:
    """Resolved local model paths used by the different stages."""

    minilm: Path
    roberta_large: Path
    llama2_7b: Path
    llama_3_2_3b_instruct: Path
    distilgpt2: Path
    gpt2_xl: Path
    c4_checkpoint: Path | None = None


@dataclass(slots=True)
class DatasetBundle:
    """All dataset resources required by the pipeline."""

    dataset_name: str
    train_texts: list[str]
    eval_texts: list[str]
    initialization_texts: list[str]
    train_client_count: int = 0
    raw_train_sample_count: int = 0
    sampled_train_sample_count: int = 0
    max_samples_per_client: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageSummary:
    """Structured stage-level execution summary."""

    stage_name: str
    output_dir: Path
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    message: str = ""
