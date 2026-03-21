from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thesis_platform.core.schemas import Sample


@dataclass(slots=True)
class RoundContext:
    """Context object shared by the server generator during one federation round."""

    round_id: int
    prompt_text: str
    public_seed_samples: list[Sample]
    config: dict[str, Any]
    output_dir: Path | None
    text_backend: Any = None
    prompt_scope: str = "global"
    cluster_id: str | None = None
    sample_id_prefix: str = "syn"
    sample_source: str = "synthetic"
    runtime_artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClientContext:
    """Per-client runtime state, including local data slices and embedding backend."""

    client_id: str
    train_samples: list[Sample]
    validation_samples: list[Sample]
    all_samples: list[Sample]
    embedder: Any
    config: dict[str, Any]
    negative_samples: list[Sample] = field(default_factory=list)
    text_backend: Any = None
    objective_type: str = "domain_probe"
    prototype_vector: list[float] | None = None
    prototype_weight: float = 1.0
    cluster_id: str | None = None
    cluster_prompt: str | None = None
    probe_state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ServerContext:
    """Global server-side state that survives across multiple rounds."""

    experiment_id: str
    prompt_text: str
    prompt_history: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None
    text_backend: Any = None
    aggregation_memory: dict[str, Any] = field(default_factory=dict)
    generated_history: list[list[str]] = field(default_factory=list)
    base_prompt: str | None = None
    cluster_prompts: dict[str, str] = field(default_factory=dict)
    client_cluster_map: dict[str, str] = field(default_factory=dict)
    prototype_feedbacks: list[Any] = field(default_factory=list)
    routing_state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalContext:
    """Evaluation context reserved for future metric runners and report builders."""

    dataset_name: str
    output_dir: Path
    config: dict[str, Any]
