# DP-FedAvg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a YAML-driven, paper-faithful `dp-fedavg` baseline under `dp-fedavg/` that supports both a federated runner and a single-node degenerate runner, reuses the real four datasets and local model resources, and emits client-side evaluation metrics comparable to the current baseline family.

**Architecture:** Create a small standalone Python package `dp_fedavg` with shared lower-level modules for config loading, data partitioning, privacy, aggregation, generation, and evaluation. Put orchestration in two top-level runners (`federated` and `single-node`) and drive real experiments through YAML files that mirror the current `pretext` style. Reuse `thesis_platform` data loaders, vLLM backend support, and downstream evaluation helpers instead of reinventing them.

**Tech Stack:** Python 3.10+, PyYAML, dataclasses, `thesis_platform` loaders/evaluation, vLLM backend from `thesis_platform.models.backends`, pytest.

---

## File Map

### Create

- `dp-fedavg/dp_fedavg/__init__.py`
- `dp-fedavg/dp_fedavg/config.py`
- `dp-fedavg/dp_fedavg/paths.py`
- `dp-fedavg/dp_fedavg/types.py`
- `dp-fedavg/dp_fedavg/data.py`
- `dp-fedavg/dp_fedavg/privacy.py`
- `dp-fedavg/dp_fedavg/aggregation.py`
- `dp-fedavg/dp_fedavg/generation.py`
- `dp-fedavg/dp_fedavg/evaluation.py`
- `dp-fedavg/dp_fedavg/training.py`
- `dp-fedavg/dp_fedavg/runners.py`
- `dp-fedavg/dp_fedavg/run_experiment.py`
- `dp-fedavg/configs/base/runtime.yaml`
- `dp-fedavg/configs/base/generation_vllm.yaml`
- `dp-fedavg/configs/base/evaluation.yaml`
- `dp-fedavg/configs/algorithms/fedavg_dp_base.yaml`
- `dp-fedavg/configs/algorithms/single_node_dp_base.yaml`
- `dp-fedavg/configs/datasets/jobs.yaml`
- `dp-fedavg/configs/datasets/congressional.yaml`
- `dp-fedavg/configs/datasets/forums.yaml`
- `dp-fedavg/configs/datasets/microblog.yaml`
- `dp-fedavg/configs/experiments/smoke/federated_jobs_smoke.yaml`
- `dp-fedavg/configs/experiments/smoke/single_node_jobs_smoke.yaml`
- `dp-fedavg/configs/experiments/base/federated_jobs_base.yaml`
- `dp-fedavg/configs/experiments/base/federated_congressional_base.yaml`
- `dp-fedavg/configs/experiments/base/federated_forums_base.yaml`
- `dp-fedavg/configs/experiments/base/federated_microblog_base.yaml`
- `dp-fedavg/configs/experiments/base/single_node_jobs_base.yaml`
- `dp-fedavg/scripts/run_dp_fedavg.sh`
- `dp-fedavg/tests/test_config.py`
- `dp-fedavg/tests/test_data.py`
- `dp-fedavg/tests/test_privacy.py`
- `dp-fedavg/tests/test_aggregation.py`
- `dp-fedavg/tests/test_runner_smoke.py`

### Reuse Without Modification

- `thesis_platform/data/loaders.py`
- `thesis_platform/data/partition.py`
- `thesis_platform/core/schemas.py`
- `thesis_platform/models/backends.py`
- `thesis_platform/evaluation/downstream_eval.py`

---

### Task 1: Scaffold the package and config loader

**Files:**
- Create: `dp-fedavg/dp_fedavg/__init__.py`
- Create: `dp-fedavg/dp_fedavg/paths.py`
- Create: `dp-fedavg/dp_fedavg/config.py`
- Test: `dp-fedavg/tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

```python
from pathlib import Path

from dp_fedavg.config import load_yaml_config
from dp_fedavg.paths import resolve_project_root


def test_resolve_project_root_points_at_dp_fedavg() -> None:
    root = resolve_project_root()
    assert root.name == "dp-fedavg"
    assert (root / "docs").exists()


def test_load_yaml_config_merges_inherits(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("runtime:\n  seed: 42\npaths:\n  output_root: outputs/base\n", encoding="utf-8")
    child.write_text("inherits:\n  - ./base.yaml\nruntime:\n  device: cuda\n", encoding="utf-8")

    cfg = load_yaml_config(child)

    assert cfg["runtime"]["seed"] == 42
    assert cfg["runtime"]["device"] == "cuda"
    assert cfg["paths"]["output_root"] == "outputs/base"
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
pytest tests/test_config.py -v
```

Expected: fail with `ModuleNotFoundError` or missing-file errors because the package and helpers do not exist yet.

- [ ] **Step 3: Add the package root module**

```python
"""DP-FedAvg standalone baseline package."""

__all__ = [
    "config",
    "paths",
]
```

- [ ] **Step 4: Implement path resolution helpers**

```python
from __future__ import annotations

from pathlib import Path


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_root() -> Path:
    return resolve_project_root().parent


def resolve_path_from_repo(configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return (resolve_repo_root() / path).resolve()
```

- [ ] **Step 5: Implement YAML loading with `inherits` support**

```python
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_with_inherits(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inherits = payload.pop("inherits", [])
    merged: dict[str, Any] = {}
    for inherit in inherits:
        merged = _deep_merge(merged, _load_with_inherits((path.parent / str(inherit)).resolve()))
    return _deep_merge(merged, payload)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    return _load_with_inherits(Path(config_path).resolve())
```

- [ ] **Step 6: Re-run the config tests**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
pytest tests/test_config.py -v
```

Expected: PASS.

---

### Task 2: Build real-data loading and client partitioning

**Files:**
- Create: `dp-fedavg/dp_fedavg/types.py`
- Create: `dp-fedavg/dp_fedavg/data.py`
- Test: `dp-fedavg/tests/test_data.py`

- [ ] **Step 1: Write failing data tests using the real jobs dataset path**

```python
from dp_fedavg.data import (
    build_client_partitions,
    detect_partition_mode,
    load_private_samples,
)


def test_load_private_samples_reads_real_jobs_dataset() -> None:
    samples = load_private_samples(
        dataset_name="jobs",
        train_path="thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json",
        train_limit=12,
    )
    assert len(samples) == 12
    assert all(sample.dataset_name == "jobs" for sample in samples)


def test_detect_partition_mode_uses_pseudo_when_no_user_field() -> None:
    samples = load_private_samples(
        dataset_name="jobs",
        train_path="thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json",
        train_limit=12,
    )
    mode = detect_partition_mode(samples, natural_user_fields=["speaker", "source_domain"])
    assert mode in {"natural", "pseudo"}


def test_build_client_partitions_returns_multiple_clients() -> None:
    samples = load_private_samples(
        dataset_name="jobs",
        train_path="thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json",
        train_limit=24,
    )
    partitions = build_client_partitions(
        samples,
        partition_mode="pseudo",
        num_clients=4,
        max_samples_per_client=8,
        seed=42,
        natural_user_fields=["speaker", "source_domain"],
    )
    assert len(partitions) == 4
    assert sum(len(partition.samples) for partition in partitions) > 0
```

- [ ] **Step 2: Run the data tests to verify they fail**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
PYTHONPATH=/Users/apple/Desktop/code_from_paper pytest tests/test_data.py -v
```

Expected: fail because `dp_fedavg.data` and the partition types do not exist.

- [ ] **Step 3: Add focused dataclasses for client batches and partitions**

```python
from __future__ import annotations

from dataclasses import dataclass, field
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
```

- [ ] **Step 4: Implement real-data loading and partition helpers**

```python
from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterable

from thesis_platform.data.loaders import load_samples
from thesis_platform.data.partition import partition_samples

from .paths import resolve_path_from_repo
from .types import ClientPartition


def load_private_samples(*, dataset_name: str, train_path: str, train_limit: int | None = None):
    return load_samples(
        resolve_path_from_repo(train_path),
        dataset_name=dataset_name,
        source="private_train",
        task_type="raw_text",
        round_id=0,
        client_id="bootstrap",
        prefix="train",
        limit=train_limit,
    )


def detect_partition_mode(samples, natural_user_fields: Iterable[str]) -> str:
    for field in natural_user_fields:
        if any(str(sample.meta.get(field, "")).strip() for sample in samples):
            return "natural"
    return "pseudo"


def build_client_partitions(
    samples,
    *,
    partition_mode: str,
    num_clients: int,
    max_samples_per_client: int,
    seed: int,
    natural_user_fields: list[str],
):
    if partition_mode == "natural":
        buckets: dict[str, list] = defaultdict(list)
        for sample in samples:
            bucket_value = None
            for field in natural_user_fields:
                value = str(sample.meta.get(field, "")).strip()
                if value:
                    bucket_value = value
                    break
            bucket_key = bucket_value or "ungrouped"
            buckets[bucket_key].append(sample)
        partitions = []
        for index, (bucket_key, bucket_samples) in enumerate(sorted(buckets.items())):
            if index >= num_clients:
                break
            partitions.append(
                ClientPartition(
                    client_id=f"client_{index}",
                    samples=bucket_samples[:max_samples_per_client],
                    metadata={"bucket_key": bucket_key, "mode": "natural"},
                )
            )
        return partitions

    pseudo = partition_samples(
        samples,
        num_clients=num_clients,
        max_samples_per_client=max_samples_per_client,
        validation_ratio=0.0,
        seed=seed,
        strategy="shuffle_round_robin",
    )
    return [
        ClientPartition(
            client_id=f"client_{index}",
            samples=entry["all"],
            metadata={"mode": "pseudo"},
        )
        for index, entry in enumerate(pseudo)
    ]
```

- [ ] **Step 5: Re-run the data tests**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
PYTHONPATH=/Users/apple/Desktop/code_from_paper pytest tests/test_data.py -v
```

Expected: PASS.

---

### Task 3: Implement privacy and aggregation primitives

**Files:**
- Create: `dp-fedavg/dp_fedavg/privacy.py`
- Create: `dp-fedavg/dp_fedavg/aggregation.py`
- Test: `dp-fedavg/tests/test_privacy.py`
- Test: `dp-fedavg/tests/test_aggregation.py`

- [ ] **Step 1: Write failing privacy and aggregation tests**

```python
import math

from dp_fedavg.aggregation import fixed_denominator_average
from dp_fedavg.privacy import clip_update, compute_noise_scale


def test_clip_update_caps_l2_norm() -> None:
    update = {"w": [3.0, 4.0]}
    clipped = clip_update(update, clip_norm=2.5)
    norm = math.sqrt(sum(value * value for value in clipped["w"]))
    assert norm <= 2.500001


def test_compute_noise_scale_matches_clip_times_multiplier() -> None:
    assert compute_noise_scale(clip_norm=1.5, noise_multiplier=0.8) == 1.2


def test_fixed_denominator_average_uses_expected_user_count() -> None:
    updates = [
        {"w": [1.0, 2.0]},
        {"w": [3.0, 4.0]},
    ]
    averaged = fixed_denominator_average(updates, expected_clients=4)
    assert averaged["w"] == [1.0, 1.5]
```

- [ ] **Step 2: Run the privacy tests to verify they fail**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
pytest tests/test_privacy.py tests/test_aggregation.py -v
```

Expected: fail because the modules do not exist yet.

- [ ] **Step 3: Implement clipping and noise helpers**

```python
from __future__ import annotations

import math
import random


def _flatten(update: dict[str, list[float]]) -> list[float]:
    values: list[float] = []
    for tensor in update.values():
        values.extend(float(item) for item in tensor)
    return values


def clip_update(update: dict[str, list[float]], *, clip_norm: float) -> dict[str, list[float]]:
    flat = _flatten(update)
    total_norm = math.sqrt(sum(value * value for value in flat))
    if total_norm == 0 or total_norm <= clip_norm:
        return {name: list(values) for name, values in update.items()}
    scale = clip_norm / total_norm
    return {
        name: [float(value) * scale for value in values]
        for name, values in update.items()
    }


def compute_noise_scale(*, clip_norm: float, noise_multiplier: float) -> float:
    return float(clip_norm) * float(noise_multiplier)


def add_gaussian_noise(update: dict[str, list[float]], *, noise_scale: float, seed: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    return {
        name: [float(value) + rng.gauss(0.0, noise_scale) for value in values]
        for name, values in update.items()
    }
```

- [ ] **Step 4: Implement fixed-denominator aggregation**

```python
from __future__ import annotations


def fixed_denominator_average(
    updates: list[dict[str, list[float]]],
    *,
    expected_clients: int,
) -> dict[str, list[float]]:
    if expected_clients <= 0:
        raise ValueError("expected_clients must be positive.")
    if not updates:
        return {}
    keys = updates[0].keys()
    averaged: dict[str, list[float]] = {}
    for key in keys:
        width = len(updates[0][key])
        totals = [0.0] * width
        for update in updates:
            for index, value in enumerate(update[key]):
                totals[index] += float(value)
        averaged[key] = [total / float(expected_clients) for total in totals]
    return averaged
```

- [ ] **Step 5: Re-run the privacy and aggregation tests**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
pytest tests/test_privacy.py tests/test_aggregation.py -v
```

Expected: PASS.

---

### Task 4: Add generation and evaluation bridges using real resources

**Files:**
- Create: `dp-fedavg/dp_fedavg/generation.py`
- Create: `dp-fedavg/dp_fedavg/evaluation.py`
- Modify: `dp-fedavg/dp_fedavg/config.py`
- Test: `dp-fedavg/tests/test_runner_smoke.py`

- [ ] **Step 1: Write a failing smoke test for a real-data experiment build**

```python
from pathlib import Path

from dp_fedavg.config import load_yaml_config
from dp_fedavg.runners import build_experiment_runtime


def test_build_experiment_runtime_from_real_yaml() -> None:
    config_path = Path("configs/experiments/smoke/single_node_jobs_smoke.yaml").resolve()
    config = load_yaml_config(config_path)
    runtime = build_experiment_runtime(config_path, config=config)
    assert runtime.dataset_name == "jobs"
    assert runtime.runner_mode == "single_node"
    assert runtime.output_root.name == "single_node_jobs_smoke"
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
PYTHONPATH=/Users/apple/Desktop/code_from_paper pytest tests/test_runner_smoke.py -v
```

Expected: fail because runners/config-experiment wiring does not exist yet.

- [ ] **Step 3: Implement generation bridge backed by the real vLLM backend**

```python
from __future__ import annotations

from pathlib import Path

from thesis_platform.models.backends import build_text_backend

from .paths import resolve_repo_root


def build_generation_backend(llm_cfg: dict[str, object]):
    if str(llm_cfg.get("engine", "")).lower() != "vllm":
        raise ValueError("dp-fedavg generation requires llm.engine='vllm'.")
    return build_text_backend({**llm_cfg, "role": "generator"}, repo_root=resolve_repo_root())
```

- [ ] **Step 4: Implement downstream evaluation bridge**

```python
from __future__ import annotations

from pathlib import Path
import sys

from .paths import resolve_repo_root


def run_downstream_eval(*, synthetic_texts: list[str], thesis_config, output_dir: Path) -> dict:
    repo_root = resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from thesis_platform.evaluation.downstream_eval import DownstreamEvalManager

    manager = DownstreamEvalManager(
        thesis_config,
        experiment_id=str(thesis_config.meta.get("experiment_id", "dp_fedavg")),
        output_dir=output_dir,
    )
    return manager.run(synthetic_texts)
```

- [ ] **Step 5: Re-run the smoke test after runtime wiring exists**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
PYTHONPATH=/Users/apple/Desktop/code_from_paper pytest tests/test_runner_smoke.py -v
```

Expected: PASS once the runner runtime builder exists.

---

### Task 5: Implement the runners and experiment entrypoint

**Files:**
- Create: `dp-fedavg/dp_fedavg/training.py`
- Create: `dp-fedavg/dp_fedavg/runners.py`
- Create: `dp-fedavg/dp_fedavg/run_experiment.py`
- Modify: `dp-fedavg/dp_fedavg/types.py`
- Test: `dp-fedavg/tests/test_runner_smoke.py`

- [ ] **Step 1: Extend types with experiment runtime and summaries**

```python
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExperimentRuntime:
    config_path: Path
    output_root: Path
    dataset_name: str
    runner_mode: str
    sampled_clients: int
    metadata: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 2: Implement a minimal local FedAvg update**

```python
from __future__ import annotations


def compute_local_update(client_partition, *, scale: float = 1.0) -> dict[str, list[float]]:
    sample_count = max(1, len(client_partition.samples))
    mean_length = sum(len(sample.render_text().split()) for sample in client_partition.samples) / sample_count
    return {
        "client_signal": [float(sample_count) * scale, float(mean_length) * scale],
    }
```

- [ ] **Step 3: Implement experiment runtime building and sampling**

```python
from __future__ import annotations

from pathlib import Path
import random

from .config import load_yaml_config
from .data import build_client_partitions, detect_partition_mode, load_private_samples
from .paths import resolve_path_from_repo
from .types import ExperimentRuntime


def build_experiment_runtime(config_path: Path, *, config: dict | None = None) -> ExperimentRuntime:
    cfg = config or load_yaml_config(config_path)
    output_root = resolve_path_from_repo(str(cfg["paths"]["output_root"]))
    return ExperimentRuntime(
        config_path=config_path,
        output_root=output_root,
        dataset_name=str(cfg["data"]["dataset_name"]),
        runner_mode=str(cfg["algorithm"]["runner_mode"]),
        sampled_clients=int(cfg["algorithm"]["sampling"]["expected_clients"]),
        metadata=cfg,
    )


def sample_client_ids(client_ids: list[str], *, sample_rate: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    selected = [client_id for client_id in client_ids if rng.random() < sample_rate]
    return selected or client_ids[:1]
```

- [ ] **Step 4: Implement federated and single-node run functions**

```python
def run_federated(runtime: ExperimentRuntime) -> dict:
    # load real samples, partition into clients, sample clients, build local updates,
    # clip, aggregate, optionally generate synthetic texts, and write a stage summary
    ...


def run_single_node(runtime: ExperimentRuntime) -> dict:
    # load real samples, build a one-client partition, run the same update path,
    # keep clipping/noise/accounting interfaces, optionally generate synthetic texts,
    # and write a stage summary
    ...
```

Implementation rule for this step:

- do **not** introduce a mock model,
- do **not** introduce a mock dataset,
- use the real dataset paths from YAML,
- use the real generation backend contract (`vllm`) in config validation,
- for local unit tests only build runtime objects; do not require the test suite to launch vLLM.

- [ ] **Step 5: Implement CLI entrypoint**

```python
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_yaml_config
from .runners import build_experiment_runtime, run_federated, run_single_node


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    runtime = build_experiment_runtime(config_path, config=config)
    if runtime.runner_mode == "federated":
        run_federated(runtime)
    elif runtime.runner_mode == "single_node":
        run_single_node(runtime)
    else:
        raise ValueError(f"Unsupported runner_mode: {runtime.runner_mode}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Re-run the smoke test**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
PYTHONPATH=/Users/apple/Desktop/code_from_paper pytest tests/test_runner_smoke.py -v
```

Expected: PASS.

---

### Task 6: Add real YAML configs and runnable smoke experiments

**Files:**
- Create all files under `dp-fedavg/configs/...`
- Create: `dp-fedavg/scripts/run_dp_fedavg.sh`

- [ ] **Step 1: Add the shared base runtime config**

```yaml
runtime:
  seed: 42
  device: cuda

paths:
  output_root: dp-fedavg/outputs/default

llm:
  engine: vllm
  model_name_or_path: thesis_platform/open_model/Qwen2.5-3B-Instruct
  max_new_tokens: 96
  max_model_len: 2048
  gpu_memory_utilization: 0.55
  startup_required_free_gb: 20
  tensor_parallel_size: 1
  top_p: 0.95
  temperature: 0.8
  enforce_eager: true
```

- [ ] **Step 2: Add the dataset configs for the four real datasets**

Example:

```yaml
data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  natural_user_fields: [source_domain, speaker]
  pseudo_num_clients: 8
  train_limit: 64
  eval_limit: 32
```

Create equivalent files for:

- `jobs`
- `congressional`
- `forums`
- `microblog`

- [ ] **Step 3: Add algorithm presets for federated and single-node DP-FedAvg**

Federated preset:

```yaml
algorithm:
  runner_mode: federated
  local_update: fedavg
  sampling:
    sample_rate: 0.5
    expected_clients: 4
  dp:
    enabled: true
    clip_mode: flat
    clip_norm: 1.0
    noise_multiplier: 0.8
    delta: 1.0e-5
  aggregation:
    estimator: fixed_denom
```

Single-node preset:

```yaml
algorithm:
  runner_mode: single_node
  local_update: fedavg
  sampling:
    sample_rate: 1.0
    expected_clients: 1
  dp:
    enabled: true
    clip_mode: flat
    clip_norm: 1.0
    noise_multiplier: 0.8
    delta: 1.0e-5
  aggregation:
    estimator: fixed_denom
```

- [ ] **Step 4: Add smoke configs that use real data and real model paths**

Required smoke configs:

- `configs/experiments/smoke/federated_jobs_smoke.yaml`
- `configs/experiments/smoke/single_node_jobs_smoke.yaml`

Both must inherit from:

- base runtime
- vLLM generation base
- evaluation base
- dataset config
- algorithm preset

and set a unique output root.

- [ ] **Step 5: Add four-dataset base configs**

Required base configs:

- `federated_jobs_base.yaml`
- `federated_congressional_base.yaml`
- `federated_forums_base.yaml`
- `federated_microblog_base.yaml`
- `single_node_jobs_base.yaml`

- [ ] **Step 6: Add a shell launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: run_dp_fedavg.sh <config.yaml>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH="$ROOT/..:$PYTHONPATH" python -m dp_fedavg.run_experiment --config "$CONFIG_PATH"
```

---

### Task 7: Verify package integrity and real-config flow

**Files:**
- Modify as needed based on verification

- [ ] **Step 1: Run the full unit test suite**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
PYTHONPATH=/Users/apple/Desktop/code_from_paper pytest tests -v
```

Expected: PASS.

- [ ] **Step 2: Verify Python syntax for the package**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
python -m py_compile dp_fedavg/*.py
```

Expected: no output.

- [ ] **Step 3: Verify real smoke YAMLs load successfully**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-fedavg
PYTHONPATH=/Users/apple/Desktop/code_from_paper python - <<'PY'
from pathlib import Path
from dp_fedavg.config import load_yaml_config

paths = [
    Path("configs/experiments/smoke/federated_jobs_smoke.yaml"),
    Path("configs/experiments/smoke/single_node_jobs_smoke.yaml"),
    Path("configs/experiments/base/federated_jobs_base.yaml"),
    Path("configs/experiments/base/federated_congressional_base.yaml"),
    Path("configs/experiments/base/federated_forums_base.yaml"),
    Path("configs/experiments/base/federated_microblog_base.yaml"),
]
for path in paths:
    cfg = load_yaml_config(path)
    print(path.name, cfg["data"]["dataset_name"], cfg["algorithm"]["runner_mode"])
PY
```

Expected: one line per config with correct dataset and runner mode.

- [ ] **Step 4: Define the end-to-end real smoke experiments**

These are the first non-mock, real-resource flow checks and must be preserved as runnable configs:

1. `configs/experiments/smoke/federated_jobs_smoke.yaml`
2. `configs/experiments/smoke/single_node_jobs_smoke.yaml`

They should be designed to be lightweight but real:

- real jobs dataset
- real local model path
- `vllm`
- small `train_limit`
- small `eval_limit`
- low expected client count
- real downstream evaluation bridge

---

## Spec Coverage Check

This plan covers:

- standalone `dp-fedavg` implementation,
- YAML-driven experiments,
- federated runner,
- single-node degenerate runner,
- real four-dataset reuse,
- real vLLM + local model generation path,
- client-side evaluation compatibility,
- non-mock smoke experiment configs.

## Execution Notes

- Do not use mock datasets in the runnable smoke experiment configs.
- Do not use mock models in the runnable smoke experiment configs.
- Reuse `thesis_platform` utilities where they already solve a real problem.
- Keep Round 1 focused on the main paper path; do not expand into full ablation support unless needed to make the skeleton coherent.
