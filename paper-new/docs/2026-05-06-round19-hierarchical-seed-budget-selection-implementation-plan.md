# Hierarchical Seed Budget Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `paper-new-round19` by copying `paper-new`, port the advanced budget-calibration stack needed for the hierarchical method into that new directory, then implement the final hierarchical, distribution-shape-aware seed-budget selection method and add a round-19 quick-comparison experiment suite plus its config files.

**Architecture:** The implementation must happen in `paper-new-round19`, but the current `paper-new` codebase is missing the advanced budget-calibration stack already present in `paper-new-round-16`. The correct round-19 flow is therefore: copy `paper-new` into `paper-new-round19`, port the advanced Stage-1 budget machinery from `paper-new-round-16` into `paper-new-round19`, then add the new hierarchical descriptor/router/resolver logic on top of that migrated stack. Quick-comparison configs should mirror the successful `round182/round183` experiment pattern while living entirely under `paper-new-round19`.

**Tech Stack:** Python 3, existing `paper_new_selector` pipeline, YAML experiment configs, `unittest`/`pytest`, existing single-node quick-eval protocol

---

## File Structure

### Planning assumption

All development code changes below target:

- `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19`

The plan also assumes two reference sources:

- implementation reference: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round-16`
- document/output location: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\docs`

All planning/design documentation outputs for this round stay under:

- `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\docs`

### Files to create

- `paper-new-round19/paper_new_selector/budget_calibration.py`
  - Ported advanced per-budget calibration core from `paper-new-round-16`.
- `paper-new-round19/paper_new_selector/shape_descriptor.py`
  - Owns lightweight private-length descriptor extraction.
- `paper-new-round19/paper_new_selector/regime_router.py`
  - Owns `z_shape` scoring and regime routing.
- `paper-new-round19/paper_new_selector/hierarchical_budget.py`
  - Owns hierarchical policy resolution on top of per-budget metrics.
- `paper-new-round19/tests/test_budget_calibration.py`
  - Ported and adapted advanced budget-calibration regression tests.
- `paper-new-round19/tests/test_shape_descriptor.py`
  - Unit tests for descriptor statistics.
- `paper-new-round19/tests/test_regime_router.py`
  - Unit tests for route decisions.
- `paper-new-round19/tests/test_hierarchical_budget.py`
  - Unit tests for broad-tail / compact / uncertain routing and resolver behavior.
- `paper-new-round19/configs/experiments/single_node_tuning_round19/_base_selector_tuning_round19.yaml`
  - Base round-19 quick-comparison config.
- `paper-new-round19/configs/experiments/single_node_tuning_round19/guard/r19_guard_forums.yaml`
- `paper-new-round19/configs/experiments/single_node_tuning_round19/guard/r19_guard_congressional.yaml`
- `paper-new-round19/configs/experiments/single_node_tuning_round19/full_run/r19_full_jobs.yaml`
- `paper-new-round19/configs/experiments/single_node_tuning_round19/full_run/r19_full_congressional.yaml`
- `paper-new-round19/configs/experiments/single_node_tuning_round19/full_run/r19_full_forums.yaml`
- `paper-new-round19/configs/experiments/single_node_tuning_round19/full_run/r19_full_microblog.yaml`
- `paper-new-round19/configs/experiments/single_node_tuning_round19/ablations/r19_ablate_no_router_forums.yaml`
- `paper-new-round19/configs/experiments/single_node_tuning_round19/ablations/r19_ablate_no_router_congressional.yaml`
- `paper-new-round19/configs/experiments/single_node_tuning_round19/ablations/r19_ablate_force_compact_forums.yaml`
- `paper-new-round19/configs/experiments/single_node_tuning_round19/ablations/r19_ablate_force_broad_congressional.yaml`
- `paper-new-round19/scripts/append_round19_summary.py`
  - Summarizes `stage1_budget_calibration.json` + downstream eval into a comparison TSV.
- `paper-new-round19/scripts/run_round19_guard_batch.sh`
  - Runs the two critical guard experiments.
- `paper-new-round19/scripts/run_round19_quick_compare.sh`
  - Runs guard, full-run, and ablations in the intended order.

### Files to modify

- `paper-new-round19/paper_new_selector/stage1_runner.py`
  - Route `seed_budget_rule.mode=hierarchical_shape_routing` into the new resolver.
- `paper-new-round19/paper_new_selector/pipeline.py`
  - Ensure the new mode writes `stage1_budget_calibration.json`.
- `paper-new-round19/tests/test_stage1_runner.py`
  - Add integration tests for `hierarchical_shape_routing`.

### Pre-implementation bootstrap step

Before any round-19 feature work begins:

1. Copy the current contents of `paper-new` into `paper-new-round19`.
2. Verify that `paper-new-round19` contains the same baseline files as `paper-new`.
3. Port the advanced budget-calibration stack from `paper-new-round-16` into `paper-new-round19`, specifically:
   - `paper_new_selector/budget_calibration.py`
   - `tests/test_budget_calibration.py`
   - the `stage1_runner.py` branches for `self_calibrated`, `self_calibrated_constrained`, and `hybrid_length_family_constrained`
   - `pipeline.py` writing of `stage1_budget_calibration.json`
4. Only after that bootstrap is complete should the new hierarchical mode be implemented.

---

## Quick Comparison Experiment Design

### Fixed protocol

Reuse the same quick-comparison protocol that underpins the existing screening/tuning evidence:

- `train_limit = 256`
- `eval_limit = 256`
- `initialization_limit = 1024`
- `bootstrap.num_prompts = 100`
- `bootstrap.max_tokens = 85`
- `eval.mode = pretext_small`
- `eval.small_epochs = 6`
- `candidate_seed_top_k = [18, 19, 20, 21, 22]`
- `meta.seed = 42` for guard and quick compare

For the final round-19 implementation, `full_run/*` keeps this full screening-scale protocol, while `guard/*` is intentionally cheaper:

- `guard.train_limit = 128`
- `guard.eval_limit = 128`
- `guard.initialization_limit = 512`
- `guard.bootstrap.num_prompts = 40`
- `guard.eval.small_epochs = 4`

### Quick experiment phases

1. **Guard batch**
   - `r19_guard_forums`
   - `r19_guard_congressional`
   - Purpose: verify broad-tail and compact-structured policies recover the two critical regime behaviors under a cheaper pre-regression sanity tier before spending on 4-dataset regression.

2. **Full quick compare**
   - `r19_full_jobs`
   - `r19_full_congressional`
   - `r19_full_forums`
   - `r19_full_microblog`
   - Purpose: confirm 4/4 > `PrE-Text` under the same screening-scale protocol.

3. **Ablation quick compare**
   - `r19_ablate_no_router_forums`
   - `r19_ablate_no_router_congressional`
   - `r19_ablate_force_compact_forums`
   - `r19_ablate_force_broad_congressional`
   - Purpose: prove the hierarchical structure is necessary, not just decorative.

### Primary quick-comparison success checks

- `r19_guard_forums`: resolved budget lands in the broad-tail band (`21/22`) and improves on the unified fallback ablation.
- `r19_guard_congressional`: resolved budget lands in the compact band (`18/19/20`) and improves on the forced-broad ablation.
- `r19_full_*`: all four datasets remain above the archived `PrE-Text` screening baseline under the same protocol.

---

## Config File Sketches

### Base round-19 config

File: `paper-new-round19/configs/experiments/single_node_tuning_round19/_base_selector_tuning_round19.yaml`

```yaml
inherits:
  - ../single_node_screening/_base_selector_screening.yaml

meta:
  stage: single_node_tuning_round19
  seed: 42

bootstrap:
  max_tokens: 85

selector:
  seed_budget_rule:
    enabled: true
    mode: hierarchical_shape_routing
    candidate_seed_top_k: [18, 19, 20, 21, 22]
    router:
      tail_threshold: 350
      short_threshold: 120
      zscore_source: screening_reference
      tau_center: 0.0
      delta_router: 0.35
    policies:
      broad_tail:
        candidate_seed_top_k: [21, 22]
        coverage_p25_ratio: 0.98
        coverage_mean_ratio: 0.998
        epsilon: 0.002
      compact_structured:
        candidate_seed_top_k: [18, 19, 20]
        coverage_p25_ratio: 0.98
        utility:
          support_weight: 1.0
          genericity_weight: 0.5
          redundancy_weight: 0.3
          budget_weight: 0.1
        epsilon: 0.01
      uncertain:
        fallback_mode: self_calibrated_constrained
        coverage_constraint:
          mode: tail_family_relative
          metrics:
            - name: coverage_p25
              relative_ratio: 0.98
              required: true
              weight: 0.7
            - name: coverage_mean
              relative_ratio: 0.998
              required: true
              weight: 0.3
```

### Guard configs

File: `paper-new-round19/configs/experiments/single_node_tuning_round19/guard/r19_guard_forums.yaml`

```yaml
inherits:
  - ../_base_selector_tuning_round19.yaml
  - ../_data_forums.yaml

meta:
  experiment_id: r19_guard_forums

paths:
  output_root: paper-new-round19/outputs/r19_guard_forums

data:
  train_limit: 128
  eval_limit: 128
  initialization_limit: 512

bootstrap:
  num_prompts: 40

eval:
  small_epochs: 4
```

File: `paper-new-round19/configs/experiments/single_node_tuning_round19/guard/r19_guard_congressional.yaml`

```yaml
inherits:
  - ../_base_selector_tuning_round19.yaml
  - ../_data_congressional.yaml

meta:
  experiment_id: r19_guard_congressional

paths:
  output_root: paper-new-round19/outputs/r19_guard_congressional

data:
  train_limit: 128
  eval_limit: 128
  initialization_limit: 512

bootstrap:
  num_prompts: 40

eval:
  small_epochs: 4
```

### Full quick-compare configs

File: `paper-new-round19/configs/experiments/single_node_tuning_round19/full_run/r19_full_jobs.yaml`

```yaml
inherits:
  - ../_base_selector_tuning_round19.yaml
  - ../_data_jobs.yaml

meta:
  experiment_id: r19_full_jobs

paths:
  output_root: paper-new-round19/outputs/r19_full_jobs
```

File: `paper-new-round19/configs/experiments/single_node_tuning_round19/full_run/r19_full_congressional.yaml`

```yaml
inherits:
  - ../_base_selector_tuning_round19.yaml
  - ../_data_congressional.yaml

meta:
  experiment_id: r19_full_congressional

paths:
  output_root: paper-new-round19/outputs/r19_full_congressional
```

File: `paper-new-round19/configs/experiments/single_node_tuning_round19/full_run/r19_full_forums.yaml`

```yaml
inherits:
  - ../_base_selector_tuning_round19.yaml
  - ../_data_forums.yaml

meta:
  experiment_id: r19_full_forums

paths:
  output_root: paper-new-round19/outputs/r19_full_forums
```

File: `paper-new-round19/configs/experiments/single_node_tuning_round19/full_run/r19_full_microblog.yaml`

```yaml
inherits:
  - ../_base_selector_tuning_round19.yaml
  - ../_data_microblog.yaml

meta:
  experiment_id: r19_full_microblog

paths:
  output_root: paper-new-round19/outputs/r19_full_microblog
```

### Ablation configs

File: `paper-new-round19/configs/experiments/single_node_tuning_round19/ablations/r19_ablate_no_router_forums.yaml`

```yaml
inherits:
  - ../_base_selector_tuning_round19.yaml
  - ../_data_forums.yaml

meta:
  experiment_id: r19_ablate_no_router_forums

selector:
  seed_budget_rule:
    mode: self_calibrated_constrained

paths:
  output_root: paper-new-round19/outputs/r19_ablate_no_router_forums
```

File: `paper-new-round19/configs/experiments/single_node_tuning_round19/ablations/r19_ablate_no_router_congressional.yaml`

```yaml
inherits:
  - ../_base_selector_tuning_round19.yaml
  - ../_data_congressional.yaml

meta:
  experiment_id: r19_ablate_no_router_congressional

selector:
  seed_budget_rule:
    mode: self_calibrated_constrained

paths:
  output_root: paper-new-round19/outputs/r19_ablate_no_router_congressional
```

File: `paper-new-round19/configs/experiments/single_node_tuning_round19/ablations/r19_ablate_force_compact_forums.yaml`

```yaml
inherits:
  - ../_base_selector_tuning_round19.yaml
  - ../_data_forums.yaml

meta:
  experiment_id: r19_ablate_force_compact_forums

selector:
  seed_budget_rule:
    router:
      tau_center: 99.0
      delta_router: 0.0

paths:
  output_root: paper-new-round19/outputs/r19_ablate_force_compact_forums
```

File: `paper-new-round19/configs/experiments/single_node_tuning_round19/ablations/r19_ablate_force_broad_congressional.yaml`

```yaml
inherits:
  - ../_base_selector_tuning_round19.yaml
  - ../_data_congressional.yaml

meta:
  experiment_id: r19_ablate_force_broad_congressional

selector:
  seed_budget_rule:
    router:
      tau_center: -99.0
      delta_router: 0.0

paths:
  output_root: paper-new-round19/outputs/r19_ablate_force_broad_congressional
```

---

## Commands

### Unit / integration verification

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m pytest tests/test_shape_descriptor.py -q
python -m pytest tests/test_regime_router.py -q
python -m pytest tests/test_hierarchical_budget.py -q
python -m pytest tests/test_budget_calibration.py -q
python -m pytest tests/test_stage1_runner.py -q
```

### Guard runs

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/guard/r19_guard_forums.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/guard/r19_guard_congressional.yaml
```

### Full quick compare

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/full_run/r19_full_jobs.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/full_run/r19_full_congressional.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/full_run/r19_full_forums.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/full_run/r19_full_microblog.yaml
```

### Ablation quick compare

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/ablations/r19_ablate_no_router_forums.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/ablations/r19_ablate_no_router_congressional.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/ablations/r19_ablate_force_compact_forums.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/ablations/r19_ablate_force_broad_congressional.yaml
```

Expected quick-run artifacts:

- `paper-new-round19/outputs/<experiment_id>/stage1_summary.json`
- `paper-new-round19/outputs/<experiment_id>/stage1_budget_calibration.json`
- `paper-new-round19/outputs/<experiment_id>/eval/downstream_eval_summary.json`

---

### Task 0: Bootstrap `paper-new-round19` from `paper-new` and port the advanced budget stack

**Files:**
- Create/Populate: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19\...`
- Reference only: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round-16\paper_new_selector\budget_calibration.py`
- Reference only: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round-16\tests\test_budget_calibration.py`

- [ ] **Step 1: Copy `paper-new` into `paper-new-round19`**

Run:

```powershell
if (-not (Test-Path "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19")) {
    New-Item -ItemType Directory -Path "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19" | Out-Null
}
robocopy "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new" "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19" /E /XD ".git" "__pycache__" "outputs"
```

Expected:

```text
Files copied from paper-new into paper-new-round19
```

- [ ] **Step 2: Verify the copied baseline layout exists**

Run:

```powershell
Get-ChildItem "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19" | Select-Object Name
```

Expected:

```text
configs
docs
paper_new_selector
scripts
tests
```

- [ ] **Step 3: Port the advanced budget stack from `paper-new-round-16`**

Run:

```powershell
Copy-Item "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round-16\paper_new_selector\budget_calibration.py" "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19\paper_new_selector\budget_calibration.py" -Force
Copy-Item "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round-16\tests\test_budget_calibration.py" "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19\tests\test_budget_calibration.py" -Force
```

Expected:

```text
Advanced budget calibration files now exist in paper-new-round19
```

- [ ] **Step 4: Port the compatible `stage1_runner.py` and `pipeline.py` advanced branches**

Run:

```powershell
Copy-Item "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round-16\paper_new_selector\stage1_runner.py" "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19\paper_new_selector\stage1_runner.py" -Force
Copy-Item "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round-16\paper_new_selector\pipeline.py" "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19\paper_new_selector\pipeline.py" -Force
Copy-Item "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round-16\tests\test_stage1_runner.py" "D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19\tests\test_stage1_runner.py" -Force
```

Expected:

```text
paper-new-round19 now contains self_calibrated / constrained / hybrid stage1 logic and stage1_budget_calibration output support
```

- [ ] **Step 5: Smoke-test the migrated advanced stack**

Run:

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m pytest tests/test_budget_calibration.py tests/test_stage1_runner.py -q
```

Expected:

```text
Migrated budget-calibration tests pass or reveal only port-specific issues to fix before Task 1
```

### Task 1: Add descriptor extraction and regime routing modules

**Files:**
- Create: `paper-new-round19/paper_new_selector/shape_descriptor.py`
- Create: `paper-new-round19/paper_new_selector/regime_router.py`
- Test: `paper-new-round19/tests/test_shape_descriptor.py`
- Test: `paper-new-round19/tests/test_regime_router.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_shape_descriptor.py
import unittest

from paper_new_selector.shape_descriptor import compute_shape_descriptor


class ShapeDescriptorTests(unittest.TestCase):
    def test_compute_shape_descriptor_reports_expected_statistics(self):
        descriptor = compute_shape_descriptor(
            private_lengths=[80, 100, 150, 220, 420],
            tail_threshold=350,
            short_threshold=120,
        )
        self.assertEqual(descriptor.median_len, 150.0)
        self.assertEqual(descriptor.p75_len, 220.0)
        self.assertAlmostEqual(descriptor.tail_ratio, 0.2)
        self.assertAlmostEqual(descriptor.short_ratio, 0.4)
        self.assertGreater(descriptor.iqr_len, 0.0)
```

```python
# tests/test_regime_router.py
import unittest

from paper_new_selector.regime_router import compute_shape_score, route_budget_regime
from paper_new_selector.shape_descriptor import ShapeDescriptor


class RegimeRouterTests(unittest.TestCase):
    def test_route_budget_regime_returns_broad_tail(self):
        descriptor = ShapeDescriptor(
            median_len=210.0,
            p75_len=460.0,
            tail_ratio=0.35,
            short_ratio=0.05,
            iqr_len=260.0,
        )
        decision = route_budget_regime(
            descriptor,
            router_cfg={
                "tau_center": 0.0,
                "delta_router": 0.35,
                "screening_reference": {
                    "median_len": {"mean": 150.0, "std": 50.0},
                    "p75_len": {"mean": 335.0, "std": 90.0},
                    "iqr_len": {"mean": 180.0, "std": 60.0},
                },
            },
        )
        self.assertEqual(decision.regime, "broad_tail")
        self.assertGreater(decision.shape_score, 0.35)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m pytest tests/test_shape_descriptor.py tests/test_regime_router.py -q
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'paper_new_selector.shape_descriptor'
```

- [ ] **Step 3: Write the minimal implementation**

```python
# paper_new_selector/shape_descriptor.py
from __future__ import annotations

from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class ShapeDescriptor:
    median_len: float
    p75_len: float
    tail_ratio: float
    short_ratio: float
    iqr_len: float


def _percentile_nearest_rank(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(int(value) for value in values)
    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 100:
        return float(sorted_values[-1])
    rank = math.ceil((float(percentile) / 100.0) * len(sorted_values))
    return float(sorted_values[max(0, rank - 1)])


def compute_shape_descriptor(
    private_lengths: list[int],
    *,
    tail_threshold: int,
    short_threshold: int,
) -> ShapeDescriptor:
    if not private_lengths:
        return ShapeDescriptor(0.0, 0.0, 0.0, 0.0, 0.0)
    q1 = _percentile_nearest_rank(private_lengths, 25)
    q3 = _percentile_nearest_rank(private_lengths, 75)
    total = float(len(private_lengths))
    return ShapeDescriptor(
        median_len=float(statistics.median(private_lengths)),
        p75_len=float(q3),
        tail_ratio=float(sum(length >= tail_threshold for length in private_lengths) / total),
        short_ratio=float(sum(length <= short_threshold for length in private_lengths) / total),
        iqr_len=float(q3 - q1),
    )
```

```python
# paper_new_selector/regime_router.py
from __future__ import annotations

from dataclasses import dataclass

from .shape_descriptor import ShapeDescriptor


@dataclass(frozen=True)
class RegimeDecision:
    regime: str
    shape_score: float


def _zscore(value: float, mean: float, std: float) -> float:
    if abs(float(std)) <= 1e-8:
        return 0.0
    return float((float(value) - float(mean)) / float(std))


def compute_shape_score(descriptor: ShapeDescriptor, router_cfg: dict) -> float:
    ref = dict(router_cfg["screening_reference"])
    return (
        _zscore(descriptor.median_len, ref["median_len"]["mean"], ref["median_len"]["std"])
        + _zscore(descriptor.p75_len, ref["p75_len"]["mean"], ref["p75_len"]["std"])
        + _zscore(descriptor.iqr_len, ref["iqr_len"]["mean"], ref["iqr_len"]["std"])
        + float(descriptor.tail_ratio)
        - float(descriptor.short_ratio)
    )


def route_budget_regime(descriptor: ShapeDescriptor, router_cfg: dict) -> RegimeDecision:
    score = compute_shape_score(descriptor, router_cfg)
    tau_center = float(router_cfg.get("tau_center", 0.0))
    delta_router = float(router_cfg.get("delta_router", 0.35))
    if score >= tau_center + delta_router:
        return RegimeDecision(regime="broad_tail", shape_score=score)
    if score <= tau_center - delta_router:
        return RegimeDecision(regime="compact_structured", shape_score=score)
    return RegimeDecision(regime="uncertain", shape_score=score)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m pytest tests/test_shape_descriptor.py tests/test_regime_router.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git -C D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19 add paper_new_selector/shape_descriptor.py paper_new_selector/regime_router.py tests/test_shape_descriptor.py tests/test_regime_router.py
git -C D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19 commit -m "feat: add shape descriptor and regime router"
```

### Task 2: Add hierarchical policy resolution on top of budget calibration

**Files:**
- Create: `paper-new-round19/paper_new_selector/hierarchical_budget.py`
- Modify: `paper-new-round19/paper_new_selector/budget_calibration.py`
- Test: `paper-new-round19/tests/test_hierarchical_budget.py`
- Test: `paper-new-round19/tests/test_budget_calibration.py`

- [ ] **Step 1: Write the failing hierarchical tests**

```python
# tests/test_hierarchical_budget.py
import unittest

from paper_new_selector.hierarchical_budget import resolve_hierarchical_budget


class HierarchicalBudgetTests(unittest.TestCase):
    def test_broad_tail_policy_prefers_high_budget_band(self):
        result = resolve_hierarchical_budget(
            private_lengths=[150, 205, 396],
            metrics_by_budget={
                21: {"coverage_p25": 0.1358, "coverage_mean": 0.2480, "support_mean": 0.82, "genericity_score": 0.18, "redundancy_score": 0.09, "budget_cost": 0.75},
                22: {"coverage_p25": 0.1358, "coverage_mean": 0.2490, "support_mean": 0.81, "genericity_score": 0.18, "redundancy_score": 0.09, "budget_cost": 1.0},
            },
            rule_cfg={
                "router": {
                    "tail_threshold": 350,
                    "short_threshold": 120,
                    "tau_center": 0.0,
                    "delta_router": 0.35,
                    "screening_reference": {
                        "median_len": {"mean": 150.0, "std": 50.0},
                        "p75_len": {"mean": 335.0, "std": 90.0},
                        "iqr_len": {"mean": 180.0, "std": 60.0},
                    },
                },
                "policies": {
                    "broad_tail": {"candidate_seed_top_k": [21, 22], "coverage_p25_ratio": 0.98, "coverage_mean_ratio": 0.998, "epsilon": 0.002},
                    "compact_structured": {"candidate_seed_top_k": [18, 19, 20], "coverage_p25_ratio": 0.98, "utility": {"support_weight": 1.0, "genericity_weight": 0.5, "redundancy_weight": 0.3, "budget_weight": 0.1}, "epsilon": 0.01},
                    "uncertain": {"fallback_mode": "self_calibrated_constrained"},
                },
            },
        )
        self.assertEqual(result["regime"], "broad_tail")
        self.assertIn(result["resolved_seed_top_k"], [21, 22])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m pytest tests/test_hierarchical_budget.py -q
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'paper_new_selector.hierarchical_budget'
```

- [ ] **Step 3: Implement the hierarchical resolver**

```python
# paper_new_selector/hierarchical_budget.py
from __future__ import annotations

from .budget_calibration import combine_feasible_budget_metrics, select_budget_by_constrained_utility
from .regime_router import route_budget_regime
from .shape_descriptor import compute_shape_descriptor


def _filter_budget_metrics(metrics_by_budget: dict[int, dict], candidate_seed_top_k: list[int]) -> dict[int, dict]:
    return {int(budget): dict(metrics_by_budget[int(budget)]) for budget in candidate_seed_top_k if int(budget) in metrics_by_budget}


def _select_broad_tail_budget(metrics_by_budget: dict[int, dict], policy_cfg: dict) -> dict:
    feasible = [
        int(budget)
        for budget, metrics in sorted(metrics_by_budget.items())
        if float(metrics["coverage_p25"]) >= float(policy_cfg["coverage_p25_ratio"]) * max(float(item["coverage_p25"]) for item in metrics_by_budget.values())
        and float(metrics["coverage_mean"]) >= float(policy_cfg["coverage_mean_ratio"]) * max(float(item["coverage_mean"]) for item in metrics_by_budget.values())
    ]
    ranked = sorted(
        feasible,
        key=lambda budget: (
            float(metrics_by_budget[budget]["coverage_p25"]),
            float(metrics_by_budget[budget]["coverage_mean"]),
            float(metrics_by_budget[budget]["support_mean"]),
            int(budget),
        ),
        reverse=True,
    )
    return {
        "selection_stage": "broad_tail_policy",
        "feasible_budgets": feasible,
        "resolved_seed_top_k": int(ranked[0]),
    }


def _select_compact_budget(metrics_by_budget: dict[int, dict], policy_cfg: dict) -> dict:
    feasible = [
        int(budget)
        for budget, metrics in sorted(metrics_by_budget.items())
        if float(metrics["coverage_p25"]) >= float(policy_cfg["coverage_p25_ratio"]) * max(float(item["coverage_p25"]) for item in metrics_by_budget.values())
    ]
    enriched = combine_feasible_budget_metrics(
        metrics_by_budget=metrics_by_budget,
        feasible_budgets=feasible,
        calibration_cfg={"utility": dict(policy_cfg["utility"])},
    )
    best_budget = max(feasible, key=lambda budget: float(enriched[budget]["feasible_utility"]))
    return {
        "selection_stage": "compact_structured_policy",
        "feasible_budgets": feasible,
        "resolved_seed_top_k": int(best_budget),
    }


def resolve_hierarchical_budget(*, private_lengths: list[int], metrics_by_budget: dict[int, dict], rule_cfg: dict) -> dict:
    descriptor = compute_shape_descriptor(
        private_lengths,
        tail_threshold=int(rule_cfg["router"]["tail_threshold"]),
        short_threshold=int(rule_cfg["router"]["short_threshold"]),
    )
    route = route_budget_regime(descriptor, dict(rule_cfg["router"]))
    if route.regime == "broad_tail":
        subset = _filter_budget_metrics(metrics_by_budget, list(rule_cfg["policies"]["broad_tail"]["candidate_seed_top_k"]))
        selected = _select_broad_tail_budget(subset, dict(rule_cfg["policies"]["broad_tail"]))
    elif route.regime == "compact_structured":
        subset = _filter_budget_metrics(metrics_by_budget, list(rule_cfg["policies"]["compact_structured"]["candidate_seed_top_k"]))
        selected = _select_compact_budget(subset, dict(rule_cfg["policies"]["compact_structured"]))
    else:
        selected = select_budget_by_constrained_utility(
            metrics_by_budget=metrics_by_budget,
            calibration_cfg={"coverage_constraint": dict(rule_cfg["policies"]["uncertain"]["coverage_constraint"]), "utility": dict(rule_cfg["policies"]["compact_structured"]["utility"]), "tiebreak": {"epsilon": 0.01, "prefer_smaller_budget": True}},
        )
        selected["selection_stage"] = "uncertain_fallback_policy"
    selected["regime"] = route.regime
    selected["shape_score"] = float(route.shape_score)
    selected["descriptor"] = descriptor.__dict__
    return selected
```

```python
# budget_calibration.py (new mode hook only)
from .hierarchical_budget import resolve_hierarchical_budget

def build_budget_metric_bundle(...):
    ...

elif mode == "hierarchical_shape_routing":
    selected = resolve_hierarchical_budget(
        private_lengths=private_lengths,
        metrics_by_budget=enriched_metrics,
        rule_cfg=calibration_cfg,
    )
```

- [ ] **Step 4: Run hierarchical tests and budget calibration regression**

Run:

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m pytest tests/test_hierarchical_budget.py tests/test_budget_calibration.py -q
```

Expected:

```text
PASS ... hierarchical_shape_routing mode covered
```

- [ ] **Step 5: Commit**

```bash
git -C D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19 add paper_new_selector/hierarchical_budget.py paper_new_selector/budget_calibration.py tests/test_hierarchical_budget.py tests/test_budget_calibration.py
git -C D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19 commit -m "feat: add hierarchical budget routing mode"
```

### Task 3: Wire the new mode through stage1 runner and pipeline outputs

**Files:**
- Modify: `paper-new-round19/paper_new_selector/stage1_runner.py`
- Modify: `paper-new-round19/paper_new_selector/pipeline.py`
- Test: `paper-new-round19/tests/test_stage1_runner.py`

- [ ] **Step 1: Write the failing stage1 runner test**

```python
def test_stage1_runner_uses_hierarchical_shape_routing_mode(self):
    ...
    with patch(
        "paper_new_selector.stage1_runner.resolve_seed_top_k_by_self_calibration"
    ) as calibration_mock, patch(
        "paper_new_selector.stage1_runner.resolve_seed_top_k_by_hierarchical_routing",
        return_value={
            "decision": decision,
            "seed_budget_summary": {
                "mode": "hierarchical_shape_routing",
                "resolved_seed_top_k": 22,
                "regime": "broad_tail",
                "selection_stage": "broad_tail_policy",
            },
        },
    ) as hierarchical_mock:
        summary = run_stage1("dummy.yaml", validate_only=False)

    hierarchical_mock.assert_called_once()
    calibration_mock.assert_not_called()
    assert summary["seed_budget"]["mode"] == "hierarchical_shape_routing"
    assert summary["seed_budget"]["regime"] == "broad_tail"
```

- [ ] **Step 2: Run the target test to verify it fails**

Run:

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m pytest tests/test_stage1_runner.py -k hierarchical_shape_routing -q
```

Expected:

```text
FAIL ... AttributeError or missing mode branch
```

- [ ] **Step 3: Write the minimal wiring**

```python
# stage1_runner.py
from .budget_calibration import (
    resolve_seed_top_k_by_hierarchical_routing,
    resolve_seed_top_k_by_self_calibration,
)

...
elif bool(rule_cfg.get("enabled", False)) and rule_mode == "hierarchical_shape_routing":
    calibration_result = resolve_seed_top_k_by_hierarchical_routing(
        selector_cfg=selector_cfg,
        candidate_vectors=candidate_vectors,
        candidate_texts=candidate_texts,
        private_vectors=private_vectors,
        private_lengths=private_lengths,
        private_support=private_support,
        genericity_penalty=genericity_penalty,
    )
    decision = calibration_result["decision"]
    seed_budget_summary = dict(calibration_result["seed_budget_summary"])
```

```python
# pipeline.py
if stage1_summary.get("seed_budget", {}).get("mode") in {
    "self_calibrated",
    "self_calibrated_constrained",
    "hybrid_length_family_constrained",
    "hierarchical_shape_routing",
}:
    write_json(output_root / "stage1_budget_calibration.json", stage1_summary["seed_budget"])
```

- [ ] **Step 4: Run the full selector pipeline test slice**

Run:

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m pytest tests/test_stage1_runner.py tests/test_budget_calibration.py tests/test_hierarchical_budget.py -q
```

Expected:

```text
PASS ... stage1 summary exposes hierarchical seed budget metadata
```

- [ ] **Step 5: Commit**

```bash
git -C D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19 add paper_new_selector/stage1_runner.py paper_new_selector/pipeline.py tests/test_stage1_runner.py
git -C D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19 commit -m "feat: wire hierarchical budget routing into stage1 pipeline"
```

### Task 4: Add round-19 quick-comparison config family and summary scripts

**Files:**
- Create: `paper-new-round19/configs/experiments/single_node_tuning_round19/...`
- Create: `paper-new-round19/scripts/append_round19_summary.py`
- Create: `paper-new-round19/scripts/run_round19_guard_batch.sh`
- Create: `paper-new-round19/scripts/run_round19_quick_compare.sh`

- [ ] **Step 1: Add the base config and leaf configs**

```yaml
# configs/experiments/single_node_tuning_round19/_base_selector_tuning_round19.yaml
inherits:
  - ../single_node_screening/_base_selector_screening.yaml

meta:
  stage: single_node_tuning_round19
  seed: 42

bootstrap:
  max_tokens: 85

selector:
  seed_budget_rule:
    enabled: true
    mode: hierarchical_shape_routing
    candidate_seed_top_k: [18, 19, 20, 21, 22]
    router:
      tail_threshold: 350
      short_threshold: 120
      zscore_source: screening_reference
      tau_center: 0.0
      delta_router: 0.35
    policies:
      broad_tail:
        candidate_seed_top_k: [21, 22]
        coverage_p25_ratio: 0.98
        coverage_mean_ratio: 0.998
        epsilon: 0.002
      compact_structured:
        candidate_seed_top_k: [18, 19, 20]
        coverage_p25_ratio: 0.98
        utility:
          support_weight: 1.0
          genericity_weight: 0.5
          redundancy_weight: 0.3
          budget_weight: 0.1
        epsilon: 0.01
      uncertain:
        fallback_mode: self_calibrated_constrained
        coverage_constraint:
          mode: tail_family_relative
          metrics:
            - name: coverage_p25
              relative_ratio: 0.98
              required: true
              weight: 0.7
            - name: coverage_mean
              relative_ratio: 0.998
              required: true
              weight: 0.3
```

- [ ] **Step 2: Add the summary extractor**

```python
# scripts/append_round19_summary.py
from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_experiment(output_root: Path) -> dict:
    stage1 = load_json(output_root / "stage1_budget_calibration.json")
    eval_summary = load_json(output_root / "eval" / "downstream_eval_summary.json")
    return {
        "experiment_id": output_root.name,
        "mode": stage1["mode"],
        "regime": stage1.get("regime"),
        "selection_stage": stage1.get("selection_stage"),
        "resolved_seed_top_k": stage1["resolved_seed_top_k"],
        "shape_score": stage1.get("shape_score"),
        "best_top1": eval_summary["best_top1"],
        "best_top3": eval_summary["best_top3"],
        "best_top5": eval_summary["best_top5"],
        "best_top10": eval_summary["best_top10"],
    }
```

- [ ] **Step 3: Add the guard and quick-compare shell entrypoints**

```bash
# scripts/run_round19_guard_batch.sh
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/guard/r19_guard_forums.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/guard/r19_guard_congressional.yaml
```

```bash
# scripts/run_round19_quick_compare.sh
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/full_run/r19_full_jobs.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/full_run/r19_full_congressional.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/full_run/r19_full_forums.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/full_run/r19_full_microblog.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/ablations/r19_ablate_no_router_forums.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/ablations/r19_ablate_no_router_congressional.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/ablations/r19_ablate_force_compact_forums.yaml
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/ablations/r19_ablate_force_broad_congressional.yaml
```

- [ ] **Step 4: Smoke-run config resolution**

Run:

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/guard/r19_guard_forums.yaml --validate-only
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round19/guard/r19_guard_congressional.yaml --validate-only
```

Expected:

```text
JSON summary containing "mode": "selector_seed_search" and hierarchical seed_budget metadata
```

- [ ] **Step 5: Commit**

```bash
git -C D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19 add configs/experiments/single_node_tuning_round19 scripts/append_round19_summary.py scripts/run_round19_guard_batch.sh scripts/run_round19_quick_compare.sh
git -C D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19 commit -m "feat: add round19 hierarchical quick-compare configs"
```

### Task 5: Run verification and the quick-comparison suite

**Files:**
- Modify: none
- Run only

- [ ] **Step 1: Run the full test suite for touched modules**

Run:

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
python -m pytest tests/test_shape_descriptor.py tests/test_regime_router.py tests/test_hierarchical_budget.py tests/test_budget_calibration.py tests/test_stage1_runner.py -q
```

Expected:

```text
All selected tests pass
```

- [ ] **Step 2: Run the guard batch**

Run:

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
bash scripts/run_round19_guard_batch.sh
```

Expected:

```text
Outputs written to paper-new-round19/outputs/r19_guard_forums and paper-new-round19/outputs/r19_guard_congressional
```

- [ ] **Step 3: Inspect guard outputs before full quick compare**

Run:

```bash
python scripts/append_round19_summary.py
```

Expected:

```text
Guard summary rows showing regime, resolved_seed_top_k, and best_top1 for both experiments
```

- [ ] **Step 4: Run full quick compare only if guard behavior matches expectations**

Run:

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19
bash scripts/run_round19_quick_compare.sh
```

Expected:

```text
Summary rows for 4 full-run experiments and 4 ablation experiments
```

- [ ] **Step 5: Commit experiment configs/scripts and plan adjustments if needed**

```bash
git -C D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19 status --short
git -C D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-round19 commit -am "chore: verify round19 hierarchical quick-compare suite"
```

---

## Self-Review

### Spec coverage

- The hierarchical design spec requires:
  - a lightweight shape descriptor
  - a continuous router with frozen development protocol
  - a policy-conditioned budget resolver
  - integration into the existing Stage-1 budget stack
  - quick-comparison experiments plus ablations
- Tasks 1-3 cover the implementation pieces.
- Task 4 covers the experiment configs and summary tooling.
- Task 5 covers verification and execution order.

### Placeholder scan

- No `TODO` / `TBD` markers remain.
- All config file paths are explicit.
- Every execution step has a concrete command.
- Every code step includes actual code rather than prose-only placeholders.

### Type consistency

- New mode string is consistently `hierarchical_shape_routing`.
- Router output uses `broad_tail`, `compact_structured`, and `uncertain`.
- Guard/full/ablation config names consistently use `r19_*`.

---

## Recommended execution order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5

The reason to keep this order is simple:

- Task 0 prepares `paper-new-round19` and ports the advanced budget stack.
- Task 1 creates the descriptor/router primitives.
- Task 2 adds the hierarchical resolver.
- Task 3 exposes the new mode through the actual pipeline.
- Task 4 creates the configs only after the mode exists.
- Task 5 spends compute only after code and config validation succeed.
