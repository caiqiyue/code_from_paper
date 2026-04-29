# Round17 Constrained Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Round17 constrained utility budget selection so Stage1 first enforces relative `coverage_p25` sufficiency, then optimizes compactness-aware utility over feasible budgets.

**Architecture:** Reuse the existing Round16 per-budget metric generation pipeline in `budget_calibration.py`, add a new `self_calibrated_constrained` mode with a feasibility stage and a feasible-set utility stage, then expose the new decision trace through `stage1_budget_calibration.json`. Keep Stage2 and eval unchanged so Round17 remains comparable to Round16 and Round15.

**Tech Stack:** Python 3.13, existing `paper_new_selector` pipeline, YAML experiment configs, `pytest`, local config loading through `thesis_bridge`.

---

## File Map

**Modify**
- `paper-new-round-16/paper_new_selector/budget_calibration.py`
  - Add constrained-selection helpers.
  - Add new mode `self_calibrated_constrained`.
  - Keep Round16 and Round16.5 paths intact.
- `paper-new-round-16/paper_new_selector/stage1_runner.py`
  - Route `seed_budget_rule.mode=self_calibrated_constrained` through the updated calibration entrypoint.
- `paper-new-round-16/tests/test_budget_calibration.py`
  - Add failing tests for coverage-feasible set selection, constrained utility ranking, and fallback behavior.
- `paper-new-round-16/tests/test_stage1_runner.py`
  - Add integration coverage that Stage1 uses the constrained mode path and writes the new summary fields.

**Create**
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/_base_selector_tuning_round17.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/_ratio_r099.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/_ratio_r098.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/_ratio_r097.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/probes/r17_probe_forums_base.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/probes/r17_probe_microblog_base.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/ratio_sweep/r17_forums_r099.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/ratio_sweep/r17_forums_r098.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/ratio_sweep/r17_forums_r097.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/full_run/r17_full_jobs.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/full_run/r17_full_congressional.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/full_run/r17_full_forums.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/full_run/r17_full_microblog.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/seed_sanity/r17_forums_seed123.yaml`
- `paper-new-round-16/configs/experiments/single_node_tuning_round17/seed_sanity/r17_forums_seed456.yaml`

---

### Task 1: Add failing constrained-selection unit tests

**Files:**
- Modify: `paper-new-round-16/tests/test_budget_calibration.py`
- Test: `paper-new-round-16/tests/test_budget_calibration.py`

- [ ] **Step 1: Write the failing tests for feasibility filtering**

```python
def test_select_feasible_budgets_by_coverage_p25_filters_by_relative_ratio(self):
    metrics_by_budget = {
        18: {"coverage_p25": 0.80},
        19: {"coverage_p25": 0.90},
        20: {"coverage_p25": 0.91},
        22: {"coverage_p25": 0.92},
    }
    summary = select_feasible_budgets_by_coverage_p25(
        metrics_by_budget=metrics_by_budget,
        calibration_cfg={"coverage_constraint": {"metric": "coverage_p25", "relative_ratio": 0.99}},
    )
    self.assertEqual(summary["feasible_budgets"], [20, 22])
```

- [ ] **Step 2: Write the failing tests for feasible-set utility selection**

```python
def test_select_budget_by_constrained_utility_chooses_best_feasible_budget(self):
    result = select_budget_by_constrained_utility(
        metrics_by_budget={
            18: {"coverage_p25": 0.80, "support_score": 0.90, "support_mean": 0.90, "genericity_score": 0.20, "redundancy_score": 0.20, "budget_cost": 0.0, "coverage_mean": 0.90},
            20: {"coverage_p25": 0.91, "support_score": 0.88, "support_mean": 0.88, "genericity_score": 0.15, "redundancy_score": 0.10, "budget_cost": 0.5, "coverage_mean": 0.93},
            22: {"coverage_p25": 0.92, "support_score": 0.84, "support_mean": 0.84, "genericity_score": 0.14, "redundancy_score": 0.09, "budget_cost": 1.0, "coverage_mean": 0.94},
        },
        calibration_cfg={
            "coverage_constraint": {"metric": "coverage_p25", "relative_ratio": 0.99},
            "utility": {"support_weight": 1.0, "genericity_weight": 0.5, "redundancy_weight": 0.3, "budget_weight": 0.1},
            "tiebreak": {"epsilon": 0.01, "prefer_smaller_budget": True},
        },
    )
    self.assertEqual(result["resolved_seed_top_k"], 20)
    self.assertEqual(result["coverage_constraint"]["feasible_budgets"], [20, 22])
```

- [ ] **Step 3: Write the failing fallback test**

```python
def test_select_budget_by_constrained_utility_falls_back_when_no_budget_is_feasible(self):
    result = select_budget_by_constrained_utility(
        metrics_by_budget={
            18: {"coverage_p25": 0.80, "support_score": 0.90, "support_mean": 0.90, "genericity_score": 0.20, "redundancy_score": 0.20, "budget_cost": 0.0, "coverage_mean": 0.90, "utility": 0.60},
            20: {"coverage_p25": 0.79, "support_score": 0.89, "support_mean": 0.89, "genericity_score": 0.18, "redundancy_score": 0.18, "budget_cost": 0.5, "coverage_mean": 0.91, "utility": 0.55},
        },
        calibration_cfg={
            "coverage_constraint": {"metric": "coverage_p25", "relative_ratio": 1.01},
            "utility": {"support_weight": 1.0, "genericity_weight": 0.5, "redundancy_weight": 0.3, "budget_weight": 0.1},
            "tiebreak": {"epsilon": 0.01, "prefer_smaller_budget": True},
        },
    )
    self.assertTrue(result["fallback_used"])
    self.assertEqual(result["selection_stage"], "fallback_argmax_utility")
```

- [ ] **Step 4: Run the budget calibration tests and confirm failure**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round-16
python -m pytest tests/test_budget_calibration.py -q
```

Expected: FAIL with missing constrained-selection helpers.

- [ ] **Step 5: Commit the red tests**

```bash
git -C /Users/apple/Desktop/code_from_paper/paper-new-round-16 add tests/test_budget_calibration.py
git -C /Users/apple/Desktop/code_from_paper/paper-new-round-16 commit -m "test: add round17 constrained selection coverage"
```

---

### Task 2: Implement constrained utility selection in `budget_calibration.py`

**Files:**
- Modify: `paper-new-round-16/paper_new_selector/budget_calibration.py`
- Test: `paper-new-round-16/tests/test_budget_calibration.py`

- [ ] **Step 1: Add feasibility helpers**

```python
def compute_relative_coverage_threshold(*, best_coverage_p25: float, relative_ratio: float) -> float:
    return float(best_coverage_p25) * float(relative_ratio)


def select_feasible_budgets_by_coverage_p25(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    calibration_cfg: dict[str, Any],
) -> dict[str, Any]:
    coverage_cfg = dict(calibration_cfg.get("coverage_constraint", {}))
    relative_ratio = float(coverage_cfg.get("relative_ratio", 0.99))
    best_coverage_p25 = max(float(metrics["coverage_p25"]) for metrics in metrics_by_budget.values())
    threshold = compute_relative_coverage_threshold(
        best_coverage_p25=best_coverage_p25,
        relative_ratio=relative_ratio,
    )
    feasible_budgets = [
        int(budget)
        for budget, metrics in sorted(metrics_by_budget.items())
        if float(metrics["coverage_p25"]) >= threshold
    ]
    return {
        "metric": "coverage_p25",
        "relative_ratio": relative_ratio,
        "best_coverage_p25": float(best_coverage_p25),
        "threshold": float(threshold),
        "feasible_budgets": feasible_budgets,
    }
```

- [ ] **Step 2: Add feasible-set utility computation**

```python
def combine_feasible_budget_metrics(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    feasible_budgets: list[int],
    calibration_cfg: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    utility_cfg = dict(calibration_cfg.get("utility", {}))
    support_weight = float(utility_cfg.get("support_weight", 1.0))
    genericity_weight = float(utility_cfg.get("genericity_weight", 0.5))
    redundancy_weight = float(utility_cfg.get("redundancy_weight", 0.3))
    budget_weight = float(utility_cfg.get("budget_weight", 0.1))
    subset = {int(b): dict(metrics_by_budget[int(b)]) for b in feasible_budgets}
    normalized_support = _normalize_metric_series({b: float(subset[b]["support_score"]) for b in subset})
    normalized_genericity = _normalize_metric_series({b: float(subset[b]["genericity_score"]) for b in subset})
    normalized_redundancy = _normalize_metric_series({b: float(subset[b]["redundancy_score"]) for b in subset})
    normalized_budget_cost = _normalize_metric_series({b: float(subset[b]["budget_cost"]) for b in subset})
    for budget in subset:
        subset[budget]["feasible_normalized_metrics"] = {
            "support_score": normalized_support[budget],
            "genericity_score": normalized_genericity[budget],
            "redundancy_score": normalized_redundancy[budget],
            "budget_cost": normalized_budget_cost[budget],
        }
        subset[budget]["feasible_utility"] = (
            support_weight * normalized_support[budget]
            - genericity_weight * normalized_genericity[budget]
            - redundancy_weight * normalized_redundancy[budget]
            - budget_weight * normalized_budget_cost[budget]
        )
    return subset
```

- [ ] **Step 3: Add final constrained selector**

```python
def select_budget_by_constrained_utility(
    *,
    metrics_by_budget: dict[int, dict[str, Any]],
    calibration_cfg: dict[str, Any],
) -> dict[str, Any]:
    coverage_constraint = select_feasible_budgets_by_coverage_p25(
        metrics_by_budget=metrics_by_budget,
        calibration_cfg=calibration_cfg,
    )
    feasible_budgets = list(coverage_constraint["feasible_budgets"])
    if not feasible_budgets:
        fallback = _select_budget_with_tiebreak(
            metrics_by_budget=metrics_by_budget,
            calibration_cfg=calibration_cfg,
        )
        fallback["coverage_constraint"] = coverage_constraint
        fallback["selection_stage"] = "fallback_argmax_utility"
        fallback["fallback_used"] = True
        return fallback
    feasible_metrics = combine_feasible_budget_metrics(
        metrics_by_budget=metrics_by_budget,
        feasible_budgets=feasible_budgets,
        calibration_cfg=calibration_cfg,
    )
    selected = _select_budget_from_feasible_metrics(
        metrics_by_budget=feasible_metrics,
        calibration_cfg=calibration_cfg,
    )
    selected["coverage_constraint"] = coverage_constraint
    selected["selection_stage"] = "feasible_set_utility"
    selected["fallback_used"] = False
    return selected
```

- [ ] **Step 4: Route `self_calibrated_constrained` through the new path**

```python
mode = str(calibration_cfg.get("mode", "self_calibrated"))
if mode == "self_calibrated":
    selected = select_budget_with_recheck(...)
elif mode == "self_calibrated_constrained":
    selected = select_budget_by_constrained_utility(
        metrics_by_budget=enriched_metrics,
        calibration_cfg=calibration_cfg,
    )
else:
    raise ValueError(f"Unsupported seed_budget_rule.mode: {mode}")
```

- [ ] **Step 5: Extend summary payload**

```python
"coverage_constraint": dict(selected.get("coverage_constraint", {})),
"selection_stage": str(selected.get("selection_stage", "argmax_utility")),
"fallback_used": bool(selected.get("fallback_used", False)),
```

- [ ] **Step 6: Run the targeted tests**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round-16
python -m pytest tests/test_budget_calibration.py -q
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git -C /Users/apple/Desktop/code_from_paper/paper-new-round-16 add paper_new_selector/budget_calibration.py tests/test_budget_calibration.py
git -C /Users/apple/Desktop/code_from_paper/paper-new-round-16 commit -m "feat: add round17 constrained utility selection"
```

---

### Task 3: Add Stage1 integration coverage

**Files:**
- Modify: `paper-new-round-16/tests/test_stage1_runner.py`
- Test: `paper-new-round-16/tests/test_stage1_runner.py`

- [ ] **Step 1: Add failing Stage1 constrained-mode integration test**

```python
def test_stage1_runner_uses_self_calibrated_constrained_mode(self):
    config["selector"]["seed_budget_rule"] = {
        "enabled": True,
        "mode": "self_calibrated_constrained",
        "candidate_seed_top_k": [18, 19, 20, 21, 22],
        "coverage_constraint": {"metric": "coverage_p25", "relative_ratio": 0.99},
    }
    with patch(
        "paper_new_selector.stage1_runner.resolve_seed_top_k_by_self_calibration",
        return_value={
            "decision": decision,
            "seed_budget_summary": {
                "configured_seed_top_k": 20,
                "resolved_seed_top_k": 20,
                "mode": "self_calibrated_constrained",
                "coverage_constraint": {"feasible_budgets": [20, 22]},
                "selection_stage": "feasible_set_utility",
                "fallback_used": False,
                "per_budget_metrics": {},
            },
        },
    ) as calibration_mock:
        summary = run_stage1("dummy.yaml", validate_only=False)
    calibration_mock.assert_called_once()
    assert summary["seed_budget"]["mode"] == "self_calibrated_constrained"
    assert summary["seed_budget"]["coverage_constraint"]["feasible_budgets"] == [20, 22]
```

- [ ] **Step 2: Run the Stage1 test to confirm failure first**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round-16
python -m pytest tests/test_stage1_runner.py::Stage1RunnerReleaseTests::test_stage1_runner_uses_self_calibrated_constrained_mode -q
```

Expected: FAIL until the mocked summary shape is asserted in code.

- [ ] **Step 3: Adjust assertions or code if needed, then rerun**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round-16
python -m pytest tests/test_stage1_runner.py -q
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git -C /Users/apple/Desktop/code_from_paper/paper-new-round-16 add tests/test_stage1_runner.py
git -C /Users/apple/Desktop/code_from_paper/paper-new-round-16 commit -m "test: cover round17 constrained stage1 summary"
```

---

### Task 4: Create Round17 experiment configs

**Files:**
- Create: `paper-new-round-16/configs/experiments/single_node_tuning_round17/*`
- Test: config loading via `paper_new_selector.thesis_bridge.load_yaml_config`

- [ ] **Step 1: Create the base Round17 config**

```yaml
inherits:
  - ../single_node_tuning_round16/_base_selector_tuning_round16.yaml

meta:
  stage: single_node_tuning_round17
  seed: 42

selector:
  seed_budget_rule:
    enabled: true
    mode: self_calibrated_constrained
    candidate_seed_top_k: [18, 19, 20, 21, 22]
    coverage_constraint:
      metric: coverage_p25
      relative_ratio: 0.99
    utility:
      support_weight: 1.0
      genericity_weight: 0.5
      redundancy_weight: 0.3
      budget_weight: 0.1
    tiebreak:
      epsilon: 0.01
      prefer_smaller_budget: true
```

- [ ] **Step 2: Create ratio sweep partials**

```yaml
# _ratio_r098.yaml
selector:
  seed_budget_rule:
    coverage_constraint:
      relative_ratio: 0.98
```

```yaml
# _ratio_r097.yaml
selector:
  seed_budget_rule:
    coverage_constraint:
      relative_ratio: 0.97
```

- [ ] **Step 3: Create probe configs**

```yaml
inherits:
  - ../_base_selector_tuning_round17.yaml
  - ../../single_node_tuning_round16/_data_forums.yaml

meta:
  experiment_id: r17_probe_forums_base

paths:
  output_root: paper-new/outputs/r17_probe_forums_base
```

```yaml
inherits:
  - ../_base_selector_tuning_round17.yaml
  - ../../single_node_tuning_round16/_data_microblog.yaml

meta:
  experiment_id: r17_probe_microblog_base

paths:
  output_root: paper-new/outputs/r17_probe_microblog_base
```

- [ ] **Step 4: Create ratio sweep and full-run configs**

```yaml
inherits:
  - ../_base_selector_tuning_round17.yaml
  - ../_ratio_r099.yaml
  - ../../single_node_tuning_round16/_data_forums.yaml

meta:
  experiment_id: r17_forums_r099

paths:
  output_root: paper-new/outputs/r17_forums_r099
```

```yaml
inherits:
  - ../_base_selector_tuning_round17.yaml
  - ../../single_node_tuning_round16/_data_jobs.yaml

meta:
  experiment_id: r17_full_jobs

paths:
  output_root: paper-new/outputs/r17_full_jobs
```

- [ ] **Step 5: Create seed sanity configs**

```yaml
inherits:
  - ../_base_selector_tuning_round17.yaml
  - ../../single_node_tuning_round16/_data_forums.yaml

meta:
  experiment_id: r17_forums_seed123
  seed: 123

paths:
  output_root: paper-new/outputs/r17_forums_seed123
```

- [ ] **Step 6: Verify all Round17 YAML files load**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round-16
python - <<'PY'
from pathlib import Path
from paper_new_selector.thesis_bridge import load_yaml_config
root = Path("configs/experiments/single_node_tuning_round17")
files = sorted(root.rglob("*.yaml"))
for path in files:
    load_yaml_config(path)
print(f"loaded {len(files)} yaml files")
PY
```

Expected: `loaded 15 yaml files`

- [ ] **Step 7: Commit**

```bash
git -C /Users/apple/Desktop/code_from_paper/paper-new-round-16 add configs/experiments/single_node_tuning_round17
git -C /Users/apple/Desktop/code_from_paper/paper-new-round-16 commit -m "feat: add round17 constrained utility configs"
```

---

### Task 5: Run full local verification

**Files:**
- Modify: none expected
- Test: whole local tree

- [ ] **Step 1: Run the full test suite**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round-16
python -m pytest tests
```

Expected: all tests PASS.

- [ ] **Step 2: Run Python syntax verification**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new-round-16
python -m py_compile paper_new_selector/*.py
```

Expected: no output, exit code 0.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git -C /Users/apple/Desktop/code_from_paper/paper-new-round-16 status --short
```

Expected: only intended Round17 code/config changes plus pre-existing unrelated files.

- [ ] **Step 4: Commit final local verification snapshot**

```bash
git -C /Users/apple/Desktop/code_from_paper/paper-new-round-16 add paper_new_selector tests configs/experiments/single_node_tuning_round17
git -C /Users/apple/Desktop/code_from_paper/paper-new-round-16 commit -m "chore: finalize round17 constrained utility implementation"
```

---

## Spec Coverage Check

- Constrained selection with `coverage_p25` relative feasibility: Task 2
- Summary trace with coverage constraint and selection stage: Task 2 and Task 3
- Round17 config tree for probes, ratio sweep, full run, seed sanity: Task 4
- Regression verification: Task 5

## Placeholder Scan

- No `TODO`, `TBD`, “implement later”, or unresolved file paths remain.
- Each code-changing task includes concrete code blocks and exact commands.

## Type Consistency

- New mode string is consistently `self_calibrated_constrained`
- Coverage constraint key is consistently `coverage_constraint`
- Summary fields are consistently `feasible_budgets`, `selection_stage`, and `fallback_used`

## Execution Handoff

Plan saved to:

`/Users/apple/Desktop/code_from_paper/paper-new/docs/2026-04-29-round17-constrained-utility-implementation-plan.md`

Execution assumption for this session:

- The user already requested direct development, so proceed with **Inline Execution** semantics from this plan in the current session.
