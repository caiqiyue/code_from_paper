# Round18 Tail-Family Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Round17 budget calibration into a configurable Round18 tail-coverage-family framework with experimental configs and launch scripts.

**Architecture:** Keep the existing `self_calibrated_constrained` pipeline entrypoint intact, but generalize the coverage constraint layer so it can evaluate multiple coverage metrics and optionally apply a conservative larger-budget recheck. Pair the code change with Round18 experiment configs and summary scripts so the new mechanism is immediately runnable.

**Tech Stack:** Python, YAML experiment configs, shell launch scripts, unittest

---

### Task 1: Generalize the constrained feasibility layer

**Files:**
- Modify: `paper_new_selector/budget_calibration.py`
- Test: `tests/test_budget_calibration.py`

- [ ] Add helper functions to normalize coverage metric specs and compute family scores.
- [ ] Keep `select_feasible_budgets_by_coverage_p25(...)` working as a compatibility wrapper.
- [ ] Add a new generalized selector that can read a metrics family from `coverage_constraint.metrics`.
- [ ] Ensure the emitted summary still includes `metric`, `relative_ratio`, `threshold`, and `feasible_budgets`, plus new per-metric trace fields.

### Task 2: Add conservative constrained recheck

**Files:**
- Modify: `paper_new_selector/budget_calibration.py`
- Test: `tests/test_budget_calibration.py`

- [ ] Add a `constrained_recheck` block that only activates when explicitly enabled in config.
- [ ] Compare the selected feasible budget against larger feasible budgets using support-drop and coverage-gain guards.
- [ ] Record recheck traces in `stage1_budget_calibration.json`.
- [ ] Add unit tests covering both “promote larger budget” and “keep current budget” cases.

### Task 3: Add Round18 experiment configs

**Files:**
- Create: `configs/experiments/single_node_tuning_round18/_base_selector_tuning_round18.yaml`
- Create: `configs/experiments/single_node_tuning_round18/_family_f1.yaml`
- Create: `configs/experiments/single_node_tuning_round18/_family_f2_loose_recheck.yaml`
- Create: `configs/experiments/single_node_tuning_round18/_family_f3_balanced_recheck.yaml`
- Create: `configs/experiments/single_node_tuning_round18/probes/*.yaml`
- Create: `configs/experiments/single_node_tuning_round18/focus/*.yaml`
- Create: `configs/experiments/single_node_tuning_round18/full_run/*.yaml`
- Create: `configs/experiments/single_node_tuning_round18/seed_sanity/*.yaml`

- [ ] Build a Round18 base config inheriting the Round17 selector foundation.
- [ ] Add focused variants for `tail family only`, `loose recheck`, and `balanced recheck`.
- [ ] Create probe configs for `forums` and `congressional`.
- [ ] Create congressional-only focus configs.
- [ ] Create a full-regression mainline that can be reused across all four datasets.

### Task 4: Add launch and summary scripts

**Files:**
- Create: `scripts/append_round18_summary.py`
- Create: `scripts/run_round18_probe_batch.sh`
- Create: `scripts/run_round18_full_regression.sh`

- [ ] Add a summary appender that reads both coverage-family trace and constrained-recheck trace.
- [ ] Add a probe launcher for the structural and congressional-focused runs.
- [ ] Add a full regression launcher for the selected Round18 mainline.

### Task 5: Add Round18 docs

**Files:**
- Create: `/Users/apple/Desktop/code_from_paper/paper-new/docs/2026-04-30-round18-tail-coverage-family-design.md`

- [ ] Document the Round17 failure mode, Round18 hypothesis, algorithm definition, and staged experiment plan.
- [ ] Make sure the doc explains why Round18 is aimed specifically at the `congressional` failure without becoming a dataset-name rule.

### Task 6: Verify locally

**Files:**
- Test: `tests/test_budget_calibration.py`

- [ ] Run `python -m unittest tests.test_budget_calibration`
- [ ] Fix any failing compatibility assumptions introduced by the new Round18 fields.
