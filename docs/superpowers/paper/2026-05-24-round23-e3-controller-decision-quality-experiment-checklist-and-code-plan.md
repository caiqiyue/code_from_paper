# E3 Controller Decision Quality Checklist and Code Plan

Updated: 2026-05-24

## 1. Goal and Non-goals

### 1.1 E3 Goal

E3 is the controller decision quality experiment for `round23`.

It should answer this mechanism question:

> Does the learned `round23` controller make better local seed-budget decisions than a fixed `k0=20` anchor and the rule-based `round19 resolver replay`, when all policies are evaluated on the same offline `k=18..22` local sweep?

The experimental unit is one controller context:

```text
dataset_name + meta_seed
```

The action space is:

```text
delta_k in {-2, -1, 0, +1, +2}
target_budget = 20 + delta_k
```

The core proof is:

- `round23 controller` has lower regret than `keep-k0=20`.
- `round23 controller` is closer to `oracle budget` than, or at least competitive with, `round19 resolver replay`.
- `round23 controller` uses the local action space instead of collapsing to a trivial constant action.
- Improvements are not caused by only one dataset.

### 1.2 What E3 Should Not Do

E3 should not repeat the E1 method-level end-to-end comparison against `PrE-Text`, `round19`, `WASP`, or `DPGA-TextSyn`.

E3 should not primarily prove one-shot absolute-`k` prediction versus two-round `delta_k`. That belongs to the later round-count or formulation justification experiment.

E3 should not evaluate arbitrary-anchor generalization. It only evaluates the formal anchored-local setup around `k0=20`.

E3 should not use E2 held-out datasets unless they also have complete `k=18..22` local sweeps and matching controller context/replay tables.

## 2. Data Sources and Existing Fields

### 2.1 Primary Input Files

Use the completed 1200 all-six collection artifact:

```text
model-train/artifacts/round23_datasets_1200_top1_delta_m0005/round23_controller_context_table.jsonl
model-train/artifacts/round23_datasets_1200_top1_delta_m0005/round19_replay_table.jsonl
model-train/artifacts/round23_datasets_1200_top1_delta_m0005/round19_replay_mapping.json
```

Observed structure from the current files:

| File | Contexts | Unique contexts | Dataset coverage |
|---|---:|---:|---|
| `round23_controller_context_table.jsonl` | 240 | 240 | 6 datasets x 40 contexts |
| `round19_replay_table.jsonl` | 240 | 240 | 6 datasets x 40 contexts |
| `round19_replay_mapping.json` | 240 | 240 | `{context_id: budget}` mapping |

Datasets:

```text
congressional, forums, imdb, jobs, microblog, openreview
```

### 2.2 Required Context Fields

Identity and scope fields:

```text
context_id
dataset_name
meta_seed
reference_budget
label_target_mode
tie_margin
```

State feature fields:

```text
shape_score
shape_regime
private_mean_length
private_p75_length
private_length_iqr
support_mean_at_k20
coverage_mean_at_k20
coverage_p25_at_k20
genericity_mean_at_k20
redundancy_mean_at_k20
```

Action reward fields for the E3 primary reward space:

```text
controller_reward_dk_neg2
controller_reward_dk_neg1
controller_reward_dk_0
controller_reward_dk_pos1
controller_reward_dk_pos2
```

Action `best_top1` fields:

```text
best_top1_dk_neg2
best_top1_dk_neg1
best_top1_dk_0
best_top1_dk_pos1
best_top1_dk_pos2
```

Oracle and fixed-anchor fields:

```text
keep_k0_reward
keep_k0_training_value
oracle_best_delta_k
oracle_best_target_budget
oracle_best_controller_reward
oracle_best_training_value
oracle_best_top1
best_top1_at_k20
```

Round19 replay fields:

```text
round19_predicted_budget
round19_predicted_delta_k
round19_replay_best_top1
round19_regime
round19_shape_score
```

Important semantic rule:

`round19_replay_table.round19_replay_reward` was computed with the legacy reward formula in `build_round23_controller_dataset.py`. For E3 primary comparisons, compute `round19` reward by using `round19_predicted_delta_k` to select the corresponding `controller_reward_dk_*` field from `round23_controller_context_table.jsonl`. Compute secondary `round19` `best_top1` in the same way, by selecting the matching `best_top1_dk_*` field from the context table. Treat `round19_replay_reward` and `round19_replay_best_top1` as audit/cross-check fields only; do not mix `round19_replay_reward` with the new `top1_delta` controller reward.

### 2.3 Model Inputs

Candidate all-six trained model:

```text
model-train/artifacts/round23_models_1200_all6_final/extratrees/no_dataset/
```

Matching runtime bundle:

```text
paper-new-round23/artifacts/controller_bundle/round23_controller_1200_all6_top1_delta_m0005_extratrees_no_dataset/
```

Observed model metadata:

```text
model_family = extratrees
feature_version = no_dataset
target_field = target_value_for_training
target_mode = top1_delta
tie_margin = 0.0005
reference_budget = 20
delta_actions = [-2, -1, 0, 1, 2]
total_features = 9
```

## 3. Current Code Support

### 3.1 Dataset Construction Support

`model-train/build_round23_controller_dataset.py` already supports:

- Building `round23_controller_samples.jsonl` with one row per context/action.
- Building `round23_controller_context_table.jsonl` with all five local action rewards and oracle labels.
- Building `round19_replay_table.jsonl` and `round19_replay_mapping.json`.
- Recording `label_target_mode=top1_delta` and `tie_margin=0.0005`.
- Recording both current target values and legacy controller reward values.

### 3.2 Round19 Replay Support

`model-train/round19_replay.py` already supports:

- Loading explicit replay mapping from `round19_replay_mapping.json`.
- Loading replay mapping from `round19_replay_table.jsonl`.
- Validating replay budget membership against `BUDGETS=[18,19,20,21,22]`.
- Reconstructing rule-based round19 replay from collection summaries when needed.

### 3.3 Feature and Model Support

`model-train/features.py` and `model-train/round23_feature_sets.py` already support:

- The formal nine state features used by the no-dataset all-six controller.
- Optional dataset one-hot feature encoding.
- `with_dataset` and `no_dataset` feature versions.

`model-train/train_round23_controller.py` already supports:

- Training one regressor per `delta_k`.
- Selecting hyperparameters by CV folds.
- Saving model files and `train_metrics.json`.

### 3.4 Existing Evaluation Support

`model-train/eval_round23_controller.py` already supports:

- Loading trained regressors from a model directory.
- Predicting rewards for all `delta_k` actions.
- Selecting `predicted_delta_k`.
- Computing per-context predicted reward and regret against oracle.
- Computing exact `delta_k` accuracy and direction accuracy for `round23`.
- Optionally loading round19 replay mapping/table and adding round19 selected budget/delta/reward to `per_context`.
- Writing `round23_controller_eval_report.json`.

Limitations for E3:

- It reports mainly `round23 controller` plus auxiliary `round19` fields, not a normalized four-policy E3 table.
- `win_rate_vs_keep_k0` is currently based on `best_top1`, while E3 primary metric needs reward-based win rate.
- It does not output Table E3-1/E3-2/E3-3 as CSV/Markdown.
- It does not compute policy rows for `keep-k0=20`, `round19 resolver replay`, `round23 controller`, and `oracle budget` with identical metric code.
- It does not compute action distributions for all policies.
- It does not compute majority-direction baseline.
- It does not provide a dedicated audit report for missing contexts, duplicate contexts, invalid deltas, reward-field consistency, and join completeness.

### 3.5 Runtime Support

`paper-new-round23/scripts/round23_controller_inference.py` supports deployed controller inference from a runtime bundle.

`paper-new-round23/scripts/run_round23_with_dynamic_controller.py` supports actual two-round runtime execution with the dynamic controller.

For E3, runtime execution is not required because all policy evaluation should be offline on the completed `k=18..22` sweep.

## 4. Experiment List

### 4.1 E3-A Overall Policy Quality

Experiment name:

```text
E3-A overall_policy_quality
```

Data source:

```text
round23_controller_context_table.jsonl
round19_replay_table.jsonl
trained all6 round23 controller model or existing eval report
```

Policies compared:

```text
keep-k0=20
round19 resolver replay
round23 controller
oracle budget
```

Metrics:

```text
contexts
mean_reward
mean_regret_vs_oracle
win_rate_vs_keep_k0_by_reward
win_rate_vs_round19_by_reward
direction_accuracy
delta_k_accuracy
mean_best_top1_regret
```

Output table:

```text
Table E3-1: Overall Controller Policy Quality
```

Proof point:

Shows whether `round23` improves over the fixed anchor and how close it is to the offline oracle and rule-based `round19`.

### 4.2 E3-B Dataset-wise Policy Quality

Experiment name:

```text
E3-B datasetwise_policy_quality
```

Data source:

Same joined per-context policy table as E3-A.

Policies compared:

```text
round19 resolver replay
round23 controller
```

Optional include:

```text
keep-k0=20
oracle budget
```

Metrics:

```text
contexts
mean_reward
mean_regret_vs_oracle
win_rate_vs_keep_k0_by_reward
direction_accuracy
delta_k_accuracy
mean_best_top1_regret
```

Output table:

```text
Table E3-2: Dataset-wise Controller Quality
```

Proof point:

Checks whether results are stable across `jobs`, `congressional`, `forums`, `microblog`, `imdb`, and `openreview`, rather than being driven by one dataset.

### 4.3 E3-C Action Distribution

Experiment name:

```text
E3-C action_distribution
```

Data source:

Same joined per-context policy table as E3-A.

Policies compared:

```text
oracle budget
round19 resolver replay
round23 controller
```

Metrics:

```text
count_delta_neg2
count_delta_neg1
count_delta_0
count_delta_pos1
count_delta_pos2
share_delta_neg2
share_delta_neg1
share_delta_0
share_delta_pos1
share_delta_pos2
entropy_optional
```

Output table:

```text
Table E3-3: Action Distribution
```

Proof point:

Shows whether the learned controller uses the action space or collapses to a single action.

### 4.4 E3-D Split-level Summary

Experiment name:

```text
E3-D split_level_summary
```

Data source:

Same joined per-context policy table as E3-A.

Dataset groups:

```text
seen4 = jobs, congressional, forums, microblog
added2 = imdb, openreview
all6 = jobs, congressional, forums, microblog, imdb, openreview
```

Policies compared:

```text
round19 resolver replay
round23 controller
```

Metrics:

```text
contexts
mean_reward
mean_regret_vs_oracle
win_rate_vs_keep_k0_by_reward
win_rate_vs_round19_by_reward
direction_accuracy
delta_k_accuracy
```

Output table:

```text
Table E3-4: Split-level Controller Quality
```

Proof point:

Separates original seen datasets from the two added controller-development datasets without turning E3 into an E2 held-out generalization experiment.

### 4.5 E3-E Direction Baseline Check

Experiment name:

```text
E3-E direction_baseline_check
```

Data source:

Same joined per-context policy table as E3-A.

Comparisons:

```text
round23 direction accuracy
round19 direction accuracy
majority oracle direction baseline
always keep direction baseline
```

Metrics:

```text
direction_accuracy
oracle_direction_distribution
majority_direction
majority_direction_accuracy
```

Output table:

```text
Table E3-5: Direction Baseline Check
```

Proof point:

Validates the interpretation rule that direction accuracy is meaningfully above a trivial majority-direction baseline.

### 4.6 E3-F Audit and Reproducibility Report

Experiment name:

```text
E3-F audit_report
```

Data source:

All E3 input files and generated per-context policy table.

Checks:

```text
input_context_count
unique_context_count
dataset_counts
missing_round19_contexts
extra_round19_contexts
invalid_round19_delta
missing_reward_fields
missing_best_top1_fields
oracle_reward_consistency
keep_k0_reward_consistency
policy_row_count
duplicate_policy_rows
```

Output table/report:

```text
e3_audit_report.json
e3_audit_report.md
```

Proof point:

Ensures the paper tables are complete, non-duplicated, and internally consistent.

## 5. Missing Scripts or Functions

### 5.1 New Summary Script

Recommended new script:

```text
model-train/summarize_round23_e3_policy_quality.py
```

Purpose:

Produce the final E3 offline policy-quality tables from a joined per-context policy table.

Inputs:

```text
--controller-context-table model-train/artifacts/round23_datasets_1200_top1_delta_m0005/round23_controller_context_table.jsonl
--round19-replay-table model-train/artifacts/round23_datasets_1200_top1_delta_m0005/round19_replay_table.jsonl
--round23-eval-report model-train/artifacts/.../round23_controller_eval_report.json
--output-dir model-train/artifacts/round23_e3_policy_quality_1200_all6_top1_delta_m0005
--scope all_contexts
```

Alternative model input:

```text
--model-dir model-train/artifacts/round23_models_1200_all6_final/extratrees/no_dataset
--model-family extratrees
--feature-version no_dataset
```

Outputs:

```text
e3_policy_contexts.jsonl
e3_table_overall_policy_quality.csv
e3_table_datasetwise_policy_quality.csv
e3_table_action_distribution.csv
e3_table_split_level_policy_quality.csv
e3_table_direction_baseline.csv
e3_audit_report.json
e3_summary.md
```

### 5.2 Suggested Functions

`load_context_rows(path) -> list[dict]`

Load `round23_controller_context_table.jsonl`.

`load_round19_rows(path) -> dict[str, dict]`

Load `round19_replay_table.jsonl` keyed by `context_id`.

`reward_field_for_delta(delta_k: int) -> str`

Map `-2,-1,0,+1,+2` to `controller_reward_dk_*`.

`best_top1_field_for_delta(delta_k: int) -> str`

Map `-2,-1,0,+1,+2` to `best_top1_dk_*`.

`direction(delta_k: int) -> int`

Return `-1`, `0`, or `+1`.

`build_policy_context_rows(context_rows, round19_rows, round23_predictions) -> list[dict]`

Build one normalized per-context row containing selected deltas, rewards, regret, top1 values, and correctness flags for every policy.

`summarize_overall(policy_rows) -> list[dict]`

Generate Table E3-1.

`summarize_datasetwise(policy_rows) -> list[dict]`

Generate Table E3-2.

`summarize_action_distribution(policy_rows) -> list[dict]`

Generate Table E3-3.

`summarize_split_level(policy_rows) -> list[dict]`

Generate Table E3-4.

`summarize_direction_baseline(policy_rows) -> list[dict]`

Generate Table E3-5.

`audit_inputs_and_outputs(...) -> dict`

Generate audit counts and consistency flags.

`write_tables(output_dir, tables) -> None`

Write CSV, JSON, and Markdown summary.

### 5.3 Suggested Normalized Per-context Fields

`e3_policy_contexts.jsonl` should include:

```text
context_id
dataset_name
meta_seed
label_target_mode
tie_margin
reference_budget
oracle_delta_k
oracle_target_budget
oracle_reward
oracle_best_top1
keep_delta_k
keep_reward
keep_best_top1
round19_delta_k
round19_target_budget
round19_reward
round19_best_top1
round23_delta_k
round23_target_budget
round23_reward
round23_best_top1
round23_predicted_rewards
round23_confidence_margin_optional
round23_regret_vs_oracle
round19_regret_vs_oracle
keep_regret_vs_oracle
round23_win_vs_keep_by_reward
round19_win_vs_keep_by_reward
round23_win_vs_round19_by_reward
round23_delta_k_correct
round19_delta_k_correct
round23_direction_correct
round19_direction_correct
round23_best_top1_regret
round19_best_top1_regret
keep_best_top1_regret
```

## 6. Code Development Plan

### 6.1 Task 1: Add E3 Policy Summarizer

File to create:

```text
model-train/summarize_round23_e3_policy_quality.py
```

Responsibilities:

- Join context table, round19 replay table, and round23 controller predictions.
- Normalize all policies into the same reward and regret semantics.
- Produce E3-A through E3-F outputs.
- Fail fast on missing contexts, duplicate contexts, invalid deltas, or missing action fields.

Implementation notes:

- Reuse constants from `model-train/common.py`: `DELTA_ACTIONS`, `REFERENCE_BUDGET`, `BUDGETS`, `read_jsonl`, `write_jsonl`, `write_csv`, `dump_json`.
- Reuse feature encoding/model loading logic from `eval_round23_controller.py` if direct model inference is needed.
- Prefer accepting an existing `round23_controller_eval_report.json` first, because it already contains `per_context` predictions.
- If both `--round23-eval-report` and `--model-dir` are supplied, use the report by default and record that choice in `e3_audit_report.json`.

Testing needed:

Yes.

### 6.2 Task 2: Extend or Reuse Evaluation Report Generation

Preferred no-risk path:

Do not modify `eval_round23_controller.py` initially. Generate the regular eval report first, then feed it into the new E3 summarizer.

Possible later extension:

Add a `--context-scope all` option to `eval_round23_controller.py`, because current evaluation requires a split payload (`final_test_context_ids` or `unseen_test_context_ids`). E3 may need all 240 contexts when using the all-six fulltrain controller.

Testing needed:

Yes, if `eval_round23_controller.py` is modified.

### 6.3 Task 3: Add Unit Tests

File to create:

```text
model-train/tests/test_round23_e3_policy_quality.py
```

Test cases:

- A tiny three-context fixture produces correct reward/regret for `keep-k0`, `round19`, `round23`, and `oracle`.
- `round19_reward` is selected from `controller_reward_dk_*`, not from `round19_replay_reward`.
- Direction accuracy maps negative, zero, positive correctly.
- Action distribution counts all five deltas and includes zero counts.
- Dataset-wise summary groups by `dataset_name`.
- Audit fails on missing round19 context.
- Audit fails on invalid `round19_predicted_delta_k`.
- Audit fails on duplicate context IDs.

Testing command:

```powershell
cd model-train
python -m pytest tests/test_round23_e3_policy_quality.py -q
```

### 6.4 Task 4: Add Markdown Summary Writer

Can be part of `summarize_round23_e3_policy_quality.py`.

Responsibilities:

- Render the five E3 result tables into `e3_summary.md`.
- Include input file paths, model/report provenance, context counts, and audit status.
- Mark `oracle budget` as offline upper bound, not deployable method.

Testing needed:

Light unit test is enough: verify output file exists and contains required table headings.

## 7. Metric Definitions

For each context and policy:

```text
policy_reward = controller_reward_dk_<selected_delta>
policy_best_top1 = best_top1_dk_<selected_delta>
regret_vs_oracle = oracle_best_controller_reward - policy_reward
best_top1_regret = oracle_best_top1 - policy_best_top1
win_vs_keep_k0_by_reward = 1 if policy_reward > keep_k0_reward else 0
win_vs_round19_by_reward = 1 if policy_reward > round19_reward else 0
delta_k_accuracy = 1 if policy_delta_k == oracle_best_delta_k else 0
direction_accuracy = 1 if sign(policy_delta_k) == sign(oracle_best_delta_k) else 0
```

Policy selected deltas:

```text
keep-k0=20: 0
round19 resolver replay: round19_predicted_delta_k
round23 controller: predicted_delta_k from model/report
oracle budget: oracle_best_delta_k
```

For `oracle budget`:

```text
mean_regret_vs_oracle = 0
delta_k_accuracy = 1
direction_accuracy = 1
```

For `keep-k0=20`, `win_rate_vs_keep_k0_by_reward` should be blank or `0.0` by definition. Prefer blank/NA in paper tables to avoid implying a meaningful self-comparison.

## 8. Execution Command Drafts

### 8.1 Generate Round23 Evaluation Report If Needed

If using the all-six final raw model directory and an all-context split file exists or is added:

```powershell
cd model-train
python eval_round23_controller.py `
  --controller-context-table artifacts/round23_datasets_1200_top1_delta_m0005/round23_controller_context_table.jsonl `
  --final-test artifacts/round23_splits_1200_all6_fulltrain_top1_delta_m0005/round23_final_test_contexts.json `
  --config configs/train_round23_controller_all6_search.yaml `
  --model-dir artifacts/round23_models_1200_all6_final/extratrees/no_dataset `
  --report-dir artifacts/round23_e3_eval_1200_all6_top1_delta_m0005/extratrees_no_dataset `
  --model-family extratrees `
  --feature-version no_dataset `
  --round19-replay-path artifacts/round23_datasets_1200_top1_delta_m0005/round19_replay_table.jsonl `
  --target-field target_value_for_training
```

Potential issue:

The all-six fulltrain split has `final_test_context_count=0` in the observed `train_metrics.json`. For E3, prefer either:

- Add an `--all-contexts` mode to `eval_round23_controller.py`.
- Or create a temporary explicit E3 split JSON containing all 240 context IDs under `final_test_context_ids`.

### 8.2 Run New E3 Summary From Existing Eval Report

```powershell
cd model-train
python summarize_round23_e3_policy_quality.py `
  --controller-context-table artifacts/round23_datasets_1200_top1_delta_m0005/round23_controller_context_table.jsonl `
  --round19-replay-table artifacts/round23_datasets_1200_top1_delta_m0005/round19_replay_table.jsonl `
  --round23-eval-report artifacts/round23_e3_eval_1200_all6_top1_delta_m0005/extratrees_no_dataset/round23_controller_eval_report.json `
  --output-dir artifacts/round23_e3_policy_quality_1200_all6_top1_delta_m0005/extratrees_no_dataset
```

### 8.3 Run New E3 Summary With Direct Model Inference

```powershell
cd model-train
python summarize_round23_e3_policy_quality.py `
  --controller-context-table artifacts/round23_datasets_1200_top1_delta_m0005/round23_controller_context_table.jsonl `
  --round19-replay-table artifacts/round23_datasets_1200_top1_delta_m0005/round19_replay_table.jsonl `
  --model-dir artifacts/round23_models_1200_all6_final/extratrees/no_dataset `
  --model-family extratrees `
  --feature-version no_dataset `
  --config configs/train_round23_controller_all6_search.yaml `
  --output-dir artifacts/round23_e3_policy_quality_1200_all6_top1_delta_m0005/extratrees_no_dataset
```

### 8.4 Test Commands

```powershell
cd model-train
python -m pytest tests/test_round23_e3_policy_quality.py -q
python -m pytest tests/test_round23_training_eval_paths.py -q
python -m pytest tests/test_round23_controller_dataset.py -q
```

### 8.5 Manual Audit Commands

```powershell
cd model-train
python - <<'PY'
import json
from pathlib import Path
base = Path('artifacts/round23_e3_policy_quality_1200_all6_top1_delta_m0005/extratrees_no_dataset')
audit = json.loads((base / 'e3_audit_report.json').read_text(encoding='utf-8'))
print(json.dumps(audit, indent=2, ensure_ascii=False))
PY
```

PowerShell-safe version:

```powershell
cd model-train
@'
import json
from pathlib import Path
base = Path('artifacts/round23_e3_policy_quality_1200_all6_top1_delta_m0005/extratrees_no_dataset')
audit = json.loads((base / 'e3_audit_report.json').read_text(encoding='utf-8'))
print(json.dumps(audit, indent=2, ensure_ascii=False))
'@ | python -
```

## 9. Review Checklist

### 9.1 Input Completeness

- `round23_controller_context_table.jsonl` exists.
- `round19_replay_table.jsonl` exists.
- `round19_replay_mapping.json` exists or is intentionally unused.
- Context table has 240 rows.
- Context table has 240 unique `context_id` values.
- Round19 replay table has 240 rows.
- Round19 replay table has 240 unique `context_id` values.
- Each of the six datasets has 40 contexts.
- Every context has all five reward fields.
- Every context has all five `best_top1` fields.

### 9.2 Join Correctness

- No context in context table is missing from round19 replay.
- No extra context exists in round19 replay.
- Every `round19_predicted_delta_k` is in `[-2,-1,0,1,2]`.
- Every `round19_predicted_budget` equals `20 + round19_predicted_delta_k`.
- Every `round23 predicted_delta_k` is in `[-2,-1,0,1,2]`.
- Every `round23 predicted_target_budget` equals `20 + predicted_delta_k`.

### 9.3 Reward Semantics

- `keep_reward` equals `controller_reward_dk_0`.
- `oracle_reward` equals the reward field selected by `oracle_best_delta_k`.
- `round19_reward` equals the reward field selected by `round19_predicted_delta_k`.
- `round23_reward` equals the reward field selected by `round23 predicted_delta_k`.
- E3 primary tables do not use `round19_replay_reward` as the primary round19 reward.
- `label_target_mode` is consistently `top1_delta`.
- `tie_margin` is consistently `0.0005`.

### 9.4 Output Completeness

- `e3_policy_contexts.jsonl` has 240 rows.
- `e3_table_overall_policy_quality.csv` has one row per policy.
- `e3_table_datasetwise_policy_quality.csv` has expected dataset-policy rows.
- `e3_table_action_distribution.csv` includes all five action columns.
- `e3_table_split_level_policy_quality.csv` includes `seen4`, `added2`, and `all6`.
- `e3_table_direction_baseline.csv` includes majority-direction baseline.
- `e3_audit_report.json` reports zero missing, duplicate, and invalid rows.
- `e3_summary.md` contains Tables E3-1 through E3-5.

### 9.5 Paper Consistency

- The E3 text calls `oracle budget` an offline upper bound, not a deployable policy.
- The E3 text does not claim arbitrary-anchor generalization.
- The E3 text does not claim one-shot-vs-delta-k superiority.
- The E3 text reports reward-based win rate as the primary win metric.
- If `best_top1` win/regret is reported, it is labeled secondary.
- Dataset-wise interpretation does not require `round23` to beat `round19` on every dataset.

## 10. Key Risks and Decisions

### 10.1 Fulltrain Model Evaluation Scope

The observed all-six final model has `train_context_count=240` and `final_test_context_count=0`. If E3 evaluates all 240 contexts with this model, the paper should describe E3 as an offline decision-quality analysis on the controller-development collection, not a held-out generalization test. E2 covers held-out generalization separately.

### 10.2 Reward Field Mismatch Risk

The context table contains current `controller_reward_dk_*` values and legacy `controller_old_reward_dk_*` values. The round19 replay table stores `round19_replay_reward` using the legacy formula. E3 must use one consistent reward semantics. Recommended: use current `controller_reward_dk_*` for all policies.

### 10.3 Existing Eval Report Is Not Final E3

`round23_controller_eval_report.json` is useful as an intermediate prediction source, but it should not be used directly as the E3 result table. A dedicated summarizer is needed to avoid missing policy rows, reward-based win rates, action distributions, and audit checks.
