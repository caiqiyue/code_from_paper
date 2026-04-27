# Stage 1 Parameter Tuning Screening Configs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `single_node_tuning` screening config set for the six approved `Stage 1` parameter experiments so the repo can run comparable tuning experiments on `jobs`, `congressional`, `forums`, and `microblog`.

**Architecture:** Reuse the existing `single_node_screening` config structure and add a parallel `single_node_tuning` config tree. Keep one shared tuning base, then add one override base per parameter group (`a1`, `a2`, `b1`, `b2`, `c1`, `d1`), and finally four dataset-specific leaf configs per group so naming, output roots, and selector overrides stay explicit and easy to audit.

**Tech Stack:** YAML config inheritance, Python `unittest`, existing `paper_new_selector.thesis_bridge.load_yaml_config` config loader.

---

## File Structure

- Create: `paper-new/configs/experiments/single_node_tuning/_base_selector_tuning.yaml`
  - Common tuning-scale config shared by every tuning experiment.
- Create: `paper-new/configs/experiments/single_node_tuning/_a1_length_floor_8.yaml`
  - Group-level override for `length_floor: 8`.
- Create: `paper-new/configs/experiments/single_node_tuning/_a2_length_lambda_010.yaml`
  - Group-level override for `length_lambda: 0.10`.
- Create: `paper-new/configs/experiments/single_node_tuning/_b1_generic_030.yaml`
  - Group-level override for `lambda_generic: 0.30`.
- Create: `paper-new/configs/experiments/single_node_tuning/_b2_generic_025.yaml`
  - Group-level override for `lambda_generic: 0.25`.
- Create: `paper-new/configs/experiments/single_node_tuning/_c1_redundancy_035.yaml`
  - Group-level override for `lambda_redundancy: 0.35`.
- Create: `paper-new/configs/experiments/single_node_tuning/_d1_combo_safe.yaml`
  - Group-level override for the approved three-parameter combo.
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a1_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a1_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a1_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a1_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a2_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a2_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a2_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a2_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b1_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b1_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b1_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b1_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b2_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b2_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b2_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b2_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_c1_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_c1_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_c1_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_c1_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_d1_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_d1_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_d1_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_d1_microblog.yaml`
- Modify: `paper-new/tests/test_config.py`
  - Add config-loading coverage for the new tuning family and critical selector overrides.

### Task 1: Add failing tests for the new tuning config family

**Files:**
- Modify: `paper-new/tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

```python
    def test_tuning_base_contract_matches_screening_scale(self):
        config = load_yaml_config("paper-new/configs/experiments/single_node_tuning/_base_selector_tuning.yaml")
        self.assertEqual(config["meta"]["stage"], "single_node_tuning")
        self.assertEqual(config["data"]["train_limit"], 256)
        self.assertEqual(config["data"]["eval_limit"], 256)
        self.assertEqual(config["data"]["initialization_limit"], 1024)
        self.assertEqual(config["generator"]["candidate_count"], 24)
        self.assertEqual(config["generator"]["generated_per_round"], 8)
        self.assertEqual(config["selector"]["seed_top_k"], 6)
        self.assertEqual(config["selector"]["hard_negative_top_k"], 6)
        self.assertEqual(config["bootstrap"]["num_prompts"], 100)
        self.assertEqual(config["eval"]["small_epochs"], 6)

    def test_tuning_group_overrides_apply_expected_selector_values(self):
        a1 = load_yaml_config("paper-new/configs/experiments/single_node_tuning/ns_tune_a1_microblog.yaml")
        a2 = load_yaml_config("paper-new/configs/experiments/single_node_tuning/ns_tune_a2_microblog.yaml")
        b1 = load_yaml_config("paper-new/configs/experiments/single_node_tuning/ns_tune_b1_forums.yaml")
        b2 = load_yaml_config("paper-new/configs/experiments/single_node_tuning/ns_tune_b2_forums.yaml")
        c1 = load_yaml_config("paper-new/configs/experiments/single_node_tuning/ns_tune_c1_jobs.yaml")
        d1 = load_yaml_config("paper-new/configs/experiments/single_node_tuning/ns_tune_d1_congressional.yaml")

        self.assertEqual(a1["selector"]["length_floor"], 8)
        self.assertEqual(a2["selector"]["length_lambda"], 0.10)
        self.assertEqual(b1["selector"]["lambda_generic"], 0.30)
        self.assertEqual(b2["selector"]["lambda_generic"], 0.25)
        self.assertEqual(c1["selector"]["lambda_redundancy"], 0.35)
        self.assertEqual(d1["selector"]["length_floor"], 8)
        self.assertEqual(d1["selector"]["lambda_generic"], 0.30)
        self.assertEqual(d1["selector"]["lambda_redundancy"], 0.35)

    def test_tuning_dataset_leaf_configs_keep_dataset_paths_and_output_roots_explicit(self):
        jobs = load_yaml_config("paper-new/configs/experiments/single_node_tuning/ns_tune_b1_jobs.yaml")
        forums = load_yaml_config("paper-new/configs/experiments/single_node_tuning/ns_tune_c1_forums.yaml")
        micro = load_yaml_config("paper-new/configs/experiments/single_node_tuning/ns_tune_d1_microblog.yaml")

        self.assertEqual(jobs["meta"]["experiment_id"], "ns_tune_b1_jobs")
        self.assertEqual(jobs["paths"]["output_root"], "paper-new/outputs/ns_tune_b1_jobs")
        self.assertEqual(jobs["data"]["dataset_name"], "jobs")
        self.assertEqual(jobs["data"]["train_path"], "thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json")

        self.assertEqual(forums["meta"]["experiment_id"], "ns_tune_c1_forums")
        self.assertEqual(forums["paths"]["output_root"], "paper-new/outputs/ns_tune_c1_forums")
        self.assertEqual(forums["data"]["dataset_name"], "forums")
        self.assertEqual(forums["data"]["eval_path"], "thesis_platform/datasets/pretext_forums/formatted/forums_eval.json")

        self.assertEqual(micro["meta"]["experiment_id"], "ns_tune_d1_microblog")
        self.assertEqual(micro["paths"]["output_root"], "paper-new/outputs/ns_tune_d1_microblog")
        self.assertEqual(micro["data"]["dataset_name"], "microblog")
        self.assertEqual(micro["data"]["train_path"], "thesis_platform/datasets/pretext_twitter/formatted/twitter_train.json")
```

- [ ] **Step 2: Run the config tests and verify they fail**

Run:

```powershell
python -m unittest discover -s tests -p "test_config.py" -v
```

Expected:

```text
FAIL: test_tuning_base_contract_matches_screening_scale
FAIL: test_tuning_group_overrides_apply_expected_selector_values
FAIL: test_tuning_dataset_leaf_configs_keep_dataset_paths_and_output_roots_explicit
```

- [ ] **Step 3: Commit the failing test**

```powershell
git add tests/test_config.py
git commit -m "test: cover stage1 tuning screening configs"
```

### Task 2: Add the shared tuning base and group override bases

**Files:**
- Create: `paper-new/configs/experiments/single_node_tuning/_base_selector_tuning.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/_a1_length_floor_8.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/_a2_length_lambda_010.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/_b1_generic_030.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/_b2_generic_025.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/_c1_redundancy_035.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/_d1_combo_safe.yaml`

- [ ] **Step 1: Create the shared tuning base**

```yaml
inherits:
  - ../single_node_formal/_base_selector_formal.yaml

meta:
  stage: single_node_tuning
  seed: 42

paths:
  output_root: paper-new/outputs/ns_tuning_default

data:
  train_limit: 256
  eval_limit: 256
  initialization_limit: 1024

llm:
  generator:
    max_new_tokens: 128

generator:
  candidate_count: 24
  generated_per_round: 8
  max_rounds: 4
  max_new_tokens: 128

selector:
  seed_top_k: 6
  hard_negative_top_k: 6

bootstrap:
  num_prompts: 100
  max_tokens: 85

eval:
  enabled: true
  mode: pretext_small
  small_eval_mode: gpt2
  max_samples_per_client: 16
  initialization_min_words: 20
  small_epochs: 6
  small_batch_size: 8
  small_eval_batch_size: 2
  small_grad_accum_steps: 4
  small_cutoff_len: 64
  small_learning_rate: 0.0002
  small_num_proc: 1
```

- [ ] **Step 2: Add the six parameter-group override bases**

```yaml
# paper-new/configs/experiments/single_node_tuning/_a1_length_floor_8.yaml
inherits:
  - ./_base_selector_tuning.yaml

selector:
  length_floor: 8
```

```yaml
# paper-new/configs/experiments/single_node_tuning/_a2_length_lambda_010.yaml
inherits:
  - ./_base_selector_tuning.yaml

selector:
  length_lambda: 0.10
```

```yaml
# paper-new/configs/experiments/single_node_tuning/_b1_generic_030.yaml
inherits:
  - ./_base_selector_tuning.yaml

selector:
  lambda_generic: 0.30
```

```yaml
# paper-new/configs/experiments/single_node_tuning/_b2_generic_025.yaml
inherits:
  - ./_base_selector_tuning.yaml

selector:
  lambda_generic: 0.25
```

```yaml
# paper-new/configs/experiments/single_node_tuning/_c1_redundancy_035.yaml
inherits:
  - ./_base_selector_tuning.yaml

selector:
  lambda_redundancy: 0.35
```

```yaml
# paper-new/configs/experiments/single_node_tuning/_d1_combo_safe.yaml
inherits:
  - ./_base_selector_tuning.yaml

selector:
  length_floor: 8
  lambda_generic: 0.30
  lambda_redundancy: 0.35
```

- [ ] **Step 3: Run the config tests and verify only the missing leaf-config test still fails**

Run:

```powershell
python -m unittest discover -s tests -p "test_config.py" -v
```

Expected:

```text
ok: test_tuning_base_contract_matches_screening_scale
ERROR or FAIL: test_tuning_group_overrides_apply_expected_selector_values
ERROR or FAIL: test_tuning_dataset_leaf_configs_keep_dataset_paths_and_output_roots_explicit
```

- [ ] **Step 4: Commit the shared tuning bases**

```powershell
git add configs/experiments/single_node_tuning tests/test_config.py
git commit -m "feat: add shared stage1 tuning config bases"
```

### Task 3: Add the twenty-four dataset leaf configs

**Files:**
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a1_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a1_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a1_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a1_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a2_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a2_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a2_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_a2_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b1_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b1_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b1_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b1_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b2_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b2_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b2_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_b2_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_c1_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_c1_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_c1_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_c1_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_d1_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_d1_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_d1_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning/ns_tune_d1_microblog.yaml`

- [ ] **Step 1: Add the four A1 leaf configs**

```yaml
# paper-new/configs/experiments/single_node_tuning/ns_tune_a1_jobs.yaml
inherits:
  - ./_a1_length_floor_8.yaml

meta:
  experiment_id: ns_tune_a1_jobs

paths:
  output_root: paper-new/outputs/ns_tune_a1_jobs

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

```yaml
# paper-new/configs/experiments/single_node_tuning/ns_tune_a1_congressional.yaml
inherits:
  - ./_a1_length_floor_8.yaml

meta:
  experiment_id: ns_tune_a1_congressional

paths:
  output_root: paper-new/outputs/ns_tune_a1_congressional

data:
  dataset_name: congressional
  train_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_train.json
  eval_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

```yaml
# paper-new/configs/experiments/single_node_tuning/ns_tune_a1_forums.yaml
inherits:
  - ./_a1_length_floor_8.yaml

meta:
  experiment_id: ns_tune_a1_forums

paths:
  output_root: paper-new/outputs/ns_tune_a1_forums

data:
  dataset_name: forums
  train_path: thesis_platform/datasets/pretext_forums/formatted/forums_train.json
  eval_path: thesis_platform/datasets/pretext_forums/formatted/forums_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

```yaml
# paper-new/configs/experiments/single_node_tuning/ns_tune_a1_microblog.yaml
inherits:
  - ./_a1_length_floor_8.yaml

meta:
  experiment_id: ns_tune_a1_microblog

paths:
  output_root: paper-new/outputs/ns_tune_a1_microblog

data:
  dataset_name: microblog
  train_path: thesis_platform/datasets/pretext_twitter/formatted/twitter_train.json
  eval_path: thesis_platform/datasets/pretext_twitter/formatted/twitter_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 2: Add the remaining A2, B1, B2, C1, and D1 leaf configs using the same pattern**

Use the same four dataset payloads as Step 1. Only change the inherited group base, `meta.experiment_id`, and `paths.output_root`.

Representative examples:

```yaml
# paper-new/configs/experiments/single_node_tuning/ns_tune_b1_jobs.yaml
inherits:
  - ./_b1_generic_030.yaml

meta:
  experiment_id: ns_tune_b1_jobs

paths:
  output_root: paper-new/outputs/ns_tune_b1_jobs

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

```yaml
# paper-new/configs/experiments/single_node_tuning/ns_tune_c1_forums.yaml
inherits:
  - ./_c1_redundancy_035.yaml

meta:
  experiment_id: ns_tune_c1_forums

paths:
  output_root: paper-new/outputs/ns_tune_c1_forums

data:
  dataset_name: forums
  train_path: thesis_platform/datasets/pretext_forums/formatted/forums_train.json
  eval_path: thesis_platform/datasets/pretext_forums/formatted/forums_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

```yaml
# paper-new/configs/experiments/single_node_tuning/ns_tune_d1_microblog.yaml
inherits:
  - ./_d1_combo_safe.yaml

meta:
  experiment_id: ns_tune_d1_microblog

paths:
  output_root: paper-new/outputs/ns_tune_d1_microblog

data:
  dataset_name: microblog
  train_path: thesis_platform/datasets/pretext_twitter/formatted/twitter_train.json
  eval_path: thesis_platform/datasets/pretext_twitter/formatted/twitter_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 3: Run the config tests and verify they pass**

Run:

```powershell
python -m unittest discover -s tests -p "test_config.py" -v
```

Expected:

```text
OK
```

- [ ] **Step 4: Run validate-only for at least two tuning configs**

Run:

```powershell
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning/ns_tune_a1_microblog.yaml --validate-only
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning/ns_tune_d1_jobs.yaml --validate-only
```

Expected:

```text
JSON summary prints with "mode": "selector_seed_search"
```

- [ ] **Step 5: Commit the tuning leaf configs**

```powershell
git add configs/experiments/single_node_tuning tests/test_config.py
git commit -m "feat: add stage1 parameter tuning screening configs"
```

## Self-Review

### Spec coverage

- The plan creates one tuning config family matching the approved design doc.
- The plan covers all six parameter groups: `a1`, `a2`, `b1`, `b2`, `c1`, `d1`.
- The plan covers all four datasets for each group.
- The plan adds config tests for base scale, selector overrides, dataset paths, and output roots.
- The plan includes validate-only checks so config inheritance is exercised by the real pipeline.

### Placeholder scan

- No `TODO`, `TBD`, or vague "handle later" language remains.
- All steps name exact files and commands.
- Code-changing steps include concrete YAML or Python snippets.

### Type consistency

- All config names consistently use `ns_tune_<group>_<dataset>`.
- All output roots consistently use `paper-new/outputs/ns_tune_<group>_<dataset>`.
- Selector keys match existing config keys: `length_floor`, `length_lambda`, `lambda_generic`, `lambda_redundancy`.
