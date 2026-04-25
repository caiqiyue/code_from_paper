# Screening Diagnosis Configs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `paper-new-xiugai` 中基于当前 `paper-new` 可运行基线，新增一套只改配置、不改 selector 主算法的诊断型 screening 实验配置，用来定位 `forums/microblog` 落后的主要原因，同时监控 `jobs/congressional` 是否被伤害。

**Architecture:** 保持 `paper_new_selector`、`Stage 1/Stage 2 shared vLLM`、`downstream eval` 代码路径不变，只通过新的 diagnosis 配置目录覆盖 selector 超参数。诊断配置分为 3 组：`genericity_off`、`redundancy_up`、`support_softened`，每组覆盖 4 个数据集，共 12 个实验文件；再通过配置解析测试保证所有配置都能正确继承和传递数值。

**Tech Stack:** Python 3.10+, YAML inheritance config loader, `paper_new_selector`, `unittest`, local Windows workspace, later remote Linux `pretext` env.

---

## File Structure

- `paper-new-xiugai/configs/experiments/single_node_diagnosis/_base_selector_diagnosis.yaml`
  - 诊断实验公共基座，继承 screening 基座，统一输出根目录。
- `paper-new-xiugai/configs/experiments/single_node_diagnosis/_variant_genericity_off.yaml`
  - 诊断 Variant 1：关闭 `genericity penalty`。
- `paper-new-xiugai/configs/experiments/single_node_diagnosis/_variant_redundancy_up.yaml`
  - 诊断 Variant 2：增强 `redundancy penalty`。
- `paper-new-xiugai/configs/experiments/single_node_diagnosis/_variant_support_softened.yaml`
  - 诊断 Variant 3：弱化 `support` 的中心化倾向。
- `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d{1,2,3}_{jobs,congressional,forums,microblog}.yaml`
  - 12 个实际可运行的诊断实验配置。
- `paper-new-xiugai/tests/test_diagnosis_configs.py`
  - 回归测试：确保 12 个配置都能被解析，关键数值与输出路径符合预期。
- `paper-new-xiugai/docs/2026-04-25-screening-diagnosis-plan.md`
  - 本计划文档。

## Diagnosis Variant Definitions

### Variant 1: `genericity_off`

目标：验证 `genericity penalty` 是否误伤 `forums/microblog` 中有任务价值但风格不规范的候选。

核心配置覆盖：

```yaml
selector:
  lambda_generic: 0.0
```

### Variant 2: `redundancy_up`

目标：验证当前 seed 集是否过于中心化，导致覆盖不足。

核心配置覆盖：

```yaml
selector:
  lambda_redundancy: 0.45
```

### Variant 3: `support_softened`

目标：在不改算法代码的前提下，削弱“高支持、中心化”偏置，观察长尾数据集是否回升。

核心配置覆盖：

```yaml
selector:
  top_q: 2
  rank_weights: [1.0, 0.4]
  density_lambda: 0.35
  novelty_lambda: 0.45
  length_lambda: 0.20
```

## Task 1: Bootstrap `paper-new-xiugai` as runnable baseline

**Files:**
- Verify: `paper-new-xiugai/configs`
- Verify: `paper-new-xiugai/paper_new_selector`
- Verify: `paper-new-xiugai/tests`

- [ ] **Step 1: Verify copied baseline tree exists**

Run:

```powershell
Get-ChildItem 'D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-xiugai'
```

Expected: contains `configs`, `paper_new_selector`, `tests`, `scripts`, `docs`.

- [ ] **Step 2: Verify screening base config is present**

Run:

```powershell
Get-Content 'D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-xiugai\configs\experiments\single_node_screening\_base_selector_screening.yaml'
```

Expected: shows `train_limit = 256`, `eval_limit = 256`, `num_prompts = 100`, `small_epochs = 6`.

- [ ] **Step 3: Commit**

```bash
git add paper-new-xiugai
git commit -m "chore: bootstrap paper-new-xiugai from paper-new baseline"
```

## Task 2: Add diagnosis config base and variant templates

**Files:**
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/_base_selector_diagnosis.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/_variant_genericity_off.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/_variant_redundancy_up.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/_variant_support_softened.yaml`

- [ ] **Step 1: Write the failing test skeleton for diagnosis config loading**

Create test scaffold:

```python
import unittest
from pathlib import Path

from paper_new_selector.thesis_bridge import load_yaml_config


class DiagnosisConfigTests(unittest.TestCase):
    def test_genericity_off_variant_loads(self):
        root = Path(__file__).resolve().parents[1]
        config = load_yaml_config(root / "configs/experiments/single_node_diagnosis/_variant_genericity_off.yaml")
        self.assertEqual(config["selector"]["lambda_generic"], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
conda run -n pretext python -m unittest paper-new-xiugai.tests.test_diagnosis_configs.DiagnosisConfigTests.test_genericity_off_variant_loads -v
```

Expected: FAIL with file-not-found for diagnosis config.

- [ ] **Step 3: Write diagnosis base config**

Create `paper-new-xiugai/configs/experiments/single_node_diagnosis/_base_selector_diagnosis.yaml`:

```yaml
inherits:
  - ../single_node_screening/_base_selector_screening.yaml

meta:
  stage: single_node_diagnosis

paths:
  output_root: paper-new-xiugai/outputs/ns_diagnosis_default
```

- [ ] **Step 4: Write diagnosis variant templates**

Create `paper-new-xiugai/configs/experiments/single_node_diagnosis/_variant_genericity_off.yaml`:

```yaml
inherits:
  - ./_base_selector_diagnosis.yaml

meta:
  diagnosis_variant: genericity_off

selector:
  lambda_generic: 0.0
```

Create `paper-new-xiugai/configs/experiments/single_node_diagnosis/_variant_redundancy_up.yaml`:

```yaml
inherits:
  - ./_base_selector_diagnosis.yaml

meta:
  diagnosis_variant: redundancy_up

selector:
  lambda_redundancy: 0.45
```

Create `paper-new-xiugai/configs/experiments/single_node_diagnosis/_variant_support_softened.yaml`:

```yaml
inherits:
  - ./_base_selector_diagnosis.yaml

meta:
  diagnosis_variant: support_softened

selector:
  top_q: 2
  rank_weights: [1.0, 0.4]
  density_lambda: 0.35
  novelty_lambda: 0.45
  length_lambda: 0.20
```

- [ ] **Step 5: Run test to verify template config passes**

Run:

```powershell
conda run -n pretext python -m unittest paper-new-xiugai.tests.test_diagnosis_configs.DiagnosisConfigTests.test_genericity_off_variant_loads -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add paper-new-xiugai/configs/experiments/single_node_diagnosis paper-new-xiugai/tests/test_diagnosis_configs.py
git commit -m "feat: add diagnosis variant templates for screening"
```

## Task 3: Add 12 dataset-specific diagnosis configs

**Files:**
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d1_jobs.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d1_congressional.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d1_forums.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d1_microblog.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d2_jobs.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d2_congressional.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d2_forums.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d2_microblog.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d3_jobs.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d3_congressional.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d3_forums.yaml`
- Create: `paper-new-xiugai/configs/experiments/single_node_diagnosis/ns_d3_microblog.yaml`

- [ ] **Step 1: Extend test file with matrix expectations**

Add test content:

```python
    def test_all_diagnosis_configs_resolve(self):
        root = Path(__file__).resolve().parents[1]
        names = [
            "ns_d1_jobs.yaml",
            "ns_d1_congressional.yaml",
            "ns_d1_forums.yaml",
            "ns_d1_microblog.yaml",
            "ns_d2_jobs.yaml",
            "ns_d2_congressional.yaml",
            "ns_d2_forums.yaml",
            "ns_d2_microblog.yaml",
            "ns_d3_jobs.yaml",
            "ns_d3_congressional.yaml",
            "ns_d3_forums.yaml",
            "ns_d3_microblog.yaml",
        ]
        for name in names:
            config = load_yaml_config(root / "configs/experiments/single_node_diagnosis" / name)
            self.assertEqual(config["data"]["train_limit"], 256)
            self.assertEqual(config["data"]["eval_limit"], 256)
            self.assertEqual(config["bootstrap"]["num_prompts"], 100)
            self.assertEqual(config["eval"]["small_epochs"], 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
conda run -n pretext python -m unittest paper-new-xiugai.tests.test_diagnosis_configs.DiagnosisConfigTests.test_all_diagnosis_configs_resolve -v
```

Expected: FAIL with missing diagnosis config files.

- [ ] **Step 3: Create D1 dataset configs**

Example `ns_d1_jobs.yaml`:

```yaml
inherits:
  - ./_variant_genericity_off.yaml

meta:
  experiment_id: ns_d1_jobs

paths:
  output_root: paper-new-xiugai/outputs/ns_d1_jobs

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

Repeat same pattern for `congressional`, `forums`, `microblog`, changing `experiment_id`, `output_root`, and dataset paths to match the screening configs already used in `paper-new`.

- [ ] **Step 4: Create D2 dataset configs**

Example `ns_d2_jobs.yaml`:

```yaml
inherits:
  - ./_variant_redundancy_up.yaml

meta:
  experiment_id: ns_d2_jobs

paths:
  output_root: paper-new-xiugai/outputs/ns_d2_jobs

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

Repeat same pattern for the other 3 datasets.

- [ ] **Step 5: Create D3 dataset configs**

Example `ns_d3_jobs.yaml`:

```yaml
inherits:
  - ./_variant_support_softened.yaml

meta:
  experiment_id: ns_d3_jobs

paths:
  output_root: paper-new-xiugai/outputs/ns_d3_jobs

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

Repeat same pattern for the other 3 datasets.

- [ ] **Step 6: Run the matrix config test**

Run:

```powershell
conda run -n pretext python -m unittest paper-new-xiugai.tests.test_diagnosis_configs.DiagnosisConfigTests.test_all_diagnosis_configs_resolve -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add paper-new-xiugai/configs/experiments/single_node_diagnosis paper-new-xiugai/tests/test_diagnosis_configs.py
git commit -m "feat: add diagnosis experiment configs for four datasets"
```

## Task 4: Add configuration value assertions for each diagnosis family

**Files:**
- Modify: `paper-new-xiugai/tests/test_diagnosis_configs.py`

- [ ] **Step 1: Add explicit assertions for diagnosis semantics**

Extend the test file:

```python
    def test_d1_genericity_off_overrides_lambda_generic(self):
        root = Path(__file__).resolve().parents[1]
        config = load_yaml_config(root / "configs/experiments/single_node_diagnosis/ns_d1_forums.yaml")
        self.assertEqual(config["selector"]["lambda_generic"], 0.0)
        self.assertEqual(config["selector"]["lambda_redundancy"], 0.25)

    def test_d2_redundancy_up_overrides_lambda_redundancy(self):
        root = Path(__file__).resolve().parents[1]
        config = load_yaml_config(root / "configs/experiments/single_node_diagnosis/ns_d2_jobs.yaml")
        self.assertEqual(config["selector"]["lambda_redundancy"], 0.45)
        self.assertEqual(config["selector"]["lambda_generic"], 0.35)

    def test_d3_support_softened_overrides_support_shape(self):
        root = Path(__file__).resolve().parents[1]
        config = load_yaml_config(root / "configs/experiments/single_node_diagnosis/ns_d3_microblog.yaml")
        self.assertEqual(config["selector"]["top_q"], 2)
        self.assertEqual(config["selector"]["rank_weights"], [1.0, 0.4])
        self.assertEqual(config["selector"]["density_lambda"], 0.35)
        self.assertEqual(config["selector"]["novelty_lambda"], 0.45)
```

- [ ] **Step 2: Run tests**

Run:

```powershell
conda run -n pretext python -m unittest paper-new-xiugai.tests.test_diagnosis_configs -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add paper-new-xiugai/tests/test_diagnosis_configs.py
git commit -m "test: lock diagnosis config semantics"
```

## Task 5: Add execution notes for diagnosis screening

**Files:**
- Modify: `paper-new-xiugai/docs/2026-04-25-screening-diagnosis-plan.md`

- [ ] **Step 1: Add experiment matrix and recommended run order**

Append this table to the plan:

```markdown
## Diagnosis Run Order

1. `ns_d1_forums`
2. `ns_d1_microblog`
3. `ns_d1_jobs`
4. `ns_d1_congressional`
5. `ns_d2_forums`
6. `ns_d2_microblog`
7. `ns_d2_jobs`
8. `ns_d2_congressional`
9. `ns_d3_forums`
10. `ns_d3_microblog`
11. `ns_d3_jobs`
12. `ns_d3_congressional`

Interpretation rule:

- 先看 `forums/microblog` 是否回升。
- 再看 `jobs/congressional` 是否明显退化。
- 只有“弱数据集改善、强数据集不明显受损”的 variant，才值得进入下一轮算法级修改。
```

- [ ] **Step 2: Run a representative config resolve check**

Run:

```powershell
conda run -n pretext python -c "from pathlib import Path; from paper_new_selector.thesis_bridge import load_yaml_config; p=Path(r'D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-xiugai\configs\experiments\single_node_diagnosis\ns_d3_forums.yaml'); c=load_yaml_config(p); print(c['meta']['experiment_id'], c['selector']['top_q'], c['bootstrap']['num_prompts'], c['eval']['small_epochs'])"
```

Expected output:

```text
ns_d3_forums 2 100 6
```

- [ ] **Step 3: Commit**

```bash
git add paper-new-xiugai/docs/2026-04-25-screening-diagnosis-plan.md
git commit -m "docs: add diagnosis screening run order"
```

## Self-Review

- Spec coverage: plan covers baseline bootstrap, diagnosis config templates, dataset configs, and parsing tests. No algorithm-code modifications are included, consistent with the “先配置诊断、后算法改造” scope.
- Placeholder scan: no `TODO` / `TBD` placeholders remain.
- Type consistency: all diagnosis overrides stay inside existing `selector` numeric fields and existing `data/bootstrap/eval` fields; no new runtime keys are introduced.

## Diagnosis Run Order

1. `ns_d1_forums`
2. `ns_d1_microblog`
3. `ns_d1_jobs`
4. `ns_d1_congressional`
5. `ns_d2_forums`
6. `ns_d2_microblog`
7. `ns_d2_jobs`
8. `ns_d2_congressional`
9. `ns_d3_forums`
10. `ns_d3_microblog`
11. `ns_d3_jobs`
12. `ns_d3_congressional`

Interpretation rule:

- 先看 `forums/microblog` 是否回升。
- 再看 `jobs/congressional` 是否明显退化。
- 只有“弱数据集改善、强数据集不明显受损”的 variant，才值得进入下一轮算法级修改。
