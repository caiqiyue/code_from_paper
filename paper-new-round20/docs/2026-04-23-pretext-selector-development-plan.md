# Paper-New Pre-Text Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `paper-new` 中实现单节点版创新算法，只重设计 `PrE-Text Stage 1 selector`，保持固定提示词、固定生成器、固定 `PrE-Text` Stage 2 bootstrap，全程不微调大模型。

**Architecture:** 新算法核心代码全部位于 `paper-new`。`thesis_platform` 只提供数据集、本地模型路径、embedder 和固定候选生成器入口，`PrE-Text` 只提供 Stage 2 bootstrap 与后续评测能力。算法主线必须落实为 `private_support - genericity_penalty - redundancy_penalty` 的动态贪心 selector，并把 `hard negative` 升级为显式 `boundary_state`，而不是简单低分样本归档。

**Tech Stack:** Python 3.10, PyTorch, SentenceTransformers MiniLM, HuggingFace Transformers, vLLM, YAML, `unittest`, active environment `pretext`.

---

## Non-Negotiables

1. 新版创新算法主代码只能写在 `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new` 下。
2. `thesis_platform` 和 `PrE-Text` 只能作为桥接依赖，不能承载新版核心 selector 逻辑。
3. 不做 LLM 微调，不做 DPO、LoRA、RLHF，不走多轮 Prompt 改写链。
4. `Stage 2` 继续复用 `PrE-Text` bootstrap，不把创新点写到生成器上。
5. `redundancy_penalty` 必须进入真实贪心选种过程，不能只停留在静态打分或文档表述。
6. `R_t` 必须统一定义为 `boundary negatives`，不能混同于“全局最低分样本尾部”。
7. `hard negative` 必须导出为 `boundary_state`，而不是只保存文本列表。
8. 候选生成器 `G` 必须只有一个合法实现来源，禁止实现者私自替换。

## Full Algorithm Contract

### Inputs

- `D_priv`: 私有训练语料
- `D_init`: 公共初始化池
- `E`: 冻结的文本编码器
- `G`: 固定候选生成器
- `Q`: `Top-Q` 排序投票的 `Q`
- `alpha_r`: rank 衰减权重列表
- `private_knn_k`: 私有样本局部密度近邻数
- `reference_top_k`: 公共初始化相似度惩罚近邻数
- `density_lambda`: 私有样本局部代表性权重
- `novelty_lambda`: 私有样本新颖性权重
- `length_lambda`: 私有样本长度稳定性权重
- `length_floor`: 文本最小稳定长度
- `length_ceiling`: 文本最大稳定长度
- `lambda_generic`: `genericity_penalty` 系数
- `lambda_redundancy`: `redundancy_penalty` 系数
- `seed_top_k`: 最终保留 seed 数量
- `hard_negative_top_k`: 最终保留 boundary negative 数量
- `candidate_count`: Stage 1 候选池大小
- `generated_per_round`: 固定生成器单轮产出数
- `exemplars_per_prompt`: few-shot exemplar 数
- `initial_prompt`: 固定提示词文本
- `bootstrap_num_prompts`: Stage 2 bootstrap 数量

### Outputs

- `S_t`: 最终 seed 集合
- `R_t`: boundary negative 集合
- `boundary_state`: 由 `R_t` 导出的拒绝边界状态
- `Y_syn`: Stage 2 生成的最终合成语料

### Stage 0: Freeze Boundary

1. 固定提示词 `initial_prompt`。
2. 固定候选生成器 `G`。
3. 固定文本编码器 `E`。
4. 明确方法只在 `Stage 1 selector` 上创新。

### Stage 1: Generate Candidate Pool

1. 从 `D_init` 采样公共初始化样本。
2. 用固定生成器 `G` 生成候选池 `C_t`。
3. 对 `C_t` 做基础清洗：
   - 去空文本
   - 去异常短文本
   - 去明显损坏文本

### Stage 2: Build Private Importance Prior `w(x)`

对每条私有样本 `x ∈ D_priv` 计算重要性先验 `w(x)`，由三类信号组成：

1. 局部代表性：样本是否接近私有分布中心。
2. 新颖性/稀缺性：样本是否代表较少出现但重要的模式。
3. 信息密度/长度稳定性：样本是否过短、模板化或口水化。

### Stage 3: Compute `private_support`

对每条私有样本 `x`：

1. 用 `E` 计算它与所有候选 `c ∈ C_t` 的相似度。
2. 取 `Top-Q` 而不是原始 `PrE-Text` 的 `Top-1`。
3. 按 rank 分配衰减权重 `alpha_r`。
4. 再乘上私有样本重要性先验 `w(x)`。

定义：

`private_support(c) = Σ_{x∈D_priv} w(x) * Σ_{r=1}^{Q} alpha_r * I[c 是 x 的第 r 个最近候选]`

### Stage 4: Compute `genericity_penalty`

对每个候选 `c`：

1. 比较 `c` 与公共初始化池 `D_init` 的相似度。
2. 若 `c` 过于接近公共初始化分布、常见模板表达、或过于安全宽泛的措辞，则提高惩罚。

输出：

`genericity_penalty(c)`

### Stage 5: Greedy Selection with Dynamic `redundancy_penalty`

对每个待选候选 `c`：

1. 将其与当前已选 seed 集合 `S_t` 比较。
2. 若 `c` 与已选 seed 高度相似，则提高 `redundancy_penalty(c)`。
3. 若 `c` 能带来新结构、新措辞、新细节，则降低惩罚。
4. `redundancy_penalty(c)` 必须在贪心选种过程中动态更新，不能在静态 scorer 中一次性写死。

### Stage 6: Final Acceptance Score and Candidate Decision

定义：

`accept_score(c) = private_support(c) - lambda_generic * genericity_penalty(c) - lambda_redundancy * redundancy_penalty(c)`

选择规则：

1. 按 `accept_score` 贪心选择高分候选进入 `S_t`。
2. 未被选中但靠近接受边界的拒绝候选进入 `R_t`。
3. 因高冗余被排除、且本来具有较高接受分数的候选也进入 `R_t`。

这里的 `R_t` 不是“全局最低分尾部”，而是 `boundary negatives`。

### Stage 7: Build `boundary_state`

`R_t` 不能只保存文本，必须进一步导出一个显式拒绝边界状态：

1. `reject_score_ceiling`: boundary negative 中的最高拒绝分数
2. `reject_score_floor`: boundary negative 中的最低拒绝分数
3. `negative_centroid`: boundary negative embedding 中心
4. `negative_pattern_stats`: 通用模板/近重复模式统计

作用：

1. 为后续轮次或后续扩展提供拒绝阈值校准。
2. 阻止与 `R_t` 高度相似的候选再次被误选。
3. 显式表达“哪些候选不该进入 seed set”。

### Stage 8: Fixed Stage 2 Bootstrap

1. 用 `S_t` 构造 bootstrap prompts。
2. 调用 `PrE-Text` 的 `build_bootstrap_prompts` 和 `generate_bootstrapped_samples`。
3. 生成最终合成语料 `Y_syn`。

## Difference from Original PrE-Text

| 维度 | 原始 `PrE-Text` | 本创新路线 |
|---|---|---|
| Stage 1 私有反馈 | `Top-1` 最近邻直方图 | 加权 `Top-Q` 排序投票 |
| 私有样本权重 | 平权 | `w(x)` 重要性先验 |
| 候选接受依据 | 相似度命中统计 + 噪声/阈值 | `private_support - genericity_penalty - redundancy_penalty` |
| 拒绝样本处理 | 直接淘汰 | `R_t` + `boundary_state` |
| Prompt 路线 | 可继续扩展 | 明确不依赖多轮 Prompt 改写 |
| Stage 2 | 固定 bootstrap | 继续固定 bootstrap |

## File Structure

### New Code Under `paper-new`

- Create: `paper-new/paper_new_selector/__init__.py`
- Create: `paper-new/paper_new_selector/contracts.py`
  - 定义 `CandidateRecord`、`SelectorDecision`、`BoundaryState`、`GeneratorContract`。
- Create: `paper-new/paper_new_selector/importance.py`
  - 计算 `w(x)`。
- Create: `paper-new/paper_new_selector/support.py`
  - 计算 `private_support`。
- Create: `paper-new/paper_new_selector/genericity.py`
  - 计算 `genericity_penalty`。
- Create: `paper-new/paper_new_selector/redundancy.py`
  - 动态计算 `redundancy_penalty`。
- Create: `paper-new/paper_new_selector/boundary.py`
  - 由 `R_t` 构造 `boundary_state`。
- Create: `paper-new/paper_new_selector/selector.py`
  - 动态贪心选种主流程，显式落实 `accept_score`。
- Create: `paper-new/paper_new_selector/thesis_bridge.py`
  - 读取 `thesis_platform` 数据、embedder、路径。
- Create: `paper-new/paper_new_selector/generator_bridge.py`
  - 唯一构造固定候选生成器 `G`，显式输出 generator contract。
- Create: `paper-new/paper_new_selector/pretext_bridge.py`
  - 解析并调用 `PrE-Text` bootstrap。
- Create: `paper-new/paper_new_selector/stage1_runner.py`
  - 单节点 Stage 1 总控。
- Create: `paper-new/paper_new_selector/pipeline.py`
  - `Stage 1 selector -> Stage 2 bootstrap -> eval` 主流水。
- Create: `paper-new/configs/single_node_jobs_selector.yaml`
- Create: `paper-new/scripts/run_selector_single_node.py`
- Create: `paper-new/tests/test_config.py`
- Create: `paper-new/tests/test_importance.py`
- Create: `paper-new/tests/test_support.py`
- Create: `paper-new/tests/test_selector.py`
- Create: `paper-new/tests/test_boundary.py`
- Create: `paper-new/tests/test_generator_bridge.py`
- Create: `paper-new/tests/test_pretext_bridge.py`
- Create: `paper-new/tests/test_pipeline_smoke.py`

### Read-Only External Dependencies

- Reuse only: `thesis_platform/datasets`
- Reuse only: `thesis_platform/open_model`
- Reuse only: `thesis_platform/models/embedding.py`
- Reuse only: `thesis_platform/adapters/generators/pretext_prompt_generator.py`
- Reuse only: `PrE-Text/pretext_platform/algorithms/bootstrap.py`
- Reuse only: `PrE-Text/pretext_platform/evaluation/*`

### External Repos Must Not Host New Core Logic

- Do not create new selector logic under `thesis_platform`.
- Do not create new selector logic under `PrE-Text`.
- Do not patch external repos just to fit the new algorithm layout unless a real blocker appears.

## Environment Rules

1. 只使用 `pretext` 环境。
2. 命令默认从 `D:\学习记录\导师项目\研究\caiqiyue_file` 运行。
3. `paper-new` 代码通过桥接模块 import `thesis_platform` 和 `pretext_platform`。
4. 本地模型路径指向 `thesis_platform/open_model`。
5. 数据集路径指向 `thesis_platform/datasets`。
6. `Stage 2` 继续遵守 `bootstrap.generator_backend='vllm'`。
7. 运行测试和脚本前显式设置 `PYTHONPATH=paper-new`。

### Task 1: Freeze the package layout and full config contract

**Files:**
- Create: `paper-new/paper_new_selector/__init__.py`
- Create: `paper-new/configs/single_node_jobs_selector.yaml`
- Create: `paper-new/tests/test_config.py`

- [ ] **Step 1: Write the failing config contract test**

```python
import unittest
from pathlib import Path
import yaml


class PaperNewSelectorConfigTests(unittest.TestCase):
    def test_config_fully_defines_algorithm_contract(self):
        config_path = Path("paper-new/configs/single_node_jobs_selector.yaml")
        self.assertTrue(config_path.exists())
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["pipeline"]["stage1_mode"], "selector_seed_search")
        self.assertEqual(config["pipeline"]["stage2_mode"], "pretext_bootstrap")
        self.assertEqual(config["paths"]["datasets_root"], "thesis_platform/datasets")
        self.assertEqual(config["paths"]["models_root"], "thesis_platform/open_model")
        self.assertEqual(config["generator"]["backend"], "thesis_pretext_prompt")
        self.assertEqual(config["generator"]["candidate_count"], 100)
        self.assertEqual(config["selector"]["rank_weights"], [1.0, 0.6, 0.3, 0.15])
        self.assertEqual(config["selector"]["private_knn_k"], 8)
        self.assertEqual(config["selector"]["reference_top_k"], 4)
        self.assertEqual(config["selector"]["density_lambda"], 0.50)
        self.assertEqual(config["selector"]["novelty_lambda"], 0.30)
        self.assertEqual(config["selector"]["length_lambda"], 0.20)
```

- [ ] **Step 2: Run the config test and verify it fails**

Run: `conda activate pretext`

Run: `cd D:\学习记录\导师项目\研究\caiqiyue_file`

Run: `$env:PYTHONPATH="paper-new"`

Run: `python -m unittest discover -s paper-new/tests -p "test_config.py" -v`

Expected: `FAIL` because the config and package marker do not exist yet.

- [ ] **Step 3: Create the package marker and config**

```yaml
pipeline:
  stage1_mode: selector_seed_search
  stage2_mode: pretext_bootstrap
  run_eval: true

paths:
  datasets_root: thesis_platform/datasets
  models_root: thesis_platform/open_model
  pretext_root: PrE-Text

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json

generator:
  backend: thesis_pretext_prompt
  source: thesis_platform/adapters/generators/pretext_prompt_generator.py
  initial_prompt: "Generate realistic job-domain texts that match the private corpus style."
  candidate_count: 100
  generated_per_round: 16
  exemplars_per_prompt: 3
  max_new_tokens: 96

selector:
  top_q: 4
  rank_weights: [1.0, 0.6, 0.3, 0.15]
  seed_top_k: 10
  hard_negative_top_k: 10
  private_knn_k: 8
  reference_top_k: 4
  density_lambda: 0.50
  novelty_lambda: 0.30
  length_lambda: 0.20
  length_floor: 12
  length_ceiling: 128
  lambda_generic: 0.35
  lambda_redundancy: 0.25

bootstrap:
  num_prompts: 1500
  generator_backend: vllm
  generator_model: llama2_7b
```

- [ ] **Step 4: Re-run the config test**

Run: `python -m unittest discover -s paper-new/tests -p "test_config.py" -v`

Expected: `OK`.

### Task 2: Implement the pure selector math

**Files:**
- Create: `paper-new/paper_new_selector/importance.py`
- Create: `paper-new/paper_new_selector/support.py`
- Create: `paper-new/paper_new_selector/genericity.py`
- Create: `paper-new/tests/test_importance.py`
- Create: `paper-new/tests/test_support.py`

- [ ] **Step 1: Write failing unit tests**

```python
import unittest

from paper_new_selector.importance import build_private_importance_weights
from paper_new_selector.support import compute_private_support
from paper_new_selector.genericity import compute_genericity_penalty


class SelectorMathTests(unittest.TestCase):
    def test_importance_weights_downweight_short_generic_private_samples(self):
        weights = build_private_importance_weights(
            private_vectors=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            private_lengths=[32, 28, 2],
            private_knn_k=2,
            density_lambda=0.5,
            novelty_lambda=0.3,
            length_lambda=0.2,
            length_floor=12,
            length_ceiling=128,
        )
        self.assertGreater(weights[0], weights[2])

    def test_private_support_uses_topq_ranked_votes(self):
        scores = compute_private_support(
            private_vectors=[[1.0, 0.0]],
            candidate_vectors=[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
            private_weights=[1.0],
            rank_weights=[1.0, 0.6],
            top_q=2,
        )
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_genericity_penalty_is_high_for_public_template_like_candidates(self):
        penalty = compute_genericity_penalty(
            candidate_vector=[1.0, 0.0],
            reference_vectors=[[0.99, 0.01], [0.98, 0.02]],
            reference_top_k=2,
        )
        self.assertGreater(penalty, 0.9)
```

- [ ] **Step 2: Run the unit tests and verify they fail**

Run: `$env:PYTHONPATH="paper-new"`

Run: `python -m unittest discover -s paper-new/tests -p "test_*.py" -v`

Expected: `FAIL`.

- [ ] **Step 3: Implement the three pure math modules**

Rules:

1. `importance.py` 只负责 `w(x)`。
2. `support.py` 只负责 `private_support`。
3. `genericity.py` 只负责 `genericity_penalty`。
4. 这三个模块都不能偷偷引入 `redundancy_penalty`。

- [ ] **Step 4: Re-run the tests**

Run: `$env:PYTHONPATH="paper-new"`

Run: `python -m unittest discover -s paper-new/tests -p "test_importance.py" -v`

Run: `python -m unittest discover -s paper-new/tests -p "test_support.py" -v`

Expected: `OK`.

### Task 3: Implement dynamic redundancy-aware selection and boundary negatives

**Files:**
- Create: `paper-new/paper_new_selector/redundancy.py`
- Create: `paper-new/paper_new_selector/boundary.py`
- Create: `paper-new/paper_new_selector/selector.py`
- Create: `paper-new/tests/test_selector.py`
- Create: `paper-new/tests/test_boundary.py`

- [ ] **Step 1: Write failing selection tests**

```python
import unittest

from paper_new_selector.selector import greedy_select_candidates
from paper_new_selector.boundary import build_boundary_state


class SelectorDecisionTests(unittest.TestCase):
    def test_redundancy_penalty_prevents_duplicate_seeds_and_marks_boundary_negative(self):
        result = greedy_select_candidates(
            candidate_vectors=[[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
            private_support=[1.0, 0.98, 0.92],
            genericity_penalty=[0.1, 0.1, 0.1],
            lambda_generic=0.35,
            lambda_redundancy=0.25,
            seed_top_k=2,
            hard_negative_top_k=1,
        )
        self.assertEqual(result.selected_indices, [0, 2])
        self.assertEqual(result.hard_negative_indices, [1])
        self.assertEqual(result.hard_negative_reason[1], "near_boundary_rejected")

    def test_boundary_state_keeps_more_than_raw_negative_texts(self):
        boundary = build_boundary_state(
            reject_scores=[0.11, 0.08, 0.03],
            reject_vectors=[[1.0, 0.0], [0.95, 0.05], [0.9, 0.1]],
        )
        self.assertIn("reject_score_ceiling", boundary)
        self.assertIn("reject_score_floor", boundary)
        self.assertIn("negative_centroid", boundary)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `$env:PYTHONPATH="paper-new"`

Run: `python -m unittest discover -s paper-new/tests -p "test_selector.py" -v`

Run: `python -m unittest discover -s paper-new/tests -p "test_boundary.py" -v`

Expected: `FAIL`.

- [ ] **Step 3: Implement the three modules**

Must hold:

1. `selector.py` 中的 `accept_score` 真正写成  
   `private_support - lambda_generic * genericity_penalty - lambda_redundancy * redundancy_penalty`
2. `redundancy_penalty` 必须在贪心过程中动态更新。
3. `R_t` 必须统一定义为“靠近接受边界的拒绝候选 + 因高冗余被拒绝的候选”。
4. `boundary.py` 必须输出 `boundary_state`，而不是只回传 `R_t` 文本列表。

- [ ] **Step 4: Re-run the tests**

Run: `$env:PYTHONPATH="paper-new"`

Run: `python -m unittest discover -s paper-new/tests -p "test_selector.py" -v`

Run: `python -m unittest discover -s paper-new/tests -p "test_boundary.py" -v`

Expected: `OK`.

### Task 4: Add bridges without moving core logic out of `paper-new`

**Files:**
- Create: `paper-new/paper_new_selector/thesis_bridge.py`
- Create: `paper-new/paper_new_selector/generator_bridge.py`
- Create: `paper-new/paper_new_selector/pretext_bridge.py`
- Create: `paper-new/tests/test_generator_bridge.py`
- Create: `paper-new/tests/test_pretext_bridge.py`

- [ ] **Step 1: Write failing bridge tests**

```python
import unittest

from paper_new_selector.generator_bridge import build_candidate_generator
from paper_new_selector.pretext_bridge import prepare_bootstrap_runtime, resolve_bootstrap_model_path
from paper_new_selector.thesis_bridge import resolve_dataset_paths


class BridgeTests(unittest.TestCase):
    def test_thesis_bridge_resolves_existing_dataset_roots(self):
        train_path, eval_path, init_path = resolve_dataset_paths("paper-new/configs/single_node_jobs_selector.yaml")
        self.assertIn("thesis_platform/datasets", train_path.as_posix())
        self.assertIn("thesis_platform/datasets", eval_path.as_posix())
        self.assertIn("thesis_platform/datasets", init_path.as_posix())

    def test_generator_bridge_uses_one_fixed_generator_source(self):
        generator = build_candidate_generator("paper-new/configs/single_node_jobs_selector.yaml")
        self.assertEqual(generator.contract["backend"], "thesis_pretext_prompt")
        self.assertIn("pretext_prompt_generator", generator.contract["source"])

    def test_pretext_bridge_uses_existing_open_model_root(self):
        model_path = resolve_bootstrap_model_path("thesis_platform/open_model", "llama2_7b")
        self.assertIn("thesis_platform/open_model", model_path.as_posix())

    def test_pretext_bridge_prepares_real_bootstrap_call_contract(self):
        runtime = prepare_bootstrap_runtime("paper-new/configs/single_node_jobs_selector.yaml")
        self.assertTrue(callable(runtime["build_bootstrap_prompts"]))
        self.assertTrue(callable(runtime["generate_bootstrapped_samples"]))
        self.assertEqual(runtime["bootstrap_cfg"]["generator_backend"], "vllm")
```

- [ ] **Step 2: Run the bridge tests and verify they fail**

Run: `$env:PYTHONPATH="paper-new"`

Run: `python -m unittest discover -s paper-new/tests -p "test_generator_bridge.py" -v`

Run: `python -m unittest discover -s paper-new/tests -p "test_pretext_bridge.py" -v`

Expected: `FAIL`.

- [ ] **Step 3: Implement the bridges**

Rules:

1. `thesis_bridge.py` 只能读取 `thesis_platform` 的现有路径、embedder、dataset 文件。
2. `generator_bridge.py` 必须唯一绑定 `thesis_platform/adapters/generators/pretext_prompt_generator.py`，并输出显式 generator contract。
3. `pretext_bridge.py` 不能只解析路径，必须真正解析出 `build_bootstrap_prompts` 和 `generate_bootstrapped_samples` 两个可调用对象，以及规范化后的 `bootstrap_cfg`。
4. 不在桥接层复制外部仓库算法主体。

- [ ] **Step 4: Re-run the bridge tests**

Run: `$env:PYTHONPATH="paper-new"`

Run: `python -m unittest discover -s paper-new/tests -p "test_generator_bridge.py" -v`

Run: `python -m unittest discover -s paper-new/tests -p "test_pretext_bridge.py" -v`

Expected: `OK`.

### Task 5: Build the single-node pipeline and validate it

**Files:**
- Create: `paper-new/paper_new_selector/stage1_runner.py`
- Create: `paper-new/paper_new_selector/pipeline.py`
- Create: `paper-new/scripts/run_selector_single_node.py`
- Create: `paper-new/tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write the failing pipeline smoke test**

```python
import unittest

from paper_new_selector.pipeline import run_pipeline


class PipelineSmokeTests(unittest.TestCase):
    def test_pipeline_returns_stage1_stage2_boundary_and_generator_contract(self):
        summary = run_pipeline("paper-new/configs/single_node_jobs_selector.yaml", validate_only=True)
        self.assertIn("stage1", summary)
        self.assertIn("stage2", summary)
        self.assertIn("boundary_state", summary["stage1"])
        self.assertIn("generator_contract", summary)
```

- [ ] **Step 2: Run the smoke test and verify it fails**

Run: `$env:PYTHONPATH="paper-new"`

Run: `python -m unittest discover -s paper-new/tests -p "test_pipeline_smoke.py" -v`

Expected: `FAIL`.

- [ ] **Step 3: Implement the pipeline**

Required execution order:

1. 通过 `thesis_bridge` 读取数据、构造 embedder。
2. 通过 `generator_bridge` 构造唯一固定候选生成器 `G`。
3. 运行 `Stage 1 selector` 得到 `S_t`、`R_t`、`boundary_state`。
4. 通过 `pretext_bridge` 调 `PrE-Text` bootstrap 得到 `Y_syn`。
5. 维持后续 small eval 接口可接。

- [ ] **Step 4: Run validate-only and the full unit suite**

Run: `conda activate pretext`

Run: `cd D:\学习记录\导师项目\研究\caiqiyue_file`

Run: `$env:PYTHONPATH="paper-new"`

Run: `python -m unittest discover -s paper-new/tests -p "test_*.py" -v`

Run: `python paper-new/scripts/run_selector_single_node.py --config paper-new/configs/single_node_jobs_selector.yaml --validate-only`

Expected:

- `unittest`: `OK`
- validate-only: summary contains `stage1_mode=selector_seed_search`, `stage2_mode=pretext_bootstrap`, resolved dataset/model paths under `thesis_platform`, one fixed generator contract, and resolved bootstrap callables

## Success Criteria

1. 新算法主代码全部位于 `paper-new`。
2. `paper-new` 能独立运行 selector 单节点 pipeline。
3. 配置文件足以唯一确定算法，不依赖关键超参硬编码。
4. `redundancy_penalty` 在真实贪心选种中生效。
5. `R_t` 被统一实现为 boundary negatives，而不是“最低分样本列表”。
6. `hard negative` 被提升为 `boundary_state`，不是简单归档。
7. `Stage 2` 由 `PrE-Text` bootstrap 生成，且不微调模型。
8. 所有数据集和本地模型仍来自 `thesis_platform/datasets` 与 `thesis_platform/open_model`。
9. 候选生成器 `G` 只有一个合法实现来源。

## Risks To Watch

1. 不要把 `paper-new` 变成对 `thesis_platform` 的复制仓。
2. 不要把 `redundancy_penalty` 偷偷降级为静态排序后的附加注释。
3. 不要把 `R_t` 和“最低分样本尾部”混为一谈。
4. 不要把 `hard negative` 简化成“最低分样本列表”。
5. 不要让桥接模块反向要求修改 `PrE-Text` 核心算法。
6. 不要让不同实现者私自替换固定生成器入口。
7. 不要在任何步骤引入生成器微调。
