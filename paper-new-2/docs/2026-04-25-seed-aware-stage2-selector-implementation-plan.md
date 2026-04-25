# Seed-Aware Stage2 Corpus Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `paper-new-2` 中实现一个基于 `PrE-Text` Stage 2 bootstrap 产物的 `Seed-Aware Synthetic Corpus Selector`，在不改变 `Stage 1` DP seed 生成和 bootstrap 主干的前提下，对合成语料做一致性、模板化和近重复控制，并跑通 screening 实验代码。

**Architecture:** `paper-new-2` 沿用 `paper-new` 的“独立 package + bridge + pipeline + tests + configs”写法。`PrE-Text` 继续作为只读依赖，负责原始 `Stage 1`、bootstrap 生成和 downstream eval；`paper-new-2` 只在 `bootstrap outputs -> eval` 之间插入 `seed-aware selector`。由于原始 `build_bootstrap_prompts()` 不保留 seed 元数据，`paper-new-2` 必须镜像该 prompt builder 的 prompt 模板与 RNG 语义，并额外保留 `prompt -> seed_texts -> generated_text` 的映射。

**Tech Stack:** Python 3.10, `paper-new-2`, `paper-new`, `PrE-Text`, `thesis_platform`, SentenceTransformers MiniLM, vLLM, YAML, `unittest`, active environment `pretext`

---

## Non-Negotiables

1. 新创新代码只能写在 `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2` 下。
2. 第一轮创新只允许是 `Stage 2` 后验质量控制器，不允许同时引入 `generation budget control`。
3. `PrE-Text` 的 `Stage 1` DP seeds 与 bootstrap 主干必须保持不变，不能悄悄改 prompt 模板、生成 backend 或 eval 协议。
4. `seed-aware` 不是拍脑袋命名，必须在实现里保留 `prompt_text`、`seed_texts`、`generated_text` 三者的映射关系。
5. 第一轮 screening 必须把 `selected_train_count` 控制为与 baseline 清洗后语料数量同量级；创新回答的是“哪些样本更该保留”，不是“多生成/少生成多少”。
6. 读取 bootstrap outputs 的最小清洗逻辑必须对齐 `PrE-Text/pretext_platform/evaluation/distilgpt2_eval.py` 的现有行为，避免出现“选择器和评测器看的是两套文本”的伪创新。
7. 除非出现真实阻塞，不修改 `paper-new` 和 `PrE-Text` 源码。

## File Structure

### New Code Under `paper-new-2`

- Create: `paper-new-2/paper_new_stage2_selector/__init__.py`
- Create: `paper-new-2/paper_new_stage2_selector/contracts.py`
  - 定义 `BootstrapPromptRecord`、`GeneratedSampleRecord`、`ScoredSampleRecord`、`Stage2SelectionResult`
- Create: `paper-new-2/paper_new_stage2_selector/thesis_bridge.py`
  - 负责加载 YAML、解析 repo root、构造 `PrE-Text` 的 `ExperimentConfig`
- Create: `paper-new-2/paper_new_stage2_selector/bootstrap_bridge.py`
  - 镜像 `PrE-Text` 的 prompt builder，同时保留 prompt metadata，并桥接原始 vLLM bootstrap
- Create: `paper-new-2/paper_new_stage2_selector/corpus_loader.py`
  - 把 bootstrap outputs 解析成带 metadata 的记录，并复刻 baseline 文本清洗
- Create: `paper-new-2/paper_new_stage2_selector/consistency.py`
  - 计算生成文本和 prompt seed 集的语义一致性
- Create: `paper-new-2/paper_new_stage2_selector/template_penalty.py`
  - 计算 prompt echo / 模板化 / 异常短文本惩罚
- Create: `paper-new-2/paper_new_stage2_selector/dedup.py`
  - 计算 exact duplicate / near duplicate 惩罚
- Create: `paper-new-2/paper_new_stage2_selector/selector.py`
  - 执行硬过滤、排序打分和 fixed-target 语料选择
- Create: `paper-new-2/paper_new_stage2_selector/eval_bridge.py`
  - 把选中的语料写成 `PrE-Text` 兼容的 stage2 产物，并复用现有小模型 eval
- Create: `paper-new-2/paper_new_stage2_selector/pipeline.py`
  - 串起 `PrE-Text Stage 1 -> PrE-Text bootstrap -> seed-aware selector -> eval`
- Create: `paper-new-2/paper_new_stage2_selector/run_stage2_seed_aware_single_node.py`
  - 单节点 CLI 入口
- Create: `paper-new-2/configs/base/stage2_seed_aware_base.yaml`
- Create: `paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml`
- Create: `paper-new-2/configs/experiments/single_node_screening/sas_s_congressional_screening.yaml`
- Create: `paper-new-2/configs/experiments/single_node_screening/sas_s_forums_screening.yaml`
- Create: `paper-new-2/configs/experiments/single_node_screening/sas_s_microblog_screening.yaml`
- Create: `paper-new-2/configs/experiments/single_node_formal/sas_c1_jobs_base.yaml`
- Create: `paper-new-2/tests/test_config.py`
- Create: `paper-new-2/tests/test_bootstrap_bridge.py`
- Create: `paper-new-2/tests/test_corpus_loader.py`
- Create: `paper-new-2/tests/test_consistency.py`
- Create: `paper-new-2/tests/test_template_penalty.py`
- Create: `paper-new-2/tests/test_selector.py`
- Create: `paper-new-2/tests/test_eval_bridge.py`
- Create: `paper-new-2/tests/test_pipeline_smoke.py`

### Read-Only External Dependencies

- Reuse only: `paper-new/paper_new_selector/thesis_bridge.py`
- Reuse only: `paper-new/paper_new_selector/eval_bridge.py`
- Reuse only: `PrE-Text/pretext_platform/core/pipeline.py`
- Reuse only: `PrE-Text/pretext_platform/algorithms/bootstrap.py`
- Reuse only: `PrE-Text/pretext_platform/evaluation/distilgpt2_eval.py`
- Reuse only: `PrE-Text/pretext_platform/evaluation/gpt2_eval.py`

### External Repos Must Not Host New Core Logic

- 不要把 selector 逻辑写回 `paper-new`
- 不要把 seed-aware metadata 逻辑写回 `PrE-Text`
- 不要在 `PrE-Text` 里新增“只为 paper-new-2 服务”的 prompt builder 变体

## Task 1: Freeze the package layout and experiment config contract

**Files:**
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\__init__.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\configs\base\stage2_seed_aware_base.yaml`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\configs\experiments\single_node_screening\sas_s_jobs_screening.yaml`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\tests\test_config.py`

- [ ] **Step 1: Write the failing config contract test**

```python
import unittest
from pathlib import Path

import yaml


class PaperNew2ConfigTests(unittest.TestCase):
    def test_jobs_screening_config_defines_stage2_seed_aware_contract(self):
        config_path = Path("paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml")
        self.assertTrue(config_path.exists())
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["pipeline"]["stage1_mode"], "pretext_stage1_passthrough")
        self.assertEqual(config["pipeline"]["stage2_mode"], "pretext_bootstrap_seed_aware_selector")
        self.assertEqual(config["selector"]["target_count_mode"], "match_baseline_clean_count")
        self.assertEqual(config["selector"]["consistency_metric"], "max_seed_cosine")
        self.assertEqual(config["bootstrap"]["generator_backend"], "vllm")
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
conda run -n pretext python -m unittest paper-new-2/tests/test_config.py -v
```

Expected:

```text
FAIL because the package marker and configs do not exist yet
```

- [ ] **Step 3: Create the package marker and base config**

`paper-new-2/paper_new_stage2_selector/__init__.py`

```python
"""Seed-aware Stage 2 selector experiment package for paper-new-2."""

__all__ = [
    "contracts",
    "bootstrap_bridge",
    "corpus_loader",
    "selector",
    "pipeline",
]
```

`paper-new-2/configs/base/stage2_seed_aware_base.yaml`

```yaml
meta:
  experiment_id: stage2_seed_aware_base
  seed: 42

pipeline:
  stage1_mode: pretext_stage1_passthrough
  stage2_mode: pretext_bootstrap_seed_aware_selector
  run_eval: true

paths:
  datasets_root: thesis_platform/datasets
  models_root: thesis_platform/open_model
  pretext_root: PrE-Text
  output_root: paper-new-2/outputs/stage2_seed_aware_base

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
  train_limit: 256
  eval_limit: 256
  initialization_limit: 1024

embedding:
  model_path: thesis_platform/open_model/all_minilm_l6_v2
  device: cpu

stage1:
  rounds: 6
  mask: 0.3
  lookahead: 4
  multiplier: 4
  seq_len: 64
  t_steps: 2
  batch_size: 64
  embed_batch_size: 128
  temperature: 1.0
  top_p: 1.0
  top_k: 0
  nearest_neighbors_print: 3
  H_multiplier: 0.25
  delta: 3e-6
  sigma: 11.3
  sensitivity: 8

bootstrap:
  enabled: true
  num_prompts: 100
  generator_backend: vllm
  generator_model: llama2_7b
  temperature: 1.0
  top_p: 1.0
  max_tokens: 85
  max_model_len: 512
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.35
  startup_required_free_gb: 2
  enforce_eager: true
  device: cuda
  batch_size: 1

selector:
  target_count_mode: match_baseline_clean_count
  consistency_metric: max_seed_cosine
  consistency_threshold: 0.42
  duplicate_threshold: 0.95
  min_words: 4
  prompt_echo_ngram: 8
  unique_token_ratio_floor: 0.45
  w_consistency: 1.0
  w_template: 0.35
  w_duplicate: 0.30

eval:
  enabled: true
  mode: pretext_small
  small_eval_mode: gpt2
  device: cuda
  max_samples_per_client: 8
  initialization_min_words: 20
  small_epochs: 6
  small_batch_size: 8
  small_eval_batch_size: 2
  small_grad_accum_steps: 8
  small_cutoff_len: 64
  small_learning_rate: 0.0002
  small_num_proc: 1
```

- [ ] **Step 4: Create the first screening config**

`paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml`

```yaml
inherits:
  - ../../base/stage2_seed_aware_base.yaml

meta:
  experiment_id: sas_s_jobs_screening
  seed: 42

paths:
  output_root: paper-new-2/outputs/sas_s_jobs_screening

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
  train_limit: 256
  eval_limit: 256
  initialization_limit: 1024

bootstrap:
  num_prompts: 100
```

- [ ] **Step 5: Re-run the config test**

Run:

```bash
conda run -n pretext python -m unittest paper-new-2/tests/test_config.py -v
```

Expected:

```text
OK
```

- [ ] **Step 6: Commit**

```bash
git add paper-new-2/paper_new_stage2_selector/__init__.py paper-new-2/configs paper-new-2/tests/test_config.py
git commit -m "feat: add paper-new-2 config contract for stage2 selector"
```

## Task 2: Build prompt-seed metadata and baseline-clean corpus records

**Files:**
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\contracts.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\bootstrap_bridge.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\corpus_loader.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\tests\test_bootstrap_bridge.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\tests\test_corpus_loader.py`

- [ ] **Step 1: Write failing tests for prompt metadata and baseline cleaning**

```python
import unittest

from paper_new_stage2_selector.bootstrap_bridge import build_bootstrap_prompt_records, attach_generated_outputs
from paper_new_stage2_selector.corpus_loader import extract_baseline_training_text


class BootstrapBridgeTests(unittest.TestCase):
    def test_prompt_records_keep_seed_metadata(self):
        records = build_bootstrap_prompt_records(
            ["alpha sample", "beta sample", "gamma sample", "delta sample"],
            num_prompts=2,
            seed=7,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].prompt_index, 0)
        self.assertEqual(len(records[0].seed_texts), 3)
        self.assertIn("Original Text Sample 1", records[0].prompt_text)

    def test_attach_generated_outputs_preserves_prompt_mapping(self):
        prompt_records = build_bootstrap_prompt_records(
            ["alpha sample", "beta sample", "gamma sample"],
            num_prompts=2,
            seed=3,
        )
        generated = attach_generated_outputs(prompt_records, ["out-one", "out-two"])
        self.assertEqual(generated[0].prompt_index, 0)
        self.assertEqual(generated[1].raw_text, "out-two")
        self.assertEqual(len(generated[0].seed_texts), 3)


class CorpusLoaderTests(unittest.TestCase):
    def test_baseline_cleaning_matches_pretext_eval_heuristic(self):
        cleaned = extract_baseline_training_text("useful synthetic text for training Orig trailing junk")
        self.assertEqual(cleaned, "useful synthetic text for training")
        self.assertEqual(extract_baseline_training_text("too short"), "")
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
conda run -n pretext python -m unittest paper-new-2/tests/test_bootstrap_bridge.py paper-new-2/tests/test_corpus_loader.py -v
```

Expected:

```text
FAIL because the bridge and loader modules do not exist yet
```

- [ ] **Step 3: Create the record contracts**

`paper-new-2/paper_new_stage2_selector/contracts.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BootstrapPromptRecord:
    prompt_index: int
    prompt_text: str
    seed_texts: list[str]


@dataclass(slots=True)
class GeneratedSampleRecord:
    record_index: int
    prompt_index: int
    prompt_text: str
    seed_texts: list[str]
    raw_text: str
    baseline_text: str
    consistency_score: float = 0.0
    template_penalty: float = 0.0
    duplicate_penalty: float = 0.0
    final_score: float = 0.0
    rejected_reason: str = ""


@dataclass(slots=True)
class Stage2SelectionResult:
    selected_records: list[GeneratedSampleRecord] = field(default_factory=list)
    rejected_records: list[GeneratedSampleRecord] = field(default_factory=list)
    raw_clean_count: int = 0
    target_count: int = 0
```

- [ ] **Step 4: Implement the bootstrap metadata bridge**

`paper-new-2/paper_new_stage2_selector/bootstrap_bridge.py`

```python
from __future__ import annotations

import random

from .contracts import BootstrapPromptRecord, GeneratedSampleRecord
from .corpus_loader import extract_baseline_training_text

PROMPT_TEMPLATE = (
    "List of 3 diverse original text samples:\n"
    "Original Text Sample 1\n{0}\n"
    "Original Text Sample 2\n{1}\n"
    "Original Text Sample 3\n{2}\n"
)


def build_bootstrap_prompt_records(seed_texts: list[str], *, num_prompts: int, seed: int) -> list[BootstrapPromptRecord]:
    if not seed_texts:
        raise ValueError("Stage 2 bootstrap requires at least 1 seed text.")
    rng = random.Random(seed)
    records: list[BootstrapPromptRecord] = []
    for prompt_index in range(num_prompts):
        if len(seed_texts) >= 3:
            examples = rng.sample(seed_texts, 3)
        else:
            examples = [rng.choice(seed_texts) for _ in range(3)]
        prompt_text = PROMPT_TEMPLATE.format(
            examples[0].replace("\n", " ").replace("\t", " "),
            examples[1].replace("\n", " ").replace("\t", " "),
            examples[2].replace("\n", " ").replace("\t", " "),
        )
        records.append(
            BootstrapPromptRecord(
                prompt_index=prompt_index,
                prompt_text=prompt_text,
                seed_texts=list(examples),
            )
        )
    return records


def attach_generated_outputs(prompt_records: list[BootstrapPromptRecord], outputs: list[str]) -> list[GeneratedSampleRecord]:
    if len(prompt_records) != len(outputs):
        raise ValueError("prompt_records and outputs must have the same length.")
    attached: list[GeneratedSampleRecord] = []
    for record_index, (prompt_record, raw_text) in enumerate(zip(prompt_records, outputs)):
        attached.append(
            GeneratedSampleRecord(
                record_index=record_index,
                prompt_index=prompt_record.prompt_index,
                prompt_text=prompt_record.prompt_text,
                seed_texts=list(prompt_record.seed_texts),
                raw_text=str(raw_text),
                baseline_text=extract_baseline_training_text(str(raw_text)),
            )
        )
    return attached
```

- [ ] **Step 5: Implement baseline cleaning to match `PrE-Text` eval**

`paper-new-2/paper_new_stage2_selector/corpus_loader.py`

```python
from __future__ import annotations

import re


def extract_baseline_training_text(raw_text: str) -> str:
    split_samples = re.split("Orig", str(raw_text))
    candidate = split_samples[0].strip().strip("\n")
    if len(candidate.split(" ")) <= 3:
        return ""
    return candidate.replace("\n\n", " ").replace("\n", " ")
```

- [ ] **Step 6: Re-run the tests**

Run:

```bash
conda run -n pretext python -m unittest paper-new-2/tests/test_bootstrap_bridge.py paper-new-2/tests/test_corpus_loader.py -v
```

Expected:

```text
OK
```

- [ ] **Step 7: Commit**

```bash
git add paper-new-2/paper_new_stage2_selector/contracts.py paper-new-2/paper_new_stage2_selector/bootstrap_bridge.py paper-new-2/paper_new_stage2_selector/corpus_loader.py paper-new-2/tests/test_bootstrap_bridge.py paper-new-2/tests/test_corpus_loader.py
git commit -m "feat: add prompt-seed metadata bridge for stage2 outputs"
```

## Task 3: Implement the three scoring primitives

**Files:**
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\consistency.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\template_penalty.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\dedup.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\tests\test_consistency.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\tests\test_template_penalty.py`

- [ ] **Step 1: Write failing tests for consistency, template penalty, and duplicate penalty**

```python
import unittest

from paper_new_stage2_selector.consistency import compute_consistency_score
from paper_new_stage2_selector.template_penalty import compute_template_penalty
from paper_new_stage2_selector.dedup import compute_duplicate_penalty


class ScoringPrimitiveTests(unittest.TestCase):
    def test_consistency_prefers_text_close_to_any_prompt_seed(self):
        score = compute_consistency_score(
            generated_vector=[1.0, 0.0],
            seed_vectors=[[0.99, 0.01], [0.0, 1.0], [0.2, 0.8]],
        )
        self.assertGreater(score, 0.95)

    def test_template_penalty_hits_prompt_echo_and_low_diversity(self):
        penalty = compute_template_penalty(
            text="List of 3 diverse original text samples original text sample original text sample",
            prompt_text="List of 3 diverse original text samples Original Text Sample 1 alpha",
            seed_texts=["alpha", "beta", "gamma"],
            min_words=4,
            prompt_echo_ngram=6,
            unique_token_ratio_floor=0.45,
        )
        self.assertGreaterEqual(penalty, 1.0)

    def test_duplicate_penalty_is_high_for_near_duplicate_vectors(self):
        penalty = compute_duplicate_penalty(
            candidate_vector=[1.0, 0.0],
            kept_vectors=[[0.999, 0.001], [0.0, 1.0]],
        )
        self.assertGreater(penalty, 0.95)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
conda run -n pretext python -m unittest paper-new-2/tests/test_consistency.py paper-new-2/tests/test_template_penalty.py -v
```

Expected:

```text
FAIL because the scoring modules do not exist yet
```

- [ ] **Step 3: Implement the consistency scorer**

`paper-new-2/paper_new_stage2_selector/consistency.py`

```python
from __future__ import annotations

import math


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def compute_consistency_score(generated_vector: list[float], seed_vectors: list[list[float]]) -> float:
    if not seed_vectors:
        return 0.0
    return max(_cosine_similarity(generated_vector, seed_vector) for seed_vector in seed_vectors)
```

- [ ] **Step 4: Implement the template penalty scorer**

`paper-new-2/paper_new_stage2_selector/template_penalty.py`

```python
from __future__ import annotations


def compute_template_penalty(
    text: str,
    prompt_text: str,
    seed_texts: list[str],
    *,
    min_words: int,
    prompt_echo_ngram: int,
    unique_token_ratio_floor: float,
) -> float:
    normalized = str(text).strip()
    words = [token for token in normalized.split() if token]
    if not words:
        return 10.0

    penalty = 0.0
    if len(words) < int(min_words):
        penalty += 1.0

    prompt_tokens = [token for token in str(prompt_text).split() if token]
    if len(prompt_tokens) >= int(prompt_echo_ngram):
        prompt_window = " ".join(prompt_tokens[: int(prompt_echo_ngram)]).lower()
        if prompt_window in normalized.lower():
            penalty += 1.0

    lowered_words = [token.lower() for token in words]
    unique_ratio = len(set(lowered_words)) / max(1, len(lowered_words))
    if unique_ratio < float(unique_token_ratio_floor):
        penalty += 0.5

    for seed_text in seed_texts:
        if normalized.strip().lower() == str(seed_text).strip().lower():
            penalty += 0.5
            break

    return penalty
```

- [ ] **Step 5: Implement the duplicate penalty scorer**

`paper-new-2/paper_new_stage2_selector/dedup.py`

```python
from __future__ import annotations

from .consistency import _cosine_similarity


def compute_duplicate_penalty(candidate_vector: list[float], kept_vectors: list[list[float]]) -> float:
    if not kept_vectors:
        return 0.0
    return max(_cosine_similarity(candidate_vector, kept_vector) for kept_vector in kept_vectors)
```

- [ ] **Step 6: Re-run the tests**

Run:

```bash
conda run -n pretext python -m unittest paper-new-2/tests/test_consistency.py paper-new-2/tests/test_template_penalty.py -v
```

Expected:

```text
OK
```

- [ ] **Step 7: Commit**

```bash
git add paper-new-2/paper_new_stage2_selector/consistency.py paper-new-2/paper_new_stage2_selector/template_penalty.py paper-new-2/paper_new_stage2_selector/dedup.py paper-new-2/tests/test_consistency.py paper-new-2/tests/test_template_penalty.py
git commit -m "feat: add stage2 selector scoring primitives"
```

## Task 4: Implement hard filters and fixed-target seed-aware selection

**Files:**
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\selector.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\tests\test_selector.py`

- [ ] **Step 1: Write the failing selector test**

```python
import unittest

from paper_new_stage2_selector.contracts import GeneratedSampleRecord
from paper_new_stage2_selector.selector import select_seed_aware_records


class Stage2SelectorTests(unittest.TestCase):
    def test_selector_rejects_low_consistency_and_near_duplicates(self):
        records = [
            GeneratedSampleRecord(0, 0, "p0", ["seed-a", "seed-b", "seed-c"], "text-a", "text-a"),
            GeneratedSampleRecord(1, 1, "p1", ["seed-a", "seed-b", "seed-c"], "text-a-dup", "text-a-dup"),
            GeneratedSampleRecord(2, 2, "p2", ["seed-x", "seed-y", "seed-z"], "text-b", "text-b"),
        ]
        result = select_seed_aware_records(
            records=records,
            generated_vectors=[[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]],
            prompt_seed_vectors=[
                [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]],
                [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]],
                [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]],
            ],
            selector_cfg={
                "target_count_mode": "match_baseline_clean_count",
                "consistency_threshold": 0.42,
                "duplicate_threshold": 0.95,
                "min_words": 1,
                "prompt_echo_ngram": 6,
                "unique_token_ratio_floor": 0.0,
                "w_consistency": 1.0,
                "w_template": 0.35,
                "w_duplicate": 0.30,
            },
        )
        self.assertEqual(result.target_count, 3)
        self.assertEqual(len(result.selected_records), 2)
        self.assertEqual(result.rejected_records[0].record_index, 1)
        self.assertEqual(result.rejected_records[0].rejected_reason, "near_duplicate")
```

- [ ] **Step 2: Run the selector test and verify it fails**

Run:

```bash
conda run -n pretext python -m unittest paper-new-2/tests/test_selector.py -v
```

Expected:

```text
FAIL because the selector does not exist yet
```

- [ ] **Step 3: Implement the fixed-target selector**

`paper-new-2/paper_new_stage2_selector/selector.py`

```python
from __future__ import annotations

from .consistency import compute_consistency_score
from .contracts import GeneratedSampleRecord, Stage2SelectionResult
from .dedup import compute_duplicate_penalty
from .template_penalty import compute_template_penalty


def select_seed_aware_records(
    *,
    records: list[GeneratedSampleRecord],
    generated_vectors: list[list[float]],
    prompt_seed_vectors: list[list[list[float]]],
    selector_cfg: dict,
) -> Stage2SelectionResult:
    if not (len(records) == len(generated_vectors) == len(prompt_seed_vectors)):
        raise ValueError("records, generated_vectors, and prompt_seed_vectors must align.")

    raw_clean_count = sum(1 for record in records if record.baseline_text)
    target_count = raw_clean_count if selector_cfg.get("target_count_mode") == "match_baseline_clean_count" else len(records)
    survivors: list[GeneratedSampleRecord] = []
    rejected: list[GeneratedSampleRecord] = []

    for record, generated_vector, seed_vectors in zip(records, generated_vectors, prompt_seed_vectors):
        record.consistency_score = compute_consistency_score(generated_vector, seed_vectors)
        if not record.baseline_text:
            record.rejected_reason = "baseline_clean_empty"
            rejected.append(record)
            continue
        if record.consistency_score < float(selector_cfg["consistency_threshold"]):
            record.rejected_reason = "low_consistency"
            rejected.append(record)
            continue
        record.template_penalty = compute_template_penalty(
            record.baseline_text,
            record.prompt_text,
            record.seed_texts,
            min_words=int(selector_cfg["min_words"]),
            prompt_echo_ngram=int(selector_cfg["prompt_echo_ngram"]),
            unique_token_ratio_floor=float(selector_cfg["unique_token_ratio_floor"]),
        )
        survivors.append(record)

    survivors.sort(
        key=lambda record: (
            record.consistency_score - float(selector_cfg["w_template"]) * record.template_penalty
        ),
        reverse=True,
    )

    kept_vectors: list[list[float]] = []
    selected: list[GeneratedSampleRecord] = []
    for record in survivors:
        vector = generated_vectors[record.record_index]
        record.duplicate_penalty = compute_duplicate_penalty(vector, kept_vectors)
        if record.duplicate_penalty >= float(selector_cfg["duplicate_threshold"]):
            record.rejected_reason = "near_duplicate"
            rejected.append(record)
            continue
        record.final_score = (
            float(selector_cfg["w_consistency"]) * record.consistency_score
            - float(selector_cfg["w_template"]) * record.template_penalty
            - float(selector_cfg["w_duplicate"]) * record.duplicate_penalty
        )
        selected.append(record)
        kept_vectors.append(vector)
        if len(selected) >= target_count:
            break

    return Stage2SelectionResult(
        selected_records=selected,
        rejected_records=rejected,
        raw_clean_count=raw_clean_count,
        target_count=target_count,
    )
```

- [ ] **Step 4: Re-run the selector test**

Run:

```bash
conda run -n pretext python -m unittest paper-new-2/tests/test_selector.py -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit**

```bash
git add paper-new-2/paper_new_stage2_selector/selector.py paper-new-2/tests/test_selector.py
git commit -m "feat: add fixed-target seed-aware stage2 selector"
```

## Task 5: Add config bridges, eval bridge, pipeline, and CLI

**Files:**
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\thesis_bridge.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\eval_bridge.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\pipeline.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\paper_new_stage2_selector\run_stage2_seed_aware_single_node.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\tests\test_eval_bridge.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\tests\test_pipeline_smoke.py`

- [ ] **Step 1: Write failing bridge and pipeline smoke tests**

```python
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_new_stage2_selector.pipeline import run_pipeline


class PipelineSmokeTests(unittest.TestCase):
    def test_validate_only_reports_stage2_selector_contract(self):
        summary = run_pipeline(
            "paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml",
            validate_only=True,
        )
        self.assertEqual(summary["stage1_mode"], "pretext_stage1_passthrough")
        self.assertEqual(summary["stage2_mode"], "pretext_bootstrap_seed_aware_selector")
        self.assertEqual(summary["stage2"]["selector"]["target_count_mode"], "match_baseline_clean_count")

    def test_pipeline_inserts_selector_between_bootstrap_and_eval(self):
        with patch(
            "paper_new_stage2_selector.pipeline.run_pretext_stage1",
            return_value={"stage1_dir": Path("stage1"), "seed_texts": ["seed-a", "seed-b", "seed-c"]},
        ), patch(
            "paper_new_stage2_selector.pipeline.prepare_bootstrap_runtime",
            return_value={
                "bootstrap_cfg": {"num_prompts": 2, "generator_backend": "vllm"},
                "generate_bootstrapped_samples": lambda prompts, _model_path, _cfg: ["good synthetic sample", "good synthetic sample"],
                "model_path": "unused",
            },
        ), patch(
            "paper_new_stage2_selector.pipeline.embed_records",
            return_value=(
                [[1.0, 0.0], [0.999, 0.001]],
                [[[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]], [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]]],
            ),
        ), patch(
            "paper_new_stage2_selector.pipeline.run_eval_from_stage2_dir",
            return_value={"enabled": True, "best_top1": 0.3},
        ):
            summary = run_pipeline(
                "paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml",
                validate_only=False,
            )
        self.assertEqual(summary["stage2"]["raw_generated_count"], 2)
        self.assertEqual(summary["stage2"]["selected_count"], 1)
        self.assertEqual(summary["eval"]["best_top1"], 0.3)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
conda run -n pretext python -m unittest paper-new-2/tests/test_eval_bridge.py paper-new-2/tests/test_pipeline_smoke.py -v
```

Expected:

```text
FAIL because the bridge and pipeline modules do not exist yet
```

- [ ] **Step 3: Implement the config bridge and `PrE-Text` passthrough adapters**

`paper-new-2/paper_new_stage2_selector/thesis_bridge.py`

```python
from __future__ import annotations

import sys
from pathlib import Path

from paper_new_selector.thesis_bridge import load_yaml_config, resolve_config_path, resolve_output_root, resolve_repo_root


def _ensure_pretext_importable(repo_root: Path) -> None:
    pretext_root = (repo_root / "PrE-Text").resolve()
    if str(pretext_root) not in sys.path:
        sys.path.insert(0, str(pretext_root))


def build_pretext_stage1_config(config_path: str | Path):
    repo_root = resolve_repo_root()
    _ensure_pretext_importable(repo_root)
    from pretext_platform.core.config import ExperimentConfig

    cfg = load_yaml_config(config_path)
    raw = {
        "meta": {"experiment_id": cfg["meta"]["experiment_id"], "seed": int(cfg["meta"]["seed"])},
        "paths": {"repo_root": str(repo_root), "output_root": str(cfg["paths"]["output_root"])},
        "data": dict(cfg["data"]),
        "runtime": {"device": str(cfg["eval"].get("device", "cuda"))},
        "stage1": dict(cfg["stage1"]) | {"enabled": True},
        "bootstrap": dict(cfg["bootstrap"]) | {"enabled": False},
        "eval_small": {"enabled": False},
        "eval_large": {"enabled": False},
    }
    return ExperimentConfig(path=resolve_config_path(config_path), raw=raw)


def run_pretext_stage1(config_path: str | Path) -> dict:
    repo_root = resolve_repo_root()
    _ensure_pretext_importable(repo_root)
    from pretext_platform.algorithms.bootstrap import load_surviving_seed_texts
    from pretext_platform.core.pipeline import run_stage1

    config = build_pretext_stage1_config(config_path)
    summary = run_stage1(config)
    stage1_dir = Path(summary.output_dir)
    seed_texts = load_surviving_seed_texts(stage1_dir, num_rounds=int(config.stage1.get("rounds", 11)))
    return {"stage1_dir": stage1_dir, "seed_texts": seed_texts}
```

- [ ] **Step 4: Implement the eval bridge**

`paper-new-2/paper_new_stage2_selector/eval_bridge.py`

```python
from __future__ import annotations

import json
from pathlib import Path

from .thesis_bridge import build_pretext_stage1_config, resolve_output_root, resolve_repo_root


def write_selected_stage2_dir(selected_texts: list[str], *, output_dir: Path) -> Path:
    stage2_dir = output_dir / "stage2_selected"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    (stage2_dir / "llama7b_text_syn.json").write_text(
        json.dumps(selected_texts, ensure_ascii=False),
        encoding="utf-8",
    )
    return stage2_dir


def run_eval_from_stage2_dir(config_path: str | Path, *, stage2_dir: Path, output_dir: Path) -> dict:
    repo_root = resolve_repo_root()
    if str(repo_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(repo_root))
    from thesis_platform.evaluation.downstream_eval import DownstreamEvalManager

    eval_config = build_pretext_stage1_config(config_path)
    eval_config.raw["eval_small"] = {"enabled": True, "eval_mode": "gpt2"}
    manager = DownstreamEvalManager(
        eval_config,
        experiment_id=str(eval_config.meta.get("experiment_id", "paper_new_2_selector_test")),
        output_dir=output_dir,
    )
    return manager.run(json.loads((stage2_dir / "llama7b_text_syn.json").read_text(encoding="utf-8")))
```

- [ ] **Step 5: Implement the pipeline and CLI**

`paper-new-2/paper_new_stage2_selector/pipeline.py`

```python
from __future__ import annotations

from pathlib import Path

from .bootstrap_bridge import attach_generated_outputs, build_bootstrap_prompt_records
from .eval_bridge import run_eval_from_stage2_dir, write_selected_stage2_dir
from .selector import select_seed_aware_records
from .thesis_bridge import load_yaml_config, resolve_output_root, run_pretext_stage1


def prepare_bootstrap_runtime(config_path: str | Path) -> dict:
    from paper_new_selector.pretext_bridge import prepare_bootstrap_runtime as prepare_runtime

    return prepare_runtime(config_path)


def embed_records(records, config_path: str | Path):
    from paper_new_selector.thesis_bridge import build_embedder_from_config

    embedder = build_embedder_from_config(config_path)
    generated_vectors = [list(map(float, row)) for row in embedder.embed_texts([record.baseline_text or record.raw_text for record in records])]
    prompt_seed_vectors = [
        [list(map(float, row)) for row in embedder.embed_texts(record.seed_texts)]
        for record in records
    ]
    return generated_vectors, prompt_seed_vectors


def run_pipeline(config_path: str | Path, *, validate_only: bool = False) -> dict:
    config = load_yaml_config(config_path)
    bootstrap_runtime = prepare_bootstrap_runtime(config_path)
    summary = {
        "stage1_mode": str(config["pipeline"]["stage1_mode"]),
        "stage2_mode": str(config["pipeline"]["stage2_mode"]),
        "stage2": {
            "bootstrap_cfg": dict(bootstrap_runtime["bootstrap_cfg"]),
            "selector": dict(config["selector"]),
        },
    }
    if validate_only:
        return summary

    stage1_runtime = run_pretext_stage1(config_path)
    prompt_records = build_bootstrap_prompt_records(
        stage1_runtime["seed_texts"],
        num_prompts=int(config["bootstrap"]["num_prompts"]),
        seed=int(config["meta"]["seed"]),
    )
    raw_outputs = bootstrap_runtime["generate_bootstrapped_samples"](
        [record.prompt_text for record in prompt_records],
        bootstrap_runtime["model_path"],
        bootstrap_runtime["bootstrap_cfg"],
    )
    generated_records = attach_generated_outputs(prompt_records, raw_outputs)
    generated_vectors, prompt_seed_vectors = embed_records(generated_records, config_path)
    selection_result = select_seed_aware_records(
        records=generated_records,
        generated_vectors=generated_vectors,
        prompt_seed_vectors=prompt_seed_vectors,
        selector_cfg=dict(config["selector"]),
    )
    output_root = resolve_output_root(config_path)
    stage2_dir = write_selected_stage2_dir(
        [record.baseline_text for record in selection_result.selected_records],
        output_dir=output_root,
    )
    summary["stage2"]["raw_generated_count"] = len(raw_outputs)
    summary["stage2"]["selected_count"] = len(selection_result.selected_records)
    summary["stage2"]["target_count"] = selection_result.target_count
    summary["eval"] = run_eval_from_stage2_dir(config_path, stage2_dir=stage2_dir, output_dir=output_root / "eval")
    return summary
```

`paper-new-2/paper_new_stage2_selector/run_stage2_seed_aware_single_node.py`

```python
from __future__ import annotations

import argparse
import json

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    summary = run_pipeline(args.config, validate_only=args.validate_only)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Re-run the bridge and pipeline tests**

Run:

```bash
conda run -n pretext python -m unittest paper-new-2/tests/test_eval_bridge.py paper-new-2/tests/test_pipeline_smoke.py -v
```

Expected:

```text
OK
```

- [ ] **Step 7: Commit**

```bash
git add paper-new-2/paper_new_stage2_selector/thesis_bridge.py paper-new-2/paper_new_stage2_selector/eval_bridge.py paper-new-2/paper_new_stage2_selector/pipeline.py paper-new-2/paper_new_stage2_selector/run_stage2_seed_aware_single_node.py paper-new-2/tests/test_eval_bridge.py paper-new-2/tests/test_pipeline_smoke.py
git commit -m "feat: add paper-new-2 stage2 selector pipeline"
```

## Task 6: Add the screening config family and end-to-end verification commands

**Files:**
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\configs\experiments\single_node_screening\sas_s_congressional_screening.yaml`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\configs\experiments\single_node_screening\sas_s_forums_screening.yaml`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\configs\experiments\single_node_screening\sas_s_microblog_screening.yaml`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\configs\experiments\single_node_formal\sas_c1_jobs_base.yaml`
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new-2\tests\test_config.py`

- [ ] **Step 1: Extend the config test to cover the matrix**

```python
def test_config_matrix_exists_for_screening_and_formal(self):
    expected = [
        "paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml",
        "paper-new-2/configs/experiments/single_node_screening/sas_s_congressional_screening.yaml",
        "paper-new-2/configs/experiments/single_node_screening/sas_s_forums_screening.yaml",
        "paper-new-2/configs/experiments/single_node_screening/sas_s_microblog_screening.yaml",
        "paper-new-2/configs/experiments/single_node_formal/sas_c1_jobs_base.yaml",
    ]
    for path in expected:
        self.assertTrue(Path(path).exists(), path)
```

- [ ] **Step 2: Add the remaining screening configs**

`paper-new-2/configs/experiments/single_node_screening/sas_s_congressional_screening.yaml`

```yaml
inherits:
  - ../../base/stage2_seed_aware_base.yaml

meta:
  experiment_id: sas_s_congressional_screening
  seed: 42

paths:
  output_root: paper-new-2/outputs/sas_s_congressional_screening

data:
  dataset_name: congressional
  train_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_train.json
  eval_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
  train_limit: 256
  eval_limit: 256
  initialization_limit: 1024

bootstrap:
  num_prompts: 100
```

`paper-new-2/configs/experiments/single_node_screening/sas_s_forums_screening.yaml`

```yaml
inherits:
  - ../../base/stage2_seed_aware_base.yaml

meta:
  experiment_id: sas_s_forums_screening
  seed: 42

paths:
  output_root: paper-new-2/outputs/sas_s_forums_screening

data:
  dataset_name: forums
  train_path: thesis_platform/datasets/pretext_forums/formatted/forums_train.json
  eval_path: thesis_platform/datasets/pretext_forums/formatted/forums_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
  train_limit: 256
  eval_limit: 256
  initialization_limit: 1024

bootstrap:
  num_prompts: 100
```

`paper-new-2/configs/experiments/single_node_screening/sas_s_microblog_screening.yaml`

```yaml
inherits:
  - ../../base/stage2_seed_aware_base.yaml

meta:
  experiment_id: sas_s_microblog_screening
  seed: 42

paths:
  output_root: paper-new-2/outputs/sas_s_microblog_screening

data:
  dataset_name: microblog
  train_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_train.json
  eval_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
  train_limit: 256
  eval_limit: 256
  initialization_limit: 1024

bootstrap:
  num_prompts: 100
```

- [ ] **Step 3: Add one formal template config for later promotion**

`paper-new-2/configs/experiments/single_node_formal/sas_c1_jobs_base.yaml`

```yaml
inherits:
  - ../../base/stage2_seed_aware_base.yaml

meta:
  experiment_id: sas_c1_jobs_base
  seed: 42

paths:
  output_root: paper-new-2/outputs/sas_c1_jobs_base

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
  train_limit: null
  eval_limit: null
  initialization_limit: null

bootstrap:
  num_prompts: 50000
```

- [ ] **Step 4: Re-run the config matrix test**

Run:

```bash
conda run -n pretext python -m unittest paper-new-2/tests/test_config.py -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Run the full unit suite**

Run:

```bash
conda run -n pretext python -m unittest discover -s paper-new-2/tests -p "test_*.py" -v
```

Expected:

```text
all tests pass
```

- [ ] **Step 6: Run validate-only on screening and formal entrypoints**

Run:

```bash
conda run -n pretext python -m paper_new_stage2_selector.run_stage2_seed_aware_single_node --config paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml --validate-only
conda run -n pretext python -m paper_new_stage2_selector.run_stage2_seed_aware_single_node --config paper-new-2/configs/experiments/single_node_formal/sas_c1_jobs_base.yaml --validate-only
```

Expected:

```text
JSON summary reports pretext_stage1_passthrough, pretext_bootstrap_seed_aware_selector, and selector target_count_mode=match_baseline_clean_count
```

- [ ] **Step 7: Run the first real screening command**

Run:

```bash
conda run -n pretext python -m paper_new_stage2_selector.run_stage2_seed_aware_single_node --config paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml
```

Expected:

```text
paper-new-2/outputs/sas_s_jobs_screening/stage2_selected/llama7b_text_syn.json exists
paper-new-2/outputs/sas_s_jobs_screening/eval/downstream_eval_summary.json exists
summary includes raw_generated_count, selected_count, and eval metrics
```

- [ ] **Step 8: Commit**

```bash
git add paper-new-2/configs/experiments paper-new-2/tests/test_config.py
git commit -m "test: add screening config matrix for paper-new-2 stage2 selector"
```

## Self-Review

### Spec coverage

本计划覆盖了以下需求：

- 创新位置严格限定在 `Stage 2` 后验质量控制
- `PrE-Text` 的 `Stage 1` 和 bootstrap 主干保持不变
- 增加了 prompt / seed / generated_text 三者映射，保证 `seed-aware` 不是空话
- 选择器包含 `Consistency`、`TemplatePenalty`、`DuplicatePenalty`
- 目标数量固定为 baseline 清洗后规模，避免把“预算变化”混入第一轮创新
- 给出了 screening 配置族和 formal promotion 模板

### Placeholder scan

本计划没有使用 `TODO`、`TBD`、`后续补充`、`类似上面` 这种占位语句。每个任务都给出了具体文件、测试、命令和预期结果。

### Type consistency

本计划统一使用以下对象和入口名：

- `BootstrapPromptRecord`
- `GeneratedSampleRecord`
- `Stage2SelectionResult`
- `build_bootstrap_prompt_records(...)`
- `attach_generated_outputs(...)`
- `select_seed_aware_records(...)`
- `run_stage2_seed_aware_single_node.py`

后续实现时，不要把这些名称改成别的变体。

---

Plan complete and saved to `paper-new-2/docs/2026-04-25-seed-aware-stage2-selector-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
