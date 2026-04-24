# Shared vLLM Stage1/Stage2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `paper-new` 的 `Stage 1` 和 `Stage 2` 共享同一个本地 `llama_2_7b_hf + vllm` engine，避免 `Stage 2` 再次初始化同一 7B 模型导致的显存峰值和 OOM。

**Architecture:** 保留现有算法语义不变，`Stage 1` 仍通过 `PretextPromptLLMGenerator + VllmTextBackend` 生成候选，`Stage 2` 仍继续复用 `PrE-Text` 的 `build_bootstrap_prompts`。唯一结构性变化是引入 `shared_vllm_session`，把 `Stage 1` 的 `VllmTextBackend` 提升到 pipeline 级别复用，并给 `Stage 2` 提供“使用现有 engine 批量生成”的入口，不再二次 `LLM(...)` 初始化。

**Tech Stack:** Python 3.10+, `paper-new`, `PrE-Text`, `thesis_platform`, `vllm`, `unittest`, 本地模型 `thesis_platform/open_model/llama_2_7b_hf`

---

## File Structure

### 需要新增的文件

- `paper-new/paper_new_selector/shared_vllm.py`
  - 负责定义共享 session 的轻量抽象
  - 封装“从 `VllmTextBackend` 暴露可复用 engine”和“Stage 2 批量生成”逻辑

- `paper-new/tests/test_shared_vllm.py`
  - 专门验证同一 engine 被 `Stage 1` 和 `Stage 2` 复用
  - 验证不会再走 `PrE-Text` 里重新 new `LLM(...)` 的路径

### 需要修改的文件

- `thesis_platform/models/backends.py`
  - 给 `VllmTextBackend` 增加公共复用接口
  - 让 `paper-new` 不需要访问 `_llm` / `_ensure_loaded()` 这类私有细节

- `paper-new/paper_new_selector/generator_bridge.py`
  - 继续构建 `Stage 1` 生成器，但要让返回句柄携带“可共享的 vLLM backend 能力”

- `paper-new/paper_new_selector/contracts.py`
  - 扩展 pipeline 层所需的共享 session 元数据契约

- `paper-new/paper_new_selector/stage1_runner.py`
  - 把当前“只返回 summary”的执行方式拆成“summary + runtime handle”
  - 允许 pipeline 在 `Stage 1` 结束后暂不释放 vLLM backend

- `paper-new/paper_new_selector/pretext_bridge.py`
  - 保留 `build_bootstrap_prompts`
  - 新增“基于已有 shared vLLM engine 的 Stage 2 生成入口”
  - 旧的 `generate_bootstrapped_samples_vllm` 保留作 fallback 之外的非共享路径测试，但 pipeline 主路径不再使用

- `paper-new/paper_new_selector/pipeline.py`
  - 接管 session 生命周期
  - 先跑 `Stage 1`
  - 再用同一个 shared engine 跑 `Stage 2`
  - `Stage 2` 完成后统一释放，再进入 eval

- `paper-new/tests/test_generator_bridge.py`
  - 增加共享 engine 能力可发现性测试

- `paper-new/tests/test_pretext_bridge.py`
  - 增加“shared session 下不再重新初始化 LLM”的测试

- `paper-new/tests/test_pipeline_smoke.py`
  - 增加 pipeline 级复用测试与释放顺序测试

---

## Task 1: 给 VllmTextBackend 增加公共共享接口

**Files:**
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\thesis_platform\models\backends.py`
- Test: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\tests\test_shared_vllm.py`

- [ ] **Step 1: 先写失败测试，要求 backend 暴露公共 session 能力**

```python
import unittest
from pathlib import Path
from unittest.mock import patch

from thesis_platform.models.backends import VllmTextBackend


class SharedVllmBackendTests(unittest.TestCase):
    def test_backend_exposes_reusable_session_methods(self):
        backend = VllmTextBackend(
            model_path=Path("demo-model"),
            max_model_len=512,
            gpu_memory_utilization=0.35,
            startup_required_free_gb=2,
            tensor_parallel_size=1,
            top_p=1.0,
            enforce_eager=True,
        )
        self.assertTrue(hasattr(backend, "ensure_session"))
        self.assertTrue(hasattr(backend, "generate_batch"))
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```bash
conda run -n pretext python -m unittest paper-new/tests/test_shared_vllm.py -v
```

Expected:

```text
FAIL: test_backend_exposes_reusable_session_methods
```

- [ ] **Step 3: 在 `VllmTextBackend` 中新增公共接口**

目标接口：

```python
def ensure_session(self) -> tuple[Any, Any]:
    return self._ensure_loaded()

def generate_batch(
    self,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float | None = None,
) -> list[str]:
    llm, sampling_params_cls = self._ensure_loaded()
    effective_temperature = temperature if temperature is not None else self._default_temperature
    sampling_params = sampling_params_cls(
        temperature=float(effective_temperature),
        top_p=float(self._top_p),
        max_tokens=int(max_new_tokens or self._default_max_new_tokens),
    )
    outputs = llm.generate(prompts, sampling_params)
    return [str(output.outputs[0].text).strip() if getattr(output, "outputs", None) else "" for output in outputs]
```

要求：
- 不破坏现有 `generate()` 行为
- `generate()` 继续作为单 prompt 封装
- `generate_batch()` 直接复用同一 `llm` 和 `SamplingParams`

- [ ] **Step 4: 运行测试，确认通过**

Run:

```bash
conda run -n pretext python -m unittest paper-new/tests/test_shared_vllm.py -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit**

```bash
git add thesis_platform/models/backends.py paper-new/tests/test_shared_vllm.py
git commit -m "feat: expose reusable vllm session on backend"
```

---

## Task 2: 在 paper-new 中引入 shared_vllm session 抽象

**Files:**
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\paper_new_selector\shared_vllm.py`
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\paper_new_selector\contracts.py`
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\paper_new_selector\generator_bridge.py`
- Test: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\tests\test_generator_bridge.py`

- [ ] **Step 1: 先写失败测试，要求 generator bridge 返回可共享 session 的 backend**

```python
import unittest

from paper_new_selector.generator_bridge import build_candidate_generator


class GeneratorBridgeSharedSessionTests(unittest.TestCase):
    def test_generator_handle_backend_supports_shared_vllm_session(self):
        handle = build_candidate_generator("configs/experiments/single_node_screening/ns_s_jobs_screening.yaml")
        self.assertTrue(hasattr(handle.text_backend, "ensure_session"))
        self.assertTrue(hasattr(handle.text_backend, "generate_batch"))
```

- [ ] **Step 2: 运行测试，确认当前失败或未覆盖**

Run:

```bash
conda run -n pretext python -m unittest paper-new/tests/test_generator_bridge.py -v
```

Expected:

```text
FAIL or missing assertion coverage
```

- [ ] **Step 3: 新增 shared session 抽象文件**

`shared_vllm.py` 最小结构：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SharedVllmSession:
    backend: Any
    llm: Any
    sampling_params_cls: Any
    model_path: str
    max_model_len: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    enforce_eager: bool


def build_shared_vllm_session(text_backend: Any) -> SharedVllmSession:
    llm, sampling_params_cls = text_backend.ensure_session()
    return SharedVllmSession(
        backend=text_backend,
        llm=llm,
        sampling_params_cls=sampling_params_cls,
        model_path=str(getattr(text_backend, "_model_path")),
        max_model_len=int(getattr(text_backend, "_max_model_len")),
        gpu_memory_utilization=float(getattr(text_backend, "_gpu_memory_utilization")),
        tensor_parallel_size=int(getattr(text_backend, "_tensor_parallel_size")),
        enforce_eager=bool(getattr(text_backend, "_enforce_eager")),
    )
```

- [ ] **Step 4: 扩展 `contracts.py`，增加 pipeline 内部共享 session 契约**

增加：

```python
@dataclass(slots=True)
class SharedSessionContract:
    llm_backend: str
    model_path: str
    max_model_len: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    enforce_eager: bool
```

- [ ] **Step 5: 在 `generator_bridge.py` 中构建并挂载 shared session 元数据**

目标：

```python
from .shared_vllm import build_shared_vllm_session

shared_session = build_shared_vllm_session(text_backend)

return CandidateGeneratorHandle(
    generator=generator,
    text_backend=text_backend,
    contract=contract,
    repo_root=repo_root,
    resource_root=resource_root,
    shared_session=shared_session,
)
```

同时更新 `CandidateGeneratorHandle`：

```python
@dataclass(slots=True)
class CandidateGeneratorHandle:
    generator: Any
    text_backend: Any
    contract: dict[str, Any]
    repo_root: Path
    resource_root: Path
    shared_session: Any | None = None
```

- [ ] **Step 6: 运行 bridge 测试，确认通过**

Run:

```bash
conda run -n pretext python -m unittest paper-new/tests/test_generator_bridge.py -v
```

Expected:

```text
OK
```

- [ ] **Step 7: Commit**

```bash
git add paper-new/paper_new_selector/shared_vllm.py paper-new/paper_new_selector/contracts.py paper-new/paper_new_selector/generator_bridge.py paper-new/tests/test_generator_bridge.py
git commit -m "feat: add shared vllm session abstraction for paper-new"
```

---

## Task 3: 改造 Stage 1 返回 runtime handle，而不是立刻释放共享 backend

**Files:**
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\paper_new_selector\stage1_runner.py`
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\paper_new_selector\runtime_cleanup.py`
- Test: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\tests\test_stage1_runner.py`

- [ ] **Step 1: 先写失败测试，要求 Stage 1 支持保留共享 backend**

```python
import unittest
from unittest.mock import patch

from paper_new_selector.stage1_runner import run_stage1_with_runtime


class Stage1RuntimeTests(unittest.TestCase):
    def test_stage1_can_keep_shared_backend_loaded_for_stage2(self):
        with patch("paper_new_selector.stage1_runner.release_runtime_memory") as release_runtime:
            summary, runtime = run_stage1_with_runtime(
                "configs/experiments/single_node_screening/ns_s_jobs_screening.yaml",
                validate_only=True,
            )
        self.assertIn("generator_contract", summary)
        self.assertIn("shared_session", runtime)
        release_runtime.assert_not_called()
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```bash
conda run -n pretext python -m unittest paper-new/tests/test_stage1_runner.py -v
```

Expected:

```text
FAIL because run_stage1_with_runtime does not exist yet
```

- [ ] **Step 3: 拆出 `run_stage1_with_runtime(...)`**

目标结构：

```python
def run_stage1_with_runtime(config_path: str | Path, *, validate_only: bool = False):
    ...
    return stage1_summary, {
        "generator_handle": generator_handle,
        "shared_session": getattr(generator_handle, "shared_session", None),
        "embedder": embedder,
    }


def run_stage1(config_path: str | Path, *, validate_only: bool = False) -> dict[str, Any]:
    summary, runtime = run_stage1_with_runtime(config_path, validate_only=validate_only)
    release_runtime_memory(
        getattr(runtime.get("generator_handle"), "text_backend", None),
        runtime.get("embedder"),
    )
    return summary
```

要求：
- `run_stage1()` 保持旧行为，兼容现有测试和调用方
- pipeline 新路径只调用 `run_stage1_with_runtime()`

- [ ] **Step 4: 运行 Stage1 测试**

Run:

```bash
conda run -n pretext python -m unittest paper-new/tests/test_stage1_runner.py -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit**

```bash
git add paper-new/paper_new_selector/stage1_runner.py paper-new/paper_new_selector/runtime_cleanup.py paper-new/tests/test_stage1_runner.py
git commit -m "refactor: split stage1 summary from runtime handles"
```

---

## Task 4: 让 Stage 2 复用 Stage 1 已加载的 vLLM engine

**Files:**
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\paper_new_selector\pretext_bridge.py`
- Create: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\tests\test_shared_vllm.py`
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\tests\test_pretext_bridge.py`

- [ ] **Step 1: 先写失败测试，要求 Stage 2 共享路径不再调用 `generate_bootstrapped_samples_vllm`**

```python
import unittest
from unittest.mock import Mock, patch

from paper_new_selector.pretext_bridge import generate_with_shared_vllm_session


class SharedStage2Tests(unittest.TestCase):
    def test_shared_stage2_uses_existing_backend_instead_of_new_llm(self):
        backend = Mock()
        backend.generate_batch.return_value = ["a", "b"]
        prompt_list = ["p1", "p2"]
        outputs = generate_with_shared_vllm_session(
            prompt_list,
            shared_session={"backend": backend},
            bootstrap_cfg={"max_tokens": 85, "temperature": 1.0},
        )
        self.assertEqual(outputs, ["a", "b"])
        backend.generate_batch.assert_called_once()
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```bash
conda run -n pretext python -m unittest paper-new/tests/test_pretext_bridge.py -v
```

Expected:

```text
FAIL because shared Stage 2 helper does not exist yet
```

- [ ] **Step 3: 在 `pretext_bridge.py` 增加共享生成路径**

目标最小实现：

```python
def generate_with_shared_vllm_session(
    prompt_list: list[str],
    *,
    shared_session: dict[str, Any] | Any,
    bootstrap_cfg: dict[str, Any],
) -> list[str]:
    backend = shared_session["backend"] if isinstance(shared_session, dict) else shared_session.backend
    return backend.generate_batch(
        prompt_list,
        max_new_tokens=int(bootstrap_cfg.get("max_tokens", 85)),
        temperature=float(bootstrap_cfg.get("temperature", 1.0)),
    )
```

然后在 `prepare_bootstrap_runtime()` 中返回两个入口：

```python
return {
    "build_bootstrap_prompts": build_bootstrap_prompts,
    "generate_bootstrapped_samples": generator_fn,
    "generate_with_shared_session": generate_with_shared_vllm_session,
    ...
}
```

要求：
- 保留原始 `generate_bootstrapped_samples_vllm` 桥接
- 共享路径只在 `paper-new pipeline` 中使用

- [ ] **Step 4: 运行 bridge/shared 测试**

Run:

```bash
conda run -n pretext python -m unittest paper-new/tests/test_pretext_bridge.py paper-new/tests/test_shared_vllm.py -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit**

```bash
git add paper-new/paper_new_selector/pretext_bridge.py paper-new/tests/test_pretext_bridge.py paper-new/tests/test_shared_vllm.py
git commit -m "feat: add shared vllm stage2 generation path"
```

---

## Task 5: 改造 pipeline，让 Stage 1/Stage 2 共用同一个 engine

**Files:**
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\paper_new_selector\pipeline.py`
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\tests\test_pipeline_smoke.py`

- [ ] **Step 1: 先写失败测试，要求 pipeline 在共享模式下不重新初始化 Stage 2 LLM**

```python
import unittest
from unittest.mock import patch

from paper_new_selector.pipeline import run_pipeline


class PipelineSharedSessionTests(unittest.TestCase):
    def test_pipeline_uses_shared_stage1_session_for_stage2(self):
        fake_runtime = {
            "generator_handle": object(),
            "shared_session": {"backend": type("B", (), {"generate_batch": lambda self, prompts, **_: ["x"] * len(prompts)})()},
            "embedder": None,
        }
        with patch("paper_new_selector.pipeline.run_stage1_with_runtime", return_value=(
            {"generator_contract": {"llm_backend": "vllm"}, "selected_texts": ["a", "b", "c"]},
            fake_runtime,
        )), patch("paper_new_selector.pipeline.prepare_bootstrap_runtime", return_value={
            "bootstrap_cfg": {"num_prompts": 4, "max_tokens": 85, "temperature": 1.0},
            "build_bootstrap_prompts": lambda seed_texts, *, num_prompts, seed: ["p1", "p2"],
            "generate_with_shared_session": lambda prompt_list, *, shared_session, bootstrap_cfg: ["o1", "o2"],
            "generate_bootstrapped_samples": lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
            "model_path": "unused",
        }), patch("paper_new_selector.pipeline.prepare_eval_runtime", return_value={"enabled": False}), patch("paper_new_selector.pipeline.release_runtime_memory") as release_runtime:
            summary = run_pipeline("configs/experiments/single_node_screening/ns_s_jobs_screening.yaml", validate_only=False)
        self.assertEqual(summary["stage2"]["generated_count"], 2)
        release_runtime.assert_called()
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```bash
conda run -n pretext python -m unittest paper-new/tests/test_pipeline_smoke.py -v
```

Expected:

```text
FAIL because pipeline still calls Stage 2 fresh bootstrap path
```

- [ ] **Step 3: 修改 `pipeline.py` 主路径**

核心改造：

```python
stage1_summary, stage1_runtime = run_stage1_with_runtime(config_path, validate_only=validate_only)
...
shared_session = stage1_runtime.get("shared_session")
...
generated_outputs = bootstrap_runtime["generate_with_shared_session"](
    prompt_list,
    shared_session=shared_session,
    bootstrap_cfg=bootstrap_runtime["bootstrap_cfg"],
)
...
release_runtime_memory(
    getattr(stage1_runtime.get("generator_handle"), "text_backend", None),
    stage1_runtime.get("embedder"),
)
```

要求：
- `validate_only=True` 时继续只返回契约
- `Stage 2` 主路径不再调用 fresh `generate_bootstrapped_samples_vllm`
- `Stage 2` 结束后统一释放共享 backend，再进入 eval

- [ ] **Step 4: 跑 pipeline smoke 测试**

Run:

```bash
conda run -n pretext python -m unittest paper-new/tests/test_pipeline_smoke.py -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit**

```bash
git add paper-new/paper_new_selector/pipeline.py paper-new/tests/test_pipeline_smoke.py
git commit -m "feat: reuse shared vllm engine across stage1 and stage2"
```

---

## Task 6: 跑完整回归并验证 shared path 契约

**Files:**
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\tests\test_pipeline_smoke.py`
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\tests\test_pretext_bridge.py`
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\tests\test_generator_bridge.py`

- [ ] **Step 1: 增加 validate-only 契约断言**

在 `test_pipeline_smoke.py` 增加：

```python
def test_validate_only_reports_shared_session_capability(self):
    summary = run_pipeline("configs/experiments/single_node_screening/ns_s_jobs_screening.yaml", validate_only=True)
    self.assertEqual(summary["generator_contract"]["llm_backend"], "vllm")
    self.assertEqual(summary["stage2"]["bootstrap_cfg"]["generator_backend"], "vllm")
```

- [ ] **Step 2: 运行 paper-new 全量测试**

Run:

```bash
conda run -n pretext python -m unittest discover -s paper-new/tests -p "test_*.py" -v
```

Expected:

```text
全部通过
```

- [ ] **Step 3: 运行关键 validate-only 命令**

Run:

```bash
conda run -n pretext python -m paper_new_selector.run_selector_single_node --config paper-new/configs/experiments/single_node_screening/ns_s_jobs_screening.yaml --validate-only
conda run -n pretext python -m paper_new_selector.run_selector_single_node --config paper-new/configs/experiments/single_node_formal/ns_c1_jobs_base.yaml --validate-only
```

Expected:

```text
Stage 1 / Stage 2 都是 vllm，配置解析成功，不报错
```

- [ ] **Step 4: Commit**

```bash
git add paper-new/tests/test_pipeline_smoke.py paper-new/tests/test_pretext_bridge.py paper-new/tests/test_generator_bridge.py
git commit -m "test: cover shared vllm session pipeline"
```

---

## Task 7: 服务器侧真实验证 shared engine 是否解决 Stage 2 二次初始化 OOM

**Files:**
- Modify: `D:\学习记录\导师项目\研究\caiqiyue_file\paper-new\docs\2026-04-25-shared-vllm-stage1-stage2-plan.md`
- Test runtime only, no new code files

- [ ] **Step 1: 同步代码到服务器 `paper-2`**

Run:

```bash
git status
git log --oneline -n 5
```

Expected:

```text
包含 shared session 相关提交
```

- [ ] **Step 2: 先跑 screening validate-only**

Run:

```bash
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_screening/ns_s_jobs_screening.yaml --validate-only
```

Expected:

```text
validate-only 成功
```

- [ ] **Step 3: 在 A6000 上跑 `NS-S-JOBS`**

Run:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export VLLM_HOST_IP=127.0.0.1
export HOST_IP=127.0.0.1
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_screening/ns_s_jobs_screening.yaml
```

Expected:

```text
Stage 1 完成后直接复用同一 7B vllm engine 进入 Stage 2，不再出现第二次模型初始化 OOM
```

- [ ] **Step 4: 检查结果文件**

检查：

```bash
ls paper-new/outputs/ns_s_jobs_screening/eval/stage2/llama7b_text_syn.json
ls paper-new/outputs/ns_s_jobs_screening/eval/downstream_eval_summary.json
```

Expected:

```text
两个文件都存在
```

- [ ] **Step 5: Commit**

```bash
git add paper-new/docs/2026-04-25-shared-vllm-stage1-stage2-plan.md
git commit -m "docs: record shared vllm validation results"
```

---

## Self-Review

### Spec coverage

本计划覆盖了以下关键要求：

- `Stage 1` 与 `Stage 2` 共用同一个 `llama_2_7b_hf + vllm` engine
- `Stage 2` 继续复用 `PrE-Text` 的 `build_bootstrap_prompts`
- `Stage 2` 不再重新初始化 `LLM(...)`
- pipeline 在 `Stage 2` 后统一释放，再进入 eval
- 增加单元测试、smoke 测试、服务器真实验证

### Placeholder scan

本计划没有使用 `TODO`、`TBD`、`后续补充` 这类占位语句；每个任务都给出了具体文件、测试、命令和预期结果。

### Type consistency

本计划统一使用以下对象命名：

- `SharedVllmSession`
- `run_stage1_with_runtime(...)`
- `generate_with_shared_vllm_session(...)`
- `generate_batch(...)`

后续实现时，不要把这些名字再改成其他变体。

---

Plan complete and saved to `paper-new/docs/2026-04-25-shared-vllm-stage1-stage2-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
