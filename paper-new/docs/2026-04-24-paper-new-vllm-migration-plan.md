# Paper-New Full vLLM Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Force every synthetic-text generation stage in `paper-new` to use local-model + `vllm`, fail fast on any non-`vllm` backend, and align key runtime parameters with `PrE-Text` formal vLLM settings.

**Architecture:** Keep the selector algorithm unchanged and tighten only the generation/runtime contract. Stage 1 candidate generation will still use `thesis_platform`'s backend abstraction, but `paper-new` will hard-require `vllm` and expose the exact Stage 1 vLLM runtime contract in validate-only mode. Stage 2 will continue to reuse `PrE-Text` bootstrap generation, but `paper-new` configs and bridge validation will be migrated to `vllm`-only with `PrE-Text`-aligned parameters.

**Tech Stack:** Python 3.10, `vllm`, `transformers`, `torch`, `unittest`, YAML inherited configs, `paper_new_selector`, `thesis_platform`, `PrE-Text`.

---

## File Structure

**Modify**
- `caiqiyue_file/thesis_platform/models/backends.py`
  - Add missing `enforce_eager` support to `VllmTextBackend` and `build_text_backend()`.
- `caiqiyue_file/thesis_platform/tests/test_thesis_platform_runtime.py`
  - Add regression tests for `enforce_eager` pass-through and vLLM config parsing.
- `caiqiyue_file/paper-new/paper_new_selector/contracts.py`
  - Extend `GeneratorContract` so validate-only output can expose Stage 1 vLLM runtime fields.
- `caiqiyue_file/paper-new/paper_new_selector/generator_bridge.py`
  - Reject non-`vllm` Stage 1 backends and validate required Stage 1 vLLM fields.
- `caiqiyue_file/paper-new/paper_new_selector/pretext_bridge.py`
  - Reject non-`vllm` Stage 2 bootstrap backends and align bootstrap runtime config with `PrE-Text`.
- `caiqiyue_file/paper-new/paper_new_selector/pipeline.py`
  - Keep validate-only output stable while exposing the stricter Stage 1/Stage 2 vLLM contract.
- `caiqiyue_file/paper-new/paper_new_selector/run_selector_single_node.py`
  - Surface clearer failure text for invalid strict-vLLM configs.
- `caiqiyue_file/paper-new/configs/base/models.yaml`
  - Migrate Stage 1 generator defaults from `transformers` to `vllm`.
- `caiqiyue_file/paper-new/configs/experiments/single_node_formal/_base_selector_formal.yaml`
  - Migrate Stage 2 from `huggingface` to `vllm` and align runtime parameters with `PrE-Text`.
- `caiqiyue_file/paper-new/configs/single_node_jobs_selector.yaml`
  - Convert the tiny smoke/test config to real local-model + `vllm`.
- `caiqiyue_file/paper-new/tests/test_generator_bridge.py`
  - Add Stage 1 strict-vLLM contract tests and reject-non-vLLM tests using temporary configs.
- `caiqiyue_file/paper-new/tests/test_pretext_bridge.py`
  - Add Stage 2 strict-vLLM contract tests and reject-non-vLLM tests using temporary configs.
- `caiqiyue_file/paper-new/tests/test_config.py`
  - Update config assertions for Stage 1/Stage 2 backend migration.
- `caiqiyue_file/paper-new/tests/test_pipeline_smoke.py`
  - Keep validate-only and config smoke checks aligned with the vLLM-only contract.
- `caiqiyue_file/paper-new/tests/test_eval_bridge.py`
  - Preserve the existing `ExperimentConfig` contract while rechecking the formal configs after migration.

**Optional docs update if behavior changes materially**
- `caiqiyue_file/paper-new/DEBUG.md`
  - Append one short note if migration introduces a new vLLM-only failure mode during real validation.

## Scope Guard

This plan only covers `paper-new` generation-path migration to `vllm` and the minimal shared backend support required in `thesis_platform`. It does **not** redesign selector scoring, bootstrap prompt logic, downstream evaluation metrics, or automation queue semantics.

### Task 1: Add Missing `enforce_eager` Support To Shared vLLM Backend

**Files:**
- Modify: `caiqiyue_file/thesis_platform/models/backends.py`
- Modify: `caiqiyue_file/thesis_platform/tests/test_thesis_platform_runtime.py`

- [ ] **Step 1: Write the failing backend regression test**

```python
def test_vllm_backend_passes_enforce_eager_to_llm_constructor(self) -> None:
    from thesis_platform.models.backends import VllmTextBackend

    captured_llm_kwargs = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured_llm_kwargs.update(kwargs)

        def generate(self, prompts, sampling_params):
            class _Leaf:
                text = "ok"

            class _Node:
                outputs = [_Leaf()]

            return [_Node()]

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams

    backend = VllmTextBackend(
        model_path=Path("local-llama"),
        max_model_len=512,
        gpu_memory_utilization=0.55,
        tensor_parallel_size=1,
        top_p=1.0,
        enforce_eager=True,
    )

    with patch.dict(sys.modules, {"vllm": fake_vllm, "torch": fake_torch}):
        with patch("thesis_platform.models.backends.ensure_vllm_generation_startup_memory", return_value={}):
            text = backend.generate("prompt", max_new_tokens=16, temperature=0.7)

    self.assertEqual(text, "ok")
    self.assertIs(captured_llm_kwargs["enforce_eager"], True)
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file
conda run -n pretext python -m unittest thesis_platform.tests.test_thesis_platform_runtime -v
```

Expected: FAIL because `VllmTextBackend` does not yet accept `enforce_eager` or does not pass it into `LLM(...)`.

- [ ] **Step 3: Implement minimal backend support**

```python
class VllmTextBackend(BaseTextBackend):
    def __init__(
        self,
        *,
        model_path: Path,
        device: str = "cuda",
        dtype: str = "auto",
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        use_chat_template: bool = False,
        max_model_len: int = 512,
        gpu_memory_utilization: float = 0.55,
        startup_required_free_gb: float | None = None,
        tensor_parallel_size: int = 1,
        top_p: float = 1.0,
        enforce_eager: bool = False,
    ) -> None:
        self._enforce_eager = bool(enforce_eager)

    def _ensure_loaded(self) -> tuple[Any, Any]:
        llm_kwargs: dict[str, Any] = {
            "model": str(self._model_path),
            "max_model_len": int(self._max_model_len),
            "tensor_parallel_size": int(self._tensor_parallel_size),
            "gpu_memory_utilization": float(self._gpu_memory_utilization),
            "enforce_eager": bool(self._enforce_eager),
        }
```

```python
if engine == "vllm":
    return VllmTextBackend(
        model_path=model_path,
        device=str(config.get("device", "cuda")),
        dtype=str(config.get("dtype", "auto")),
        temperature=float(config.get("temperature", 0.2)),
        max_new_tokens=int(config.get("max_new_tokens", 256)),
        use_chat_template=bool(config.get("use_chat_template", False)),
        max_model_len=int(config.get("max_model_len", 512)),
        gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.55)),
        startup_required_free_gb=_optional_float(config.get("startup_required_free_gb")),
        tensor_parallel_size=int(config.get("tensor_parallel_size", 1)),
        top_p=float(config.get("top_p", 1.0)),
        enforce_eager=bool(config.get("enforce_eager", False)),
    )
```

- [ ] **Step 4: Re-run the targeted test**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file
conda run -n pretext python -m unittest thesis_platform.tests.test_thesis_platform_runtime -v
```

Expected: PASS, and the test now asserts `captured_llm_kwargs["enforce_eager"] is True`.

- [ ] **Step 5: Commit**

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file
git add thesis_platform/models/backends.py thesis_platform/tests/test_thesis_platform_runtime.py
git commit -m "feat: add enforce eager support to shared vllm backend"
```

### Task 2: Migrate Shared Configs To Strict vLLM Before Tightening Runtime Checks

**Files:**
- Modify: `caiqiyue_file/paper-new/configs/base/models.yaml`
- Modify: `caiqiyue_file/paper-new/configs/experiments/single_node_formal/_base_selector_formal.yaml`
- Modify: `caiqiyue_file/paper-new/configs/single_node_jobs_selector.yaml`
- Modify: `caiqiyue_file/paper-new/tests/test_config.py`

- [ ] **Step 1: Write the failing config assertions**

```python
def test_formal_config_uses_vllm_for_stage1_and_stage2(self):
    config = load_yaml_config("paper-new/configs/experiments/single_node_formal/ns_c1_jobs_base.yaml")
    self.assertEqual(config["llm"]["generator"]["engine"], "vllm")
    self.assertEqual(config["bootstrap"]["generator_backend"], "vllm")
```

```python
def test_smoke_config_uses_real_local_vllm_models_only(self):
    config = load_yaml_config("paper-new/configs/single_node_jobs_selector.yaml")
    self.assertEqual(config["llm"]["generator"]["engine"], "vllm")
    self.assertEqual(config["llm"]["generator"]["model_name_or_path"], "thesis_platform/open_model/llama_2_7b_hf")
    self.assertEqual(config["bootstrap"]["generator_backend"], "vllm")
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m unittest tests.test_config -v
```

Expected: FAIL because Stage 1 still points at `transformers` and Stage 2 still points at `huggingface`.

- [ ] **Step 3: Migrate the shared config defaults**

```yaml
# paper-new/configs/base/models.yaml
llm:
  generator:
    engine: vllm
    model_name_or_path: thesis_platform/open_model/llama_2_7b_hf
    device: cuda
    dtype: auto
    use_chat_template: false
    temperature: 1.0
    top_p: 1.0
    max_new_tokens: 192
    max_model_len: 512
    gpu_memory_utilization: 0.55
    startup_required_free_gb: 26
    tensor_parallel_size: 1
    enforce_eager: true
    role: generator
```

```yaml
# paper-new/configs/experiments/single_node_formal/_base_selector_formal.yaml
llm:
  generator:
    engine: vllm
    model_name_or_path: thesis_platform/open_model/llama_2_7b_hf
    device: cuda
    dtype: auto
    use_chat_template: false
    temperature: 1.0
    top_p: 1.0
    max_new_tokens: 192
    max_model_len: 512
    gpu_memory_utilization: 0.55
    startup_required_free_gb: 26
    tensor_parallel_size: 1
    enforce_eager: true
    role: generator

bootstrap:
  num_prompts: 1500
  generator_backend: vllm
  generator_model: llama2_7b
  max_tokens: 85
  temperature: 1.0
  top_p: 1.0
  max_model_len: 512
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.55
  startup_required_free_gb: 26
  enforce_eager: true
  device: cuda
```

```yaml
# paper-new/configs/single_node_jobs_selector.yaml
llm:
  generator:
    engine: vllm
    model_name_or_path: thesis_platform/open_model/llama_2_7b_hf
    device: cuda
    dtype: auto
    use_chat_template: false
    temperature: 1.0
    top_p: 1.0
    max_new_tokens: 64
    max_model_len: 512
    gpu_memory_utilization: 0.55
    startup_required_free_gb: 26
    tensor_parallel_size: 1
    enforce_eager: true
    role: generator

bootstrap:
  generator_backend: vllm
  generator_model: llama2_7b
  max_tokens: 64
  temperature: 1.0
  top_p: 1.0
  max_model_len: 512
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.55
  startup_required_free_gb: 26
  enforce_eager: true
  device: cuda
```

- [ ] **Step 4: Re-run the config tests**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m unittest tests.test_config -v
```

Expected: PASS, and all formal/test configs now declare `vllm` for every generation stage.

- [ ] **Step 5: Commit**

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file
git add paper-new/configs/base/models.yaml paper-new/configs/experiments/single_node_formal/_base_selector_formal.yaml paper-new/configs/single_node_jobs_selector.yaml paper-new/tests/test_config.py
git commit -m "feat: migrate paper-new configs to strict vllm generation"
```

### Task 3: Force Stage 1 Candidate Generation To Use vLLM Only

**Files:**
- Modify: `caiqiyue_file/paper-new/paper_new_selector/contracts.py`
- Modify: `caiqiyue_file/paper-new/paper_new_selector/generator_bridge.py`
- Modify: `caiqiyue_file/paper-new/tests/test_generator_bridge.py`

- [ ] **Step 1: Write the failing Stage 1 contract tests**

```python
def test_stage1_generator_rejects_non_vllm_engine_for_real_configs(self):
    config_path = self._write_temp_config(
        """
        inherits:
          - ../configs/single_node_jobs_selector.yaml
        llm:
          generator:
            engine: transformers
        """
    )
    with self.assertRaisesRegex(ValueError, "Stage 1 candidate generation requires llm.generator.engine='vllm'"):
        build_candidate_generator(str(config_path))
```

```python
def test_stage1_generator_contract_reports_vllm_runtime_fields(self):
    handle = build_candidate_generator("paper-new/configs/single_node_jobs_selector.yaml")
    self.assertEqual(handle.contract["llm_backend"], "vllm")
    self.assertEqual(handle.contract["max_model_len"], 512)
    self.assertEqual(handle.contract["gpu_memory_utilization"], 0.55)
    self.assertEqual(handle.contract["tensor_parallel_size"], 1)
    self.assertTrue(handle.contract["enforce_eager"])
```

- [ ] **Step 2: Run the Stage 1 tests to verify failure**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m unittest tests.test_generator_bridge -v
```

Expected: FAIL because the contract dataclass and generator bridge do not yet expose the full Stage 1 vLLM runtime fields.

- [ ] **Step 3: Extend the Stage 1 contract and tighten the bridge**

```python
@dataclass(slots=True)
class GeneratorContract:
    backend: str
    source: str
    initial_prompt: str
    candidate_count: int
    generated_per_round: int
    exemplars_per_prompt: int
    max_new_tokens: int
    max_prompt_chars: int
    max_exemplar_chars: int
    llm_backend: str
    max_model_len: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    enforce_eager: bool
```

```python
llm_engine = str(llm_cfg.get("engine", "")).strip().lower()
if llm_engine != "vllm":
    raise ValueError(
        "Stage 1 candidate generation requires llm.generator.engine='vllm'. "
        "Transformers and other backends are rejected for paper-new formal/test runs."
    )

required_keys = [
    "model_name_or_path",
    "max_model_len",
    "gpu_memory_utilization",
    "startup_required_free_gb",
    "tensor_parallel_size",
    "top_p",
    "temperature",
    "enforce_eager",
]
missing = [key for key in required_keys if key not in llm_cfg]
if missing:
    raise ValueError(f"Missing required llm.generator vLLM settings: {missing}")

contract = GeneratorContract(
    backend=str(generator_cfg["backend"]),
    source=str(generator_cfg["source"]),
    initial_prompt=str(generator_cfg["initial_prompt"]),
    candidate_count=int(generator_cfg["candidate_count"]),
    generated_per_round=int(generator_cfg["generated_per_round"]),
    exemplars_per_prompt=int(generator_cfg["exemplars_per_prompt"]),
    max_new_tokens=int(generator_cfg["max_new_tokens"]),
    max_prompt_chars=int(generator_cfg.get("max_prompt_chars", 0)),
    max_exemplar_chars=int(generator_cfg.get("max_exemplar_chars", 0)),
    llm_backend=llm_engine,
    max_model_len=int(llm_cfg["max_model_len"]),
    gpu_memory_utilization=float(llm_cfg["gpu_memory_utilization"]),
    tensor_parallel_size=int(llm_cfg["tensor_parallel_size"]),
    enforce_eager=bool(llm_cfg["enforce_eager"]),
)
```

- [ ] **Step 4: Re-run the Stage 1 tests**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m unittest tests.test_generator_bridge -v
```

Expected: PASS with the Stage 1 contract now hard-wired to `vllm`.

- [ ] **Step 5: Commit**

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file
git add paper-new/paper_new_selector/contracts.py paper-new/paper_new_selector/generator_bridge.py paper-new/tests/test_generator_bridge.py
git commit -m "feat: require vllm for paper-new stage1 generation"
```

### Task 4: Force Stage 2 Bootstrap To Use vLLM Only And Match `PrE-Text`

**Files:**
- Modify: `caiqiyue_file/paper-new/paper_new_selector/pretext_bridge.py`
- Modify: `caiqiyue_file/paper-new/tests/test_pretext_bridge.py`

- [ ] **Step 1: Write the failing Stage 2 bridge tests**

```python
def test_bootstrap_runtime_rejects_non_vllm_backend(self):
    config_path = self._write_temp_config(
        """
        inherits:
          - ../configs/single_node_jobs_selector.yaml
        bootstrap:
          generator_backend: huggingface
        """
    )
    with self.assertRaisesRegex(ValueError, "bootstrap.generator_backend must be 'vllm'"):
        prepare_bootstrap_runtime(str(config_path))
```

```python
def test_bootstrap_runtime_matches_pretext_formal_vllm_defaults(self):
    runtime = prepare_bootstrap_runtime("paper-new/configs/experiments/single_node_formal/ns_c1_jobs_base.yaml")
    cfg = runtime["bootstrap_cfg"]
    self.assertEqual(cfg["generator_backend"], "vllm")
    self.assertEqual(cfg["max_model_len"], 512)
    self.assertEqual(cfg["gpu_memory_utilization"], 0.55)
    self.assertEqual(cfg["tensor_parallel_size"], 1)
    self.assertEqual(cfg["temperature"], 1.0)
    self.assertEqual(cfg["top_p"], 1.0)
    self.assertEqual(cfg["max_tokens"], 85)
    self.assertTrue(cfg["enforce_eager"])
    self.assertEqual(cfg["startup_required_free_gb"], 26)
```

- [ ] **Step 2: Run the Stage 2 bridge tests to verify failure**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m unittest tests.test_pretext_bridge -v
```

Expected: FAIL because the bridge does not yet enforce the vLLM-only bootstrap contract or expose all aligned runtime fields.

- [ ] **Step 3: Tighten the Stage 2 runtime contract**

```python
backend = str(config["bootstrap"]["generator_backend"]).strip().lower()
if backend != "vllm":
    raise ValueError(
        "paper-new Stage 2 bootstrap requires bootstrap.generator_backend='vllm'. "
        "Non-vLLM backends are rejected to keep formal/test runs aligned with PrE-Text."
    )

bootstrap_cfg = {
    "num_prompts": int(config["bootstrap"]["num_prompts"]),
    "generator_backend": "vllm",
    "generator_model": str(config["bootstrap"]["generator_model"]),
    "max_tokens": int(config["bootstrap"].get("max_tokens", 85)),
    "temperature": float(config["bootstrap"].get("temperature", 1.0)),
    "top_p": float(config["bootstrap"].get("top_p", 1.0)),
    "max_model_len": int(config["bootstrap"].get("max_model_len", 512)),
    "tensor_parallel_size": int(config["bootstrap"].get("tensor_parallel_size", 1)),
    "gpu_memory_utilization": float(config["bootstrap"].get("gpu_memory_utilization", 0.55)),
    "startup_required_free_gb": float(config["bootstrap"].get("startup_required_free_gb", 26)),
    "enforce_eager": bool(config["bootstrap"].get("enforce_eager", True)),
    "device": str(config["bootstrap"].get("device", "cuda")),
}
```

- [ ] **Step 4: Re-run the Stage 2 bridge tests**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m unittest tests.test_pretext_bridge -v
```

Expected: PASS and bootstrap runtime now mirrors `PrE-Text` formal vLLM fields.

- [ ] **Step 5: Commit**

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file
git add paper-new/paper_new_selector/pretext_bridge.py paper-new/tests/test_pretext_bridge.py
git commit -m "feat: align paper-new stage2 bootstrap with pretext vllm runtime"
```

### Task 5: Keep Validate-Only Output Stable While Tightening CLI And Eval Assertions

**Files:**
- Modify: `caiqiyue_file/paper-new/paper_new_selector/pipeline.py`
- Modify: `caiqiyue_file/paper-new/paper_new_selector/run_selector_single_node.py`
- Modify: `caiqiyue_file/paper-new/tests/test_pipeline_smoke.py`
- Modify: `caiqiyue_file/paper-new/tests/test_eval_bridge.py`

- [ ] **Step 1: Write the failing smoke/validate tests**

```python
def test_validate_only_reports_vllm_contract_for_both_generation_stages(self):
    result = run_pipeline("paper-new/configs/single_node_jobs_selector.yaml", validate_only=True)
    self.assertEqual(result["generator_contract"]["llm_backend"], "vllm")
    self.assertEqual(result["stage2"]["bootstrap_cfg"]["generator_backend"], "vllm")
```

```python
def test_all_formal_configs_still_build_eval_config_with_none_limits(self):
    for config_name in [
        "ns_c1_jobs_base.yaml",
        "ns_c2_congressional_base.yaml",
        "ns_c3_forums_base.yaml",
        "ns_c4_microblog_base.yaml",
        "ns_c5_jobs_eps05.yaml",
        "ns_c6_jobs_eps758.yaml",
        "ns_c7_jobs_no_privacy.yaml",
        "ns_c8_jobs_seed123.yaml",
        "ns_c9_jobs_seed456.yaml",
    ]:
        cfg = _build_thesis_eval_config(f"paper-new/configs/experiments/single_node_formal/{config_name}")
        self.assertIsNone(cfg.data.get("train_limit"))
        self.assertIsNone(cfg.data.get("eval_limit"))
        self.assertIsNone(cfg.data.get("initialization_limit"))
```

- [ ] **Step 2: Run the smoke/eval tests to verify failure if any contract is still stale**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m unittest tests.test_pipeline_smoke tests.test_eval_bridge -v
```

Expected: FAIL until validate-only output and config assertions reflect the vLLM-only contract.

- [ ] **Step 3: Tighten CLI failure text without breaking the existing summary shape**

```python
try:
    summary = run_pipeline(args.config, validate_only=args.validate_only)
except ValueError as exc:
    raise SystemExit(
        "paper-new configuration is invalid for strict vLLM generation. "
        f"{exc}"
    ) from exc
```

```python
summary: dict[str, Any] = {
    "stage1_mode": str(config["pipeline"]["stage1_mode"]),
    "stage2_mode": str(config["pipeline"]["stage2_mode"]),
    "generator_contract": dict(stage1_summary["generator_contract"]),
    "stage1": stage1_summary,
    "stage2": {
        "bootstrap_cfg": dict(bootstrap_runtime["bootstrap_cfg"]),
        "model_path": str(bootstrap_runtime["model_path"]),
        "build_bootstrap_prompts": bootstrap_runtime["build_bootstrap_prompts"].__name__,
        "generate_bootstrapped_samples": bootstrap_runtime["generate_bootstrapped_samples"].__name__,
    },
    "eval": eval_runtime,
}
```

- [ ] **Step 4: Re-run the smoke/eval tests**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m unittest tests.test_pipeline_smoke tests.test_eval_bridge -v
```

Expected: PASS, and validate-only now exposes the exact vLLM contract used by both generation stages without breaking the existing top-level `generator_contract`.

- [ ] **Step 5: Run the full paper-new suite**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: PASS for the complete `paper-new` suite.

- [ ] **Step 6: Commit**

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file
git add paper-new/paper_new_selector/pipeline.py paper-new/paper_new_selector/run_selector_single_node.py paper-new/tests/test_pipeline_smoke.py paper-new/tests/test_eval_bridge.py
git commit -m "test: lock paper-new validation to strict vllm contract"
```

### Task 6: Server-Side Preflight Before Re-Running Formal Experiments

**Files:**
- No new source files required.
- Reuse: `caiqiyue_file/paper-new/configs/single_node_jobs_selector.yaml`
- Reuse: `caiqiyue_file/paper-new/configs/experiments/single_node_formal/ns_c1_jobs_base.yaml`

- [ ] **Step 1: Validate the tiny vLLM config locally**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m paper_new_selector.run_selector_single_node --config configs/single_node_jobs_selector.yaml --validate-only
```

Expected: PASS, and output includes Stage 1 `llm_backend=vllm` and Stage 2 `generator_backend=vllm`.

- [ ] **Step 2: Validate one formal config locally**

Run:
```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_formal/ns_c1_jobs_base.yaml --validate-only
```

Expected: PASS with no fallback text and no `huggingface`/`transformers` remnants.

- [ ] **Step 3: Run one real tiny smoke on the Linux server**

Run:
```bash
cd /mnt/public/caiqiyue_file/code_from_paper/paper-new
source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
python -m paper_new_selector.run_selector_single_node --config configs/single_node_jobs_selector.yaml
```

Expected: PASS, generate the tiny Stage 2 corpus and downstream eval artifacts without any non-vLLM backend usage.

- [ ] **Step 4: Re-run the first formal experiment**

Run:
```bash
cd /mnt/public/caiqiyue_file/code_from_paper/paper-new
source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_formal/ns_c1_jobs_base.yaml
```

Expected: Stage 1 and Stage 2 both use local-model + `vllm`; any backend mismatch aborts immediately with a clear error.

- [ ] **Step 5: Commit final validation notes**

```bash
cd D:\学习记录\导师项目\研究\caiqiyue_file
git add paper-new/DEBUG.md
git commit -m "docs: record vllm migration validation notes"
```

## Self-Review

- Spec coverage checked:
  - All synthetic generation stages forced to `vllm`: covered by Tasks 2, 3, 4.
  - Fail-fast behavior for non-`vllm`: covered by Tasks 3, 4, 5.
  - Parameter alignment with `PrE-Text`: covered by Tasks 1, 2, 4.
  - Formal/test config migration: covered by Task 2.
  - Server readiness before rerunning formal experiments: covered by Task 6.
- Placeholder scan completed:
  - No `TODO` or “implement later” placeholders remain.
  - Every code-affecting task includes concrete snippets and exact commands.
- Type/name consistency checked:
  - `llm.generator.engine`
  - `bootstrap.generator_backend`
  - `startup_required_free_gb`
  - `enforce_eager`
  - `gpu_memory_utilization`
  - `tensor_parallel_size`
  - `max_model_len`
  stay consistent across tasks.
