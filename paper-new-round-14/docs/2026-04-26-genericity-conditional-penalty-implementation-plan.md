# Genericity Conditional Penalty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the current `jobs / congressional / microblog` gains from round3 weighted-reference smoothing while improving `forums` by changing `genericity penalty` from uniform global application to a three-band conditional penalty.

**Architecture:** Preserve the round3 weighted `genericity reference` design and modify only the penalty application stage inside `Stage 1`. The new design introduces a three-band gate over the raw `genericity score`, so low-score candidates are barely penalized, mid-score candidates are lightly penalized, and high-score candidates keep near-full penalty. This isolates the change to the `genericity` path and avoids touching `support`, `boundary`, or `Stage 2`.

**Tech Stack:** Python 3.10, `paper_new_selector`, YAML experiment configs, `unittest`, existing `paper-new` config loader and screening pipeline.

---

## Design Summary

## 1. What stays unchanged

- Keep round3 weighted reference smoothing:
  - `reference_top_k`
  - `reference_rank_weights`
  - weighted mean reference aggregation in [genericity.py](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/paper_new_selector/genericity.py)
- Keep current `support`, `importance prior`, `boundary`, `Stage 2`, and downstream eval logic unchanged.
- Keep the current screening scale and output conventions.

## 2. What changes

Current behavior is effectively:

```python
penalty = lambda_generic * genericity_score
```

Planned behavior:

```python
penalty = lambda_generic * gate(genericity_score) * genericity_score
```

Where `gate(score)` is a three-band piecewise function:

- Low band: almost no penalty
- Mid band: reduced penalty
- High band: full penalty

## 3. Proposed first-pass gate

Add these selector parameters:

- `genericity_gate_low`
- `genericity_gate_high`
- `genericity_gate_low_scale`
- `genericity_gate_mid_scale`

Default first-pass values:

```yaml
selector:
  genericity_gate_low: 0.78
  genericity_gate_high: 0.90
  genericity_gate_low_scale: 0.10
  genericity_gate_mid_scale: 0.45
```

Interpretation:

- `score <= 0.78`: apply only `10%` of the original genericity penalty
- `0.78 < score <= 0.90`: apply `45%` of the original genericity penalty
- `score > 0.90`: apply `100%` of the original genericity penalty

This is deliberately conservative:

- It preserves strong suppression for clearly public-template-like candidates
- It reduces over-penalization of mid-generic candidates, which is the most likely source of `forums` underperformance
- It should be safer than globally reducing `lambda_generic`

## 4. Expected behavior by dataset

- `jobs`: should stay close to round3 `f1/f3`, because very high genericity candidates still receive full penalty
- `congressional`: should stay close to round3 `f3`, for the same reason
- `microblog`: should remain at or near round3 gains, because weighted reference smoothing already helped and the new gate should not re-harden penalty
- `forums`: should improve relative to round3 because medium-generic, non-template candidates should no longer be over-penalized

## 5. Round4 experiment plan

Create a new round of screening configs, keeping round3 weighted reference as the baseline structure.

Planned groups:

- `g1`: weighted reference + default three-band gate
- `g2`: more permissive middle band
- `g3`: earlier high-band trigger
- `g4`: `g1` + `length_lambda=0.10`

Recommended concrete values:

```yaml
# g1 default
genericity_gate_low: 0.78
genericity_gate_high: 0.90
genericity_gate_low_scale: 0.10
genericity_gate_mid_scale: 0.45

# g2 softer mid band
genericity_gate_low: 0.78
genericity_gate_high: 0.90
genericity_gate_low_scale: 0.10
genericity_gate_mid_scale: 0.30

# g3 earlier hard penalty
genericity_gate_low: 0.75
genericity_gate_high: 0.86
genericity_gate_low_scale: 0.10
genericity_gate_mid_scale: 0.45

# g4 g1 + prior robust length setting
genericity_gate_low: 0.78
genericity_gate_high: 0.90
genericity_gate_low_scale: 0.10
genericity_gate_mid_scale: 0.45
length_lambda: 0.10
```

That produces:

- `4` groups
- `4` datasets per group
- `16` new experiments total

---

### Task 1: Add Genericity Gate Helpers

**Files:**
- Modify: [paper-new/paper_new_selector/genericity.py](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/paper_new_selector/genericity.py)
- Test: [paper-new/tests/test_support.py](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/tests/test_support.py)

- [ ] **Step 1: Write the failing tests**

Add these tests to `test_support.py`:

```python
def test_genericity_gate_uses_low_mid_high_scales():
    from paper_new_selector.genericity import apply_genericity_gate

    self = None
    assert apply_genericity_gate(
        score=0.70,
        gate_low=0.78,
        gate_high=0.90,
        low_scale=0.10,
        mid_scale=0.45,
    ) == 0.10
    assert apply_genericity_gate(
        score=0.84,
        gate_low=0.78,
        gate_high=0.90,
        low_scale=0.10,
        mid_scale=0.45,
    ) == 0.45
    assert apply_genericity_gate(
        score=0.95,
        gate_low=0.78,
        gate_high=0.90,
        low_scale=0.10,
        mid_scale=0.45,
    ) == 1.0


def test_genericity_penalty_applies_gate_to_raw_score():
    from paper_new_selector.genericity import compute_genericity_penalty

    penalty = compute_genericity_penalty(
        candidate_vector=[1.0, 0.0],
        reference_vectors=[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        reference_top_k=3,
        reference_rank_weights=[1.0, 0.5, 0.1],
        gate_low=0.78,
        gate_high=0.90,
        low_scale=0.10,
        mid_scale=0.45,
        apply_gate=True,
    )
    raw_score = (1.0 * 1.0 + 0.9701425001453318 * 0.5 + 0.0 * 0.1) / (1.0 + 0.5 + 0.1)
    expected = raw_score * 0.45
    assert abs(penalty - expected) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n pretext python -m unittest discover -s tests -p 'test_support.py' -v
```

Expected: FAIL with `ImportError` or `TypeError` because `apply_genericity_gate` and the new gated signature do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add these functions and signature changes in `genericity.py`:

```python
def apply_genericity_gate(
    *,
    score: float,
    gate_low: float,
    gate_high: float,
    low_scale: float,
    mid_scale: float,
) -> float:
    if score <= gate_low:
        return float(low_scale)
    if score <= gate_high:
        return float(mid_scale)
    return 1.0


def compute_genericity_penalty(
    *,
    candidate_vector: list[float],
    reference_vectors: list[list[float]],
    reference_top_k: int,
    reference_rank_weights: list[float] | None = None,
    apply_gate: bool = False,
    gate_low: float = 0.0,
    gate_high: float = 1.0,
    low_scale: float = 1.0,
    mid_scale: float = 1.0,
) -> float:
    if not reference_vectors:
        return 0.0
    top_scores = sorted(
        (_cosine(candidate_vector, reference) for reference in reference_vectors),
        reverse=True,
    )[: max(1, reference_top_k)]
    weights = _resolve_reference_rank_weights(
        count=len(top_scores),
        reference_rank_weights=reference_rank_weights,
    )
    denominator = float(sum(weights))
    if denominator <= 0.0:
        return 0.0
    weighted_mean = sum(score * weight for score, weight in zip(top_scores, weights)) / denominator
    raw_score = max(0.0, min(1.0, float(weighted_mean)))
    if not apply_gate:
        return raw_score
    gate_scale = apply_genericity_gate(
        score=raw_score,
        gate_low=gate_low,
        gate_high=gate_high,
        low_scale=low_scale,
        mid_scale=mid_scale,
    )
    return raw_score * gate_scale
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run -n pretext python -m unittest discover -s tests -p 'test_support.py' -v
```

Expected: PASS for the two new tests plus existing genericity tests.

- [ ] **Step 5: Commit**

```bash
git add paper-new/paper_new_selector/genericity.py paper-new/tests/test_support.py
git commit -m "feat: add conditional genericity gate"
```

---

### Task 2: Wire Genericity Gate Into Stage1

**Files:**
- Modify: [paper-new/paper_new_selector/stage1_runner.py](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/paper_new_selector/stage1_runner.py)
- Test: [paper-new/tests/test_stage1_runner.py](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/tests/test_stage1_runner.py)

- [ ] **Step 1: Write the failing test**

Add this test to `test_stage1_runner.py`:

```python
def test_stage1_runner_passes_genericity_gate_config_to_genericity():
    fake_backend = _FakeTextBackend()
    fake_embedder = _FakeEmbedder()
    config = {
        "pipeline": {"stage1_mode": "selector_seed_search"},
        "generator": {
            "initial_prompt": "prompt",
            "candidate_count": 2,
            "max_rounds": 1,
            "exemplars_per_prompt": 1,
        },
        "meta": {"seed": 42},
        "selector": {
            "private_knn_k": 1,
            "density_lambda": 0.0,
            "novelty_lambda": 0.0,
            "length_lambda": 0.0,
            "length_floor": 1,
            "length_ceiling": 100,
            "rank_weights": [1.0],
            "top_q": 1,
            "reference_top_k": 6,
            "reference_rank_weights": [1.0, 0.8, 0.6, 0.4, 0.25, 0.1],
            "genericity_gate_low": 0.78,
            "genericity_gate_high": 0.90,
            "genericity_gate_low_scale": 0.10,
            "genericity_gate_mid_scale": 0.45,
            "lambda_generic": 0.2,
            "lambda_redundancy": 0.3,
            "seed_top_k": 1,
            "hard_negative_top_k": 1,
        },
        "privacy": {"enabled": False, "delta": 1e-5},
        "stage1": {"sigma": 0.0, "delta": 1e-5},
    }
    sample_bundle = {
        "train_samples": [_FakeSample("private alpha"), _FakeSample("private beta")],
        "eval_samples": [_FakeSample("eval alpha")],
        "init_samples": [_FakeSample("seed alpha"), _FakeSample("seed beta")],
    }
    decision = SimpleNamespace(
        selected_indices=[0],
        hard_negative_indices=[1],
        hard_negative_reason={1: "boundary_negative"},
        accept_scores=[0.9, 0.2],
        to_dict=lambda: {"selected_indices": [0], "hard_negative_indices": [1]},
    )

    with patch("paper_new_selector.stage1_runner.load_yaml_config", return_value=config), patch(
        "paper_new_selector.stage1_runner.load_text_samples",
        return_value=sample_bundle,
    ), patch(
        "paper_new_selector.stage1_runner.build_candidate_generator",
        return_value=SimpleNamespace(generator=_FakeGenerator(), text_backend=fake_backend, contract={"llm_backend": "vllm"}),
    ), patch(
        "paper_new_selector.stage1_runner.build_embedder_from_config",
        return_value=fake_embedder,
    ), patch(
        "paper_new_selector.stage1_runner.build_private_importance_weights",
        return_value=[1.0, 1.0],
    ), patch(
        "paper_new_selector.stage1_runner.compute_private_support",
        return_value=[0.9, 0.2],
    ), patch(
        "paper_new_selector.stage1_runner.apply_gaussian_privacy_noise",
        side_effect=lambda scores, **_: scores,
    ), patch(
        "paper_new_selector.stage1_runner.compute_genericity_penalties",
        return_value=[0.1, 0.3],
    ) as genericity_mock, patch(
        "paper_new_selector.stage1_runner.greedy_select_candidates",
        return_value=decision,
    ), patch(
        "paper_new_selector.stage1_runner.build_boundary_state",
        return_value={"negative_pattern_stats": {"count": 1}},
    ):
        run_stage1("dummy.yaml", validate_only=False)

    assert genericity_mock.call_args.kwargs["apply_gate"] is True
    assert genericity_mock.call_args.kwargs["gate_low"] == 0.78
    assert genericity_mock.call_args.kwargs["gate_high"] == 0.90
    assert genericity_mock.call_args.kwargs["low_scale"] == 0.10
    assert genericity_mock.call_args.kwargs["mid_scale"] == 0.45
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n pretext python -m unittest discover -s tests -p 'test_stage1_runner.py' -v
```

Expected: FAIL because `stage1_runner` is not yet forwarding the new gate parameters.

- [ ] **Step 3: Write minimal implementation**

Update the `compute_genericity_penalties(...)` call in `stage1_runner.py`:

```python
genericity_penalty = compute_genericity_penalties(
    candidate_vectors=candidate_vectors,
    reference_vectors=reference_vectors,
    reference_top_k=int(selector_cfg["reference_top_k"]),
    reference_rank_weights=list(selector_cfg.get("reference_rank_weights", [])),
    apply_gate=True,
    gate_low=float(selector_cfg.get("genericity_gate_low", 0.0)),
    gate_high=float(selector_cfg.get("genericity_gate_high", 1.0)),
    low_scale=float(selector_cfg.get("genericity_gate_low_scale", 1.0)),
    mid_scale=float(selector_cfg.get("genericity_gate_mid_scale", 1.0)),
)
```

And extend `compute_genericity_penalties(...)` in `genericity.py`:

```python
def compute_genericity_penalties(
    *,
    candidate_vectors: list[list[float]],
    reference_vectors: list[list[float]],
    reference_top_k: int,
    reference_rank_weights: list[float] | None = None,
    apply_gate: bool = False,
    gate_low: float = 0.0,
    gate_high: float = 1.0,
    low_scale: float = 1.0,
    mid_scale: float = 1.0,
) -> list[float]:
    return [
        compute_genericity_penalty(
            candidate_vector=candidate_vector,
            reference_vectors=reference_vectors,
            reference_top_k=reference_top_k,
            reference_rank_weights=reference_rank_weights,
            apply_gate=apply_gate,
            gate_low=gate_low,
            gate_high=gate_high,
            low_scale=low_scale,
            mid_scale=mid_scale,
        )
        for candidate_vector in candidate_vectors
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run -n pretext python -m unittest discover -s tests -p 'test_stage1_runner.py' -v
```

Expected: PASS for the new forwarding test and existing runtime tests.

- [ ] **Step 5: Commit**

```bash
git add paper-new/paper_new_selector/genericity.py paper-new/paper_new_selector/stage1_runner.py paper-new/tests/test_stage1_runner.py
git commit -m "feat: wire genericity gate into stage1"
```

---

### Task 3: Add Base Config Contract For Conditional Genericity

**Files:**
- Modify: [paper-new/configs/single_node_jobs_selector.yaml](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/configs/single_node_jobs_selector.yaml)
- Modify: [paper-new/configs/experiments/single_node_formal/_base_selector_formal.yaml](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/configs/experiments/single_node_formal/_base_selector_formal.yaml)
- Test: [paper-new/tests/test_config.py](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/tests/test_config.py)

- [ ] **Step 1: Write the failing test**

Update config assertions in `test_config.py`:

```python
self.assertEqual(config["selector"]["genericity_gate_low"], 0.78)
self.assertEqual(config["selector"]["genericity_gate_high"], 0.90)
self.assertEqual(config["selector"]["genericity_gate_low_scale"], 0.10)
self.assertEqual(config["selector"]["genericity_gate_mid_scale"], 0.45)
```

Apply these checks to:

- `test_config_fully_defines_algorithm_contract`
- `test_formal_config_supports_inherits_and_jobs_base_contract`

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n pretext python -m unittest discover -s tests -p 'test_config.py' -v
```

Expected: FAIL because the base configs do not yet define the new gate parameters.

- [ ] **Step 3: Write minimal implementation**

Add these values to both base configs:

```yaml
selector:
  reference_top_k: 6
  reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]
  genericity_gate_low: 0.78
  genericity_gate_high: 0.90
  genericity_gate_low_scale: 0.10
  genericity_gate_mid_scale: 0.45
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run -n pretext python -m unittest discover -s tests -p 'test_config.py' -v
```

Expected: PASS for the updated base contract checks.

- [ ] **Step 5: Commit**

```bash
git add paper-new/configs/single_node_jobs_selector.yaml paper-new/configs/experiments/single_node_formal/_base_selector_formal.yaml paper-new/tests/test_config.py
git commit -m "feat: add genericity gate defaults to selector configs"
```

---

### Task 4: Add Round4 Screening Config Set

**Files:**
- Create: `paper-new/configs/experiments/single_node_tuning_round4/_base_selector_tuning_round4.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/_g1_conditional_genericity_default.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/_g2_conditional_genericity_soft_mid.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/_g3_conditional_genericity_early_high.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/_g4_conditional_genericity_plus_a2.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g1_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g1_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g1_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g1_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g2_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g2_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g2_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g2_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g3_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g3_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g3_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g3_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g4_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g4_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g4_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g4_microblog.yaml`
- Test: [paper-new/tests/test_config.py](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/tests/test_config.py)

- [ ] **Step 1: Write the failing config tests**

Add these new assertions to `test_config.py`:

```python
def test_tuning_round4_base_contract_matches_screening_scale(self):
    config = load_yaml_config(
        "paper-new/configs/experiments/single_node_tuning_round4/_base_selector_tuning_round4.yaml"
    )
    self.assertEqual(config["meta"]["stage"], "single_node_tuning_round4")
    self.assertEqual(config["selector"]["genericity_gate_low"], 0.78)
    self.assertEqual(config["selector"]["genericity_gate_high"], 0.90)
    self.assertEqual(config["selector"]["genericity_gate_low_scale"], 0.10)
    self.assertEqual(config["selector"]["genericity_gate_mid_scale"], 0.45)
    self.assertEqual(config["data"]["train_limit"], 256)
    self.assertEqual(config["data"]["eval_limit"], 256)
    self.assertEqual(config["data"]["initialization_limit"], 1024)


def test_tuning_round4_group_overrides_apply_expected_selector_values(self):
    g1 = load_yaml_config("paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g1_jobs.yaml")
    g2 = load_yaml_config("paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g2_jobs.yaml")
    g3 = load_yaml_config("paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g3_jobs.yaml")
    g4 = load_yaml_config("paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g4_jobs.yaml")

    self.assertEqual(g1["selector"]["genericity_gate_mid_scale"], 0.45)
    self.assertEqual(g2["selector"]["genericity_gate_mid_scale"], 0.30)
    self.assertEqual(g3["selector"]["genericity_gate_low"], 0.75)
    self.assertEqual(g3["selector"]["genericity_gate_high"], 0.86)
    self.assertEqual(g4["selector"]["length_lambda"], 0.10)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n pretext python -m unittest discover -s tests -p 'test_config.py' -v
```

Expected: FAIL because round4 config files do not exist yet.

- [ ] **Step 3: Write the base and override configs**

Base config:

```yaml
inherits:
  - ../single_node_formal/_base_selector_formal.yaml

meta:
  stage: single_node_tuning_round4
  seed: 42

paths:
  output_root: paper-new/outputs/ns_tuning_round4_default

data:
  train_limit: 256
  eval_limit: 256
  initialization_limit: 1024

generator:
  candidate_count: 24
  generated_per_round: 8
  max_rounds: 4

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

selector:
  seed_top_k: 6
  hard_negative_top_k: 6
  reference_top_k: 6
  reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]
  genericity_gate_low: 0.78
  genericity_gate_high: 0.90
  genericity_gate_low_scale: 0.10
  genericity_gate_mid_scale: 0.45
```

Override examples:

```yaml
# _g2_conditional_genericity_soft_mid.yaml
inherits:
  - ./_base_selector_tuning_round4.yaml

selector:
  genericity_gate_mid_scale: 0.30
```

```yaml
# _g3_conditional_genericity_early_high.yaml
inherits:
  - ./_base_selector_tuning_round4.yaml

selector:
  genericity_gate_low: 0.75
  genericity_gate_high: 0.86
```

```yaml
# _g4_conditional_genericity_plus_a2.yaml
inherits:
  - ./_base_selector_tuning_round4.yaml

selector:
  length_lambda: 0.10
```

Leaf config example:

```yaml
inherits:
  - ./_g1_conditional_genericity_default.yaml

meta:
  experiment_id: ns_tune4_g1_jobs

paths:
  output_root: paper-new/outputs/ns_tune4_g1_jobs

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_init.json
```

- [ ] **Step 4: Run config tests and validate-only checks**

Run:

```bash
conda run -n pretext python -m unittest discover -s tests -p 'test_config.py' -v
conda run -n pretext python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round4/ns_tune4_g1_forums.yaml --validate-only
conda run -n pretext python -m paper_new_selector.run_selector_single_node --config configs/experiments/single_node_tuning_round4/ns_tune4_g4_jobs.yaml --validate-only
```

Expected:

- `test_config.py` PASS
- both `--validate-only` commands return a valid JSON summary without config errors

- [ ] **Step 5: Commit**

```bash
git add paper-new/configs/experiments/single_node_tuning_round4 paper-new/tests/test_config.py
git commit -m "feat: add round4 conditional genericity screening configs"
```

---

### Task 5: Run Full Regression Suite

**Files:**
- Modify: none
- Test: [paper-new/tests/test_support.py](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/tests/test_support.py)
- Test: [paper-new/tests/test_stage1_runner.py](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/tests/test_stage1_runner.py)
- Test: [paper-new/tests/test_config.py](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/tests/test_config.py)

- [ ] **Step 1: Run targeted tests**

Run:

```bash
conda run -n pretext python -m unittest discover -s tests -p 'test_support.py' -v
conda run -n pretext python -m unittest discover -s tests -p 'test_stage1_runner.py' -v
conda run -n pretext python -m unittest discover -s tests -p 'test_config.py' -v
```

Expected: all three targeted suites PASS.

- [ ] **Step 2: Run full suite**

Run:

```bash
conda run -n pretext python -m unittest discover -s tests -v
```

Expected: full `paper-new` suite PASS with no regressions.

- [ ] **Step 3: Commit verification-only checkpoint**

```bash
git add paper-new
git commit -m "test: verify conditional genericity implementation"
```

---

### Task 6: Prepare Round4 Result Documentation Skeleton

**Files:**
- Create: `paper-new/docs/2026-04-26-round4-conditional-genericity-results.md`

- [ ] **Step 1: Create result template before running experiments**

Create this skeleton:

```markdown
# 2026-04-26 Round4 Conditional Genericity Results

## 1. Round4 Design

- `g1`: weighted reference + default three-band gate
- `g2`: softer middle band
- `g3`: earlier high-band trigger
- `g4`: `g1` + `length_lambda=0.10`

## 2. Final Results

| experiment | synthetic_train_count | eval_count | best_top1 | best_top3 | best_top5 | best_top10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

## 3. Cross-Dataset Comparison

- `jobs`:
- `congressional`:
- `forums`:
- `microblog`:

## 4. Decision

- Did `forums` improve?
- Were `jobs / congressional / microblog` preserved?
- Is another `genericity` iteration justified?
```

- [ ] **Step 2: Commit**

```bash
git add paper-new/docs/2026-04-26-round4-conditional-genericity-results.md
git commit -m "docs: add round4 conditional genericity results template"
```

---

## Self-Review

### Spec coverage

This plan covers:

- keeping round3 weighted reference unchanged
- changing only the `genericity` penalty application path
- introducing a three-band conditional gate
- wiring the gate through config and runtime
- creating a new round of screening experiments
- preserving existing tests and adding focused regression coverage

### Placeholder scan

No `TBD`, `TODO`, or unspecified file targets remain. Every task lists exact files, commands, and concrete config values.

### Type consistency

The planned new interface names are used consistently throughout:

- `apply_genericity_gate`
- `genericity_gate_low`
- `genericity_gate_high`
- `genericity_gate_low_scale`
- `genericity_gate_mid_scale`

