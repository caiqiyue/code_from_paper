# Genericity Reference Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `genericity` top-k simple mean with a rank-weighted top-k reference smoother, then expose that structure through configs and a new round of screening experiments.

**Architecture:** Keep the existing `genericity penalty` formula and selector pipeline intact. Only change how the public reference neighborhood is aggregated: widen the neighborhood to `reference_top_k = 6` by default and replace the hard simple mean with a normalized rank-weighted mean. Then add config coverage and a dedicated round-3 screening config family to validate whether this structure improves `forums` / `microblog` without materially hurting `jobs` / `congressional`.

**Tech Stack:** Python 3, YAML config inheritance, `unittest`, existing `paper_new_selector` runtime and config loader

---

## File Structure

### Files to modify

- Modify: `paper-new/paper_new_selector/genericity.py`
  - Add rank-weighted reference aggregation while preserving backward-safe behavior when no weights are provided.
- Modify: `paper-new/paper_new_selector/stage1_runner.py`
  - Thread `reference_rank_weights` from config into `compute_genericity_penalties`.
- Modify: `paper-new/tests/test_support.py`
  - Add focused unit tests for weighted reference smoothing.
- Modify: `paper-new/tests/test_stage1_runner.py`
  - Verify the new selector config field is forwarded correctly.
- Modify: `paper-new/tests/test_config.py`
  - Extend config contract assertions to cover `reference_rank_weights` and new round-3 configs.
- Modify: `paper-new/configs/single_node_jobs_selector.yaml`
  - Update the canonical smoke config to the new default structure.
- Modify: `paper-new/configs/experiments/single_node_formal/_base_selector_formal.yaml`
  - Set the new genericity-reference default for formal runs.
- Modify: `paper-new/configs/experiments/single_node_tuning/_base_selector_tuning.yaml`
  - Keep tuning base aligned with the new structure.
- Modify: `paper-new/configs/experiments/single_node_tuning_round2/_base_selector_tuning_round2.yaml`
  - Keep round-2 baseline compatible with the new structure.

### Files to create

- Create: `paper-new/configs/experiments/single_node_tuning_round3/_base_selector_tuning_round3.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/_f1_weighted_ref_k6.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/_f2_weighted_ref_k6_steeper.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/_f3_weighted_ref_k8_tail.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/_f4_weighted_ref_k6_plus_a2e5.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f2_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f2_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f2_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f2_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f3_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f3_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f3_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f3_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f4_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f4_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f4_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f4_microblog.yaml`

### Round-3 design locked in by this plan

The structure change itself is fixed as:

- `reference_top_k = 6`
- `reference_rank_weights = [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]`

The round-3 screening family intentionally tests only a narrow local neighborhood around that structure:

- `F1`: direct structure change only
- `F2`: same `k=6`, steeper tail suppression
- `F3`: slightly wider `k=8` neighborhood with soft tail
- `F4`: `F1` structure plus the two most balanced parameter signals from prior rounds:
  - `length_lambda = 0.10`
  - `density_lambda = 0.45`
  - `novelty_lambda = 0.35`

No additional `support`-geometry changes belong in this round.

### Target config contract after implementation

Every selector config should define:

```yaml
selector:
  reference_top_k: 6
  reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]
```

`reference_rank_weights` semantics:

- Ordered from nearest neighbor rank to farthest included rank
- Normalized at runtime, not in YAML
- If `reference_top_k` is greater than the provided list length, reuse the last weight for the remaining ranks
- If `reference_top_k` is smaller than the provided list length, ignore the extra tail

---

### Task 1: Add failing unit tests for weighted genericity smoothing

**Files:**
- Modify: `paper-new/tests/test_support.py`
- Test: `paper-new/tests/test_support.py`

- [ ] **Step 1: Add a failing test for weighted reference aggregation**

Append these tests to `paper-new/tests/test_support.py`:

```python
    def test_genericity_penalty_supports_rank_weighted_reference_mean(self):
        penalty = compute_genericity_penalty(
            candidate_vector=[1.0, 0.0],
            reference_vectors=[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
            reference_top_k=3,
            reference_rank_weights=[1.0, 0.5, 0.1],
        )
        expected = (1.0 * 1.0 + 0.9701425001453318 * 0.5 + 0.0 * 0.1) / (1.0 + 0.5 + 0.1)
        self.assertAlmostEqual(penalty, expected, places=6)

    def test_genericity_penalty_reuses_last_weight_when_topk_exceeds_weight_count(self):
        penalty = compute_genericity_penalty(
            candidate_vector=[1.0, 0.0],
            reference_vectors=[[1.0, 0.0], [0.8, 0.2], [0.6, 0.4]],
            reference_top_k=3,
            reference_rank_weights=[1.0, 0.5],
        )
        expected = (
            1.0 * 1.0
            + 0.9701425001453318 * 0.5
            + 0.8320502943378436 * 0.5
        ) / (1.0 + 0.5 + 0.5)
        self.assertAlmostEqual(penalty, expected, places=6)
```

- [ ] **Step 2: Run the support tests to verify they fail**

Run:

```powershell
conda run -n pretext python -m unittest paper-new.tests.test_support -v
```

Expected:

- FAIL with `TypeError` because `compute_genericity_penalty()` does not yet accept `reference_rank_weights`

- [ ] **Step 3: Commit the failing test**

```bash
git add paper-new/tests/test_support.py
git commit -m "test: cover weighted genericity reference smoothing"
```

---

### Task 2: Implement weighted top-k reference smoothing in `genericity.py`

**Files:**
- Modify: `paper-new/paper_new_selector/genericity.py`
- Test: `paper-new/tests/test_support.py`

- [ ] **Step 1: Implement the minimal weighted-reference helper**

Update `paper-new/paper_new_selector/genericity.py` to:

```python
from __future__ import annotations

import math


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator == 0:
        return 0.0
    return _dot(left, right) / denominator


def _resolve_reference_rank_weights(
    *,
    count: int,
    reference_rank_weights: list[float] | None,
) -> list[float]:
    if count <= 0:
        return []
    if not reference_rank_weights:
        return [1.0] * count
    weights: list[float] = []
    tail = float(reference_rank_weights[-1])
    for index in range(count):
        if index < len(reference_rank_weights):
            weights.append(float(reference_rank_weights[index]))
        else:
            weights.append(tail)
    return weights


def compute_genericity_penalty(
    *,
    candidate_vector: list[float],
    reference_vectors: list[list[float]],
    reference_top_k: int,
    reference_rank_weights: list[float] | None = None,
) -> float:
    \"\"\"Estimate how close a candidate stays to the public initialization distribution.\"\"\"

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
    return max(0.0, min(1.0, float(weighted_mean)))


def compute_genericity_penalties(
    *,
    candidate_vectors: list[list[float]],
    reference_vectors: list[list[float]],
    reference_top_k: int,
    reference_rank_weights: list[float] | None = None,
) -> list[float]:
    return [
        compute_genericity_penalty(
            candidate_vector=candidate_vector,
            reference_vectors=reference_vectors,
            reference_top_k=reference_top_k,
            reference_rank_weights=reference_rank_weights,
        )
        for candidate_vector in candidate_vectors
    ]
```

- [ ] **Step 2: Run the unit tests to verify the new behavior**

Run:

```powershell
conda run -n pretext python -m unittest paper-new.tests.test_support -v
```

Expected:

- PASS for both new tests

- [ ] **Step 3: Commit the implementation**

```bash
git add paper-new/paper_new_selector/genericity.py paper-new/tests/test_support.py
git commit -m "feat: add weighted genericity reference smoothing"
```

---

### Task 3: Thread the new selector field through stage-1 runtime

**Files:**
- Modify: `paper-new/paper_new_selector/stage1_runner.py`
- Modify: `paper-new/tests/test_stage1_runner.py`
- Test: `paper-new/tests/test_stage1_runner.py`

- [ ] **Step 1: Add a failing runner test that asserts config forwarding**

In `paper-new/tests/test_stage1_runner.py`, add this test:

```python
    def test_stage1_runner_passes_reference_rank_weights_to_genericity(self):
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
            return_value=SimpleNamespace(
                generator=_FakeGenerator(),
                text_backend=fake_backend,
                contract={"llm_backend": "vllm"},
            ),
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

        self.assertEqual(
            genericity_mock.call_args.kwargs["reference_rank_weights"],
            [1.0, 0.8, 0.6, 0.4, 0.25, 0.1],
        )
```

- [ ] **Step 2: Run the stage-1 runner tests to verify they fail**

Run:

```powershell
conda run -n pretext python -m unittest paper-new.tests.test_stage1_runner -v
```

Expected:

- FAIL because `reference_rank_weights` is not yet passed into `compute_genericity_penalties`

- [ ] **Step 3: Pass the new field from config into runtime**

Update the `compute_genericity_penalties(...)` call in `paper-new/paper_new_selector/stage1_runner.py`:

```python
        genericity_penalty = compute_genericity_penalties(
            candidate_vectors=candidate_vectors,
            reference_vectors=reference_vectors,
            reference_top_k=int(selector_cfg["reference_top_k"]),
            reference_rank_weights=list(selector_cfg.get("reference_rank_weights", [])),
        )
```

- [ ] **Step 4: Re-run runner tests**

Run:

```powershell
conda run -n pretext python -m unittest paper-new.tests.test_stage1_runner -v
```

Expected:

- PASS

- [ ] **Step 5: Commit the runtime wiring**

```bash
git add paper-new/paper_new_selector/stage1_runner.py paper-new/tests/test_stage1_runner.py
git commit -m "feat: wire weighted genericity reference config through stage1"
```

---

### Task 4: Update canonical config contracts to expose the new structure

**Files:**
- Modify: `paper-new/configs/single_node_jobs_selector.yaml`
- Modify: `paper-new/configs/experiments/single_node_formal/_base_selector_formal.yaml`
- Modify: `paper-new/configs/experiments/single_node_tuning/_base_selector_tuning.yaml`
- Modify: `paper-new/configs/experiments/single_node_tuning_round2/_base_selector_tuning_round2.yaml`
- Modify: `paper-new/tests/test_config.py`
- Test: `paper-new/tests/test_config.py`

- [ ] **Step 1: Add failing config assertions**

Update `paper-new/tests/test_config.py` with these assertion changes:

```python
        self.assertEqual(config["selector"]["reference_top_k"], 6)
        self.assertEqual(
            config["selector"]["reference_rank_weights"],
            [1.0, 0.8, 0.6, 0.4, 0.25, 0.1],
        )
```

Also extend tuning-base and round2-base contract tests with:

```python
        self.assertEqual(config["selector"]["reference_top_k"], 6)
        self.assertEqual(
            config["selector"]["reference_rank_weights"],
            [1.0, 0.8, 0.6, 0.4, 0.25, 0.1],
        )
```

- [ ] **Step 2: Run config tests to verify they fail**

Run:

```powershell
conda run -n pretext python -m unittest paper-new.tests.test_config -v
```

Expected:

- FAIL because the selector configs still use `reference_top_k: 4` and do not define `reference_rank_weights`

- [ ] **Step 3: Update the canonical selector configs**

Apply these YAML edits:

`paper-new/configs/single_node_jobs_selector.yaml`

```yaml
selector:
  top_q: 4
  rank_weights: [1.0, 0.6, 0.3, 0.15]
  seed_top_k: 3
  hard_negative_top_k: 3
  private_knn_k: 8
  reference_top_k: 6
  reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]
  density_lambda: 0.50
  novelty_lambda: 0.30
  length_lambda: 0.20
  length_floor: 12
  length_ceiling: 128
  lambda_generic: 0.35
  lambda_redundancy: 0.25
```

`paper-new/configs/experiments/single_node_formal/_base_selector_formal.yaml`

```yaml
selector:
  top_q: 4
  rank_weights: [1.0, 0.6, 0.3, 0.15]
  seed_top_k: 10
  hard_negative_top_k: 10
  private_knn_k: 8
  reference_top_k: 6
  reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]
  density_lambda: 0.50
  novelty_lambda: 0.30
  length_lambda: 0.20
  length_floor: 12
  length_ceiling: 128
  lambda_generic: 0.35
  lambda_redundancy: 0.25
```

`paper-new/configs/experiments/single_node_tuning/_base_selector_tuning.yaml`

```yaml
selector:
  seed_top_k: 6
  hard_negative_top_k: 6
  reference_top_k: 6
  reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]
```

`paper-new/configs/experiments/single_node_tuning_round2/_base_selector_tuning_round2.yaml`

```yaml
selector:
  seed_top_k: 6
  hard_negative_top_k: 6
  reference_top_k: 6
  reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]
```

- [ ] **Step 4: Re-run config tests**

Run:

```powershell
conda run -n pretext python -m unittest paper-new.tests.test_config -v
```

Expected:

- PASS for config contract tests

- [ ] **Step 5: Commit the config contract updates**

```bash
git add \
  paper-new/configs/single_node_jobs_selector.yaml \
  paper-new/configs/experiments/single_node_formal/_base_selector_formal.yaml \
  paper-new/configs/experiments/single_node_tuning/_base_selector_tuning.yaml \
  paper-new/configs/experiments/single_node_tuning_round2/_base_selector_tuning_round2.yaml \
  paper-new/tests/test_config.py
git commit -m "chore: expose weighted genericity reference config defaults"
```

---

### Task 5: Add round-3 screening configs for the structure change

**Files:**
- Create: `paper-new/configs/experiments/single_node_tuning_round3/_base_selector_tuning_round3.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/_f1_weighted_ref_k6.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/_f2_weighted_ref_k6_steeper.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/_f3_weighted_ref_k8_tail.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/_f4_weighted_ref_k6_plus_a2e5.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f2_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f2_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f2_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f2_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f3_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f3_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f3_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f3_microblog.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f4_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f4_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f4_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f4_microblog.yaml`
- Modify: `paper-new/tests/test_config.py`
- Test: `paper-new/tests/test_config.py`

- [ ] **Step 1: Create the round-3 base config**

Create `paper-new/configs/experiments/single_node_tuning_round3/_base_selector_tuning_round3.yaml`:

```yaml
inherits:
  - ../single_node_formal/_base_selector_formal.yaml

meta:
  stage: single_node_tuning_round3
  seed: 42

paths:
  output_root: paper-new/outputs/ns_tuning_round3_default

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
  reference_top_k: 6
  reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]

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

- [ ] **Step 2: Create the four group overrides**

Create `paper-new/configs/experiments/single_node_tuning_round3/_f1_weighted_ref_k6.yaml`:

```yaml
inherits:
  - ./_base_selector_tuning_round3.yaml
```

Create `paper-new/configs/experiments/single_node_tuning_round3/_f2_weighted_ref_k6_steeper.yaml`:

```yaml
inherits:
  - ./_base_selector_tuning_round3.yaml

selector:
  reference_rank_weights: [1.0, 0.7, 0.45, 0.25, 0.12, 0.05]
```

Create `paper-new/configs/experiments/single_node_tuning_round3/_f3_weighted_ref_k8_tail.yaml`:

```yaml
inherits:
  - ./_base_selector_tuning_round3.yaml

selector:
  reference_top_k: 8
  reference_rank_weights: [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.12, 0.05]
```

Create `paper-new/configs/experiments/single_node_tuning_round3/_f4_weighted_ref_k6_plus_a2e5.yaml`:

```yaml
inherits:
  - ./_base_selector_tuning_round3.yaml

selector:
  density_lambda: 0.45
  novelty_lambda: 0.35
  length_lambda: 0.10
```

- [ ] **Step 3: Create one full set of leaf configs**

Create `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_jobs.yaml`:

```yaml
inherits:
  - ./_f1_weighted_ref_k6.yaml

meta:
  experiment_id: ns_tune3_f1_jobs

paths:
  output_root: paper-new/outputs/ns_tune3_f1_jobs

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

Create `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_congressional.yaml`:

```yaml
inherits:
  - ./_f1_weighted_ref_k6.yaml

meta:
  experiment_id: ns_tune3_f1_congressional

paths:
  output_root: paper-new/outputs/ns_tune3_f1_congressional

data:
  dataset_name: congressional
  train_path: thesis_platform/datasets/congressional/formatted/congressional_train.json
  eval_path: thesis_platform/datasets/congressional/formatted/congressional_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

Create `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_forums.yaml`:

```yaml
inherits:
  - ./_f1_weighted_ref_k6.yaml

meta:
  experiment_id: ns_tune3_f1_forums

paths:
  output_root: paper-new/outputs/ns_tune3_f1_forums

data:
  dataset_name: forums
  train_path: thesis_platform/datasets/pretext_forums/formatted/forums_train.json
  eval_path: thesis_platform/datasets/pretext_forums/formatted/forums_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

Create `paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_microblog.yaml`:

```yaml
inherits:
  - ./_f1_weighted_ref_k6.yaml

meta:
  experiment_id: ns_tune3_f1_microblog

paths:
  output_root: paper-new/outputs/ns_tune3_f1_microblog

data:
  dataset_name: microblog
  train_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_train.json
  eval_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 4: Duplicate the leaf pattern for `f2`, `f3`, and `f4`**

Use the exact same dataset-path layout as `f1`, changing only:

```text
inherits:
  - ./_f2_weighted_ref_k6_steeper.yaml
meta.experiment_id: ns_tune3_f2_<dataset>
paths.output_root: paper-new/outputs/ns_tune3_f2_<dataset>
```

```text
inherits:
  - ./_f3_weighted_ref_k8_tail.yaml
meta.experiment_id: ns_tune3_f3_<dataset>
paths.output_root: paper-new/outputs/ns_tune3_f3_<dataset>
```

```text
inherits:
  - ./_f4_weighted_ref_k6_plus_a2e5.yaml
meta.experiment_id: ns_tune3_f4_<dataset>
paths.output_root: paper-new/outputs/ns_tune3_f4_<dataset>
```

- [ ] **Step 5: Add round-3 config tests**

Append these tests to `paper-new/tests/test_config.py`:

```python
    def test_tuning_round3_base_contract_matches_screening_scale(self):
        config = load_yaml_config(
            "paper-new/configs/experiments/single_node_tuning_round3/_base_selector_tuning_round3.yaml"
        )
        self.assertEqual(config["meta"]["stage"], "single_node_tuning_round3")
        self.assertEqual(config["selector"]["reference_top_k"], 6)
        self.assertEqual(
            config["selector"]["reference_rank_weights"],
            [1.0, 0.8, 0.6, 0.4, 0.25, 0.1],
        )
        self.assertEqual(config["data"]["train_limit"], 256)
        self.assertEqual(config["eval"]["small_epochs"], 6)

    def test_tuning_round3_group_overrides_apply_expected_selector_values(self):
        f1 = load_yaml_config("paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_jobs.yaml")
        f2 = load_yaml_config("paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f2_jobs.yaml")
        f3 = load_yaml_config("paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f3_jobs.yaml")
        f4 = load_yaml_config("paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f4_jobs.yaml")

        self.assertEqual(f1["selector"]["reference_top_k"], 6)
        self.assertEqual(f1["selector"]["reference_rank_weights"], [1.0, 0.8, 0.6, 0.4, 0.25, 0.1])
        self.assertEqual(f2["selector"]["reference_rank_weights"], [1.0, 0.7, 0.45, 0.25, 0.12, 0.05])
        self.assertEqual(f3["selector"]["reference_top_k"], 8)
        self.assertEqual(f3["selector"]["reference_rank_weights"], [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.12, 0.05])
        self.assertEqual(f4["selector"]["length_lambda"], 0.10)
        self.assertEqual(f4["selector"]["density_lambda"], 0.45)
        self.assertEqual(f4["selector"]["novelty_lambda"], 0.35)

    def test_tuning_round3_leaf_configs_keep_dataset_paths_and_output_roots_explicit(self):
        jobs = load_yaml_config("paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_jobs.yaml")
        forums = load_yaml_config("paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f3_forums.yaml")
        micro = load_yaml_config("paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f4_microblog.yaml")

        self.assertEqual(jobs["meta"]["experiment_id"], "ns_tune3_f1_jobs")
        self.assertEqual(jobs["paths"]["output_root"], "paper-new/outputs/ns_tune3_f1_jobs")
        self.assertEqual(forums["data"]["eval_path"], "thesis_platform/datasets/pretext_forums/formatted/forums_eval.json")
        self.assertEqual(micro["data"]["train_path"], "thesis_platform/datasets/pretext_microblog/formatted/microblog_train.json")
```

- [ ] **Step 6: Run config tests and validate two leaf configs**

Run:

```powershell
conda run -n pretext python -m unittest paper-new.tests.test_config -v
conda run -n pretext python -m paper_new_selector.run_selector_single_node --config paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f1_forums.yaml --validate-only
conda run -n pretext python -m paper_new_selector.run_selector_single_node --config paper-new/configs/experiments/single_node_tuning_round3/ns_tune3_f4_jobs.yaml --validate-only
```

Expected:

- config tests PASS
- both `--validate-only` commands succeed without schema or path errors

- [ ] **Step 7: Commit the round-3 screening configs**

```bash
git add paper-new/configs/experiments/single_node_tuning_round3 paper-new/tests/test_config.py
git commit -m "feat: add round3 weighted genericity screening configs"
```

---

### Task 6: Run the focused verification suite

**Files:**
- Test: `paper-new/tests/test_support.py`
- Test: `paper-new/tests/test_stage1_runner.py`
- Test: `paper-new/tests/test_config.py`

- [ ] **Step 1: Run the targeted Python test suite**

Run:

```powershell
conda run -n pretext python -m unittest `
  paper-new.tests.test_support `
  paper-new.tests.test_stage1_runner `
  paper-new.tests.test_config -v
```

Expected:

- PASS

- [ ] **Step 2: Run the full paper-new test suite**

Run:

```powershell
cd D:\学习记录\导师项目\研究\caiqiyue_file\paper-new
conda run -n pretext python -m unittest discover -s tests -v
```

Expected:

- PASS

- [ ] **Step 3: Record the verification summary**

Capture in the implementation notes or PR summary:

```text
Weighted genericity reference smoothing added.
Config contract updated to include reference_rank_weights.
Round-3 tuning configs validated successfully.
```

- [ ] **Step 4: Commit the final verification-safe state**

```bash
git add paper-new
git commit -m "test: verify weighted genericity reference smoothing rollout"
```

---

## Self-Review

### Spec coverage

This plan covers:

- Structure change from simple mean to weighted top-k mean
- Runtime wiring
- Config contract updates
- New screening config family for validating the structure change
- Unit, integration, and config verification

### Placeholder scan

No `TODO`, `TBD`, or implied steps remain. All file paths, code snippets, and commands are explicit.

### Type consistency

The plan uses one stable new field name everywhere:

- `reference_rank_weights`

It is threaded consistently through:

- YAML selector configs
- `compute_genericity_penalty`
- `compute_genericity_penalties`
- `stage1_runner`
- config tests
- runner tests

---

**Execution note:** This plan is intentionally scoped to the first structural micro-change only. It does not include conditional genericity, adaptive weighting, or importance-prior restructuring. Those belong in a later plan only if this round still fails to produce a cross-dataset improvement.
