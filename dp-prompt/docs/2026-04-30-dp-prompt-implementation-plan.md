# DP-Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a YAML-driven, independent `dp-prompt` experiment framework that faithfully reproduces the paper-style document sanitization loop with local open-source generation, utility evaluation, and text-based privacy attacks.

**Architecture:** We will keep the original repository's algorithmic intent, but replace notebook/script-first execution with a layered experiment runner. The new package will load a real document dataset, build prompts, generate sanitized text with a local model, evaluate utility, run static/adaptive text attacks, and write structured summaries including privacy-control metadata.

**Tech Stack:** Python, PyYAML, pandas, scikit-learn, optional `transformers` backend for generation and classifier-style evaluation, JSONL/CSV datasets, shell runner scripts.

---

## File Map

### New package files

- Create: `dp-prompt/dp_prompt/__init__.py`
- Create: `dp-prompt/dp_prompt/cli.py`
- Create: `dp-prompt/dp_prompt/config.py`
- Create: `dp-prompt/dp_prompt/data/__init__.py`
- Create: `dp-prompt/dp_prompt/data/schema.py`
- Create: `dp-prompt/dp_prompt/data/loader.py`
- Create: `dp-prompt/dp_prompt/prompting/__init__.py`
- Create: `dp-prompt/dp_prompt/prompting/templates.py`
- Create: `dp-prompt/dp_prompt/decoding/__init__.py`
- Create: `dp-prompt/dp_prompt/decoding/privacy.py`
- Create: `dp-prompt/dp_prompt/generation/__init__.py`
- Create: `dp-prompt/dp_prompt/generation/backend.py`
- Create: `dp-prompt/dp_prompt/evaluation/__init__.py`
- Create: `dp-prompt/dp_prompt/evaluation/utility.py`
- Create: `dp-prompt/dp_prompt/attacks/__init__.py`
- Create: `dp-prompt/dp_prompt/attacks/text_attacks.py`
- Create: `dp-prompt/dp_prompt/runners/__init__.py`
- Create: `dp-prompt/dp_prompt/runners/document_pipeline.py`
- Create: `dp-prompt/dp_prompt/utils/__init__.py`
- Create: `dp-prompt/dp_prompt/utils/io.py`

### New config files

- Create: `dp-prompt/configs/base/document_dp_prompt_base.yaml`
- Create: `dp-prompt/configs/datasets/imdb_document.yaml`
- Create: `dp-prompt/configs/models/local_open_source_llm.yaml`
- Create: `dp-prompt/configs/experiments/r1_imdb_base.yaml`
- Create: `dp-prompt/configs/experiments/r1_imdb_temp_low.yaml`
- Create: `dp-prompt/configs/experiments/r1_imdb_temp_mid.yaml`
- Create: `dp-prompt/configs/experiments/r1_imdb_temp_high.yaml`

### New scripts

- Create: `dp-prompt/scripts/run_dp_prompt.sh`

### New tests

- Create: `dp-prompt/tests/test_config.py`
- Create: `dp-prompt/tests/test_data_loader.py`
- Create: `dp-prompt/tests/test_privacy_controls.py`
- Create: `dp-prompt/tests/test_attack_splits.py`
- Create: `dp-prompt/tests/test_runner_build.py`

### Existing files to reference or lightly integrate

- Read/Reuse: `dp-prompt/attacks/text_attack.py`
- Read/Reuse: `dp-prompt/attacks/text_attacker_architecture.py`
- Read/Reuse: `dp-prompt/mechanisms/documentlevel/opensource_paraphrase_generation.py`
- Read/Reuse: `dp-prompt/README.md`
- Read/Reuse: `dp-prompt/docs/2026-04-30-dp-prompt-design.md`

---

## Task 1: Build configuration loading and experiment wiring

**Files:**
- Create: `dp-prompt/dp_prompt/config.py`
- Create: `dp-prompt/dp_prompt/utils/io.py`
- Create: `dp-prompt/dp_prompt/cli.py`
- Create: `dp-prompt/configs/base/document_dp_prompt_base.yaml`
- Create: `dp-prompt/configs/models/local_open_source_llm.yaml`
- Test: `dp-prompt/tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

```python
from pathlib import Path

from dp_prompt.config import load_experiment_config


def test_load_experiment_config_merges_inherits(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "runtime:\n  seed: 42\nmodel:\n  name: base\nprivacy:\n  temperature: 1.0\n",
        encoding="utf-8",
    )
    child.write_text(
        f"inherits:\n  - {base}\nprivacy:\n  temperature: 1.5\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(child)

    assert cfg["runtime"]["seed"] == 42
    assert cfg["model"]["name"] == "base"
    assert cfg["privacy"]["temperature"] == 1.5


def test_load_experiment_config_records_source_paths(tmp_path: Path):
    cfg_file = tmp_path / "standalone.yaml"
    cfg_file.write_text("runtime:\n  seed: 7\n", encoding="utf-8")

    cfg = load_experiment_config(cfg_file)

    assert str(cfg_file) in cfg["_meta"]["config_chain"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dp_prompt'` or missing `load_experiment_config`.

- [ ] **Step 3: Implement minimal config loader and CLI entry**

Implementation requirements:

```python
# dp_prompt/config.py
def load_experiment_config(config_path: str | Path) -> dict:
    ...


def deep_merge_dicts(base: dict, override: dict) -> dict:
    ...
```

```python
# dp_prompt/cli.py
def main() -> int:
    # parse --config
    # load config
    # dispatch to document pipeline runner
    ...
```

Base YAML should include:

```yaml
runtime:
  seed: 42
  output_root: outputs

privacy:
  temperature: 1.0
  logits_clipping:
    enabled: false
    lower_bound: null
    upper_bound: null
  max_generated_tokens: 128
  stop_sequences: []
  report_privacy_summary: true
```

- [ ] **Step 4: Run tests again**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dp_prompt/config.py dp_prompt/utils/io.py dp_prompt/cli.py configs/base/document_dp_prompt_base.yaml configs/models/local_open_source_llm.yaml tests/test_config.py
git commit -m "feat: add dp-prompt config loader and cli skeleton"
```

---

## Task 2: Implement dataset schema and split-aware loader

**Files:**
- Create: `dp-prompt/dp_prompt/data/schema.py`
- Create: `dp-prompt/dp_prompt/data/loader.py`
- Create: `dp-prompt/configs/datasets/imdb_document.yaml`
- Test: `dp-prompt/tests/test_data_loader.py`

- [ ] **Step 1: Write the failing data-loader tests**

```python
from pathlib import Path

import pandas as pd

from dp_prompt.data.loader import load_document_dataset


def test_load_document_dataset_reads_jsonl_and_assigns_splits(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text('{"text":"a","label":1,"author_id":"u1"}\n', encoding="utf-8")
    val.write_text('{"text":"b","label":0,"author_id":"u2"}\n', encoding="utf-8")
    test.write_text('{"text":"c","label":1,"author_id":"u3"}\n', encoding="utf-8")

    cfg = {
        "dataset": {
            "text_field": "text",
            "label_field": "label",
            "author_field": "author_id",
            "splits": {
                "train": str(train),
                "validation": str(val),
                "test": str(test),
            },
        }
    }

    bundle = load_document_dataset(cfg)

    assert set(bundle.dataframe["split"]) == {"train", "validation", "test"}
    assert list(bundle.dataframe.columns)[:4] == ["sample_id", "text", "label", "author_id"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_data_loader.py -v
```

Expected: FAIL because loader/schema are missing.

- [ ] **Step 3: Implement dataset bundle and loader**

Implementation requirements:

- standardize columns to:
  - `sample_id`
  - `text`
  - `label`
  - `author_id`
  - `split`
- support `.jsonl`, `.json`, `.csv`
- preserve original fields in `metadata`
- record split indices for:
  - utility evaluation
  - static attack
  - adaptive attack

The IMDb config should look like:

```yaml
dataset:
  name: imdb
  text_field: text
  label_field: label
  author_field: author_id
  splits:
    train: ${DP_PROMPT_IMDB_TRAIN}
    validation: ${DP_PROMPT_IMDB_VALIDATION}
    test: ${DP_PROMPT_IMDB_TEST}
```

- [ ] **Step 4: Run tests again**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_data_loader.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dp_prompt/data/schema.py dp_prompt/data/loader.py configs/datasets/imdb_document.yaml tests/test_data_loader.py
git commit -m "feat: add split-aware document dataset loader"
```

---

## Task 3: Add prompt templates and privacy-control summaries

**Files:**
- Create: `dp-prompt/dp_prompt/prompting/templates.py`
- Create: `dp-prompt/dp_prompt/decoding/privacy.py`
- Test: `dp-prompt/tests/test_privacy_controls.py`

- [ ] **Step 1: Write the failing privacy-control tests**

```python
from dp_prompt.decoding.privacy import build_privacy_controls_summary
from dp_prompt.prompting.templates import render_document_prompt


def test_render_document_prompt_uses_review_template():
    prompt = render_document_prompt("hello world", template_name="review_paraphrase")
    assert "hello world" in prompt
    assert "Paraphrase" in prompt


def test_build_privacy_controls_summary_contains_reproducible_fields():
    summary = build_privacy_controls_summary(
        {
            "temperature": 1.25,
            "logits_clipping": {"enabled": True, "lower_bound": -3.0, "upper_bound": 3.0},
            "max_generated_tokens": 96,
            "stop_sequences": ["\n\n"],
        }
    )

    assert summary["temperature"] == 1.25
    assert summary["logits_clipping"]["enabled"] is True
    assert summary["max_generated_tokens"] == 96
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_privacy_controls.py -v
```

Expected: FAIL because modules are missing.

- [ ] **Step 3: Implement prompt rendering and privacy summary helpers**

Implementation requirements:

- `render_document_prompt(text, template_name="review_paraphrase")`
- template registry with at least one paper-style template
- `build_privacy_controls_summary(cfg)` returns:
  - temperature
  - logits clipping enabled flag
  - clipping bounds
  - max generated tokens
  - stop sequences
  - privacy reporting mode

- [ ] **Step 4: Run tests again**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_privacy_controls.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dp_prompt/prompting/templates.py dp_prompt/decoding/privacy.py tests/test_privacy_controls.py
git commit -m "feat: add prompt templates and privacy control summaries"
```

---

## Task 4: Build local generation backend with lazy transformer loading

**Files:**
- Create: `dp-prompt/dp_prompt/generation/backend.py`
- Modify: `dp-prompt/dp_prompt/cli.py`
- Test: `dp-prompt/tests/test_runner_build.py`

- [ ] **Step 1: Write the failing backend-build test**

```python
from dp_prompt.generation.backend import build_generation_request


def test_build_generation_request_includes_sampling_controls():
    request = build_generation_request(
        prompt="Document: a\nParaphrase:",
        cfg={
            "temperature": 1.5,
            "max_generated_tokens": 80,
            "stop_sequences": ["\n\n"],
        },
    )

    assert request["temperature"] == 1.5
    assert request["max_new_tokens"] == 80
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_runner_build.py::test_build_generation_request_includes_sampling_controls -v
```

Expected: FAIL because backend is missing.

- [ ] **Step 3: Implement generation backend**

Implementation requirements:

- keep `transformers` imports inside runtime methods
- expose:
  - `build_generation_request(...)`
  - `LocalTransformersGenerator.from_config(...)`
  - `generate_batch(prompts)`
- output:
  - generated text
  - prompt text
  - per-sample metadata

- [ ] **Step 4: Run the test again**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_runner_build.py::test_build_generation_request_includes_sampling_controls -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dp_prompt/generation/backend.py dp_prompt/cli.py tests/test_runner_build.py
git commit -m "feat: add local generation backend abstraction"
```

---

## Task 5: Implement utility evaluation and static/adaptive text attacks

**Files:**
- Create: `dp-prompt/dp_prompt/evaluation/utility.py`
- Create: `dp-prompt/dp_prompt/attacks/text_attacks.py`
- Test: `dp-prompt/tests/test_attack_splits.py`

- [ ] **Step 1: Write the failing attack-split tests**

```python
import pandas as pd

from dp_prompt.attacks.text_attacks import build_attack_views


def test_build_attack_views_matches_static_semantics():
    df = pd.DataFrame(
        [
            {"sample_id": "1", "text": "a", "sanitized_text": "sa", "label": 1, "author_id": "u1", "split": "train"},
            {"sample_id": "2", "text": "b", "sanitized_text": "sb", "label": 0, "author_id": "u2", "split": "validation"},
            {"sample_id": "3", "text": "c", "sanitized_text": "sc", "label": 1, "author_id": "u3", "split": "test"},
        ]
    )

    views = build_attack_views(df, attack_mode="static")

    assert views["train"]["text_field"] == "text"
    assert views["test"]["text_field"] == "sanitized_text"


def test_build_attack_views_matches_adaptive_semantics():
    df = pd.DataFrame(
        [
            {"sample_id": "1", "text": "a", "sanitized_text": "sa", "label": 1, "author_id": "u1", "split": "train"},
            {"sample_id": "2", "text": "b", "sanitized_text": "sb", "label": 0, "author_id": "u2", "split": "validation"},
            {"sample_id": "3", "text": "c", "sanitized_text": "sc", "label": 1, "author_id": "u3", "split": "test"},
        ]
    )

    views = build_attack_views(df, attack_mode="adaptive")

    assert views["train"]["text_field"] == "sanitized_text"
    assert views["validation"]["text_field"] == "sanitized_text"
    assert views["test"]["text_field"] == "sanitized_text"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_attack_splits.py -v
```

Expected: FAIL because attack module is missing.

- [ ] **Step 3: Implement utility and text attack runners**

Implementation requirements:

- Utility evaluation:
  - accept sanitized train / validation / test splits
  - train a paper-style text classifier over `label`
  - return accuracy / macro-f1
- Text attacks:
  - static: clean train/validation + sanitized test
  - adaptive: sanitized train/validation/test
  - return author-id metrics
- preserve compatibility with existing attack code where useful, but the new runner must have one stable YAML-facing interface

- [ ] **Step 4: Run tests again**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_attack_splits.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dp_prompt/evaluation/utility.py dp_prompt/attacks/text_attacks.py tests/test_attack_splits.py
git commit -m "feat: add utility evaluation and text attack runners"
```

---

## Task 6: Implement the full document pipeline runner and output artifacts

**Files:**
- Create: `dp-prompt/dp_prompt/runners/document_pipeline.py`
- Modify: `dp-prompt/dp_prompt/utils/io.py`
- Create: `dp-prompt/configs/experiments/r1_imdb_base.yaml`
- Create: `dp-prompt/configs/experiments/r1_imdb_temp_low.yaml`
- Create: `dp-prompt/configs/experiments/r1_imdb_temp_mid.yaml`
- Create: `dp-prompt/configs/experiments/r1_imdb_temp_high.yaml`
- Create: `dp-prompt/scripts/run_dp_prompt.sh`
- Test: `dp-prompt/tests/test_runner_build.py`

- [ ] **Step 1: Write the failing pipeline-build test**

```python
from dp_prompt.runners.document_pipeline import build_pipeline_components


def test_build_pipeline_components_exposes_required_sections():
    cfg = {
        "dataset": {"name": "imdb"},
        "model": {"backend": "local_transformers"},
        "privacy": {"temperature": 1.0, "max_generated_tokens": 32},
    }

    components = build_pipeline_components(cfg)

    assert "privacy_controls" in components
    assert components["privacy_controls"]["temperature"] == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_runner_build.py::test_build_pipeline_components_exposes_required_sections -v
```

Expected: FAIL because runner is missing.

- [ ] **Step 3: Implement the end-to-end pipeline**

The runner must:

1. load dataset bundle
2. render prompts for all splits
3. generate sanitized train / validation / test documents
4. save `sanitized_corpus.json`
5. run utility evaluation and save `utility_summary.json`
6. run static/adaptive text attacks and save `privacy_attack_summary.json`
7. save `privacy_controls_summary.json`
8. save consolidated `experiment_summary.json`

The experiment YAMLs should define:

- `r1_imdb_base.yaml`
- `r1_imdb_temp_low.yaml`
- `r1_imdb_temp_mid.yaml`
- `r1_imdb_temp_high.yaml`

Each should inherit the same dataset/model base and only override privacy controls / experiment id.

- [ ] **Step 4: Run the pipeline-build tests again**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_runner_build.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dp_prompt/runners/document_pipeline.py dp_prompt/utils/io.py configs/experiments/r1_imdb_base.yaml configs/experiments/r1_imdb_temp_low.yaml configs/experiments/r1_imdb_temp_mid.yaml configs/experiments/r1_imdb_temp_high.yaml scripts/run_dp_prompt.sh tests/test_runner_build.py
git commit -m "feat: add dp-prompt document pipeline runner"
```

---

## Task 7: Verification and smoke-experiment readiness

**Files:**
- Modify: `dp-prompt/docs/2026-04-30-dp-prompt-design.md` only if implementation reality forces a correction
- Create: `dp-prompt/docs/2026-04-30-dp-prompt-test-experiments.md`

- [ ] **Step 1: Run the local tests that do not require heavyweight model dependencies**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m pytest tests/test_config.py tests/test_data_loader.py tests/test_privacy_controls.py tests/test_attack_splits.py tests/test_runner_build.py -v
```

Expected: PASS in the current local environment for the pure-Python parts.

- [ ] **Step 2: Run Python syntax verification**

Run:

```bash
cd /Users/apple/Desktop/code_from_paper/dp-prompt
python -m py_compile $(find dp_prompt -name '*.py' -print)
```

Expected: PASS with no output.

- [ ] **Step 3: Write the real smoke-experiment instructions**

The testing doc must define:

- required real dataset env vars:
  - `DP_PROMPT_IMDB_TRAIN`
  - `DP_PROMPT_IMDB_VALIDATION`
  - `DP_PROMPT_IMDB_TEST`
- required real local model env vars:
  - `DP_PROMPT_LOCAL_MODEL_PATH`
- smoke commands:

```bash
bash scripts/run_dp_prompt.sh configs/experiments/r1_imdb_base.yaml
bash scripts/run_dp_prompt.sh configs/experiments/r1_imdb_temp_low.yaml
bash scripts/run_dp_prompt.sh configs/experiments/r1_imdb_temp_mid.yaml
bash scripts/run_dp_prompt.sh configs/experiments/r1_imdb_temp_high.yaml
```

- expected outputs:
  - `sanitized_corpus.json`
  - `utility_summary.json`
  - `privacy_attack_summary.json`
  - `privacy_controls_summary.json`
  - `experiment_summary.json`

- [ ] **Step 4: Commit**

```bash
git add docs/2026-04-30-dp-prompt-test-experiments.md
git commit -m "docs: add dp-prompt smoke experiment instructions"
```

---

## Spec Coverage Self-Review

- Independent project under `dp-prompt/`: covered by Tasks 1-6.
- YAML-driven experiments: covered by Tasks 1, 2, and 6.
- Paper-style document sanitization loop: covered by Tasks 3-6.
- Local open-source backend first: covered by Task 4 and model config files.
- Utility evaluation: covered by Task 5.
- Text attacks only, static + adaptive: covered by Task 5 and attack-split tests.
- Privacy-control metadata and paper-style reproducibility surface: covered by Tasks 3 and 6.
- Structured outputs: covered by Task 6.
- Real smoke experiments, no mock model/dataset: covered by Task 7.

## Placeholder Scan

- No `TODO`, `TBD`, or “implement later” placeholders remain.
- Every task includes exact files, commands, and required outputs.

## Type Consistency Check

- Standard dataset columns are consistently named `sample_id`, `text`, `label`, `author_id`, `split`.
- Sanitized output column is consistently named `sanitized_text`.
- Privacy summary artifact is consistently named `privacy_controls_summary.json`.

## Execution Handoff

Plan complete and saved to `/Users/apple/Desktop/code_from_paper/dp-prompt/docs/2026-04-30-dp-prompt-implementation-plan.md`.

The user has already chosen inline execution, so the next step is to implement this plan in the current session using the executing-plans workflow.
