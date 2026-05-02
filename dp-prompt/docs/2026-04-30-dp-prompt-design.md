# DP-Prompt Design

## 1. Goal

Implement `DP-Prompt` as a **fully independent project** inside:

`/Users/apple/Desktop/code_from_paper/dp-prompt`

The first stage should **faithfully reproduce the paper's original task chain**, not immediately adapt to the current four-dataset comparison pipeline.

The project should borrow the engineering strengths of the current `pretext` workflow:

- YAML-driven experiments
- repeatable experiment runners
- clean output directories
- structured summaries

But it should remain algorithmically and structurally independent from `PrE-Text`, `paper-new`, and `dp-fedavg`.

## 2. Algorithm Positioning

`DP-Prompt` is fundamentally different from `DP-FedAvg`.

### `DP-Prompt`

- does **not** fine-tune the generation model,
- uses a pretrained language model in a **zero-shot prompting** setup,
- takes a private document as input,
- generates a sanitized paraphrase as output,
- controls the privacy-utility tradeoff through decoding behavior, especially **temperature** and optional **logit clipping**.

### Core paper-style flow

```text
private document
-> prompt template
-> pretrained local model
-> DP-aware decoding / temperature-controlled sampling
-> sanitized document
-> utility evaluation
-> privacy attack evaluation
```

This means the primary output of the algorithm is **sanitized text**, not a trained or fine-tuned model.

## 3. Round 1 Scope

Round 1 should reproduce the **original DP-Prompt document-sanitization task**.

It should support a complete runnable pipeline:

1. read private documents,
2. construct prompts,
3. generate sanitized paraphrases with a local open-source model,
4. run utility evaluation,
5. run privacy attack evaluation,
6. write structured summaries.

Round 1 should prioritize:

- document-level DP-Prompt generation,
- YAML-driven experiment control,
- local open-source generation backend,
- **text attack** privacy evaluation only,
- structured outputs suitable for future sweep analysis.

Round 1 should **not require**:

- OpenAI API integration,
- multi-model comparison sweeps,
- embedding attack implementation,
- direct adaptation to the four current project datasets,
- implementation of unrelated DP mechanisms from the repository.

## 4. Core Constraints

The implementation must respect the following constraints:

1. **Independent project**
   - All new engineering for this algorithm should remain under `dp-prompt/`.

2. **Paper-faithful first**
   - Stage 1 is for reproducing the paper's original task structure before adapting it to the broader project comparison ecosystem.

3. **YAML-driven experiments**
   - Experiments must be configured through layered YAML files rather than notebook-only or ad hoc script parameters.

4. **Open-source local model first**
   - The first implementation should use a local open-source generation backend.
   - Round 1 should be faithful to the paper's **algorithmic structure and experiment loop**, while explicitly allowing the generation backend to differ from the paper's original closed-model setup.
   - This means Round 1 is a **paper-style open-source reproduction**, not a vendor-identical reproduction.

5. **Full paper-style evaluation loop**
   - Generation alone is not enough.
   - Round 1 must include both utility evaluation and privacy attack evaluation.

6. **Privacy attacks in Round 1**
   - Round 1 should implement **text attack only**:
     - static attack
     - adaptive attack

## 5. High-Level Architecture

Recommended project structure:

```text
dp-prompt/
  configs/
    base/
    datasets/
    models/
    experiments/
  docs/
  dp_prompt/
    data/
    prompting/
    decoding/
    generation/
    evaluation/
    attacks/
    runners/
    utils/
  scripts/
  tests/
```

## 6. Module Responsibilities

### 6.1 `data/`

Responsible for loading and standardizing the paper task datasets.

Round 1 responsibilities:

- load raw text records,
- extract:
  - document text
  - label
  - author identity
- produce standardized examples for generation, utility eval, and privacy attack eval.

This layer is critical because the algorithm simultaneously needs:

- document text for sanitization,
- labels for utility evaluation,
- author identifiers for deanonymization attacks.

### 6.2 `prompting/`

Responsible for constructing prompts from private documents.

Responsibilities:

- define prompt templates,
- format prompts consistently,
- keep prompt logic independent from generation backend logic.

This separation matters because prompt construction is part of the algorithmic identity of DP-Prompt.

### 6.3 `decoding/`

Responsible for DP-aware decoding configuration.

Responsibilities:

- temperature configuration,
- optional logits clipping configuration,
- decoding parameter normalization,
- explicit representation of privacy-utility tradeoff controls,
- privacy accounting metadata capture sufficient to reproduce the paper-style privacy setting.

This layer should isolate the algorithm's main privacy control knobs from the rest of the generation code.

Round 1 should persist all privacy-relevant decoding controls needed to interpret or recompute the run:

- temperature,
- whether logits clipping is enabled,
- clipping bounds if enabled,
- maximum generated token budget,
- stop-token configuration,
- any paper-specific privacy constants exposed by the implementation.

### 6.4 `generation/`

Responsible for executing paraphrase generation with the local open-source model.

Responsibilities:

- load or attach to the configured local model backend,
- run batched paraphrase generation,
- output sanitized text,
- preserve generation metadata for experiment summaries.

This is the execution core of the DP-Prompt mechanism.

### 6.5 `evaluation/`

Responsible for utility evaluation.

Round 1 focus:

- reproduce the paper-style utility task first,
- report utility metrics such as classification quality in structured form.

This layer should be independent from privacy attack logic.

### 6.6 `attacks/`

Responsible for privacy evaluation.

Round 1 focus:

- text attack only
- static attacker
- adaptive attacker

This layer must measure deanonymization effectiveness on sanitized text outputs.

Round 1 attack semantics should be fixed in the design rather than left implicit:

- **Utility evaluation split**
  - Utility evaluation should use the same train / validation / test partition as the paper-style task setup.
  - Sanitized text should replace original private text only for the evaluation view being measured.

- **Static text attack**
  - Attacker train split: clean train documents.
  - Attacker validation split: clean validation documents.
  - Attacker test split: sanitized test documents.
  - Intuition: the attacker is trained on clean public-style text but tested against sanitized private outputs.

- **Adaptive text attack**
  - Attacker train split: sanitized train documents.
  - Attacker validation split: sanitized validation documents.
  - Attacker test split: sanitized test documents.
  - Intuition: the attacker adapts to the sanitization mechanism by observing sanitized training data.

- **Shared indexing rule**
  - Utility evaluation and privacy attacks should share the same dataset split indices.
  - The sanitized corpus must preserve sample identity so that attacks and utility evaluation refer to the same examples.

### 6.7 `runners/`

Responsible for orchestrating the full experiment pipeline.

Expected flow:

```text
load config
-> load dataset
-> build prompts
-> generate sanitized documents
-> run utility evaluation
-> run privacy attacks
-> write summaries
```

This layer should coordinate modules, not re-implement generation, prompting, or attacks internally.

### 6.8 `utils/`

Shared support code only.

Responsibilities:

- YAML loading
- path resolution
- seed management
- logging
- output writing
- summary helpers

## 7. YAML Configuration Design

The configuration system should mirror the strengths of the current `pretext` workflow while staying independent.

### 7.1 `configs/base/`

Shared defaults for all experiments.

Recommended contents:

- runtime device
- random seed
- output root rules
- generation defaults
- DP decoding defaults
- evaluation defaults
- privacy-reporting defaults

### 7.2 `configs/datasets/`

One YAML per paper dataset.

Each dataset config should specify:

- train / validation / test paths
- text field
- label field
- author field
- utility task metadata
- privacy attack metadata
- explicit split files or reproducible split seed

Round 1 should stay aligned with the datasets needed to reproduce the paper task chain.

### 7.3 `configs/models/`

Model backend configuration should be isolated from experiment wiring.

Recommended contents:

- backend type
- local model path
- tokenizer path if needed
- dtype
- max context length
- generation batch size
- generation token budget

### 7.4 `configs/experiments/`

Actual runnable experiment entry points.

Each experiment should combine:

- one base config,
- one dataset config,
- one model config,
- optional experiment-specific overrides.

Each runnable experiment should also explicitly declare the privacy-control surface being exercised:

- temperature value,
- logits clipping mode,
- clipping bounds if active,
- generated token budget,
- enabled attack modes,
- whether privacy metrics are only recorded or also converted to a reported paper-style summary.

Examples of useful Round 1 experiment types:

- base experiment
- temperature sweep
- attack mode comparison

## 8. Round 1 Minimum Runnable Experiment Loop

The first runnable loop should be:

1. input real private documents,
2. build prompts,
3. generate sanitized documents with local model inference,
4. run utility evaluation,
5. run text-based deanonymization attacks,
6. save all artifacts and summaries, including privacy-control metadata and privacy-reporting outputs.

Round 1 should specifically include:

### 8.1 Base experiment

One canonical paper-style DP-Prompt run.

### 8.2 Temperature sweep

At least a small three-point sweep:

- low temperature
- medium temperature
- high temperature

Because temperature is the primary privacy-utility control in the paper.

### 8.3 Fixed-model first pass

Round 1 should fix the generation model to a single local open-source model.

Multi-model comparisons can be added later, but the code structure should allow them.

## 9. Output Artifacts

Each experiment should produce four core outputs.

### 9.1 `sanitized_corpus.json`

This is the most important intermediate artifact.

Each record should contain at least:

- sample id
- original text
- sanitized text
- label
- author id
- generation metadata
- split name

### 9.2 `utility_summary.json`

Contains:

- utility task name
- evaluation config
- main utility metrics

### 9.3 `privacy_attack_summary.json`

Contains:

- attack mode
- attacker type
- privacy metrics such as deanonymization performance
- split definitions used for attacker train / validation / test

### 9.4 `privacy_controls_summary.json`

Contains the paper-style privacy control surface and any reported privacy quantities the run can support.

At minimum:

- temperature
- logits clipping enabled flag
- clipping bounds if enabled
- generated token budget
- stop-token policy
- paper-style privacy metadata needed for recomputation
- reported privacy summary if the implementation computes one directly

### 9.5 `experiment_summary.json`

A final consolidated summary containing:

- dataset
- model
- temperature
- document count
- utility results
- privacy attack results
- privacy-controls summary reference
- references to output artifact paths

## 10. Comparison Semantics

Although Round 1 is focused on paper-faithful reproduction rather than immediate adaptation to the current four-dataset system, the design should still preserve a future path toward comparison.

This future compatibility should come from:

- structured sanitized text outputs,
- stable evaluation summaries,
- YAML-driven reproducibility,
- modular generation and attack code.

Round 1 should not distort the paper task just to align early with the broader comparison pipeline.

## 11. Round 1 Success Criteria

Round 1 is successful if:

1. real private documents can be sanitized with DP-Prompt-style paraphrase generation,
2. utility evaluation runs successfully on sanitized documents,
3. text-based privacy attacks run successfully,
4. privacy-control parameters are explicitly persisted in structured outputs,
5. experiment outputs are structured and reproducible,
6. YAML-driven experiments replace notebook-only execution as the primary workflow.

## 12. Non-Goals for Round 1

Round 1 does not aim to:

- adapt immediately to the current four-dataset comparison system,
- add OpenAI API support,
- implement embedding attacks,
- reproduce every model sweep from the paper,
- merge the algorithm into `pretext` or `paper-new`,
- turn `dp-prompt` into a general-purpose privacy framework.

Round 1 should stay focused on building a faithful, maintainable, YAML-driven reproduction of the DP-Prompt paper's original document-level task loop.
