# DP-FedAvg Design

## 1. Goal

Implement the algorithm from the `dp-fedavg` paper as an independent baseline in:

`/Users/apple/Desktop/code_from_paper/dp-fedavg`

This project is not an extension of `PrE-Text` or `paper-new`. It is a standalone comparison method that should:

- reproduce the paper's core algorithmic structure,
- run in the existing server `pretext` environment when possible,
- reuse the current four datasets and local model resources from `thesis_platform`,
- output client-side evaluation results that can be compared directly against the second-stage results of `PrE-Text` and later selector-based methods.

## 2. Positioning in the Existing Experimental System

Within the current project ecosystem:

- `PrE-Text` and `paper-new` are treated as synthetic-data style baselines or improved methods.
- `dp-fedavg` is treated as an independent algorithm baseline.

Its comparison role is:

- **server side**: run federated/private training and aggregation,
- **client side**: perform downstream task evaluation,
- **main comparison target**: client-side metrics such as `top1/top3/top5/top10`.

The client-side result of `dp-fedavg` is considered comparable to the current "second-stage" evaluation output used by `PrE-Text` and the new selector algorithm.

## 3. Scope of Round 1

Round 1 is a **minimum viable paper-faithful implementation**, not a full ablation-complete system.

Round 1 must support:

1. **Federated runner**
   - multiple clients / users,
   - Poisson-style user sampling,
   - local user updates,
   - DP clipping and Gaussian noise,
   - server aggregation,
   - client-side downstream evaluation.

2. **Single-node degenerate runner**
   - one client / one user,
   - acts as a `DP local training baseline`,
   - keeps the same clipping / noise / accounting interfaces.

Round 1 will implement the main paper path only:

- `FedAvg`
- flat clipping
- fixed-denominator aggregation estimator
- Gaussian noise
- basic privacy accountant
- YAML-driven experiments

Round 1 will **not require full implementation yet** of:

- `FedSGD`
- per-layer clipping
- clipped-denominator estimator
- late-stage clipping/noise schedule

These should be anticipated in interfaces, but do not need to be fully runnable in the first implementation wave.

## 4. Constraints

The implementation must respect these constraints:

1. **Code location**
   - all new experiment code lives inside `dp-fedavg`.

2. **Environment**
   - should run directly in the server-side `pretext` environment whenever possible.

3. **Datasets**
   - use the current four experimental datasets already used in the broader project.

4. **Models and generation**
   - all generation-related behavior should use `vllm + local model`,
   - model paths and related runtime conventions should align with the current local/server setup.

5. **Experiment control**
   - all experiments must be driven by YAML files,
   - configuration style should be close to the current `pretext` workflow.

6. **Evaluation**
   - client-side downstream evaluation should reuse the current evaluation style and output format as much as possible, so that results remain directly comparable.

## 5. High-Level Architecture

Recommended project layout:

```text
dp-fedavg/
  configs/
    base/
    datasets/
    algorithms/
    experiments/
  docs/
  dp_fedavg/
    runners/
    data/
    models/
    training/
    privacy/
    aggregation/
    generation/
    evaluation/
    utils/
  scripts/
  tests/
```

### 5.1 `runners/`

Top-level execution flow only.

- `federated_runner`
  - full paper-style multi-client training
- `single_node_runner`
  - degenerate one-client baseline

This layer should orchestrate the experiment, not implement DP math directly.

### 5.2 `data/`

Responsible for dataset adaptation into user/client training units.

Responsibilities:

- read the four datasets from current project resources,
- determine whether a dataset has a natural user/client structure,
- if not, build pseudo-clients,
- construct round-level sampled client batches,
- feed local training data to client updates.

Client construction policy:

- if a natural user-like field exists, use it,
- otherwise create pseudo-clients.

### 5.3 `models/`

Round 1 should stay close to the paper.

Responsibilities:

- provide the core model wrapper used by DP-FedAvg training,
- expose a stable training/evaluation interface,
- support local model loading in a way compatible with the current environment.

The design should avoid prematurely over-abstracting model support.

### 5.4 `training/`

Responsible for local client update logic.

Round 1 requires:

- `FedAvg` local update path

Responsibilities:

- start from current global model state,
- train on one client's local data for the configured local steps/epochs,
- return a user update / model delta to the server.

This layer should not own clipping or noise logic.

### 5.5 `privacy/`

Core DP mechanism layer.

Responsibilities:

- flat clipping,
- Gaussian noise injection,
- privacy accountant,
- DP parameter validation.

This layer should be mathematically isolated so that later ablations are easy.

### 5.6 `aggregation/`

Server-side update combination.

Round 1 requires:

- fixed-denominator estimator

Responsibilities:

- receive local client updates,
- apply clipping/noise outputs in the correct order,
- aggregate and apply the server update.

### 5.7 `generation/`

This project must align with the current resource setup.

Responsibilities:

- route generation-related operations through `vllm + local model`,
- keep a unified interface for prompt building and generation calls,
- provide a reusable generation layer for any synthetic or text output needed by the experiment flow.

### 5.8 `evaluation/`

Client-side evaluation is the primary comparison output.

Round 1 should output:

- `top1`
- `top3`
- `top5`
- `top10`

Where possible, evaluation style and reporting should align with the current project conventions so that `dp-fedavg` can be inserted into existing result tables without special-case treatment.

### 5.9 `utils/`

Shared support code only.

Responsibilities:

- YAML loading
- experiment wiring
- seed control
- logging
- path resolution
- summary writing

## 6. Dual Runner Strategy

The project should use **shared lower-level modules + two runners**.

This is the required structure because the project must support both:

1. **full federated architecture**
2. **single-node degenerate DP local training baseline**

The lower-level algorithm components should be implemented only once, including:

- client update logic,
- clipping,
- noising,
- aggregation,
- accounting,
- evaluation plumbing.

The two runners differ mainly in:

- client sampling behavior,
- number of participating clients,
- aggregation semantics,
- logging semantics.

## 7. YAML Configuration Design

Experiments must be YAML-driven.

Recommended configuration split:

### 7.1 `configs/base/`

Shared runtime and infrastructure defaults.

Example contents:

- runtime device
- seeds
- output conventions
- vLLM backend selection
- local model path defaults
- logging defaults

### 7.2 `configs/datasets/`

One file per dataset.

Each dataset config should define:

- dataset name
- train/eval paths
- whether a natural user/client field exists
- how pseudo-clients should be built if needed
- any dataset-specific evaluation settings

### 7.3 `configs/algorithms/`

Algorithm presets for DP-FedAvg.

Round 1 config fields should include at least:

- runner mode
- local update type
- client sampling mode and rate
- local steps / local epochs
- clipping mode
- clip norm
- noise multiplier
- privacy delta
- aggregation estimator

### 7.4 `configs/experiments/`

Concrete experiment entry points.

These should combine:

- one base runtime config,
- one dataset config,
- one algorithm preset,
- optional overrides.

## 8. Data Flow

### 8.1 Federated runner

```text
dataset
-> user/client partition
-> sampled clients per round
-> local client updates
-> clipping
-> Gaussian noise
-> server aggregation
-> updated server state
-> client-side downstream evaluation
-> summaries
```

### 8.2 Single-node runner

```text
dataset
-> single client or single logical user
-> local update
-> clipping
-> Gaussian noise
-> update apply
-> client-side downstream evaluation
-> summaries
```

## 9. Comparison Semantics with `PrE-Text`

`dp-fedavg` does not need to mirror `PrE-Text`'s explicit two-stage synthetic-data pipeline.

Instead:

- the DP-FedAvg server side is treated as the algorithmic learning/generation side,
- the client side is treated as the downstream evaluation side,
- the **final client-side metrics** are the direct comparison target.

Therefore, the main comparison table should align on:

- `top1`
- `top3`
- `top5`
- `top10`

Intermediate server statistics should still be preserved, but they are secondary.

## 10. Required Outputs

Each experiment should produce two classes of output.

### 10.1 Server-side auxiliary outputs

- sampled clients per round
- clip norm statistics
- noise scale
- privacy `(epsilon, delta)`
- number of rounds
- convergence summary

### 10.2 Client-side primary outputs

- `top1`
- `top3`
- `top5`
- `top10`

These client-side outputs are the main baseline-comparison results.

## 11. Round 1 Minimum Experiment Set

Round 1 should include the following experiment types:

1. **federated base**
2. **single-node degenerate base**
3. **small noise sweep**
4. **four-dataset base runs**

This set is sufficient to answer:

- whether the implementation is runnable end-to-end,
- whether it produces comparable outputs on the current four datasets,
- whether the DP baseline can be meaningfully inserted into the current evaluation framework.

## 12. Testing Strategy

Round 1 tests should focus on mechanism correctness and pipeline integrity.

Required test groups:

1. user/client sampling correctness
2. flat clipping correctness
3. Gaussian noise injection interface behavior
4. fixed-denominator aggregation behavior
5. YAML loading and experiment wiring
6. federated runner smoke test
7. single-node runner smoke test

The goal of Round 1 testing is not exhaustive benchmark validation, but confidence that the algorithmic skeleton and experiment plumbing are correct.

## 13. Non-Goals for Round 1

Round 1 does not aim to:

- become a general-purpose federated learning platform,
- replace `PrE-Text`,
- merge into `paper-new`,
- finish every ablation from the paper,
- optimize for best possible benchmark performance before the baseline is structurally correct.

Round 1 is successful if it produces a clean, paper-faithful, YAML-driven `dp-fedavg` baseline that can run in the current environment and output client-side metrics comparable to the current baseline family.
