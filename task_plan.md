# Task Plan: Pre-Text and Single-Node Innovation Workflow Trace

## Goal
Read `PrE-Text`, `thesis_platform`, and `execute/new` formal experiment docs/configs, then reconstruct:
- the single-node innovation algorithm workflow,
- the pre-text baseline workflow,
- the concrete models used at each stage,
- and especially which model generates the synthetic data.

## Current Phase
Complete

## Phases
### Phase 1: Workflow and Doc Discovery
- [x] Read the formal experiment docs under `execute/new`
- [x] Identify which docs describe single-node innovation vs pre-text baseline
- [x] Record the mapping in `findings.md`
- **Status:** complete

### Phase 2: Single-Node Innovation Trace
- [x] Trace the entrypoint, config chain, and runtime path in `thesis_platform`
- [x] Reconstruct the algorithm steps from selection to generation to evaluation
- [x] Identify the generation model, scorer model, embedding model, and evaluation model
- **Status:** complete

### Phase 3: Pre-Text Baseline Trace
- [x] Trace the entrypoint, config chain, and runtime path in `PrE-Text`
- [x] Reconstruct the stage-wise pre-text pipeline
- [x] Identify which model performs synthetic text generation and which models are used for evaluation
- **Status:** complete

### Phase 4: Cross-Check and Synthesis
- [x] Cross-check code behavior against the formal experiment docs
- [x] Note any doc/code mismatch or ambiguity
- [x] Draft a concise comparison of the two workflows
- **Status:** complete

### Phase 5: Delivery
- [x] Answer the user in Chinese with algorithm steps and model names
- [x] Include file references and line numbers for the main claims
- [x] Call out residual uncertainty explicitly
- **Status:** complete

## Key Questions
1. In the formal single-node setup, what exactly is the innovation algorithm doing step by step?
2. In the formal pre-text setup, what exactly is the experiment pipeline doing step by step?
3. Which models are used for generation, scoring, prototype construction, privacy, and evaluation?
4. Which model specifically generates the synthetic data in each workflow?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Treat `execute/new` docs as the intended experiment blueprint | The user explicitly pointed to them as the formal experiment design |
| Use code as the ground truth when docs and code differ | The user asked for the actual implementation |
| Separate “single-node innovation algorithm” and “pre-text baseline” into two traced pipelines | They live in different codebases and use different abstractions |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| PowerShell rejected `for ... in` syntax and wildcard `rg` on Windows paths | 1 | Switched to `foreach` and explicit file enumeration |

## Notes
- Keep findings tied to formal configs under `configs/experiments/single_node_formal`.
- Prefer direct runtime code over README summaries.
