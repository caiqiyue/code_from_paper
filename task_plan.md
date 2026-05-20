# Task Plan

## Goal
Assess whether the current `round19` / `round23` code, the `round19_round23_collection_repeat40` 1200-record experiment set, and the surrounding training/runtime flow are sufficient to support a minimal new `anchor robustness` experiment package.

The review must answer:
- whether `paper-new-round23` is structurally hardcoded to `k0=20`, and where
- whether the 1200 collection records are sufficient for fixed-anchor analysis and/or controller training around `k0=19/20/21`
- which minimal experiment path is easiest now: offline replay analysis, config-only runtime smoke/quick compare, or fresh collection
- which experiments are already supported by code versus which need development

## Phases

1. Audit `round23` code/config/controller bundle for `k0=20` dependencies. in_progress
2. Audit `round19_round23_collection_repeat40` manifests and record files for coverage and field completeness. pending
3. Cross-check training/runtime scripts to see what can reuse the existing collection without code changes. pending
4. Write the final readiness assessment with concrete file/dir evidence. pending

## Current Status

- Phase 1: in_progress
- Phase 2: pending
- Phase 3: pending
- Phase 4: pending

## Risks To Track

- The "1200 records" may refer to generated configs rather than materialized result rows.
- `round23` may expose `reference_budget` as a CLI/config argument while still baking `k20` into feature schema and model artifacts.
- Existing collection records may be sufficient for offline analysis but not for retraining the current controller bundle without schema changes.

## Errors Encountered

- None yet for this audit.
