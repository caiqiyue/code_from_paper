# Single-Node Stage A And Aggregation Design

Date: 2026-04-16
Scope: `thesis_platform` single-node workflow
Status: Draft for review

## 1. Background

The current single-node workflow keeps the high-level structure of:

1. generate candidate samples
2. score bad samples in Stage A
3. critique selected bad samples
4. aggregate critique rules into a prompt update
5. iterate

The current problem is not the outer workflow itself. The problem is that the Stage A scoring logic in the single-node setting has been mixed with a domain-probe interpretation that belongs more naturally to the federated setting.

For the single-node workflow, the main practical goal is simpler:

- find bad samples from the generated batch
- if the chosen algorithm cannot find meaningful bad samples, fall back to random selection

At the same time, the implementation should remain compatible with experimentation:

- Stage A must support multiple interchangeable scoring algorithms
- aggregation must also support multiple interchangeable critique-fusion algorithms

## 2. Goals

This design changes the single-node workflow in the following way:

- redefine Stage A as a bad-sample discovery stage, not a distribution-alignment stage
- make Stage A scorer configurable through the experiment config
- make aggregation configurable through the experiment config
- add explicit Stage A failure detection
- add random fallback when the selected scorer produces no useful ranking signal
- keep the rest of the workflow stable so algorithms can be compared fairly

## 3. Non-Goals

This design does not attempt to:

- redesign the full federated workflow
- remove or rename the existing algorithms for publication purposes
- settle which scorer is best before experiments
- change the critique or retrieval stage semantics beyond what is needed for cleaner interfaces

## 4. Stage A Role In Single-Node Mode

### 4.1 New Stage A definition

In single-node mode, Stage A has one responsibility:

- rank the current batch of generated samples by badness and select `top_k` samples for critique

Stage A does not attempt to:

- align multiple client distributions
- approximate personalized client-specific distributions
- use domain-shift detection as its primary success criterion

This is because the single-node setting has one server and one client. The personalized distribution-alignment motivation from the federated setting does not directly apply here.

### 4.2 Expected Stage A input/output

Input:

- real training samples for the single client
- generated candidate samples from the current prompt
- scorer configuration

Output:

- a ranked list of candidate samples with scores and metadata
- a selected set of `top_k` bad samples
- a record of whether selection came from algorithmic scoring or random fallback

## 5. Unified Stage A Scorer Interface

Stage A should treat all scorers as plugins behind one interface.

Conceptual contract:

- input: `train_samples`, `generated_samples`, `client_ctx`, `config`
- output: `[(sample, score, meta)]`

Required supported scorers:

- `datainf`
- `gradmm`
- `ira`
- `random`

Common interface rules:

- all scorers return the same result structure
- all scores use one shared direction: `larger_is_worse`
- the Stage A runner does not know algorithm internals
- scorer-specific details stay inside scorer implementations

This keeps the experiment framework clean and makes scorer comparison fair.

## 6. Stage A Failure Detection And Random Fallback

The scorer returning numbers does not mean those numbers are useful for bad-sample selection. Stage A therefore needs a separate signal-quality check.

### 6.1 Failure conditions

Stage A scoring is treated as failed if either condition holds:

1. all scores are equal or nearly equal
2. the gap between the `top_k` region and the overall median is too small

The first condition catches the obvious case:

- all `0`
- all identical
- nearly constant scores

The second condition catches the softer failure case:

- the scorer returns different values, but the ranking has no meaningful separation

### 6.2 Fallback behavior

If Stage A scoring fails:

- randomly sample `top_k` items from the current generated batch
- record that the selection mode was `random_fallback`
- record the failure reason in round artifacts

If Stage A scoring succeeds:

- sort by descending score
- select the worst `top_k`
- record the selection mode as `scored`

### 6.3 Why fallback is required

The Stage A loop must always provide samples to critique. If it cannot identify bad samples confidently, random fallback is preferable to a fake ranking because:

- it keeps the prompt-update loop alive
- it avoids over-interpreting invalid score patterns
- it creates a reproducible baseline for ablation experiments

## 7. DataInf Position In Single-Node Mode

### 7.1 Design principle

Single-node `datainf` should stay as close as possible to the original DataInf idea:

- measure how a sample affects a validation objective

It should not primarily be implemented as:

- a domain probe with synthetic negative samples
- a proxy classifier whose main job is distinguishing in-domain versus out-of-domain text

That interpretation is more natural in the federated setting, where multiple client distributions exist and personalization matters.

### 7.2 Single-node DataInf semantics

For single-node Stage A, `datainf` should be interpreted as:

- reference set: real single-client samples, or a held-out validation subset derived from them
- candidate set: generated samples from the current prompt
- score meaning: estimated harmfulness of a candidate with respect to the reference validation objective

Bad samples are therefore candidates that appear more likely to:

- hurt the reference objective
- be unhelpful to the reference objective
- produce unstable or negative influence relative to the reference objective

This keeps the single-node usage aligned with the original DataInf spirit even though the task carrier is synthetic-sample filtering rather than the exact benchmark tasks in the original repository.

### 7.3 Consequence for current implementation

The current single-node `domain_probe` interpretation should no longer be treated as the defining implementation for Stage A.

It may still exist as an experimental option in the future, but it should not define the default single-node Stage A logic.

## 8. Aggregation As A Configurable Plugin

Aggregation should be treated the same way as Stage A scoring: as a plugin selected by configuration.

### 8.1 Aggregation role

Aggregation has one responsibility:

- fuse critique rules from selected bad samples into the next prompt update

### 8.2 Required behavior

The single-node pipeline should allow aggregation strategy switching through config.

Initial target setup:

- default: `dbscan_attn_tsgdm`
- plus one or two simpler alternatives for controlled experiments

Recommended initial options:

- `dbscan_attn_tsgdm`
- `uid`
- `summarization`
- optional debug baseline: `identity` or `noop`

### 8.3 Why plugin aggregation matters

If Stage A is configurable but aggregation is fixed, then the experiment matrix is incomplete. Making aggregation configurable enables:

- fair comparison of critique-fusion strategies
- separation of ranking quality from fusion quality
- more defensible ablation studies

## 9. End-To-End Single-Node Loop After This Design

The revised single-node loop is:

1. generate `N` candidate samples using the current prompt
2. run the configured Stage A scorer
3. run Stage A failure detection
4. if scoring succeeded, take `top_k` worst samples by score
5. if scoring failed, randomly sample `top_k` candidates
6. retrieve anchor examples for the selected samples
7. generate critiques
8. run the configured aggregation strategy
9. update the prompt
10. continue until convergence or the configured iteration limit

The workflow skeleton remains stable. Only the scorer and aggregator are plugin points.

## 10. Configuration Changes

The experiment config should explicitly expose the following controls.

### 10.1 Stage A

- `stage_a.scorer`
- `stage_a.failure_equal_epsilon`
- `stage_a.failure_margin_threshold`
- `stage_a.random_fallback_seed`
- `stage_a.log_failure_reason`

Supported `stage_a.scorer` values in the first version:

- `datainf`
- `gradmm`
- `ira`
- `random`

### 10.2 Aggregation

- `aggregator.name`

Initial supported values:

- `dbscan_attn_tsgdm`
- `uid`
- `summarization`
- optional `identity`

## 11. Logging And Artifacts

Each Stage A round should save enough information to support debugging and ablation analysis.

Recommended fields:

- scorer name
- selected aggregator name
- score summary statistics
- whether failure detection triggered
- failure reason
- selection mode: `scored` or `random_fallback`
- selected sample ids
- selected sample scores

These artifacts are required because Stage A quality must be diagnosed independently from downstream metrics.

## 12. Experimental Plan Enabled By This Design

This design enables two clean experiment axes.

### 12.1 Scorer comparison

Fix one aggregator and compare:

- `datainf`
- `gradmm`
- `ira`
- `random`

Questions answered:

- which scorer most often produces a usable Stage A ranking
- which scorer most often triggers fallback
- which scorer leads to better later prompt updates

### 12.2 Aggregation comparison

Fix one scorer and compare:

- `dbscan_attn_tsgdm`
- `uid`
- `summarization`

Questions answered:

- which aggregation strategy best fuses critiques into prompt updates
- whether aggregation quality or scorer quality dominates the final result

## 13. Recommended First Implementation Scope

The first implementation pass should stay narrow.

Priority order:

1. redefine single-node Stage A semantics
2. add configurable scorer switching
3. add failure detection and random fallback
4. add configurable aggregation switching
5. add logging for failure reasons and selection mode

This order keeps the single-node workflow functional while minimizing risk.

## 14. Acceptance Criteria

The design is considered successfully implemented when:

1. single-node Stage A can select `datainf`, `gradmm`, `ira`, or `random` from config
2. Stage A failure detection can trigger random fallback automatically
3. selected samples always reach critique even when scoring fails
4. aggregation strategy can be switched from config
5. round artifacts clearly show scorer, aggregator, failure status, and selection mode
6. the single-node workflow no longer depends on domain-probe semantics as its primary Stage A definition

## 15. Open Decision Already Resolved

The following design choices have already been agreed for this document:

- single-node Stage A focuses on bad-sample discovery, not distribution alignment
- Stage A scorer is configurable through config
- supported Stage A scorer set starts with `datainf / gradmm / ira / random`
- scoring failure uses a combined rule:
  - equal or nearly equal scores
  - or weak separation from the median
- failure triggers random fallback
- aggregation is also configurable through config
- aggregation should start with the current default plus one or two alternatives

## 16. Summary

This design keeps the single-node innovation workflow intact while making the two key decision points configurable:

- which algorithm identifies bad samples
- which algorithm fuses critiques into prompt updates

The result is a cleaner single-node story:

- Stage A finds bad samples
- if it cannot, it falls back to random
- aggregation fuses critique rules using a configurable strategy
- experiments can compare both axes cleanly

That gives a more defensible implementation for the single-node setting and a cleaner path for later experiments.
