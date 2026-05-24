# E3 Controller Decision Quality Experiment Design

Updated: 2026-05-24

## 1. Position in the Thesis Experiment Matrix

E3 should be defined as the **controller decision quality experiment**.

E1 has already answered the end-to-end performance question: whether `round23` is competitive with `PrE-Text`, `round19`, `WASP`, and `DPGA-TextSyn` on the controller-development datasets.

E2 answers the held-out generalization question: whether the same `round23` controller can transfer to extra unseen datasets.

Therefore, E3 should not repeat another method-level `best_top1` comparison. Its job is to answer a more specific mechanism question:

> Does the learned `round23` controller make better seed-budget decisions, or are the final scores only caused by random variation?

## 2. Main Claim

E3 should mainly prove two claims:

1. Compared with fixed `keep-k0=20`, `round23` makes useful local budget corrections.
2. Compared with the rule-based `round19 resolver replay`, the learned `round23` controller is closer to the offline oracle budget.

This directly supports the thesis line:

`PrE-Text` exposes the seed-budget bottleneck.

`round19` solves it with a rule-based adaptive resolver.

`round23` upgrades the resolver into a learned dynamic controller.

E3 is the experiment that must show the learned controller is not just a model wrapper around the old rule system; it has measurable budget-decision quality.

## 3. What E3 Should Not Try to Prove

E3 should not primarily prove that two-round `delta_k` is better than one-shot absolute-`k` prediction.

That question belongs to the later **round-count justification** experiment, because it answers a different reviewer concern:

> Why is the algorithm specifically two-round instead of one-shot or three-round?

Keeping this separation avoids mixing three different claims in one section:

- controller quality;
- round-count structure;
- one-shot-vs-delta-k formulation.

E3 can briefly mention that round-count justification will be handled separately, but it should not carry that proof.

## 4. Experimental Unit

The experimental unit is one controller context:

`dataset_name + meta_seed`

The current E3 artifact scope is the six controller-development datasets with 40 contexts per dataset:

`6 datasets x 40 contexts = 240 contexts`

For each context, the 1200-record collection provides a local budget sweep around `k0=20`:

- `k=18`
- `k=19`
- `k=20`
- `k=21`
- `k=22`

Equivalently, the controller action space is:

- `delta_k=-2`
- `delta_k=-1`
- `delta_k=0`
- `delta_k=+1`
- `delta_k=+2`

This allows offline policy evaluation without rerunning all vLLM experiments.

## 5. Data Source

Primary source:

- `model-train/artifacts/round23_datasets_1200_top1_delta_m0005/round23_controller_context_table.jsonl`
- `model-train/artifacts/round23_datasets_1200_top1_delta_m0005/round19_replay_table.jsonl`
- `model-train/artifacts/round23_datasets_1200_top1_delta_m0005/round19_replay_mapping.json`

These files are derived from the completed 1200 collection experiments.

The context table already contains:

- controller state features at `k0=20`;
- reward for each `delta_k`;
- `best_top1` for each `delta_k`;
- oracle best `delta_k`;
- oracle best target budget;
- oracle reward;
- keep-k0 reward.

The replay table records the `round19` resolver decision for each context.

Important reward semantics:

- use the context table's `controller_reward_dk_*` fields to compute the selected reward for `keep-k0`, `round19`, `round23`, and `oracle`;
- use the context table's `best_top1_dk_*` fields for secondary `best_top1` regret;
- use `round19_replay_table` only to obtain the `round19` selected budget / `delta_k`;
- do not use `round19_replay_reward` as a primary E3 reward, because it was generated with the legacy reward formula.

## 6. Policies to Compare

E3 should compare four budget policies.

| Policy | Meaning | Role |
|---|---|---|
| `keep-k0=20` | Always use the reference budget without correction | fixed-budget lower baseline |
| `round19 resolver replay` | Reconstruct the rule-based round19 budget decision on the same collection contexts | rule-based adaptive baseline |
| `round23 controller` | Use the trained all6 controller to choose `delta_k` | learned policy under evaluation |
| `oracle budget` | Pick the best observed action from the local sweep | offline upper bound |

`oracle budget` is not a deployable method. It is only the ceiling used to calculate regret.

## 7. Metrics

E3 should report both aggregate and dataset-wise results.

### 7.1 Primary Metrics

| Metric | Definition | Purpose |
|---|---|---|
| `mean controller reward` | Mean reward achieved by the policy-selected `delta_k` | Measures policy utility directly |
| `mean regret vs oracle` | `oracle_reward - policy_reward` | Measures distance from best local budget |
| `win rate vs keep-k0` | Share of contexts where policy reward is higher than `delta_k=0` | Tests whether correction helps |
| `direction accuracy` | Whether predicted direction matches oracle direction: decrease / keep / increase | Tests coarse decision quality |
| `delta_k accuracy` | Exact match with oracle `delta_k` | Tests strict action accuracy |

### 7.2 Secondary Metrics

| Metric | Definition | Purpose |
|---|---|---|
| `mean best_top1 regret` | Oracle `best_top1` minus policy-selected `best_top1` | Links policy quality back to downstream result |
| `win rate vs round19` | Share of contexts where `round23` reward is higher than `round19` reward | Direct learned-vs-rule comparison |
| `dataset-wise regret` | Regret split by dataset | Checks whether the controller only works on one dataset |

## 8. Recommended Result Tables

### Table E3-1: Overall Controller Policy Quality

| Policy | Contexts | Mean Reward | Mean Regret vs Oracle | Win Rate vs keep-k0 | Win Rate vs round19 | Direction Acc. | Delta-k Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| `keep-k0=20` |  |  |  |  |  |  |  |
| `round19 resolver replay` |  |  |  |  |  |  |  |
| `round23 controller` |  |  |  |  |  |  |  |
| `oracle budget` |  |  |  |  |  |  |  |

### Table E3-2: Dataset-wise Controller Quality

| Dataset | Policy | Contexts | Mean Reward | Mean Regret vs Oracle | Win Rate vs keep-k0 | Direction Acc. |
|---|---|---:|---:|---:|---:|---:|
| `jobs` | `round19 resolver replay` |  |  |  |  |  |
| `jobs` | `round23 controller` |  |  |  |  |  |
| `congressional` | `round19 resolver replay` |  |  |  |  |  |
| `congressional` | `round23 controller` |  |  |  |  |  |
| `forums` | `round19 resolver replay` |  |  |  |  |  |
| `forums` | `round23 controller` |  |  |  |  |  |
| `microblog` | `round19 resolver replay` |  |  |  |  |  |
| `microblog` | `round23 controller` |  |  |  |  |  |
| `imdb` | `round19 resolver replay` |  |  |  |  |  |
| `imdb` | `round23 controller` |  |  |  |  |  |
| `openreview` | `round19 resolver replay` |  |  |  |  |  |
| `openreview` | `round23 controller` |  |  |  |  |  |

### Table E3-3: Action Distribution

| Policy | `delta=-2` | `delta=-1` | `delta=0` | `delta=+1` | `delta=+2` |
|---|---:|---:|---:|---:|---:|
| `oracle budget` |  |  |  |  |  |
| `round19 resolver replay` |  |  |  |  |  |
| `round23 controller` |  |  |  |  |  |

This table is useful because it shows whether the learned controller collapses to a single action or actually uses the local action space.

## 9. Scope and Splits

The default E3 scope should be the full six controller-development datasets:

- `jobs`
- `congressional`
- `forums`
- `microblog`
- `imdb`
- `openreview`

The paper can also report split-level summaries:

- original seen four datasets;
- controller-dev two added datasets;
- all six datasets.

E3 should not use the extra held-out E2 datasets unless a later collection exists for them. E2 datasets are for end-to-end held-out generalization, not for offline oracle evaluation unless they also have complete `k=18..22` sweeps.

## 10. Interpretation Rules

E3 supports the mechanism claim if:

- `round23` has lower mean regret than `keep-k0=20`;
- `round23` has lower or comparable regret than `round19 resolver replay`;
- `round23` has higher win rate vs `keep-k0` than `round19` or at least competitive win rate with lower regret;
- direction accuracy is meaningfully above a trivial majority-direction baseline;
- dataset-wise results do not show that all gains come from only one dataset.

E3 does not require `round23` to beat `round19` on every single dataset. A defensible conclusion is:

> The learned controller produces a budget policy that is closer to the local oracle and more adaptive than a fixed anchor, while remaining competitive with or better than the rule-based resolver.

## 11. Expected Writing Role in the Paper

E3 should appear after E1 and E2.

Suggested narrative:

1. E1 shows the complete method is competitive end-to-end.
2. E2 shows the method can be applied to extra held-out datasets.
3. E3 explains why the controller mechanism is meaningful: it makes measurable budget decisions closer to the local oracle.
4. Later experiments handle round-count justification, anchor robustness, and ablation.

## 12. Risk Control

Potential reviewer concern:

> If `round23` does not dramatically beat `round19` in final `best_top1`, why is it needed?

E3 answers:

- final score is noisy and expensive;
- offline policy metrics reveal whether the budget decision itself is better;
- `round23` replaces hand-coded rules with a learned policy that can be evaluated by regret, action accuracy, and win rate against fixed budget.

Potential reviewer concern:

> Is this just overfitting to `k0=20`?

E3 should not overclaim. It should explicitly say:

- E3 evaluates the controller inside the current formal local action space around `k0=20`;
- anchor robustness is handled in a separate experiment;
- E3 proves controller quality under the formal anchored-local setup, not arbitrary-anchor generalization.
