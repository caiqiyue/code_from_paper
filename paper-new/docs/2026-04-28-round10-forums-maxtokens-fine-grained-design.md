# Round 10: Forums Max Tokens Fine-Grained Search

## Hypothesis

max_tokens=85 is optimal, but we should fine-tune around it:
- 85 ± 4 range (81-89)
- Step size of 1 to find the precise optimal

## Experiment Design

| Config | seed_top_k | max_tokens |
|--------|------------|------------|
| ns_tune10_f1 | 23 | 81 |
| ns_tune10_f2 | 23 | 82 |
| ns_tune10_f3 | 23 | 83 |
| ns_tune10_f4 | 23 | 84 |
| ns_tune10_f5 | 23 | 86 |
| ns_tune10_f6 | 23 | 87 |
| ns_tune10_f7 | 23 | 88 |
| ns_tune10_f8 | 23 | 89 |

## Configuration

- Base: single_node_tuning_round9 (inherits round8 → round4)
- Dataset: forums
- seed_top_k: 23 (optimal from Round 7)
- max_tokens: 81-89 (test points)

## Expected Outcome

Find the precise optimal max_tokens value around 85.

## Files Modified

- `configs/experiments/single_node_tuning_round10/_base_selector_tuning_round10.yaml`
- `configs/experiments/single_node_tuning_round10/ns_tune10_f{1-8}_forums_mt*.yaml`

## Results

| Config | max_tokens | best_top1 | vs PrE-Text (0.2501) |
|--------|------------|-----------|----------------------|
| ns_tune10_f1 | 81 | - | - |
| ns_tune10_f2 | 82 | - | - |
| ns_tune10_f3 | 83 | - | - |
| ns_tune10_f4 | 84 | - | - |
| ns_tune10_f5 | 86 | - | - |
| ns_tune10_f6 | 87 | - | - |
| ns_tune10_f7 | 88 | - | - |
| ns_tune10_f8 | 89 | - | - |
| Round 9 ref | 85 | 0.2498 | -0.0003 |