# Round 9: Forums Smaller Max Tokens Experiment

## Hypothesis

From Round 8: **Shorter synthesis text is better for forums**
- max_tokens=85: best_top1=0.2498
- max_tokens=150: best_top1=0.2465

Further reduce max_tokens to 50-60 to test if even shorter synthesis helps.

## Experiment Design

| Config | seed_top_k | max_tokens | Expected Synth Length |
|--------|------------|------------|----------------------|
| ns_tune9_f1_forums_mt50 | 23 | 50 | ~30-35 words |
| ns_tune9_f2_forums_mt60 | 23 | 60 | ~35-40 words |

## Configuration

- Base: single_node_tuning_round8
- Dataset: forums
- seed_top_k: 23 (optimal from Round 7)
- max_tokens: 50, 60 (test points)

## Expected Outcome

If shorter = better holds, we should see 0.2498 → potentially higher with mt=50/60.

## Files Modified

- `configs/experiments/single_node_tuning_round9/_base_selector_tuning_round9.yaml`
- `configs/experiments/single_node_tuning_round9/ns_tune9_f1_forums_mt50.yaml`
- `configs/experiments/single_node_tuning_round9/ns_tune9_f2_forums_mt60.yaml`