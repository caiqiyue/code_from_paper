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

---

## Results (2026-04-28)

| Config | max_tokens | best_top1 | vs PrE-Text (0.2501) |
|--------|------------|-----------|----------------------|
| ns_tune9_f1 | **50** | 0.2456 | -0.0045 |
| ns_tune9_f2 | **60** | 0.2449 | -0.0052 |
| Round 8 (ref) | 85 | 0.2498 | -0.0003 |
| Round 8 (ref) | 150 | 0.2465 | -0.0036 |

## Analysis

**结论**: max_tokens=85 仍是最优值，"越短越好"假设被否定。

- mt=50: 明显更差 (0.2456 vs 0.2498)
- mt=60: 明显更差 (0.2449 vs 0.2498)
- mt=85: 最佳 (0.2498)
- mt=150: 较差 (0.2465)

**Curve is concave**: 存在一个最优中间值 (约85)，过短或过长都会降低性能。

## Next Steps

1. **接受现状**: forums 差距仅 0.0003，可能属于实验误差范围
2. **尝试 seed_top_k=6**: 之前 congressional 最优值，可能对 forums 也有帮助
3. **调整 genericity gate 参数**: 针对 forums 非结构化特点