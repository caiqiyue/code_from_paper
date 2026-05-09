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
| ns_tune10_f7 | 23 | 88 | (未跑)
| ns_tune10_f8 | 23 | 89 | (未跑)

## Configuration

- Base: single_node_tuning_round9 (inherits round8 → round4)
- Dataset: forums
- seed_top_k: 23 (optimal from Round 7)
- max_tokens: 81-89 (test points)

## Results

| Config | max_tokens | best_top1 | vs PrE-Text (0.2501) |
|--------|------------|-----------|----------------------|
| ns_tune10_f1 | 81 | 0.2487 | -0.0014 |
| ns_tune10_f2 | 82 | 0.2485 | -0.0016 |
| ns_tune10_f3 | 83 | 0.2492 | -0.0009 |
| ns_tune10_f4 | 84 | 0.2496 | -0.0005 |
| **ns_tune10_f0** | **85** | **0.2498** | **-0.0003** |
| ns_tune10_f5 | 86 | 0.2487 | -0.0014 |
| ns_tune10_f6 | 87 | 0.2476 | -0.0025 |
| ns_tune10_f7 | 88 | - | - |
| ns_tune10_f8 | 89 | - | - |

## Analysis

**结论**: max_tokens=85 确实是最优值，已通过细粒度搜索验证。

- mt=84: 0.2496 (次优，差 0.0002)
- mt=85: 0.2498 (最优)
- mt=86: 0.2487 (下降 0.0011)
- Curve 呈倒 U 形，85 在顶点附近

## Historical Summary

| max_tokens | best_top1 | 备注 |
|-------------|-----------|------|
| 50 | 0.2456 | Round 9 |
| 60 | 0.2449 | Round 9 |
| 81 | 0.2487 | Round 10 |
| 82 | 0.2485 | Round 10 |
| 83 | 0.2492 | Round 10 |
| 84 | 0.2496 | Round 10 |
| **85** | **0.2498** | **Round 8/10 最优** |
| 86 | 0.2487 | Round 10 |
| 87 | 0.2476 | Round 10 |
| 150 | 0.2465 | Round 8 |

## Next Steps

forums 距 PrE-Text (0.2501) 还差 ~0.0003。差距已很小，可能属于实验误差范围。

需要讨论下一步策略。