# Round 7 seed_top_k 调优实验

**日期**: 2026-04-28  
**实验方向**: 通过调整 `seed_top_k` 参数探索 forums 数据集性能上限  
**实验配置**: 6 configs × 1 dataset = 6 experiments  
**服务器**: A6000 GPU, paper-2-generic 分支, pretext 虚拟环境

---

## 背景

在 Round 6 实验中发现，所有 40 个配置（c01-c10）产生完全相同的结果，原因是原 scoring formula `base_score = private_support - λ_generic × genericity_penalty` 中，penalty 的量级（~0.02-0.04）远小于 private_support 的候选间差异，导致 λ_generic 在 [0.01, 0.35] 范围内变化无法翻转候选排名。

修复后的 formula 改为：
```python
base_scores = [
    float(private_support[index])
    * (1.0 - float(lambda_generic) * float(genericity_penalty[index]))
    for index in range(count)
]
```

然而 debug 发现 forums 的 genericity_penalty 值非常小（0.02-0.04），即使新公式也难以产生显著差异。因此转而探索 `seed_top_k` 参数对 forums 性能的影响。

---

## PrE-Text 基线

| 数据集 | PrE-Text macro_f1 |
|--------|-------------------|
| jobs | 0.2718 |
| forums | 0.2501 |
| microblog | 0.2683 |
| congressional | 0.2906 |

---

## 实验设计

| 实验 ID | seed_top_k | 继承链 | 描述 |
|---------|-----------|--------|------|
| ns_tune7_s01_forums | 6 | base→_s01→leaf | 基线对比 |
| ns_tune7_s02_forums | 15 | base→_s02→leaf | 中等规模 |
| ns_tune7_s03_forums | 20 | base→_s03→leaf | 较大规模 |
| ns_tune7_s04_forums | 25 | base→_s04→leaf | 继续增大 |
| ns_tune7_s05_forums | 30 | base→_s05→leaf | 继续增大 |
| ns_tune7_s06_forums | 40 | base→_s06→leaf | 最大规模 |

**基础配置**: 继承 `single_node_tuning_round4/_base_selector_tuning_round4.yaml`，其他参数固定不变：
- λ_generic: 0.35 (使用 `_forums_lambda_generic` override)
- λ_redundancy: 0.25
- gate_low: 0.78
- gate_high: 0.90
- mid_scale: 0.45

---

## 实验结果

### 完整结果表

| seed_top_k | best_top1 | vs PrE-Text (0.2501) | 状态 |
|-----------|-----------|----------------------|------|
| 6 | 0.2476 | -0.0025 | ❌ |
| 15 | 0.2482 | -0.0019 | ❌ |
| **20** | **0.2496** | **-0.0005** | ❌ 最接近基线 |
| 25 | 0.2488 | -0.0013 | ❌ |
| 30 | 0.2473 | -0.0028 | ❌ |
| 40 | 0.2473 | -0.0028 | ❌ |

### 趋势分析

```
seed_top_k:   6    15    20    25    30    40
best_top1:  0.2476 → 0.2482 → 0.2496 → 0.2488 → 0.2473 → 0.2473
                         ↑ 峰值
```

---

## 关键发现

### 1. seed_top_k=20 是 forums 的最优值

性能曲线呈倒 U 型：
- seed_top_k=6-20：性能随参数增大而提升
- seed_top_k=20：达到峰值 0.2496
- seed_top_k>20：性能反而下降

### 2. 最优值仍未能突破 PrE-Text 基线

- forums 最优结果：0.2496 (seed_top_k=20)
- PrE-Text 基线：0.2501
- 差距：0.0005（仅 0.2%）

### 3. seed_top_k 的影响机制

较小的 seed_top_k（如 6）可能选择的候选过于集中，导致多样性不足；较大的 seed_top_k（如 30+）可能引入过多低质量候选，稀释了核心样本的效果。

---

## 结论

1. **seed_top_k=20 是 forums 的最优配置**，但仍未能突破基线
2. **forums 的性能上限可能受限于算法框架**，而非参数调优
3. **jobs/microblog/congressional 已稳定超越基线**，forums 需要其他方向改进

---

## 待探索方向

1. **候选生成策略改进**：调整 candidate_count、generated_per_round 等参数
2. **gate 参数组合**：尝试不同的 gate_low/gate_high 组合
3. **bootstrap 策略**：调整 num_prompts、max_tokens 等生成参数
4. **候选初始化**：尝试不同的 initialization 数据源

---

## 配置文件路径

```
paper-new-round5/configs/experiments/single_node_tuning_round7/
├── _base_selector_tuning_round7.yaml    # 基础配置
├── _s01.yaml ~ _s06.yaml                 # 6 个 group configs
└── ns_tune7_s{01~06}_forums.yaml        # 6 个 leaf configs
```

---

## 执行记录

- 实验完成时间: 2026-04-28
- 成功: 6/6
- 每实验平均耗时: ~5-6 分钟
- GPU: A6000 (GPU 1)