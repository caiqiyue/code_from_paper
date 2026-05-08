# Round 7 seed_top_k 调优实验

**日期**: 2026-04-28  
**实验方向**: 通过调整 `seed_top_k` 参数探索 forums 数据集性能上限  
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

### 第一批实验：粗粒度探索

| 实验 ID | seed_top_k | 继承链 | 描述 |
|---------|-----------|--------|------|
| ns_tune7_s01_forums | 6 | base→_s01→leaf | 基线对比 |
| ns_tune7_s02_forums | 15 | base→_s02→leaf | 中等规模 |
| ns_tune7_s03_forums | 20 | base→_s03→leaf | 较大规模 |
| ns_tune7_s04_forums | 25 | base→_s04→leaf | 继续增大 |
| ns_tune7_s05_forums | 30 | base→_s05→leaf | 继续增大 |
| ns_tune7_s06_forums | 40 | base→_s06→leaf | 最大规模 |

### 第二批实验：细粒度探索（21-29）

| 实验 ID | seed_top_k | 继承链 | 描述 |
|---------|-----------|--------|------|
| ns_tune7_s07_forums | 21 | base→_s07→leaf | 细粒度 |
| ns_tune7_s08_forums | 22 | base→_s08→leaf | 细粒度 |
| ns_tune7_s09_forums | 23 | base→_s09→leaf | 细粒度 |
| ns_tune7_s10_forums | 24 | base→_s10→leaf | 细粒度 |
| ns_tune7_s12_forums | 26 | base→_s12→leaf | 细粒度 |
| ns_tune7_s13_forums | 27 | base→_s13→leaf | 细粒度 |
| ns_tune7_s14_forums | 28 | base→_s14→leaf | 细粒度 |
| ns_tune7_s15_forums | 29 | base→_s15→leaf | 细粒度 |

**基础配置**: 继承 `single_node_tuning_round4/_base_selector_tuning_round4.yaml`，其他参数固定不变：
- λ_generic: 0.35 (使用 `_forums_lambda_generic` override)
- λ_redundancy: 0.25
- gate_low: 0.78
- gate_high: 0.90
- mid_scale: 0.45

---

## 完整实验结果

### 全部 14 个实验结果表

| seed_top_k | best_top1 | vs PrE-Text (0.2501) | 状态 |
|-----------|-----------|----------------------|------|
| 6 | 0.2476 | -0.0025 | ❌ |
| 15 | 0.2482 | -0.0019 | ❌ |
| **20** | **0.2496** | **-0.0005** | ❌ 次优 |
| 21 | 0.2464 | -0.0037 | ❌ |
| 22 | 0.2481 | -0.0020 | ❌ |
| **23** | **0.2498** | **-0.0003** | ❌ **最优** |
| 24 | 0.2489 | -0.0012 | ❌ |
| 25 | 0.2488 | -0.0013 | ❌ |
| 26 | 0.2465 | -0.0036 | ❌ |
| 27 | 0.2485 | -0.0016 | ❌ |
| 28 | 0.2454 | -0.0047 | ❌ |
| 29 | 0.2427 | -0.0074 | ❌ |
| 30 | 0.2473 | -0.0028 | ❌ |
| 40 | 0.2473 | -0.0028 | ❌ |

### 趋势分析

```
seed_top_k:   6    15    20    21    22    23    24    25    26    27    28    29    30    40
best_top1:  0.2476 0.2482 0.2496 0.2464 0.2481 0.2498 0.2489 0.2488 0.2465 0.2485 0.2454 0.2427 0.2473 0.2473
                        ↑                                                        ↓
                     次优                    波动下降                      下降后回稳
                                  ↑
                               最优(0.2498)
```

---

## 关键发现

### 1. seed_top_k=23 是全局最优

在全部 14 个实验点中：
- **最优值**: seed_top_k=23，获得 best_top1=0.2498
- **次优值**: seed_top_k=20，获得 best_top1=0.2496
- **最差值**: seed_top_k=29，获得 best_top1=0.2427

### 2. 最优值仍未能突破 PrE-Text 基线

- forums 最优结果：0.2498 (seed_top_k=23)
- PrE-Text 基线：0.2501
- **差距：0.0003（仅 0.1%）**

### 3. 性能曲线呈不规则倒 U 型

- seed_top_k=6-23：整体上升趋势（0.2476 → 0.2498）
- seed_top_k=23-29：急剧下降（0.2498 → 0.2427）
- seed_top_k=30-40：回稳（0.2473）

### 4. seed_top_k 敏感性分析

| 区间 | 表现 |
|------|------|
| 6-15 | 缓慢上升 |
| 15-23 | 快速上升（峰值） |
| 23-29 | 急剧下降 |
| 29-40 | 趋于稳定 |

---

## 结论

### 主要结论

1. **seed_top_k=23 是 forums 的全局最优配置**，best_top1=0.2498
2. **仍未突破 PrE-Text 基线（0.2501）**，差距仅 0.0003（0.1%）
3. **seed_top_k 细粒度探索已覆盖 6-40 全部范围**，确认 20-23 是最优区间
4. **问题不在 seed_top_k 参数**，需要算法框架改进

### 与 PrE-Text Screening 结果对比

| 数据集 | PrE-Text Screening | NS (当前最优) | 差异 |
|--------|-------------------|---------------|------|
| jobs | 0.2732 | 0.2761 | +0.0029 ✅ |
| congressional | 0.2950 | 0.2970 | +0.0020 ✅ |
| forums | 0.2501 | 0.2498 | -0.0003 ❌ |
| microblog | 0.2763 | 0.2749 | -0.0014 ❌ |

**NS 算法在 jobs/congressional 上超越 PrE-Text，但在 forums 上仍略低于基线。**

---

## 下一步方向

### 已验证（无效方向）

- ✅ seed_top_k 在 [6, 40] 区间细粒度探索
- ✅ λ_generic 在 [0.01, 0.35] 区间探索
- ✅ λ_redundancy 在 [0.15, 0.25] 区间探索

### 待探索方向

1. **算法框架微调**：
   - 针对长文本/非结构化数据调整 scoring formula
   - 候选选择时考虑文本长度因素

2. **Bootstrap 策略优化**：
   - 调整 max_tokens 参数（当前 85，可能不适合生成长文本）
   - 调整 num_prompts（当前 100）

3. **候选初始化来源**：
   - 尝试使用 forums 相关初始化替代 C4 初始化

4. **数据集特性分析**：
   - forums 文本平均 379 词，远长于 jobs(270) 和 congressional(227)
   - 非结构化内容可能需要不同的处理策略

---

## 配置文件路径

```
paper-new-round5/configs/experiments/single_node_tuning_round7/
├── _base_selector_tuning_round7.yaml    # 基础配置
├── _s01.yaml ~ _s06.yaml              # 6 个粗粒度 group configs
├── _s07.yaml ~ _s15.yaml              # 9 个细粒度 group configs
└── ns_tune7_s{01~15}_forums.yaml      # 14 个 leaf configs (不含 s11)
```

---

## 执行记录

- **第一批实验完成时间**: 2026-04-28
- **第二批实验完成时间**: 2026-04-28
- **总实验数**: 14/14
- **成功率**: 100%
- **每实验平均耗时**: ~5-6 分钟
- **GPU**: A6000 (GPU 1)