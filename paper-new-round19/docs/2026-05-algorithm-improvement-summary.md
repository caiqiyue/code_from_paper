# Round15 之后算法改善总结：从硬编码到自适应预算选择

## 1. 背景：Round14 发现的规律

Round14 通过 dataset-family seed budget rule 首次实现四个数据集全部超过 PrE-Text screening 基准：

| 数据集 | seed_top_k | best_top1 | PrE-Text | vs PrE-Text |
|--------|------------|-----------|----------|-------------|
| jobs | 20 | 0.2786 | 0.2732 | +0.0054 |
| congressional | 19 | 0.2955 | 0.2950 | +0.0005 |
| forums | 22 | 0.2507 | 0.2501 | +0.0005 |
| microblog | 18 | 0.2767 | 0.2763 | +0.0004 |

这个结果本身是有效的，但存在一个根本性问题：**19/20/22/18 这个分配是经验性的硬编码**——我们不知道为什么 forums 需要 22、microblog 需要 18，只知道这样配置能过线。

## 2. Round15：算法化尝试与硬编码残余

Round15 尝试把配置级的 family rule 升级为算法级的 adaptive seed budget rule，引入了基于 private text 长度统计的 threshold rule：

```
private length stats -> threshold rule -> resolved seed_top_k
```

结果：Round15 也实现了 4/4 超过 PrE-Text，但方法论上仍然依赖手工阈值，不是真正的自适应。

**核心问题**：
1. `19/20/22/18` 的分配像 empirical rule，不像 selector 的自然自适应行为
2. 如果换数据切片或新数据集，阈值规则可能需要人工修补
3. 论文叙事上无法解释"为什么 forums 应该是 22 而不是 20"

## 3. Round16：Self-Calibration 主框架

### 3.1 核心思路

Round16 不再依据 private text 长度统计决定 budget，而是让 selector 在同一批候选样本上，对多个候选 budget 进行轻量自校准：

```
candidate budget -> internal utility -> argmax
```

utility 函数为：
```
utility(k) = w_support * support_score
           - w_genericity * genericity_score
           - w_redundancy * redundancy_score
           + w_coverage * coverage_score
           - w_budget * budget_cost
```

### 3.2 实验结果（50 个实验）

| 数据集 | c1 resolved | c1 best_top1 | 结论 |
|--------|-------------|--------------|------|
| forums | **18** | 0.2501 | 偏保守，被压到 18 |
| congressional | 19 | 0.2958 | 正常 |
| jobs | 20 | 0.2784 | 正常 |
| microblog | **18** | 0.2772 | 偏保守，被压到 18 |

**关键发现**：utility_gap 很大（forums=0.302, microblog=0.243），说明 k=18 确实是当前权重下的全局 utility 最优解，不是边界误判。

**失败原因**：主 utility 偏向紧凑性，coverage 只是其中一项奖励，容易被 support/genericity/redundancy 压掉。forums 需要更大 budget 的本质不是"utility 最大"，而是"private manifold 更散、尾部覆盖更不足"——这个信号在单层 utility 中无法优先表达。

### 3.3 汇总结论

- forums: c1-c5 在 0.246-0.250 之间，c5 最高（0.2500）
- congressional: c1-c5 在 0.293-0.296 之间，r16_full_c1 最高（0.2958）
- jobs（seed=123）: r16_full_c2 最高（0.2808）
- microblog（seed=123）: r16_full_c1 最高（0.2772）
- seed robustness：forums 在 seed 42/123/456 下 best_top1 稳定在 0.247-0.249
- Ablation：no_budget_cost 对 forums 无显著影响，说明自校准结论稳健

## 4. Round16.5：Near-Boundary Recheck（未成功）

### 4.1 设计思路

保留 Round16 self-calibration 主框架，在 top-2 utility 接近时注入 second-stage recheck：

```
primary: self-calibrated utility argmax
secondary: if utility_gap <= trigger_gap, recheck coverage_p25_gain and support_drop
```

### 4.2 实验结果

| 实验 | trigger_gap | recheck_triggered | resolved | best_top1 |
|------|------------|-------------------|----------|-----------|
| r165_forums_rg08 | 0.08 | False | 18 | 0.2485 |
| r165_forums_rg12 | 0.12 | False | 18 | **0.2500** |
| r165_forums_rg16 | 0.16 | False | 18 | 0.2494 |
| r165_forums_loose_guard | 0.12 | False | 18 | 0.2460 |

### 4.3 失败原因

recheck 机制从未触发——utility_gap=0.302 远大于 trigger_gap 阈值，说明问题不是"近边界误判"，而是**主目标函数本身就系统性地偏向小预算**。recheck 只在边界情形有效，但 forums 的 utility_gap 并不小。

## 5. Round17：Constrained Utility（部分成功）

### 5.1 核心思路

将 budget 选择重构为 constrained optimization：

```
Step 1: 计算 per-budget coverage_p25
Step 2: 定义 relative coverage sufficiency：feasible(k) = coverage_p25(k) >= tau * best_coverage_p25
Step 3: 只在 feasible budgets 上优化 compactness-aware utility
```

关键升级：**coverage 从"utility 中的一项"升级为"可行性条件"**。

### 5.2 实验结果

#### forums ratio sweep

| 实验 | ratio | feasible_budgets | resolved | best_top1 |
|------|-------|-----------------|----------|-----------|
| r17_forums_r099 | 0.99 | 21,22 | 21 | 0.2462 |
| r17_forums_r098 | 0.98 | 21,22 | 21 | **0.2505** ✅ |
| r17_forums_r097 | 0.97 | 21,22 | 21 | 0.2492 |

#### 四数据集统一回归（ratio=0.98）

| 数据集 | feasible_budgets | resolved | best_top1 | PrE-Text | 差值 |
|--------|-----------------|----------|-----------|----------|------|
| forums | 21,22 | 21 | **0.2505** | 0.2501 | +0.0004 ✅ |
| microblog | 22 | 22 | **0.2770** | 0.2763 | +0.0007 ✅ |
| jobs | 18-22 | 18 | **0.2782** | 0.2732 | +0.0050 ✅ |
| congressional | 18-22 | 18 | 0.2928 | 0.2950 | **-0.0022** ❌ |

### 5.3 关键机制证据

**forums**：`coverage_p25(18)=0.1309, coverage_p25(21)=0.1359`，18/19/20 全部被 feasibility 直接排除。Round17 的价值首先在于把 coverage 不足的小 budget 从候选集中剔除，不在于"偏向大 budget"。

**congressional**：`coverage_p25(18)=coverage_p25(19)=...=coverage_p25(22)=0.1699`，所有 budget 的 coverage_p25 完全相同，feasibility 失效，退回到 utility 偏好选 18。

### 5.4 结论

Round17 **部分成功**：成功修复了 forums 的预算压缩问题（3/4 超过 PrE-Text），但在 congressional 上失败——单一 coverage_p25 约束对不同 budget 缺乏区分力时，算法退化为 compactness 偏好。

## 6. Round18：Tail-Coverage Family（进行中）

### 6.1 设计升级

Round18 不再扫 ratio，而是升级约束信号本身：

```
Step 1: Tail-Coverage Family Feasibility
  feasible(k) = coverage_p25(k) >= tau_p25 * best_coverage_p25
             AND coverage_mean(k) >= tau_mean * best_coverage_mean

Step 2: Family Score Trace（记录用，不硬决策）

Step 3: Feasible-Set Utility（沿用 Round17）

Step 4: Constrained Recheck（保守后处理）
  如果选了较小 budget，但仍有 larger feasible budget，
  检查 larger budget 是否在 coverage_mean_gain / coverage_p25_gain / coverage_min_gain / support_drop 上通过保守 guard
```

### 6.2 目标

- 保住 forums（继续在 21 附近）
- 修复 congressional（通过 dual-metric feasibility 或 constrained recheck）
- 不破坏 jobs/microblog 的现有优势

### 6.3 预期行为差异

| 数据集 | Round17 问题 | Round18 修复思路 |
|--------|------------|----------------|
| forums | 基本正常 | 保持稳定 |
| congressional | coverage_p25 完全平坦，feasibility 失效 | dual-metric feasibility + constrained recheck |
| jobs | 已经是强最优 18 | 保持稳定 |
| microblog | 需要严格约束才能稳定在 22 | 保持稳定 |

## 7. 方法论演进总结

### 7.1 各 Round 决策结构对比

| Round | 决策方式 | 是否依赖硬编码 |
|-------|---------|-------------|
| Round14 | dataset-family -> threshold -> budget | 是（完全硬编码） |
| Round15 | private length stats -> threshold rule | 是（长度阈值） |
| Round16 | 单层 utility argmax | 否，但系统性偏向小预算 |
| Round16.5 | utility argmax + near-boundary recheck | 否，recheck 未触发 |
| Round17 | feasibility stage + feasible-set utility | 否（部分成功） |
| Round18 | dual-metric feasibility + family trace + constrained recheck | 仍在探索 |

### 7.2 核心算法演进

**从单层 utility 到两层 constrained optimization**：

```
Round16:   argmax_k utility(k)                    # coverage 只是奖励项
Round17:   feasible(k) by coverage_p25, then argmax # coverage 是硬约束
Round18:   feasible(k) by dual-metric, then argmax # 升级为 multi-metric 约束
           + constrained recheck                  # 增加保守纠偏层
```

### 7.3 剩余问题

1. **congressional 的 coverage_p25 在不同 budget 上完全平坦**：单一尾部指标无法区分 budget，需要更强的约束信号
2. **internal utility 和 downstream best_top1 不完全一致**：utility 选出的 budget 评估不一定最优
3. **没有找到统一的 ratio 参数能同时满足 4/4**：ratio=0.98 对 forums 有效但对 congressional 过松

### 7.4 论文叙事

最终的方法论表述应当是：

> We formulate seed-budget selection as a constrained internal model-selection problem. Instead of directly maximizing a scalar utility over all candidate budgets, we first define a feasible set using a relative tail-coverage sufficiency condition. The selector then optimizes a compactness-aware utility only over the feasible budgets. For datasets where the primary tail-coverage metric lacks discriminative power across budgets, a conservative constrained recheck provides a secondary safeguard against overly compact selections.

## 8. 时间线

```
2026-04-28  Round16 design (self-calibration)
2026-04-28  Round16 实验完成（50 个实验，3/4 成功）
2026-04-29  Round16.5 design + 实验（recheck 未触发）
2026-04-29  Round17 design (constrained utility)
2026-04-29  Round17 实现计划
2026-04-30  Round17 ratio sweep 完成（3/4 成功，congressional 失败）
2026-04-30  Round18 design（tail-coverage family）
```
