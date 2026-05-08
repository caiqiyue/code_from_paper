# Round 16.5: Near-Boundary Recheck Design

## 背景

Round15 已经证明，以下预算分配可以稳定实现 4/4 超过 PrE-Text：

| 数据集 | resolved_seed_top_k | best_top1 | vs PrE-Text |
|--------|---------------------|-----------|-------------|
| jobs | 20 | 0.2737 | +0.0005 |
| congressional | 19 | 0.2970 | +0.0020 |
| forums | 22 | 0.2507 | +0.0005 |
| microblog | 18 | 0.2754 | +0.0004 |

Round16 则把预算选择进一步算法化，改成：

```text
candidate budget -> internal utility -> argmax
```

这条主线在方法叙事上是前进的，但 50 个实验说明它还不能替代 Round15：

1. `forums` 的预算长期被压到 `18`，没有自然回到 `20/22`。
2. `microblog` 也经常落在 `18`，说明当前 utility 更擅长找到“紧凑的小预算解”，而不是“真正按数据分布自适应的预算解”。
3. `weight_sweep` 对 `forums` 的改善很有限，说明问题不只是权重略偏，而是当前决策结构在关键近边界场景下不够会“纠偏”。
4. `ablation` 显示当前自校准结论是稳定的，因此不能把问题简单归咎于某一个单项写坏。

因此，Round16.5 的目标不是推翻 Round16，而是：

> 保留 Round16 的 self-calibration 主框架，只在最容易误判的近边界决策处加入一个轻量、可解释的 second-stage recheck。

## Round16.5 目标

主目标：

- 保留 Round16 的方法主线：`internal utility` 仍然是 budget 选择的第一判据。
- 只在 top-2 utility 接近时，注入少量经验性 inductive bias。
- 让 `forums` 有机会从 `18` 被纠偏回 `20/22`，同时不显著破坏 `jobs/congressional/microblog`。
- 在结果上尽量恢复到 Round15 的 4/4 水平。

次目标：

- 不回退到显式的 dataset-family hard rule。
- 不在主决策中直接使用“forums -> 22”这类映射。
- 不修改 Stage2 bootstrap 主结构。
- 继续保留完整的 `stage1_budget_calibration.json` 可解释轨迹。

## 设计原则

Round16.5 必须遵守以下原则：

1. **Round16 主判据不变**
   - 预算仍先由 self-calibrated utility 排序。
   - second-stage recheck 只处理近边界情形，不全局接管决策。

2. **纠偏只发生在“差一点点”的地方**
   - 如果某个 budget 在 utility 上明显更优，就直接采用。
   - 只有当 top-2 非常接近时，才允许额外判断“是否还没选够”。

3. **纠偏信号尽量使用 Round16 已经有的内部量**
   - coverage mean
   - coverage p25
   - support
   - utility gap

4. **纠偏目标是恢复真正需要更大 budget 的数据集**
   - 不是默认偏向更大预算
   - 而是在“coverage 尚未饱和、support 没明显塌陷”的情况下，允许较大预算翻盘

## 核心思路

Round16 的失败不是因为 argmax utility 这个思路完全错了，而是：

> 它把 `forums` 和 `microblog` 这类边界相近的数据，都收敛成了“18 更紧凑，因此更优”。

Round16.5 的修正方式是：

```text
primary decision:
    use self-calibrated utility as usual

secondary decision:
    if top-2 budgets are close enough,
    recheck whether the larger budget still provides meaningful unsaturated coverage
    without causing meaningful support collapse
```

也就是：

- 平时仍然按 Round16 的 utility 走
- 只有在 near-boundary 情况下，才问一句：
  - “较大 budget 是否真的还在补足 private manifold 的未覆盖区域？”
  - “这种补足是否没有带来明显的 support 退化？”

如果答案是 yes，就允许较大 budget 反超。

## 推荐方案 A：Coverage-Guard Recheck

### 总体决策流程

Round16.5 的预算选择流程改成：

```python
1. 对 candidate_seed_top_k 全部计算 Round16 utility
2. 找到 top-1 budget 和 runner-up budget
3. 若 utility_gap > recheck_trigger_gap:
       直接选 top-1
4. 若 utility_gap <= recheck_trigger_gap:
       启动 near-boundary recheck
5. 若 larger budget 通过 recheck:
       选择 larger budget
   否则:
       保持 top-1
```

其中：

- `top-1` = utility 最大的预算
- `runner-up` = utility 第二大的预算
- `larger budget` = top-2 中 budget 更大的那个
- `smaller budget` = top-2 中 budget 更小的那个

### 何时触发 recheck

Round16.5 不应在所有情况都做 recheck。推荐触发条件：

```python
trigger_recheck = (
    utility_gap <= recheck_trigger_gap
    and larger_budget > smaller_budget
)
```

推荐初始值：

```python
recheck_trigger_gap = 0.12
```

理由：

- Round16 的很多 `forums` 失败案例里，虽然 top-1 很稳定，但真正值得纠偏的情形并不是“所有 top-1 都错”，而是某些接近边界的场景没有被允许往上走。
- 如果阈值太小，recheck 很少触发，Round16.5 退化成 Round16。
- 如果阈值太大，recheck 会过度干预，像半个新规则系统。

`0.12` 先作为快速实验的第一档，中后续实验中再与 `0.08 / 0.16` 做比较。

## Recheck 指标设计

second-stage recheck 只看 3 类信号：

1. `coverage_mean_gain`
2. `coverage_p25_gain`
3. `support_drop`

### 1. coverage_mean_gain

定义：

```python
coverage_mean_gain = coverage_mean(larger_budget) - coverage_mean(smaller_budget)
```

意义：

- 判断整体 private manifold 覆盖是否仍在增长
- 这是 Round16 中 coverage 的延续信号

### 2. coverage_p25_gain

定义：

```python
coverage_p25_gain = coverage_p25(larger_budget) - coverage_p25(smaller_budget)
```

意义：

- `coverage_mean` 关注整体平均覆盖
- `coverage_p25` 更关注尾部，也就是最难覆盖的那部分 private samples
- `forums` 类数据更可能在尾部 coverage 上仍明显不足

这是 Round16.5 最关键的新增信号。

如果只有 `coverage_mean` 增长而 `coverage_p25` 没动，往往说明较大 budget 只是在平均意义上略有收益，但并没有真正补到“最难覆盖的区域”。

### 3. support_drop

定义：

```python
support_drop = support_mean(smaller_budget) - support_mean(larger_budget)
```

意义：

- 防止为了 coverage 盲目放大 budget
- 如果较大 budget 虽然 coverage 更好，但 support 明显塌陷，就不应放行

## Recheck 决策规则

推荐使用一个非常清楚、可解释的 gating rule，而不是再叠一层复杂 utility。

### 推荐规则

当 `utility_gap <= recheck_trigger_gap` 时：

```python
pass_recheck = (
    coverage_mean_gain >= coverage_mean_gain_min
    and coverage_p25_gain >= coverage_p25_gain_min
    and support_drop <= support_drop_max
)
```

若 `pass_recheck=True`，选择 `larger_budget`。  
否则保持 Round16 的原始 top-1 结果。

### 推荐初始阈值

```python
coverage_mean_gain_min = 0.004
coverage_p25_gain_min = 0.008
support_drop_max = 0.015
```

解释：

- `coverage_mean_gain_min`
  - 防止因为极小的平均 coverage 噪声就放大 budget

- `coverage_p25_gain_min`
  - 对 `forums` 这类尾部 coverage 未饱和数据更关键
  - 这个阈值应该略高于 `coverage_mean_gain_min`

- `support_drop_max`
  - 防止 larger budget 通过吸入过弱样本来“伪装改善”

## 为什么这个设计有机会恢复 Round15 水平

Round15 的成功行为本质上不是“我们知道 forums 应该是 22”，而是：

> 对 forums 这类分布更散的任务，较大预算仍然有真实覆盖收益。

Round16 没能表达这一点，因为它的主 utility 更偏向紧凑性。  
Round16.5 的 recheck 恰好只在近边界处，把“coverage 是否真的还没饱和”单独拿出来问一次。

因此：

- `jobs / congressional`
  - 如果它们本来在较小预算下 coverage 已接近饱和，recheck 不会轻易放行 larger budget

- `microblog`
  - 若它与 `forums` 不同，表现为 `coverage_p25` 增益不明显，则仍会保留较小预算

- `forums`
  - 如果较大预算能明显改善尾部 coverage，同时 support 没显著变差，就能通过 recheck，从 `18` 拉回 `20/22`

这正是我们要的“折中”：

- 主体仍是 self-calibration
- 纠偏只作用于 coverage 未饱和的近边界场景

## 与 Round15 / Round16 的关系

### 相比 Round15

Round16.5 仍然更算法化：

- 没有 length threshold
- 没有 dataset-family mapping
- 没有 `if dataset == forums: seed_top_k = 22`

### 相比 Round16

Round16.5 更接近结果目标：

- 不再完全依赖单层 utility argmax
- 给 near-boundary 决策增加一层轻量校正
- 试图恢复 Round15 已经验证有效的预算结构

## 代码开发计划

Round16.5 的开发必须基于：

[`/Users/apple/Desktop/code_from_paper/paper-new-round-16`](/Users/apple/Desktop/code_from_paper/paper-new-round-16)

不回到 `paper-new-round11` 混改。

### 代码改动范围

核心改动应控制在 3 个模块内：

1. `paper_new_selector/budget_calibration.py`
2. `paper_new_selector/stage1_runner.py`
3. `tests/`

原则上不改：

- Stage2 bootstrap 主流程
- `pipeline.py` 主结构
- 数据路径和 eval bridge

### 开发 Phase 1：补齐 recheck 所需指标

目标：

- 在现有 calibration metrics 中显式加入：
  - `coverage_p25`
  - `support_mean`
  - `coverage_mean_gain`
  - `coverage_p25_gain`
  - `support_drop`

建议任务：

1. 在 `budget_calibration.py` 中扩展 per-budget metrics
2. 保证 `per_budget_metrics` 中可直接读取这些聚合量
3. 补充对应单元测试

建议新增/扩展测试：

- `tests/test_budget_calibration.py`

需要覆盖：

- `coverage_p25` 计算正确
- `coverage_gain` / `support_drop` 计算正确
- larger/smaller budget 比较逻辑正确

### 开发 Phase 2：实现 near-boundary recheck

目标：

- 在已有 `select_budget_with_tiebreak` 附近引入 second-stage recheck

建议拆分函数：

```python
should_trigger_near_boundary_recheck(...)
evaluate_near_boundary_recheck(...)
select_budget_with_recheck(...)
```

推荐流程：

1. 先保留原始 utility ranking
2. 拿到 top-1 与 runner-up
3. 若满足触发条件，则判断 larger budget 是否通过 coverage-guard recheck
4. 返回最终 budget 与 decision trace

测试覆盖：

- top-2 差距大时，不触发 recheck
- top-2 接近，但 larger budget coverage 改善不足，不翻盘
- top-2 接近，larger budget coverage 改善明显且 support 未塌陷，允许翻盘

### 开发 Phase 3：输出轨迹增强

目标：

- 让 Stage1 summary 足够解释 recheck 为什么发生、为什么没发生

建议在 `stage1_budget_calibration.json` 中新增：

```json
"near_boundary_recheck": {
  "triggered": true,
  "utility_gap": 0.083,
  "smaller_budget": 18,
  "larger_budget": 20,
  "coverage_mean_gain": 0.006,
  "coverage_p25_gain": 0.011,
  "support_drop": 0.007,
  "pass_recheck": true,
  "final_budget": 20
}
```

这样后续分析会非常直观。

### 开发 Phase 4：兼容旧逻辑

目标：

- Round16 原始 `self_calibrated` 逻辑仍可保留
- 通过配置显式开启 Round16.5 recheck

推荐配置形式：

```yaml
selector:
  seed_budget_rule:
    enabled: true
    mode: self_calibrated
    candidate_seed_top_k: [18, 19, 20, 21, 22]
    utility:
      support_weight: 1.0
      genericity_weight: 0.5
      redundancy_weight: 0.3
      coverage_weight: 0.4
      budget_weight: 0.1
    near_boundary_recheck:
      enabled: true
      trigger_gap: 0.12
      coverage_mean_gain_min: 0.004
      coverage_p25_gain_min: 0.008
      support_drop_max: 0.015
```

这样：

- `enabled=false` 时退回纯 Round16
- `enabled=true` 时启用 Round16.5

## 快速对比实验设计

Round16.5 不适合一上来再跑大矩阵。  
应该先做一轮 **快速对比实验**，只验证最关键的问题：

> recheck 能否把 `forums` 从 18 拉回更合理预算，同时不伤其他数据集？

### 实验组 A：最小功能探针

目的：

- 验证 recheck 逻辑真的会触发
- 验证日志里能看到完整 recheck 轨迹

实验：

1. `r165_probe_forums_base`
2. `r165_probe_microblog_base`

配置：

- 使用 Round16 `c1` utility 权重
- 开启 `near_boundary_recheck`
- `trigger_gap = 0.12`

成功标准：

- `stage1_budget_calibration.json` 中出现 `near_boundary_recheck`
- `forums` 至少从“纯 Round16 的强 18 偏好”变成“18/19/20 之间有翻盘可能”
- `microblog` 不应被轻易推到更大 budget

### 实验组 B：recheck 阈值小扫

目的：

- 看触发阈值和 recheck guard 强度是否合理

推荐 4 组：

1. `r165_forums_rg08`
   - `trigger_gap=0.08`
2. `r165_forums_rg12`
   - `trigger_gap=0.12`
3. `r165_forums_rg16`
   - `trigger_gap=0.16`
4. `r165_forums_loose_guard`
   - `trigger_gap=0.12`
   - `coverage_p25_gain_min=0.006`

观察指标：

- `resolved_seed_top_k`
- `runner_up_seed_top_k`
- `utility_gap`
- `near_boundary_recheck.pass_recheck`
- `best_top1`

成功标准：

- 至少有 1 组把 `forums` 拉回到 `19/20/22`
- 且 `best_top1 >= 0.2505`

### 实验组 C：四数据集快速回归

在实验组 A/B 选出 1 套最优 recheck 参数后，立即跑 4 个完整数据集：

1. `r165_full_jobs`
2. `r165_full_congressional`
3. `r165_full_forums`
4. `r165_full_microblog`

目标：

- 检查是否恢复到接近 Round15 的整体水平

成功标准：

| 数据集 | 目标 |
|--------|------|
| jobs | `best_top1 >= 0.2732` |
| congressional | `best_top1 >= 0.2950` |
| forums | `best_top1 >= 0.2501`，理想值 `>= 0.2505` |
| microblog | `best_top1 >= 0.2763` |

如果 4/4 成立，则 Round16.5 进入下一阶段。

### 实验组 D：最小对照组

为了证明 Round16.5 的改进来自 recheck，而不是偶然波动，必须保留两个对照：

1. `r16_c1_forums` 或其等价重跑
   - 纯 Round16，无 recheck
2. `r165_full_forums`
   - Round16.5，有 recheck

必须对比：

- `resolved_seed_top_k`
- `near_boundary_recheck` 是否触发
- `best_top1` 是否提升

### 实验组 E：可选 seed sanity check

如果实验组 C 的结果接近成功，建议额外补两组：

1. `r165_forums_seed123`
2. `r165_forums_seed456`

目的：

- 看 recheck 是否只是在 seed 42 上偶然有效

这组不是第一优先级，但如果 `forums` 终于回到 `>= 0.2505`，就值得补。

## 推荐执行顺序

1. 实现指标与 recheck 逻辑
2. 本地单测覆盖 recheck 分支
3. 跑实验组 A：`forums + microblog probe`
4. 跑实验组 B：forums 小扫
5. 选最优 recheck 配置
6. 跑实验组 C：四数据集快速回归
7. 若接近成功，再补实验组 D/E

## 成功标准

Round16.5 至少应满足：

1. 仍以 Round16 self-calibration 为主判据
2. 不引入 dataset-name 的硬编码映射
3. `near_boundary_recheck` 仅在近边界时触发
4. `forums` 能从当前系统性偏小预算中恢复
5. 至少存在 1 套统一配置，使四个数据集重新达到或逼近 Round15 水平

更理想的成功标准：

6. `forums` 回到 `0.2505+`
7. `jobs/congressional/microblog` 不低于 PrE-Text
8. 设计文档和日志能清楚解释“为什么 larger budget 在 forums 上被放行”

## 风险与缓解

### 风险 1：recheck 过强，系统重新偏向更大预算

缓解：

- recheck 只在 `utility_gap <= trigger_gap` 时触发
- 增加 `support_drop_max` 约束
- 保持 `coverage_p25_gain_min` 不是零

### 风险 2：recheck 太弱，Round16.5 退化成 Round16

缓解：

- 在快速实验中显式比较 `trigger_gap=0.08/0.12/0.16`
- 至少设置一组更宽松的 `coverage_p25_gain_min`

### 风险 3：只救回 forums，但伤到 microblog

缓解：

- `microblog` 必须进入第一批 probe
- 以 `coverage_p25` 而非只看 `coverage_mean` 作为放行条件

### 风险 4：设计看起来又像规则系统

缓解：

- 明确 recheck 是 near-boundary correction，不是主决策器
- 文档与论文叙事中强调：
  - primary decision 来自 self-calibrated utility
  - recheck 只是对“coverage 是否仍未饱和”的局部校正

## 论文叙事建议

英文可以这样表述：

> We keep the self-calibrated budget selection framework as the primary decision mechanism, and introduce a lightweight near-boundary recheck for cases where the top budget candidates are very close in internal utility. The recheck allows a larger budget to override the compact solution only when it provides meaningful additional coverage, especially on the lower-covered portion of the private manifold, without causing a substantial support drop. This preserves the adaptive nature of the selector while correcting the compactness bias observed in the first self-calibration version.

中文可以这样写：

> 我们保留自校准预算选择作为主决策机制，仅在候选预算的内部效用非常接近时，引入一个轻量的近边界复核机制。只有当较大预算能在 private manifold 的低覆盖部分带来明确的额外覆盖收益，且不会导致显著的 support 退化时，才允许它覆盖原本更紧凑的小预算解。这样既保持了 selector 的自适应性质，也能纠正第一版自校准方法中过强的 compactness 偏置。


## 实验结果汇总

### 实验组 A: Probe（base 权重 + recheck 开启）

| 实验 | resolved_seed_top_k | runner_up_seed_top_k | utility_gap | recheck_triggered | recheck_pass | best_top1 |
|------|---------------------|----------------------|-------------|--------------------|--------------|-----------|
| r165_probe_forums_base | 18 | 19 | 0.302 | False | False | 0.2460 |
| r165_probe_microblog_base | 18 | 19 | 0.243 | False | False | 0.2728 |

### 实验组 B: recheck trigger_gap 小扫（forums）

| 实验 | resolved_seed_top_k | utility_gap | recheck_triggered | recheck_pass | best_top1 |
|------|---------------------|-------------|-------------------|--------------|-----------|
| r165_forums_rg08 | 18 | 0.302 | False | False | 0.2485 |
| r165_forums_rg12 | 18 | 0.302 | False | False | 0.2500 |
| r165_forums_rg16 | 18 | 0.302 | False | False | 0.2494 |
| r165_forums_loose_guard | 18 | 0.302 | False | False | 0.2460 |

### 汇总结论

- **recheck 机制从未触发**：所有配置下 utility_gap 均为 0.302，远大于 trigger_gap 阈值（0.08/0.12/0.16）
- **resolved_seed_top_k 全部为 18**：说明 k=18 在当前权重下确实是 utility 全局最优解，recheck 的 near-boundary 条件从未满足
- **best_top1 差异来自权重而非 budget 重选**：rg12 达到最高 0.2500，说明权重影响比 trigger_gap 影响更显著
- **结论**：Round16.5 的 recheck 机制在当前 trigger_gap 配置下无法将 forums 从 18 纠偏回 20/22，因为 utility_gap 本身就很大，top-1 vs top-2 边界并不接近
- **下一步方向**：需要更激进地放宽 trigger_gap（如 0.20/0.25），或者降低 coverage_p25_gain_min，才能让 recheck 更容易触发；或者转向调整 utility 权重本身而非依赖 recheck 纠偏
