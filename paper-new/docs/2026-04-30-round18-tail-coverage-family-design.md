# Round 18: Tail-Coverage Family and Conservative Recheck Design

## 背景

Round17 已经证明了一件重要的事：

- 用 `coverage_p25` 做相对可行性约束，确实能修复 `forums` 被过小 budget 提前压缩的问题。

但 Round17 也暴露出一个明确边界：

- `congressional` 上，`coverage_p25` 在 `18-22` 之间几乎完全持平；
- 一旦约束指标失去分辨力，feasibility stage 就退化为“全预算可行”；
- 最终 selector 会重新回到 compactness 偏好，选出过小的 `18`，从而输给 `PrE-Text`。

因此 Round18 的目标不是继续扫 `0.99 / 0.98 / 0.97`，而是升级约束信号本身。

## Round18 核心目标

主目标：

- 保留 Round17 的“先做 feasibility，再做 utility”主结构；
- 把单一 `coverage_p25` 约束升级为可配置的 tail-coverage family；
- 引入一个保守的 constrained recheck，专门处理“主约束太平、预算被过度压缩”的情况；
- 在实验上争取保住 `forums / microblog / jobs`，同时修复 `congressional`。

次目标：

- 不引入 dataset-name 映射；
- 不回退到 Round14 的长度家族规则；
- 继续让预算选择保留可解释的 calibration trace；
- 代码层面尽量在 Round17 结构上增量演化，不重写整个 selector。

## Round17 的失败点复盘

在 `r17_congressional_r098` 中：

- `coverage_p25(18) = coverage_p25(19) = ... = coverage_p25(22)`
- `coverage_min(18) = coverage_min(19) = ... = coverage_min(22)`
- `coverage_mean` 虽然略有增长，但增长极小

这意味着：

1. 单一 `coverage_p25` 约束没有区分力；
2. 所有 budget 都进入可行集；
3. 第二阶段 utility 重新偏向更小 budget；
4. 结果退回 `18`，而不是对 downstream 更友好的 `19/20`。

因此 Round18 需要解决两个问题：

1. feasibility signal 不能只看一个尾部指标；
2. 当多个 budget 都可行时，不能完全没有“保守防过压缩”的后备机制。

## Round18 算法结构

Round18 继续保留：

```text
candidate budgets -> feasibility filter -> feasible-set utility
```

但把 feasibility 与后处理扩成两层。

### Step 1：Tail-Coverage Family Feasibility

对每个预算 `k`，仍然计算：

- `coverage_p25(k)`
- `coverage_mean(k)`
- `coverage_min(k)`

然后不再只用一个指标，而是使用一个 metrics family：

```text
M = {coverage_p25, coverage_mean, coverage_min}
```

Round18 第一版不要求三个指标都强约束启用，而是采用“主约束 + 辅助约束”的可配置形式：

- 主约束：`coverage_p25`
- 辅助约束：`coverage_mean`
- 观测项：`coverage_min`

形式化写法：

```text
feasible(k) = [
  coverage_p25(k) >= r_p25 * best_coverage_p25
] AND [
  coverage_mean(k) >= r_mean * best_coverage_mean
]
```

其中：

- `r_p25` 控制尾部低覆盖区域是否被充分照顾；
- `r_mean` 控制整体覆盖不能过早塌缩；
- `coverage_min` 暂时先保留在 trace 和 recheck 中，不直接作为强硬门槛。

### 为什么加入 `coverage_mean`

因为 Round17 的失败说明：

- 仅靠 `coverage_p25`，有些数据集上不同 budget 的差异会被压平；
- 但 `coverage_mean` 至少还能反映整体覆盖是否继续小幅增长。

它不能单独替代 `coverage_p25`，但可以作为第二条 guard：

- 防止“尾部指标看似持平，但整体覆盖仍在缓慢改善”的预算过早被压掉。

### Step 2：Family Score Trace

Round18 不把 family score 直接作为硬决策主目标，但会记录：

```text
family_score(k) = weighted_normalized_sum(
  coverage_p25(k),
  coverage_mean(k),
  coverage_min(k)
)
```

推荐初始权重：

- `coverage_p25`: `0.7`
- `coverage_mean`: `0.3`
- `coverage_min`: `0.0` 或仅观测

这个分数有两个用途：

1. 帮助后续分析“为什么某个 budget 看起来更像 coverage-sufficient”；
2. 为下一轮可能的 family-score 主导约束提供 trace。

### Step 3：Feasible-Set Utility

只在可行 budget 上继续使用 Round17 的 compactness-aware utility：

```text
utility_feasible(k) =
  w_support * support_score(k)
  - w_genericity * genericity_score(k)
  - w_redundancy * redundancy_score(k)
  - w_budget * budget_cost(k)
```

默认仍沿用：

- `support_weight = 1.0`
- `genericity_weight = 0.5`
- `redundancy_weight = 0.3`
- `budget_weight = 0.1`

这样做的原因是：

- `forums` 和 `microblog` 的成功，主要来自 feasibility stage 修正了预算候选集；
- 第二阶段 utility 本身不一定要大改，先保留它的已有优势。

### Step 4：Constrained Recheck

Round18 新增一个保守后处理：

> constrained recheck over larger feasible budgets

触发场景：

- 已经选出一个较小 budget；
- 但仍存在更大的 feasible budget；
- 我们怀疑当前 budget 过于紧凑，可能把 coverage 压得过头。

Round18 的第一版 recheck 不看 dataset name，也不直接强推大预算，而是要求 larger budget 同时满足：

```text
support_drop <= support_drop_max
coverage_mean_gain >= coverage_mean_gain_min
coverage_p25_gain >= coverage_p25_gain_min
coverage_min_gain >= coverage_min_gain_min
```

只要满足这组 guard，才允许把当前解提升到更大的 feasible budget。

它的作用是：

- 对 `forums`：通常不会破坏已成功的 `21/22` 区域；
- 对 `jobs`：如果 larger budget 带来的 coverage 改善太小且 support 掉得太多，就不会乱提升；
- 对 `congressional`：即使 `coverage_p25` 平，也给了系统一次“保守避免过压缩”的机会。

## Round18 的预期行为

### forums

预期保持 Round17 的核心收益：

- 小预算继续被 feasibility stage 排除；
- `21/22` 继续作为主要可行集；
- 最终性能不低于 `Round17 r098` 的 `0.2505`。

### microblog

预期继续依赖严格约束：

- 避免重新回落到 `21`；
- 在主配置下继续稳定在 `22` 左右。

### jobs

预期允许继续保留 `18` 的强最优情况：

- 如果 larger budget 没有足够 coverage family 改善，则 recheck 不应强推更大 budget。

### congressional

这是 Round18 的唯一核心修复点：

- 若 tail-family feasibility 能排除部分过小 budget，则直接成功；
- 若 feasibility 仍过宽，则 constrained recheck 应该提供第二层保守纠偏机会。

## 代码开发范围

Round18 继续基于：

[`/Users/apple/Desktop/code_from_paper/paper-new-round-16`](/Users/apple/Desktop/code_from_paper/paper-new-round-16)

核心改动文件：

- [paper_new_selector/budget_calibration.py](/Users/apple/Desktop/code_from_paper/paper-new-round-16/paper_new_selector/budget_calibration.py)
- [tests/test_budget_calibration.py](/Users/apple/Desktop/code_from_paper/paper-new-round-16/tests/test_budget_calibration.py)

新增实验资产：

- `configs/experiments/single_node_tuning_round18/`
- `scripts/append_round18_summary.py`
- `scripts/run_round18_probe_batch.sh`
- `scripts/run_round18_full_regression.sh`

## 实验设计

Round18 不追求一口气大扫，而是按三组推进。

### 实验组 A：结构探针

目的：

- 验证 tail-family feasibility 是否真的改变 `congressional` 的可行集；
- 验证它不会破坏 `forums` 的成功结构。

实验：

1. `r18_probe_congressional_f1`
2. `r18_probe_congressional_f2`
3. `r18_probe_forums_f1`

关注：

- `coverage_constraint.metrics`
- `feasible_budgets`
- `family_score_by_budget`
- `constrained_recheck`
- `resolved_seed_top_k`

### 实验组 B：Congressional Focus

目的：

- 只围绕 `congressional` 测试“tail family + recheck”是否能把预算从 `18` 拉到更稳妥的位置。

实验：

1. `r18_congressional_f1`
2. `r18_congressional_f2`
3. `r18_congressional_f3`

配置意义：

- `f1`: 只做 tail-family feasibility
- `f2`: feasibility + loose constrained recheck
- `f3`: feasibility + balanced constrained recheck

成功标准：

- 至少有 1 套配置在 `congressional` 上达到 `best_top1 >= 0.2950`
- 同时 `resolved_seed_top_k` 不再机械停在 `18`

### 实验组 C：四数据集统一回归

当 B 组选出主配置后，跑：

1. `r18_full_forums`
2. `r18_full_microblog`
3. `r18_full_jobs`
4. `r18_full_congressional`

成功标准：

- `forums >= 0.2501`
- `microblog >= 0.2763`
- `jobs >= 0.2732`
- `congressional >= 0.2950`

只有四个条件同时成立，Round18 才算真正超越 Round15 结构能力。

## 推荐执行顺序

1. 先跑 `probes`
2. 再跑 `congressional focus`
3. 选出主配置
4. 最后做 `full regression`

## 判断标准

如果 Round18 成功，应当看到：

1. `forums` 不丢；
2. `microblog` 不回落；
3. `jobs` 不被错误放大 budget；
4. `congressional` 至少有一套统一配置被救回。

如果 Round18 仍失败，则说明：

- 单纯增强 tail-coverage family 仍不足以修复 internal utility 与 downstream utility 的偏差；
- 下一步就要考虑把“budget 选择”进一步从单次 calibration 推向更显式的 internal validation proxy。
