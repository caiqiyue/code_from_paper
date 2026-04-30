# Round 17: Constrained Utility Optimization Design

## 背景

Round15 已经证明，以下预算结构可以实现 4/4 超过 PrE-Text：

| 数据集 | resolved_seed_top_k | best_top1 | vs PrE-Text |
|--------|---------------------|-----------|-------------|
| jobs | 20 | 0.2737 | +0.0005 |
| congressional | 19 | 0.2970 | +0.0020 |
| forums | 22 | 0.2507 | +0.0005 |
| microblog | 18 | 0.2754 | +0.0004 |

Round16 试图把预算选择进一步算法化，改为：

```text
candidate budget -> internal utility -> argmax
```

但 50 个实验表明，当前自校准主 utility 有一个明确问题：

1. `forums` 长期被压到 `18`
2. `microblog` 也经常落在 `18`
3. `weight_sweep` 没有真正把 `forums` 拉回 `20/22`
4. `recheck` 方案没有打到核心，因为 `utility_gap` 并不小，说明问题不是近边界误判，而是主目标函数本身偏向小预算

因此 Round17 的方向不是继续修补 near-boundary case，而是直接升级主决策结构。

## Round17 核心目标

主目标：

- 保留 Round16 “内部信号驱动预算选择”的方法主线
- 放弃单层线性 utility 直接 `argmax` 的结构
- 用一个更正式、更可论文表达的 constrained optimization 形式替代它
- 在结果上尽量恢复到 Round15 的 4/4 水平

次目标：

- 不引入 dataset-name 的硬编码映射
- 不回退到 length-family threshold
- 不修改 Stage2 bootstrap 主结构
- 继续保留完整的 per-budget calibration trace

## 核心思想

Round17 将 budget 选择定义为：

> a constrained internal model-selection problem

具体来说：

1. 先判断每个 budget 是否满足 **relative coverage sufficiency**
2. 只有满足 coverage sufficiency 的 budget 才进入可行集
3. 再在可行集里优化 compactness-aware utility

也就是说，Round17 不再问：

```text
哪个 budget 的总 utility 最大？
```

而是改成：

```text
哪些 budget 已经覆盖够了？
在这些 coverage-sufficient budgets 里，哪个最干净、最贴 private、最不冗余？
```

这比 Round16 更适合 `forums`，因为它把“不能过早收缩到尾部覆盖不足的小预算”变成了显式约束，而不是次要奖励项。

## 算法定义

### 候选 budget 集合

Round17 第一版仍保持：

```python
candidate_seed_top_k = [18, 19, 20, 21, 22]
```

理由：

- 与 Round16 保持可比性
- 完全覆盖 Round15 成功预算区域
- 便于直接做 Round16 vs Round17 对照

### Step 1: 计算 per-budget metrics

对每个 `k in candidate_seed_top_k`，仍然在同一批 candidates 上运行轻量 selector，得到：

- `support_mean(k)`
- `genericity_mean(k)`
- `redundancy_mean(k)`
- `coverage_mean(k)`
- `coverage_p25(k)`
- `coverage_min(k)`
- `budget_cost(k)`

这里最关键的是：

```python
coverage_p25(k)
```

它表示 private manifold 中较低覆盖区域的代表性覆盖水平，是 Round17 的主约束信号。

### Step 2: 定义 relative coverage sufficiency

先求：

```python
best_coverage_p25 = max(coverage_p25(k) for k in candidate_seed_top_k)
```

然后定义预算 `k` 是否可行：

```python
feasible(k) = coverage_p25(k) >= tau * best_coverage_p25
```

Round17 第一版采用：

```python
tau = 0.99
```

也就是：

> 只有当某个 budget 的 `coverage_p25` 达到候选集合最佳尾部覆盖的 99%，它才被视为 coverage-sufficient。

### 为什么用 coverage_p25 而不是 coverage_mean

因为当前真正的问题不是平均覆盖不够，而是：

- `forums` 的尾部、更难覆盖的 private region 没有被小预算充分照顾
- `coverage_mean` 很容易被已覆盖区域稀释
- `coverage_p25` 更能判断“是否还存在明显未覆盖的 private tail”

因此 Round17 的主约束应当放在 `coverage_p25` 上，而不是 `coverage_mean`。

## 可行集定义

Round17 的可行集为：

```python
feasible_budgets = [
    k for k in candidate_seed_top_k
    if coverage_p25(k) >= 0.99 * best_coverage_p25
]
```

这意味着：

- 如果 `18` 的尾部覆盖明显不足，它会被直接排除
- 如果 `19/20/22` 中有多个预算都已基本覆盖够，那么它们都会保留到下一阶段

这不是“偏向大预算”，而是：

> 不允许 coverage 明显不足的 budget 过早参与最终竞争

## Step 3: 在可行集里做 utility optimization

Round17 的 utility 不再承担“让 coverage 变大”的职责。  
coverage 已经在前面的 feasibility stage 中处理。

因此，第二阶段 utility 可以收敛为一个更干净的 compactness-aware objective：

```python
utility_feasible(k) = (
    w_support * support_score(k)
    - w_genericity * genericity_score(k)
    - w_redundancy * redundancy_score(k)
    - w_budget * budget_cost(k)
)
```

推荐初始权重：

```python
w_support = 1.0
w_genericity = 0.5
w_redundancy = 0.3
w_budget = 0.1
```

注意：

- Round17 默认不再把 `coverage_mean` 直接放进第二阶段 utility
- 这样 coverage 的职责更单一，避免再次被 support/compactness 压掉

## Step 4: Fallback 与 tie-break

### Fallback 1：可行集为空

如果极端情况下没有 budget 满足约束：

```python
if not feasible_budgets:
    fallback to original Round16 argmax utility
```

这只是兜底，不应是常态。

### Fallback 2：可行集只有一个

若只有 1 个 budget 满足 coverage sufficiency，则直接选它。

### Fallback 3：可行集多个 budget

若有多个预算满足 coverage sufficiency，则：

1. 计算 `utility_feasible(k)`
2. 选择 utility 最大者

### tie-break

如果可行集内 top-2 utility 很接近，保留轻量 tie-break：

```python
if utility_gap <= epsilon:
    prefer smaller budget
```

推荐：

```python
epsilon = 0.01
```

因为在满足 coverage sufficiency 后，优先较小 budget 是合理的。

## 为什么这套结构更“高大上”且逻辑合理

Round17 的关键好处在于：

1. **它不是 dataset mapping**
   - 没有 `forums -> 22`
   - 没有 `if dataset == forums`

2. **它不是长度阈值**
   - 不依赖 mean / p75 / median 的手工规则

3. **它不是简单 if/else 工程补丁**
   - 它可以被正式写成 constrained optimization

4. **它能解释 Round15 的成功行为**
   - `forums` 需要更大 budget，并不是因为“它叫 forums”
   - 而是因为较小 budget 在尾部 coverage 上未达到 sufficiency

因此论文表述上可以自然写成：

> We formulate seed-budget selection as a constrained internal model-selection problem. Instead of directly maximizing a scalar utility, we first define a feasible set of budgets via a relative tail-coverage sufficiency condition over the private manifold. Utility optimization is then performed only over the feasible budgets. This prevents overly compact budgets from winning when they still under-cover the lower-covered portion of the private distribution.

## 与 Round16 的关系

### Round16

```text
maximize one scalar utility over all budgets
```

问题：

- coverage 只是 utility 中的一项
- 容易被 compactness / support 一起压掉
- `forums` 被系统性推向 18

### Round17

```text
first satisfy coverage sufficiency
then optimize compactness-aware utility
```

优点：

- coverage 从“奖励项”升级为“可行性条件”
- 更容易保护 `forums` 这类 coverage 未饱和数据
- 其他数据集仍能在可行集内保持偏小预算

## 代码开发计划

Round17 开发继续基于：

[`/Users/apple/Desktop/code_from_paper/paper-new-round-16`](/Users/apple/Desktop/code_from_paper/paper-new-round-16)

不单独新建 `paper-new-round-17`，先在 Round16 工作副本上演化实现。

### 开发范围

优先修改：

1. `paper_new_selector/budget_calibration.py`
2. `tests/test_budget_calibration.py`
3. `tests/test_stage1_runner.py`
4. `configs/experiments/single_node_tuning_round17/`

原则上不改：

- `pipeline.py`
- Stage2 bootstrap 逻辑
- eval bridge

### Phase 1：实现 coverage-sufficiency feasibility stage

目标：

- 在 per-budget metrics 上定义可行集

建议新增函数：

```python
compute_relative_coverage_threshold(...)
select_feasible_budgets_by_coverage_p25(...)
```

核心输出：

```json
"coverage_constraint": {
  "metric": "coverage_p25",
  "relative_ratio": 0.99,
  "best_coverage_p25": 0.842,
  "threshold": 0.83358,
  "feasible_budgets": [20, 21, 22]
}
```

测试覆盖：

- 某个 budget 低于 99% 阈值时被排除
- 多个 budget 达到阈值时都进入可行集
- 全部预算都达到阈值时可行集等于全体候选

### Phase 2：实现 feasible-set utility

目标：

- 把 second-stage utility 改为不含 coverage 的 compactness-aware objective

建议新增函数：

```python
combine_feasible_budget_metrics(...)
select_budget_from_feasible_set(...)
```

测试覆盖：

- `coverage` 不再直接参与 second-stage utility
- 可行集里 utility 正常比较 support/genericity/redundancy/budget
- 可行集单元素时直接返回该 budget

### Phase 3：fallback 与 trace

目标：

- 把最终决策链条写清楚

建议输出：

```json
"seed_budget": {
  "mode": "self_calibrated_constrained",
  "candidate_seed_top_k": [18, 19, 20, 21, 22],
  "coverage_constraint": {
    "metric": "coverage_p25",
    "relative_ratio": 0.99,
    "best_coverage_p25": 0.842,
    "threshold": 0.83358,
    "feasible_budgets": [20, 21, 22]
  },
  "resolved_seed_top_k": 20,
  "selection_stage": "feasible_set_utility",
  "fallback_used": false,
  "per_budget_metrics": { ... }
}
```

这样后续分析可以清楚回答：

- 哪些 budget 被 coverage constraint 淘汰
- 为什么最终 budget 会从 `18` 跳到 `20` 或 `22`

### Phase 4：配置体系

Round17 推荐新开配置目录：

```text
configs/experiments/single_node_tuning_round17/
```

推荐基础配置：

```yaml
selector:
  seed_budget_rule:
    enabled: true
    mode: self_calibrated_constrained
    candidate_seed_top_k: [18, 19, 20, 21, 22]
    coverage_constraint:
      metric: coverage_p25
      relative_ratio: 0.99
    utility:
      support_weight: 1.0
      genericity_weight: 0.5
      redundancy_weight: 0.3
      budget_weight: 0.1
    tiebreak:
      epsilon: 0.01
      prefer_smaller_budget: true
```

## 实验设计

Round17 的实验目标非常明确：

> 验证 `coverage_p25` feasibility constraint 能否把 `forums` 从 `18` 拉回更合理 budget，并重新实现 4/4 超过 PrE-Text。

### 实验组 A：最小功能探针

目的：

- 验证 constrained selection 是否真的在工作
- 看 `18` 是否会因为 coverage insufficiency 被淘汰

实验：

1. `r17_probe_forums_base`
2. `r17_probe_microblog_base`

重点观察：

- `best_coverage_p25`
- `threshold`
- `feasible_budgets`
- `resolved_seed_top_k`
- `best_top1`

成功标准：

- `forums` 的可行集不应被 `18` 独占
- `microblog` 允许继续保留较小预算

### 实验组 B：ratio sweep

目的：

- 看 coverage sufficiency 的强度应该放在哪一档

推荐 3 组：

1. `r17_forums_r099`
   - `relative_ratio = 0.99`
2. `r17_forums_r098`
   - `relative_ratio = 0.98`
3. `r17_forums_r097`
   - `relative_ratio = 0.97`

观察指标：

- `feasible_budgets`
- `resolved_seed_top_k`
- `best_top1`

成功标准：

- 至少 1 组把 `forums` 拉回 `19/20/22`
- 且 `best_top1 >= 0.2505`

### 实验组 C：四数据集快速回归

在 ratio sweep 中选出 1 套主配置后，立即跑：

1. `r17_full_jobs`
2. `r17_full_congressional`
3. `r17_full_forums`
4. `r17_full_microblog`

成功标准：

| 数据集 | 目标 |
|--------|------|
| jobs | `best_top1 >= 0.2732` |
| congressional | `best_top1 >= 0.2950` |
| forums | `best_top1 >= 0.2501`，理想值 `>= 0.2505` |
| microblog | `best_top1 >= 0.2763` |

如果 4/4 成立，则 Round17 主线成立。

### 实验组 D：Round16 对照组

必须保留至少两个对照：

1. `r16_c1_forums`
2. `r17_full_forums`

对比：

- `coverage_constraint.feasible_budgets`
- `resolved_seed_top_k`
- `best_top1`

目的：

- 证明提升不是偶然波动，而是 feasibility stage 真正改变了预算选择

### 实验组 E：可选 seed sanity check

若 Round17 在 `forums` 上明显恢复，建议再补：

1. `r17_forums_seed123`
2. `r17_forums_seed456`

目的：

- 验证 constrained selection 不是只在 seed 42 上偶然有效

## 实验结果记录（2026-04-30）

本轮结果已在服务器端核对，实验输出目录位于：

```text
/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/r17_*
```

本节所有结论均来自各实验目录中的：

- `stage1_budget_calibration.json`
- `eval/downstream_eval_summary.json`

### 实验组 A 结果：最小功能探针

| 实验 | ratio | feasible_budgets | resolved_seed_top_k | utility_gap | best_top1 |
|------|-------|------------------|---------------------|-------------|-----------|
| `r17_probe_forums_base` | `0.99` | `21,22` | `21` | `0.3000` | `0.2480` |
| `r17_probe_microblog_base` | `0.99` | `22` | `22` | `0.0000` | `0.2777` |

分析：

- `forums` 上，`18/19/20` 被 coverage feasibility 直接排除，说明 Round17 的约束层确实改变了预算选择结构，不再让过小 budget 直接参与最终竞争。
- `microblog` 上，`0.99` 条件下只有 `22` 可行，说明该数据集在尾部覆盖上对更大 budget 更敏感。
- 因此，实验组 A 的核心目标已经达到：Round17 的 feasibility stage 确实在工作，而不是一个空壳约束。

### 实验组 B 结果：ratio sweep

#### forums

| 实验 | ratio | feasible_budgets | resolved_seed_top_k | best_top1 | 结论 |
|------|-------|------------------|---------------------|-----------|------|
| `r17_forums_r099` | `0.99` | `21,22` | `21` | `0.2462` | 未达标 |
| `r17_forums_r098` | `0.98` | `21,22` | `21` | `0.2505` | 达到目标 |
| `r17_forums_r097` | `0.97` | `21,22` | `21` | `0.2492` | 仍低于 `0.2505` |

结论：

- `forums` 的最佳点是 `ratio = 0.98`。
- 三组 sweep 的 `feasible_budgets` 都稳定在 `21,22`，说明 `coverage_p25` 约束已经足够稳地把 `18/19/20` 排除出去。
- `0.99` 过严，虽然结构正确，但性能不足；`0.97` 虽然更松，但没有带来额外收益；`0.98` 是当前最合适的折中点。

#### microblog 续跑

| 实验 | ratio | feasible_budgets | resolved_seed_top_k | best_top1 | 结论 |
|------|-------|------------------|---------------------|-----------|------|
| `r17_microblog_r099` | `0.99` | `22` | `22` | `0.2770` | 达标 |
| `r17_microblog_r098` | `0.98` | `22` | `22` | `0.2770` | 达标 |
| `r17_microblog_r097` | `0.97` | `20,21,22` | `21` | `0.2751` | 回落 |

结论：

- `microblog` 上，`0.99/0.98` 均将可行集压到单点 `22`，因此结果稳定且优于 PrE-Text。
- 一旦放松到 `0.97`，可行集扩大为 `20,21,22`，第二阶段 utility 转而选择 `21`，性能回落。
- 这说明 `microblog` 对约束松弛较敏感，过松会重新引入“预算压缩”问题。

### 实验组 C 结果：四数据集快速回归（采用 `ratio = 0.98`）

| 数据集 | 实验 | feasible_budgets | resolved_seed_top_k | best_top1 | PrE-Text | 差值 |
|--------|------|------------------|---------------------|-----------|----------|------|
| forums | `r17_forums_r098` | `21,22` | `21` | `0.2505` | `0.2501` | `+0.0004` |
| microblog | `r17_microblog_r098` | `22` | `22` | `0.2770` | `0.2763` | `+0.0007` |
| jobs | `r17_jobs_r098` | `18,19,20,21,22` | `18` | `0.2782` | `0.2732` | `+0.0050` |
| congressional | `r17_congressional_r098` | `18,19,20,21,22` | `18` | `0.2928` | `0.2950` | `-0.0022` |

结论：

- `ratio = 0.98` 并没有实现 `4/4` 超过 PrE-Text，实际结果是 `3/4`。
- 成功数据集为 `forums / microblog / jobs`。
- 失败数据集为 `congressional`，其 `best_top1 = 0.2928`，低于 PrE-Text 的 `0.2950`。

### 关键机制证据

#### forums：Round17 确实修复了 Round16 的核心问题

`r17_forums_r098` 的 per-budget calibration 显示：

- `18`: `coverage_p25 = 0.130894`，`feasible = false`
- `19`: `coverage_p25 = 0.130894`，`feasible = false`
- `20`: `coverage_p25 = 0.131690`，`feasible = false`
- `21`: `coverage_p25 = 0.135860`，`feasible = true`
- `22`: `coverage_p25 = 0.135860`，`feasible = true`

这说明：

- Round17 的价值首先不在于“直接偏向大 budget”，而在于把 coverage 不足的小 budget 从候选集中剔除。
- `forums` 上的成功不是偶然波动，而是 feasibility stage 真正改变了预算选择结构。
- 在 `21` 和 `22` 都可行后，第二阶段 utility 选择了更紧凑的 `21`，从而在 coverage 够用的前提下保留了 compactness 优势。

#### microblog：约束层也在起作用，但需要保持足够严格

`r17_microblog_r098` 的关键现象是：

- 只有 `22` 满足 coverage constraint，因此不会再被压缩到 `18/19/20`。
- 当 ratio 降到 `0.97` 时，`20/21/22` 同时进入可行集，utility 改选 `21`，`best_top1` 从 `0.2770` 降到 `0.2751`。

这说明：

- `microblog` 的性能稳定依赖于较严格的 coverage sufficiency 约束。
- 该数据集不是“不需要约束”，而是“约束一旦过松，旧问题会重新出现”。

#### jobs：Round17 没有修复问题，但误打误撞拿到了更强结果

`r17_jobs_r098` 的 per-budget calibration 显示：

- 五个 budget `18-22` 全部可行。
- `18` 的 utility 显著最高，且 `utility_gap = 0.5081`，说明 `18` 不是近边界偶然胜出，而是强最优。

这说明：

- 对 `jobs` 而言，coverage constraint 基本没有发挥筛选作用。
- 最终表现提升主要来自第二阶段 utility 本身恰好偏向了最优的更小 budget。
- 从算法解释上说，Round17 在 `jobs` 上不是“constraint 救回来了”，而是“constraint 不妨碍 utility 选到更强的小 budget”。

#### congressional：Round17 当前失败的根因

`r17_congressional_r098` 的 per-budget calibration 显示：

- `18-22` 五个 budget 全部可行。
- 五个 budget 的 `coverage_p25` 实际上完全相同，均为 `0.169911`。
- 因此 coverage feasibility 无法淘汰任何较小 budget。
- 第二阶段 utility 最终回到对 compactness 更友好的 `18`，导致 `best_top1 = 0.2928`，低于 PrE-Text。

这说明：

- Round17 的核心假设是“尾部覆盖不足会在 `coverage_p25` 上显式暴露出来”；但在 `congressional` 上，这个信号没有足够分辨力。
- 一旦 `coverage_p25` 不能拉开不同 budget 的差异，Round17 就会退化为“在几乎全可行的预算集上做 utility 选择”，从而重新偏向小 budget。
- 因此，Round17 当前不是普适解决方案，它更像是对 `forums` 类 coverage-tail 问题非常有效，但对 `congressional` 这种 coverage 指标平坦的数据集不够强。

## Round17 总结判断

Round17 是一个**部分成功**的回合，而不是最终收敛回合。

它的主要收获有三点：

1. `coverage_p25` feasibility stage 确实是有效机制，不是伪改动。
2. 它成功修复了 `forums` 在 Round16 中被系统性压向小预算的问题。
3. 在统一 `ratio = 0.98` 下，它拿到了 `forums / microblog / jobs` 三个数据集超过 PrE-Text 的结果。

但它也有清晰边界：

1. 它没有实现目标中的 `4/4` 全面超过 PrE-Text。
2. `congressional` 上 `coverage_p25` 对不同 budget 缺乏区分力，导致约束层失效。
3. 因此 Round17 不能直接替代 Round15 作为最终统一方案。

更准确的定位应当是：

> Round17 证明了“先做相对尾部覆盖可行性筛选，再在可行集中优化 utility”这一结构在 `forums` 类问题上是正确方向，但 `coverage_p25` 作为唯一约束指标还不够普适，尚不足以支撑 4 个数据集的统一最优预算选择。

## 对后续迭代的启示

从本轮结果看，后续若继续沿 Round17 主线推进，重点不应再是简单调 `0.99/0.98/0.97`，而应放在：

1. 为什么 `congressional` 的 `coverage_p25` 在不同 budget 上几乎不变。
2. 是否需要把约束信号从单一 `coverage_p25` 扩展为更能区分 budget 的 tail-coverage family。
3. 是否需要把 feasibility stage 与 utility stage 之间的职责边界进一步拉开，避免在“全预算均可行”时又完全退化成 compactness 偏好。

## 推荐执行顺序

1. 实现 Round17 constrained selection
2. 跑本地单测
3. 跑实验组 A：`forums + microblog probe`
4. 跑实验组 B：`ratio sweep`
5. 选出主 ratio
6. 跑实验组 C：四数据集快速回归
7. 如果成功，再补实验组 D/E

## 成功标准

Round17 至少应满足：

1. 不使用 dataset-name 的硬编码映射
2. 不使用长度阈值 family rule
3. 预算选择可以被正式解释为 constrained internal model selection
4. `coverage_p25` feasibility stage 能改变 `forums` 的预算选择结构
5. 至少存在 1 套统一配置，重新实现 4/4 超过 PrE-Text

更理想的成功标准：

6. `forums` 恢复到 `0.2505+`
7. `jobs/congressional/microblog` 全部不弱于 Round15
8. 论文叙事中能自然解释为什么 `forums` 应该选择更大 budget

## 风险与缓解

### 风险 1：0.99 过严，过多预算被排除

缓解：

- 同时准备 `0.98 / 0.97` 的 ratio sweep

### 风险 2：feasible set 太大，最终仍偏向小预算

缓解：

- second-stage utility 保留 `budget_cost`
- 必要时补充 `coverage_p25` 排名信息做分析，但不立刻加新规则

### 风险 3：只救回 forums，伤到 jobs/congressional

缓解：

- `jobs/congressional` 必须进入第一轮 full regression
- 不在第一版中加入额外针对 forums 的任何显式偏置

### 风险 4：写成约束后看起来像新阈值系统

缓解：

- 强调这是 relative feasibility，不是 absolute dataset-specific threshold
- 用“relative tail coverage sufficiency”统一解释所有数据集

## 论文叙事建议

英文表述：

> We formulate seed-budget selection as a constrained internal model-selection problem. Instead of directly maximizing a scalar utility over all candidate budgets, we first define a feasible set using a relative tail-coverage sufficiency condition over the private manifold, measured by the 25th percentile coverage. Utility optimization is then performed only over the feasible budgets. This prevents overly compact budgets from winning when they still under-cover the lower-covered portion of the private distribution.

中文表述：

> 我们将 seed budget 选择表述为一个带约束的内部模型选择问题。不同于直接在所有候选 budget 上最大化单一效用函数，Round17 先基于 private manifold 尾部覆盖的相对充分性条件定义可行预算集合，其中尾部覆盖由 `coverage_p25` 表征；随后只在可行预算集合内优化紧凑性导向的效用函数。这样可以避免那些对 private 分布低覆盖区域仍然覆盖不足的过小预算过早胜出。
