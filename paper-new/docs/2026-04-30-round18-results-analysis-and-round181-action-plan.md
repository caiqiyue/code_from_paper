# Round18 Results Analysis and Round18.1 Action Plan

## 1. 结论先行

Round18 当前 **没有达到目标**。

必须明确的是：我的对标基准算法始终是 `PrE-Text`，因此判断标准不是“比 Round17 好一点”或“机制更复杂”，而是：

> 最终实验结果必须高于 `PrE-Text`。

从本轮已经完成的实验看，Round18 目前不满足这一标准，原因有两点：

1. `congressional` 仍然没有超过 `PrE-Text`。
2. `forums` 也没有稳定超过 `PrE-Text`。

因此，Round18 现在不能进入“四数据集统一主线”，更不能替代 Round15。

## 2. 对标基准

当前 `PrE-Text` 基线为：

| 数据集 | PrE-Text best_top1 |
|--------|--------------------|
| forums | `0.2501` |
| microblog | `0.2763` |
| jobs | `0.2732` |
| congressional | `0.2950` |

后续所有判断都必须围绕这 4 个值展开。

## 3. Round18 已完成实验结果

### 3.1 Congressional 相关实验

| 实验 | resolved_seed_top_k | best_top1 | 与 PrE-Text 比较 |
|------|---------------------|-----------|------------------|
| `r18_probe_congressional_f1` | `18` | `0.2928` | 低于 `0.2950` |
| `r18_probe_congressional_f2` | `18` | `0.2928` | 低于 `0.2950` |
| `r18_congressional_f1` | `18` | `0.2928` | 低于 `0.2950` |
| `r18_congressional_f2` | `18` | `0.2928` | 低于 `0.2950` |
| `r18_congressional_f3` | `18` | `0.2932` | 仍低于 `0.2950` |

结论：

- Round18 的 `congressional` 修复目标失败。
- 即使 `f3` 有轻微提升，也仍然没有超过 `PrE-Text`。
- 因此不能说 Round18 已经解决了 Round17 的失败点。

### 3.2 Forums 相关实验

| 实验 | resolved_seed_top_k | best_top1 | 与 PrE-Text 比较 |
|------|---------------------|-----------|------------------|
| `r18_probe_forums_f1` | `21` | `0.2481` | 低于 `0.2501` |
| `r18_forums_seed123` | `21` | `0.2491` | 低于 `0.2501` |
| `r18_forums_seed456` | `22` | `0.2478` | 低于 `0.2501` |

结论：

- Round18 也没有稳住 `forums`。
- 这说明当前设计不仅没救回 `congressional`，还削弱了本该保住的 `forums` 优势。

## 4. 为什么 Round18 失败

Round18 的设计初衷是：

1. 用 `tail-coverage family feasibility` 替代 Round17 的单一 `coverage_p25` 约束；
2. 再用 `constrained recheck` 防止 budget 被过度压缩。

但是从结果看，这两层都没有真正解决问题。

### 4.1 失败点一：tail-family feasibility 对 congressional 仍然没有分辨力

在 `r18_congressional_f1/f2/f3` 中，关键现象是：

- `coverage_p25(18) = coverage_p25(19) = ... = coverage_p25(22)`
- `coverage_min(18) = coverage_min(19) = ... = coverage_min(22)`
- `coverage_mean` 虽然略有增加，但增幅极小

因此：

- `18-22` 仍然全部进入可行集；
- feasibility stage 没有真正缩小候选预算集合；
- 算法仍会退化为“在全可行集里按 utility 偏向更小 budget”；
- 最终还是停在 `18`。

这说明：

> Round18 虽然把约束从单指标扩成了 family，但 family 中的信号对 `congressional` 仍然不够强，无法真正改变预算选择结构。

### 4.2 失败点二：constrained recheck 实际上没有起作用

以 `r18_congressional_f3` 为例，日志显示：

- `resolved_seed_top_k = 18`
- `promoted_budget = 22`
- `pass_recheck = false`
- `support_drop = 3.5023`
- `coverage_mean_gain = 0.000240`
- `coverage_p25_gain = 0.0`
- `coverage_min_gain = 0.0`

这里最关键的发现是：

`support_drop` 的量纲与配置阈值不一致。

当前配置里使用的是类似：

- `support_drop_max = 0.015`
- `support_drop_max = 0.02`

这种“小数阈值”。

但代码里实际比较的是原始 `support_mean` 差值，而不是归一化后的差值，所以在真实实验里：

- `support_drop` 会直接变成 `0.5` 甚至 `3.5` 这种数量级；
- recheck 几乎不可能通过；
- 于是“理论上开启了 recheck”，实际上它并没有真正工作。

这不是简单参数不合适，而是：

> 当前 Round18 的 constrained recheck 存在明显的量纲错位问题。

### 4.3 失败点三：forums 也没有被稳住

Round18 原本希望做到：

- 不破坏 `forums` 上已经验证有效的较大 budget 结构；
- 同时给 `congressional` 更多修复空间。

但结果表明：

- `forums` 的可行集仍然主要落在 `21,22`；
- 结构层面看似合理；
- 但最终 `best_top1` 并没有回到超过 `PrE-Text` 的水平。

因此当前 Round18 不能被表述为“只差 congressional 一点点”，更准确的说法应是：

> Round18 目前同时没有稳住 `forums`，也没有修复 `congressional`。

## 5. 这轮实验带来的关键发现

本轮最重要的不是“又试了一批配置”，而是得到以下几个更明确的发现：

### 发现 1

单纯把 `coverage_p25` 扩展为 `coverage_p25 + coverage_mean + coverage_min`，并不能自动解决 `congressional`。

原因是这些信号在 `congressional` 上整体都太平，仍不足以把小 budget 排除出去。

### 发现 2

只靠“global relative ratio”这一类全局相对阈值，不足以修复 `congressional`。

因为：

- 它只关心“某预算是否接近最优 coverage”；
- 但 `congressional` 的问题是不同预算之间 coverage 变化太小，导致“接近最优”几乎对所有 budget 都成立。

### 发现 3

Round18 真正还没有测试到它本来想测试的 recheck 能力，因为当前 recheck 基本被量纲问题锁死了。

也就是说，这轮失败不能简单理解为：

> recheck 这个思想没用。

更准确地说是：

> 目前的 recheck 实现方式还没有被正确测试到。

### 发现 4

在没有先单点超过 `PrE-Text` 之前，不应继续直接做 full regression。

否则会带来两个问题：

1. 算力浪费；
2. 结果解释混乱，因为主失败点还没修掉。

## 6. 接下来该怎么做

基于以上分析，下一步不应该继续扩跑 Round18，而应该收缩为一个更小、更明确的修正版回合，这里记为：

`Round18.1`

其目标不是追求大而全，而是先满足两个硬条件：

1. `congressional >= 0.2950`
2. `forums >= 0.2501`

只有这两个条件先满足，才有资格继续做四数据集统一回归。

## 7. Round18.1 设计的代码改动

Round18.1 的代码改动应当聚焦在真正失效的部分，而不是再做大范围结构重写。

### 7.1 改动一：修正 constrained recheck 中 support_drop 的量纲

当前问题：

- recheck 用的是原始 `support_mean` 差值；
- 配置却使用归一化尺度的小数阈值。

改法建议有两种：

#### 方案 A：改用归一化 support drop

把 recheck 比较项从：

```text
support_drop = support_mean(smaller) - support_mean(larger)
```

改成：

```text
support_drop_norm =
  normalized_support(smaller) - normalized_support(larger)
```

优点：

- 和当前 `0.01 / 0.02` 这一类阈值量纲匹配；
- 更容易跨数据集解释；
- 和 constrained utility 中的归一化逻辑更一致。

这是更推荐的方案。

#### 方案 B：保留原始 support drop，但重新标定阈值

即继续使用原始值，但把：

- `support_drop_max`

改成与原始 `support_mean` 尺度一致的范围。

缺点：

- 参数含义更不直观；
- 不同数据集之间可能更不稳定。

因此建议优先采用方案 A。

### 7.2 改动二：从“全局相对阈值”转向“相邻 budget 增益判断”

Round18 当前的问题是：

- `best_coverage * ratio` 这种形式，对 `congressional` 区分力太弱。

因此 Round18.1 应补入一个更直接的问题判断：

> 从当前较小 budget 增加到更大 budget，是否真的带来了足够的覆盖增益？

建议新增一类增益判断信号：

- `coverage_mean_gain(k -> k+1)`
- `coverage_p25_gain(k -> k+1)`
- 可选：`family_score_gain(k -> k+1)`

具体逻辑可以是：

```text
如果 larger feasible budget 相比当前 budget 的 coverage 增益足够，
并且 support 损失可接受，
则允许提升到 larger budget。
```

这比“是否接近全局最优 coverage”更适合当前的 `congressional` 问题。

### 7.3 改动三：保留 tail-family trace，但弱化其“单独决定一切”的职责

Round18 已经证明：

- `coverage_p25 / mean / min` family trace 是有分析价值的；
- 但它单独不足以稳定决定最终 budget。

所以 Round18.1 不应该把它完全删掉，而应调整定位：

- feasibility family 继续保留；
- 但最终是否提升 budget，要更多依赖“相邻 budget 增益 + support 损失”的组合判断。

### 7.4 改动四：保证 recheck trace 清晰可解释

Round18.1 输出中应明确保留：

- 当前 budget
- 候选 larger budget
- `support_drop`
- `support_drop_normalized`
- `coverage_mean_gain`
- `coverage_p25_gain`
- `family_score_gain`
- 是否最终 promoted

这样后续分析才能直接回答：

- 为什么被提升；
- 为什么没被提升；
- 是 support 损失过大，还是 coverage 增益太小。

## 8. Round18.1 设计的实验

Round18.1 不应一开始就跑全量四数据集，而应分三步。

### 8.1 实验组 A：代码修复后的 congressional 诊断实验

目的：

- 验证修正量纲后的 recheck 是否终于真正开始工作；
- 验证相邻 budget 增益判断能否把 `18` 推向 `19/20`。

建议实验：

1. `r181_congressional_g1`
   - normalized support drop
   - loose gain guard
2. `r181_congressional_g2`
   - normalized support drop
   - balanced gain guard
3. `r181_congressional_g3`
   - normalized support drop
   - slightly stronger coverage gain guard

关注指标：

- `resolved_seed_top_k`
- `constrained_recheck.pass_recheck`
- `promoted_budget`
- `best_top1`

成功标准：

- 至少 1 组达到 `best_top1 >= 0.2950`

### 8.2 实验组 B：forums 保护回归

目的：

- 确保修复 `congressional` 的动作没有再次伤到 `forums`。

建议实验：

1. `r181_forums_g1`
2. `r181_forums_g2`

成功标准：

- 至少 1 组达到 `best_top1 >= 0.2501`

### 8.3 实验组 C：四数据集统一回归

只有在 A 和 B 都成功后，才跑：

1. `r181_full_forums`
2. `r181_full_microblog`
3. `r181_full_jobs`
4. `r181_full_congressional`

唯一成功标准是：

- `forums >= 0.2501`
- `microblog >= 0.2763`
- `jobs >= 0.2732`
- `congressional >= 0.2950`

只要有一个没超过 `PrE-Text`，就不能宣称成功。

## 9. 推荐执行顺序

接下来最合理的行动顺序是：

1. 停止继续扩跑当前 Round18 full regression
2. 修改代码，先修正 recheck 的 support 量纲
3. 增加相邻 budget 增益判断
4. 只跑 `congressional` 诊断实验
5. 再跑 `forums` 保护回归
6. 只有当两者都超过 `PrE-Text` 后，才做四数据集统一回归

## 10. 最终判断

这轮 Round18 可以总结为：

> Round18 当前失败。它没有达到“超过 PrE-Text”这一核心标准。具体表现为：没有修复 congressional，同时也未能稳定保持 forums 的优势。因此，当前 Round18 不能进入统一四数据集主线。下一步应收缩为 Round18.1，优先修复 constrained recheck 的量纲问题，并将预算提升判断从单纯的全局相对 coverage 阈值，升级为相邻 budget 增益驱动的保守提升机制。

