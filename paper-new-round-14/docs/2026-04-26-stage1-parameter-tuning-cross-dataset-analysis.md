# `Stage 1` 参数调优跨数据集分析与下一步建议

更新时间：2026-04-26

## 1. 文档目的

本文档基于以下两份结果文档，做一次跨四个数据集的统一分析：

- [2026-04-24-pretext-screening-results.md](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/docs/2026-04-24-pretext-screening-results.md)
- [2026-04-26-stage1-parameter-tuning-screening-results-full.md](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/docs/2026-04-26-stage1-parameter-tuning-screening-results-full.md)

分析目标不是只看 `forums` / `microblog` 有没有涨，而是回答更严格的问题：

> 哪些参数调整真正有助于弱数据集，同时又不会明显伤害 `jobs` / `congressional`；如果做不到，下一步应不应该继续全局调参，还是应开始改算法结构。

## 2. 比较基线

本文统一使用两条参考线：

### 2.1 `PrE-Text` 基线 `SP-S-*`

| 数据集 | `best_top1` |
| --- | ---: |
| `jobs` | 0.2731984829329962 |
| `congressional` | 0.2949640287769784 |
| `forums` | 0.25014487154722814 |
| `microblog` | 0.2762705387848682 |

### 2.2 当前创新算法旧基线 `NS-S-*`

| 数据集 | `best_top1` |
| --- | ---: |
| `jobs` | 0.2761061946902655 |
| `congressional` | 0.2969732322250308 |
| `forums` | 0.2470542785396948 |
| `microblog` | 0.27493312953763854 |

之所以同时保留两条基线，是因为它们回答的是两个不同问题：

- 相对 `NS-S-*`：参数有没有把当前创新算法本身调得更好
- 相对 `SP-S-*`：参数有没有让创新算法真正超过 `PrE-Text`

## 3. 最重要的整体结论

先给结论，再展开。

### 3.1 没有任何一组全局参数，能让四个数据集同步上升

这次两轮调参中，没有一组参数满足下面这个条件：

- `jobs` 不下降
- `congressional` 不下降
- `forums` 提升
- `microblog` 提升

也就是说，当前问题已经不是“还没找到那组完美参数”，而更像是：

> 这套 `Stage 1` 全局静态参数，本身就难以同时适配正式文本数据集和非正式文本数据集。

### 3.2 当前最明显的冲突点在 `congressional`

几乎所有对 `forums` / `microblog` 有帮助的改动，都会不同程度伤到 `congressional`。

典型例子：

- `A2 length_lambda: 0.20 -> 0.10`
  - 相对 `NS-S-*`
  - `jobs +0.00341`
  - `congressional -0.00052`
  - `forums +0.00187`
  - `microblog +0.00217`

- `E5 density 0.50 -> 0.45, novelty 0.30 -> 0.35`
  - `jobs +0.00335`
  - `congressional -0.00149`
  - `forums +0.00103`
  - `microblog +0.00236`

- `E4 reference_top_k: 4 -> 6`
  - `jobs -0.00038`
  - `congressional -0.00207`
  - `forums +0.00238`
  - `microblog +0.00057`

这说明当前参数张力最强的地方不是 `jobs`，而是：

- 一旦朝有利于 `forums` / `microblog` 的方向调
- `congressional` 往往先掉

### 3.3 纯全局调参已经逼近收益上限

从结果形态上看，这两轮调参已经把几个主要方向都摸过了：

- 长度约束
- `genericity`
- `redundancy`
- `support` 几何
- `reference` 邻域
- `density / novelty` 平衡

但最后最好的结果仍然是：

- `microblog` 可以被拉起来，甚至超过 `PrE-Text`
- `forums` 可以被改善，但仍未稳定超过 `PrE-Text`
- 而且一旦弱集改善，强集里的 `congressional` 常常回落

因此，下一步不应再把主要精力放在“大范围全局调参”上，而应开始转向结构层修改。

## 4. 哪些参数方向是有效的

这里的“有效”指的是：对 `forums` / `microblog` 至少有清晰正向信号。

### 4.1 `length_lambda` 降低是第一类有效方向

实验：

- `A2: length_lambda 0.20 -> 0.10`

结果：

- 相对 `NS-S-*`
  - `jobs +0.00341`
  - `congressional -0.00052`
  - `forums +0.00187`
  - `microblog +0.00217`

解释：

- 这是这轮里最接近“跨数据集稳健提升”的单参数结果之一
- 它同时改善了两个弱集
- 对 `jobs` 还有增益
- 对 `congressional` 只有很小回落

判断：

- `length_lambda` 当前确实偏高
- 短文本与非正式文本在现有 `importance prior` 中被轻微压制

结论：

- `length_lambda` 是值得保留到下一版的参数方向

### 4.2 降低 `lambda_generic` 对 `microblog` 明显有效

实验：

- `B1: lambda_generic 0.35 -> 0.30`
- `B2: lambda_generic 0.35 -> 0.25`

结果：

- `B1` 相对 `NS-S-*`
  - `jobs +0.00442`
  - `congressional -0.00564`
  - `forums +0.00039`
  - `microblog +0.00408`

- `B2` 相对 `NS-S-*`
  - `jobs +0.00259`
  - `congressional -0.00499`
  - `forums +0.00000`
  - `microblog +0.00344`

解释：

- `lambda_generic` 的确压制了 `microblog`
- 对 `forums` 只有很弱帮助
- 但代价是 `congressional` 明显退化

判断：

- `genericity penalty` 的方向判断是对的
- 但“全局统一减弱 genericity”不是最终答案

结论：

- `genericity` 需要继续研究
- 但下一步不应再做全局降低，而应考虑条件化或自适应化

### 4.3 增大 `reference_top_k` 是改善 `forums` 的最有效方向

实验：

- `E4: reference_top_k 4 -> 6`

结果：

- 相对 `NS-S-*`
  - `jobs -0.00038`
  - `congressional -0.00207`
  - `forums +0.00238`
  - `microblog +0.00057`

解释：

- 这是 `forums` 上最有效的参数改动
- 说明当前 `genericity` 的参考邻域太窄
- 放宽参考邻域后，`forums` 中一些原本被误罚的候选被保留下来了

限制：

- 虽然 `forums` 被明显改善
- 但 `congressional` 同时下滑
- 而且 `forums` 最终仍没超过 `PrE-Text`

结论：

- `reference_top_k` 方向是有效的
- 但仍然提示下一步应走“结构化改 genericity”，不是继续纯全局调参

### 4.4 `density / novelty` 重平衡是第二个比较稳的方向

实验：

- `E5: density_lambda 0.50 -> 0.45`, `novelty_lambda 0.30 -> 0.35`

结果：

- 相对 `NS-S-*`
  - `jobs +0.00335`
  - `congressional -0.00149`
  - `forums +0.00103`
  - `microblog +0.00236`

解释：

- 它和 `A2` 一样，都属于“两个弱集都提升，`jobs` 也提升，`congressional` 小幅回落”
- 这说明当前 `importance prior` 里“密度偏好过强”的判断是成立的

结论：

- `density / novelty` 是有效方向
- 值得保留为下一版默认倾向之一

## 5. 哪些参数方向无效或不值得继续深挖

### 5.1 `length_floor: 12 -> 8` 不值得继续

实验：

- `A1`

结果：

- `forums` 变差
- `congressional` 变差
- `microblog` 没提升

结论：

- 不是 `length_floor` 的阈值问题
- 更可能是长度项整体权重问题，也就是 `length_lambda`

### 5.2 `lambda_redundancy: 0.25 -> 0.35` 基本无效

实验：

- `C1`

结果：

- 对 `forums`、`microblog` 都几乎没有帮助
- 只是维持了 `jobs` / `congressional`

结论：

- 当前弱集问题不是简单的“冗余惩罚不够”
- 至少不是靠全局增大 `lambda_redundancy` 就能解决

### 5.3 `top_q: 4 -> 3` 明显不值得继续

实验：

- `E1`

结果：

- 四个数据集全部变差或近似变差

结论：

- 当前 `Top-Q` 扩散投票不是主要矛盾

### 5.4 `private_knn_k: 8 -> 6` 风险过大

实验：

- `E3`

结果：

- `microblog` 有提升
- 但 `jobs` 和 `congressional` 都下降
- `forums` 还略降

结论：

- 不能作为下一步主方向

## 6. 有没有“最接近跨数据集稳健”的参数组

有，但要注意，“最接近”不等于“已经够好”。

### 6.1 `A2` 是第一轮最平衡的组

- `length_lambda 0.20 -> 0.10`

优点：

- `forums` 提升
- `microblog` 提升
- `jobs` 提升

缺点：

- `congressional` 仍有轻微下降

判断：

- 这是目前最像“默认安全方向”的单参数组

### 6.2 `E5` 是第二轮最平衡的组

- `density 0.50 -> 0.45`
- `novelty 0.30 -> 0.35`

优点：

- 两个弱集都提升
- `jobs` 也提升

缺点：

- `congressional` 仍下降

判断：

- 它说明 `importance prior` 的平衡确实还可以继续往“少一点密度、多一点新颖性”方向调

### 6.3 但没有任何组达到真正的 Pareto 改进

如果按严格标准看：

- `forums` 和 `microblog` 同时提升
- `jobs` 和 `congressional` 同时不下降

那么本轮没有任何参数组满足。

这点非常关键，因为它决定了下一步策略：

> 问题已经不再是继续搜全局参数，而是这套全局静态参数形式本身不够表达当前跨数据集需求。

## 7. 下一步建议：开始修改算法结构，而不是继续大范围全局调参

我的判断是：

> 下一步应该开始改 `Stage 1` 机制结构，而不是再继续大范围全局参数搜索。

原因有三条：

### 7.1 证据已经显示“全局参数共享”本身是冲突源

同一组参数要同时服务：

- `jobs` / `congressional` 这种更规整、更正式的数据
- `forums` / `microblog` 这种更松散、更长尾的数据

现在结果已经表明，很多参数一旦向弱集方向调，`congressional` 就先掉。

这说明问题不是“参数值还差一点”，而是：

- 不同数据分布需要不同的惩罚强度或不同的参考方式

### 7.2 参数空间的主方向已经被摸过

你这两轮其实已经把最主要的可解释参数都试过了。

继续第三轮全局调参，大概率只会得到：

- 更细的局部波动
- 更复杂的 trade-off
- 而不是一个能让四个数据集一起变好的统一设置

### 7.3 当前已经能定位到结构问题大概在哪

最值得改结构的不是整个 selector，而是这两块：

1. `genericity penalty`
现在最像的问题是：
- 对不同数据分布使用了同一种全局惩罚形式
- 参考邻域太固定
- 对非正式文本容易误罚

2. `importance prior`
现在最像的问题是：
- 密度偏好在正式数据集上有利
- 但在长尾分布上不够灵活

## 8. 推荐的下一版结构修改方向

### 8.1 优先改 `genericity`，不是继续全局降 `lambda_generic`

推荐方向：

- 不再全局下调 `lambda_generic`
- 改成条件化 `genericity penalty`
- 或改成更宽、更平滑的参考邻域机制

最直接的结构思路是：

- 保留 `reference_top_k` 放宽这个方向
- 但不要全数据集统一用更大的 `reference_top_k`
- 而是让 `genericity` 对候选分布更分段地生效

一句话说：

- 不是“把 genericity 变弱”
- 而是“让 genericity 少误伤非正式文本”

### 8.2 次优先改 `importance prior`

推荐方向：

- 保留 `A2` 和 `E5` 的经验
- 让 `length` 与 `density / novelty` 的权重更柔性，而不是固定常数

如果下一版不想大改，至少可以把下一版默认起点设成：

- `length_lambda = 0.10`
- `density_lambda = 0.45`
- `novelty_lambda = 0.35`

但这更适合作为“结构修改后的默认起点”，不建议再把它当成最终解法。

### 8.3 `support` 主干暂时不要优先动

因为：

- `top_q` 明显无效
- `rank_weights` 也没有给出强信号
- `support` 主干不是当前最值得投入的主矛盾

## 9. 最终建议

如果你要我给出一个明确的执行结论，我的建议是：

### 9.1 不再做大范围第三轮全局参数搜索

原因：

- 本轮已经证明没有全局参数能让四个数据集同步上升
- 继续大范围调参，投入产出比会很低

### 9.2 开始改算法结构

优先顺序：

1. 改 `genericity penalty` 的作用形式
2. 再改 `importance prior` 的平衡方式
3. `support` 主干最后再看

### 9.3 如果一定要保留一个“下一版默认参数起点”

我会选这个起点，而不是继续盲搜：

- `length_lambda = 0.10`
- `density_lambda = 0.45`
- `novelty_lambda = 0.35`
- `reference_top_k = 6`
- `lambda_generic` 先不要继续全局降低到 `0.25`

原因：

- `A2`、`E4`、`E5` 是目前最有解释力的三个正向方向
- 它们分别对应：
  - 减轻长度约束
  - 放宽 `genericity` 参考邻域
  - 减弱密度偏置、增强新颖性

但要再次强调：

- 这组只能当“结构修改后的起始点”
- 不能把它当成已经验证完毕的最终全局最优参数

## 10. 一句话总结

本次两轮调参的结论不是“还没调到位”，而是：

> 纯全局静态参数已经很难同时兼顾 `jobs/congressional` 和 `forums/microblog`；下一步应该开始修改 `Stage 1` 的 `genericity` 与 `importance prior` 结构，而不是继续做大范围全局参数搜索。
