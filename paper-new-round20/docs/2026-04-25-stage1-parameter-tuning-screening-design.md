# `Stage 1` 参数调优 screening 设计

更新时间：2026-04-25

## 1. 文档目的

本文档用于固定当前这一轮 `parameter-only screening` 的设计。

这轮工作不修改 `paper-2` / `paper-new` 当前版本创新算法的结构，不改 `Stage 2`，不改 downstream eval，只针对 `Stage 1 selector` 中已有参数做受控微调，判断是否存在一组更稳健的参数，使：

- `forums` / `microblog` 的表现回升；
- 同时 `jobs` / `congressional` 不明显退化；
- 从而让当前创新算法在 4 个数据集上表现得更稳定。

这轮实验的研究目标不是“重新设计新算法”，而是优先回答一个更现实的问题：

> 在不改算法结构的前提下，是否可以仅通过小幅调整 `Stage 1` 参数，得到跨数据集更稳健的结果。

---

## 2. 当前问题背景

根据当前 `screening-balanced` 结果：

- `jobs`：创新算法优于 `PrE-Text`
- `congressional`：创新算法优于 `PrE-Text`
- `forums`：创新算法劣于 `PrE-Text`
- `microblog`：创新算法在 `best_top1` 上仍略低于 `PrE-Text`

这说明当前版本不是“完全无效”，但也没有达到“已经稳定优于基线”的程度。

因此，下一步最合理的动作不是直接扩大参数进入 formal experiment，而是先做一轮更保守的 `Stage 1` 参数诊断：

- 先看能否通过简单微调参数修复弱数据集；
- 同时验证这些调整会不会破坏 `jobs` / `congressional` 上已有的优势；
- 若参数层已经能形成更稳健的趋势，再决定是否继续放大或改算法结构。

---

## 3. 本轮调参的目标与边界

### 3.1 目标

本轮调参的核心目标定义为：

> `forums` / `microblog` 要提升，但 `jobs` / `congressional` 不能明显变差。

这不是“优先救弱数据集”的单边目标，而是“跨数据集稳健”的目标。

### 3.2 边界

本轮实验必须满足以下约束：

- 不改算法结构；
- 不改 `Stage 2` bootstrap；
- 不改 downstream eval；
- 只改 `Stage 1 selector` 已有参数；
- 第一轮只优先调低回归风险参数；
- 不做大规模参数网格搜索；
- 不用这一轮结果直接替代 formal experiment。

### 3.3 第一轮优先调参范围

第一轮只聚焦以下参数：

- `length_floor`
- `length_lambda`
- `lambda_generic`
- `lambda_redundancy`

这些参数之所以优先，是因为它们更可能解释当前 `forums` / `microblog` 的劣势，同时对 `jobs` / `congressional` 的破坏风险相对更低。

### 3.4 第一轮暂不调整的参数

以下参数暂不纳入第一轮：

- `top_q`
- `rank_weights`
- `private_knn_k`
- `reference_top_k`
- `density_lambda`
- `novelty_lambda`

原因是这些参数会更直接地改变 `private_support` 或 `genericity_penalty` 的几何结构，影响面更大，更容易把当前在正式文本数据集上的已有优势一起打掉。

---

## 4. 参数实验矩阵

本轮不采用全组合网格搜索，而采用“单因素为主、少量组合兜底”的方式。

保留当前 screening 基线参数不变，作为所有新组的唯一对照。

在基线之外，新增 6 组参数实验。

### 4.1 基线组

- 使用当前 `single_node_screening` 默认参数；
- 不做任何额外调整；
- 作用是为所有新参数组提供统一对照。

### 4.2 新增参数组

#### A1 `length_floor_8`

- 修改：
  - `length_floor: 12 -> 8`
- 目的：
  - 减轻短文本样本在 `importance prior` 中被系统性降权的程度；
  - 优先观察 `microblog` 是否因此回升。

#### A2 `length_lambda_010`

- 修改：
  - `length_lambda: 0.20 -> 0.10`
- 目的：
  - 降低长度稳定性约束在私有样本重要性中的占比；
  - 验证弱集问题是否来自“对短文本惩罚过强”。

#### B1 `generic_030`

- 修改：
  - `lambda_generic: 0.35 -> 0.30`
- 目的：
  - 小幅减弱 `genericity penalty`；
  - 检验 `forums` / `microblog` 中是否存在被误杀的高价值候选。

#### B2 `generic_025`

- 修改：
  - `lambda_generic: 0.35 -> 0.25`
- 目的：
  - 在 B1 基础上进一步减弱 `genericity penalty`；
  - 观察 `genericity` 对 4 个数据集的敏感性曲线。

#### C1 `redundancy_035`

- 修改：
  - `lambda_redundancy: 0.25 -> 0.35`
- 目的：
  - 增强 seed 集覆盖约束；
  - 检验 `forums` / `microblog` 的落后是否主要来自 seed 过于中心化。

#### D1 `combo_safe`

- 修改：
  - `length_floor: 12 -> 8`
  - `lambda_generic: 0.35 -> 0.30`
  - `lambda_redundancy: 0.25 -> 0.35`
- 目的：
  - 用一组保守的小组合验证“轻微放松短文本约束 + 轻微放松 genericity + 轻微增强 diversity”是否能形成更稳健的整体收益；
  - 这是本轮唯一的组合组，不用于替代单因素实验，而用于验证三者是否存在协同增益。

---

## 5. 为什么优先选择这几类参数

### 5.1 `length_floor` / `length_lambda`

当前 `importance prior` 中显式使用了长度稳定性项。

这在 `jobs` / `congressional` 这类文本更规整的数据集上通常问题不大，但在 `microblog` 这类天然更短的文本分布上，可能会把一些真实有价值的样本权重压低。

因此，先动长度相关参数是低风险且最有解释力的选择。

### 5.2 `lambda_generic`

当前 `genericity penalty` 的设计目标是抑制过于模板化、过于接近初始化分布的候选。

但对于 `forums` / `microblog`，很多表面上更“普通”或更“松散”的表达，可能恰恰是任务分布中的正常模式，而不是应被压制的坏候选。

因此，小幅下调 `lambda_generic` 有望回补弱数据集，同时回归风险低于直接修改 `reference_top_k` 或 penalty 结构。

### 5.3 `lambda_redundancy`

如果当前问题主要来自 seed 集过于中心化、覆盖不足，那么适度提高 `lambda_redundancy` 有机会帮助 `forums` / `microblog`，且不会像直接动 `support` 主干那样破坏整个排序逻辑。

因此，`lambda_redundancy` 适合作为第一轮的 diversity 补偿项。

---

## 6. 本轮判定标准

### 6.1 主指标

本轮主判定指标仍使用：

- `best_top1`

同时必须保留并核查：

- `best_top3`
- `best_top5`
- `best_top10`

### 6.2 跨数据集稳健判定

一组新参数如果满足以下条件，则判定为“值得继续跟进”：

- `forums` / `microblog` 至少有 1 个数据集出现明确回升；
- `jobs` / `congressional` 不出现双双明显退化；
- `top3 / top5 / top10` 没有出现明显的全面恶化。

### 6.3 不通过的情形

以下情况判定为不通过：

- 虽然 `forums` / `microblog` 有提升，但 `jobs` / `congressional` 双双明显变差；
- 仅主指标略高，但 `top3 / top5 / top10` 全面退化；
- 4 个数据集结果整体接近基线，没有形成清晰趋势。

### 6.4 结果解释原则

本轮 screening 的目标是判断“参数调整方向是否值得继续投入”，而不是直接给出正式论文级结论。

因此，结果解释必须聚焦：

- 哪一类参数更可能对应当前机制问题；
- 哪一类调整更容易带来稳健提升；
- 哪一类调整虽能补弱集，但会明显伤强集；
- 下一轮是否仍值得继续停留在参数层，还是需要进入结构层修改。

---

## 7. 配置落地方案

本轮实验配置按以下原则组织：

- 保留当前 screening 基线配置不动；
- 新建一组参数调优用的 screening 配置；
- 每个参数组分别展开为：
  - `jobs`
  - `congressional`
  - `forums`
  - `microblog`
- 命名中显式带参数组代号，避免结果目录混乱；
- 输出目录同样要与参数组代号一一对应。

建议命名方式：

- `ns_tune_a1_jobs.yaml`
- `ns_tune_a1_congressional.yaml`
- `ns_tune_a1_forums.yaml`
- `ns_tune_a1_microblog.yaml`

其余参数组同理：

- `a2`
- `b1`
- `b2`
- `c1`
- `d1`

这样后续结果目录、日志目录和文档汇总都能直接按参数组归档。

---

## 8. 本轮调参设计的一句话结论

这轮 `Stage 1` 调参 screening 的核心策略不是激进改动，而是：

> 先用低回归风险参数做受控微调，验证是否能在不破坏 `jobs` / `congressional` 的前提下，补回 `forums` / `microblog` 的弱势表现。

如果这一轮已经出现稳定的跨数据集改善，再考虑扩大实验或进入下一层机制改进；如果参数层仍不能形成稳健收益，再转向算法结构诊断与修改。
