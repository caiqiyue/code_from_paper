# `Stage 1` 参数调优 screening 设计（第二轮）

更新时间：2026-04-25

## 1. 文档目的

本文档用于固定第二轮 `Stage 1 parameter-only screening` 的设计。

第一轮 tuning 主要聚焦低回归风险参数：

- `length_floor`
- `length_lambda`
- `lambda_generic`
- `lambda_redundancy`

第二轮的目的不是替代第一轮，而是在第一轮之外，针对此前暂不调整的一组高影响参数，做一轮更保守的机制诊断：

- `top_q`
- `rank_weights`
- `private_knn_k`
- `reference_top_k`
- `density_lambda`
- `novelty_lambda`

本轮仍然保持以下边界不变：

- 不改算法结构
- 不改 `Stage 2`
- 不改 downstream eval
- 只改 `Stage 1 selector` 已有参数

## 2. 当前问题背景

根据当前 screening 现象，创新算法在不同数据集上的表现并不一致：

- `jobs`：优于 `PrE-Text`
- `congressional`：优于 `PrE-Text`
- `forums`：劣于 `PrE-Text`
- `microblog`：在 `best_top1` 上仍偏弱

这一结果说明当前版本不是“完全无效”，但也还没有达到“跨数据集稳定优于基线”的程度。

第一轮 tuning 已优先测试低风险参数，第二轮则转向此前暂缓的一组高影响参数，目的是回答更具体的问题：

> 当前版本在 `forums` / `microblog` 上的劣势，是否来自 `support` 几何、`genericity` 参考约束，或者 `importance prior` 的密度偏置？

## 3. 第二轮目标与原则

### 3.1 目标

第二轮的主目标仍然定义为：

> `forums` / `microblog` 要提升，同时 `jobs` / `congressional` 不能明显变差。

因此，本轮仍然按 `4` 个数据集全跑，而不是只跑弱数据集。

### 3.2 原则

本轮不采用激进设置，而采用：

- `5` 组单因素轻改
- `1` 组保守组合
- 不引入极端参数
- 不进行大规模网格搜索

这样做的原因是，这一组参数的回归风险明显高于第一轮；如果一开始就上激进改法，结果更可能只说明“参数很危险”，而不是准确定位哪一类机制存在问题。

## 4. 基线参数

当前第二轮关注参数的基线值如下：

- `top_q = 4`
- `rank_weights = [1.0, 0.6, 0.3, 0.15]`
- `private_knn_k = 8`
- `reference_top_k = 4`
- `density_lambda = 0.50`
- `novelty_lambda = 0.30`

## 5. 机制假设

### 5.1 `support` 几何可能过强

如果 `top_q` 较大、低位 `rank_weights` 衰减不够快，那么更多私有样本会把支持票分散给一组“主流候选”。
这会让 `support` 更偏向中心模式，对 `jobs` / `congressional` 可能有利，但可能压制 `forums` / `microblog` 中更分散、更长尾的 seed。

### 5.2 `genericity` 参考约束可能过紧

如果 `reference_top_k` 太小，`genericity penalty` 会更接近局部最近邻参考。
这可能会对更自由、更口语化的数据分布形成误罚，使 `forums` / `microblog` 中本来合理的候选被压低。

### 5.3 `importance prior` 可能偏密度、轻新颖性

当前 `importance prior` 中：

- `density_lambda = 0.50`
- `novelty_lambda = 0.30`

这意味着 `density` 比 `novelty` 更强。
如果这一偏置过强，则 selector 会更容易偏向私有分布中心区，而不利于长尾覆盖。

## 6. 第二轮实验矩阵

本轮新建一组独立的 round-2 tuning 配置，命名为 `ns_tune2_*`。

### 6.1 E1 `topq_3`

- 修改：
  - `top_q: 4 -> 3`
- 目的：
  - 减少每个私有样本向多个候选扩散支持票的程度
  - 检查当前 `Top-Q` 投票是否让中心候选获得了过强支持

### 6.2 E2 `rank_steeper`

- 修改：
  - `rank_weights: [1.0, 0.6, 0.3, 0.15] -> [1.0, 0.45, 0.20, 0.10]`
- 目的：
  - 保留 `Top-1` 支持
  - 更快衰减低位排名候选的贡献
  - 检查弱集问题是否来自低位候选累积支持过强

### 6.3 E3 `knn_6`

- 修改：
  - `private_knn_k: 8 -> 6`
- 目的：
  - 轻微减弱 `importance prior` 对高密度区域的偏好
  - 检查是否是私有分布中心区被系统性放大

### 6.4 E4 `refk_6`

- 修改：
  - `reference_top_k: 4 -> 6`
- 目的：
  - 让 `genericity penalty` 参考更多 public initialization 邻居
  - 检查当前参考集是否过于局部、过于严格

### 6.5 E5 `density_novelty_rebalance`

- 修改：
  - `density_lambda: 0.50 -> 0.45`
  - `novelty_lambda: 0.30 -> 0.35`
- 保持：
  - `length_lambda = 0.20`
- 目的：
  - 在不改长度项的前提下，轻微把 `importance prior` 从“密度优先”拉向“密度 / 新颖性更平衡”

### 6.6 E6 `combo_safe2`

- 修改：
  - `top_q: 4 -> 3`
  - `reference_top_k: 4 -> 6`
  - `density_lambda: 0.50 -> 0.45`
  - `novelty_lambda: 0.30 -> 0.35`
- 目的：
  - 以一组保守小组合验证三条机制线是否存在协同收益
  - 这组组合只用于验证“多处轻改是否比单点轻改更稳”，不用于替代单因素判断

## 7. 本轮不做的事

本轮仍然不做以下操作：

- 不引入激进设置
- 不直接调到极端 `top_q`
- 不大幅重写 `rank_weights`
- 不同时大幅改 `support` 与 `genericity`
- 不做结构性修改，如 cluster-balanced selector 或条件化 penalty

这些内容应在第二轮仍不能形成稳定改进时，再进入后续机制改造阶段。

## 8. 判定标准

### 8.1 主指标

主指标仍然使用：

- `best_top1`

同时必须记录：

- `best_top3`
- `best_top5`
- `best_top10`

### 8.2 值得继续跟进的条件

一组新参数如果满足以下条件，则判定为“值得继续跟进”：

- `forums` / `microblog` 至少有 `1` 个数据集明显提升
- `jobs` / `congressional` 不出现双双明显退化
- `top3 / top5 / top10` 不出现明显全面恶化

### 8.3 不通过条件

以下情况判定为不通过：

- 弱数据集虽有提升，但 `jobs` / `congressional` 明显一起变差
- 主指标略好，但 `top3 / top5 / top10` 全面恶化
- 四个数据集整体接近基线，没有形成可解释趋势

## 9. 配置落地方案

第二轮配置采用独立目录：

- `paper-new/configs/experiments/single_node_tuning_round2`

目录组织方式：

- `1` 个 round-2 base
- `6` 个 group override
- `24` 个 dataset leaf config

命名方式：

- `ns_tune2_e1_jobs.yaml`
- `ns_tune2_e1_congressional.yaml`
- `ns_tune2_e1_forums.yaml`
- `ns_tune2_e1_microblog.yaml`

其余 `e2 / e3 / e4 / e5 / e6` 同理。

结果目录一一对应：

- `paper-new/outputs/ns_tune2_e1_jobs`
- `paper-new/outputs/ns_tune2_e1_congressional`
- ...

## 10. 一句话结论

第二轮 `Stage 1` tuning screening 的核心策略是：

> 不改算法结构，只对此前暂缓的高影响参数做保守轻改，判断当前版本在 `forums` / `microblog` 上的劣势，究竟更接近 `support` 几何问题、`genericity` 约束问题，还是 `importance prior` 的密度偏置问题。
