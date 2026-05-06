# Hierarchical Distribution-Shape-Aware Seed Budget Selection Design

**日期**: 2026-05-06  
**适用项目**: `paper-new`  
**目标定位**: 形成一个既能稳定超过 `PrE-Text`，又能被包装成正式论文创新点的最终版算法主线

---

## 1. 设计目标

本文档给出当前创新算法的最终收敛方向。核心要求只有两条：

1. **实验结果必须稳定全面超过 `PrE-Text`**
2. **算法必须能被表述为有依据、可解释、可发表的正式创新，而不是零散调参**

在这两个目标中，**结果优先**。也就是说，方法形式上的整洁性必须服从“4 个数据集稳定超过 `PrE-Text`”这一硬标准；但在结果达标的前提下，方法需要被重新组织成一个更强、更完整、更像论文方法的框架。

---

## 2. 当前证据链与核心判断

基于已有实验文档，可以得出以下稳定结论：

### 2.1 已经被充分验证的事实

1. 保留 `PrE-Text` 的两阶段框架、重构 `Stage 1 selector` 是有效方向。
2. `Top-Q weighted support + importance prior + genericity penalty + dynamic redundancy penalty` 这条 `Stage 1` 主线已经证明能显著提升 `jobs / congressional / microblog`，并具备论文可写性。
3. 单纯依赖全局静态参数，无法让 4 个异构数据集同时稳定变好。
4. `seed budget` 不是普通超参数，而是决定四数据集能否统一超过 `PrE-Text` 的核心结构变量。
5. `Round14 / Round15` 已经证明：只要 budget 路径正确，创新算法可以实现四数据集全面超过 `PrE-Text`。
6. `Round16 / 17 / 18` 进一步证明：将 budget 决策完全压缩为一个统一的单层 global optimizer，虽然形式上更“算法化”，但结果不稳定。

### 2.2 已经被精确定位的问题

1. `forums` 当前最优行为已被 `Round18.2` 明确验证：固定 `k=22` 时可以达到 `0.2514`，高于 `PrE-Text 0.2501`。
2. 因此，`forums` 当前的主要问题不是 selector 主体退化，而是**自动 budget 层把它压到了错误的 `k=21`**。
3. `congressional` 则更适合较小、较紧凑的 budget 路径，说明它与 `forums` 的最优 budget principle 本来就不同。

### 2.3 结论

最终版方法**不应再追求“所有数据集共用一个完全统一的 budget argmax 公式”**，而应追求：

> 一个统一主框架下的层级式预算决策：先识别数据分布属于哪类 regime，再在该 regime 下运行对应的 budget calibration。

---

## 3. 最终版方法总表述

最终版算法建议命名为：

> **Hierarchical Distribution-Shape-Aware Seed Budget Selection**

核心表述如下：

> We formulate seed-budget selection as a hierarchical, distribution-shape-aware decision process rather than a single global budget optimization problem. The method first identifies the private-data regime through a lightweight shape descriptor, and then performs regime-conditioned budget calibration before the final Stage-1 seed selection.

中文表述建议为：

> 我们将 seed budget 选择表述为一个层级式、分布形态感知的决策过程，而不是一个对所有数据统一适用的单层全局优化问题。算法首先通过轻量级 private distribution shape descriptor 识别当前数据所属的 budget regime，再在对应 regime 下执行条件化 budget calibration，最终完成 Stage 1 seed selection。

---

## 4. 最终版三条核心创新点

最终论文中的创新点不应再拆成很多细碎技巧，而应明确收敛成 3 条主创新。

### 创新点一：Quality-Aware Stage-1 Seed Selector

保留并正式化当前已经成熟的 `Stage 1` 主体：

1. `Top-Q weighted private support`
2. `importance prior`
3. `genericity penalty`
4. `dynamic redundancy penalty`
5. `boundary-aware negative retention`

这条创新的论文意义是：

- 把 `PrE-Text` 的 `Top-1` 最近邻投票式 seed collection，升级成一个同时考虑代表性、非通用性、非冗余性和边界信息的质量感知选择器。

### 创新点二：Distribution-Shape-Aware Regime Identification

新增一个 lightweight `shape descriptor`，不直接输出具体 `k`，而是识别当前 private data 更接近哪一类 budget regime。

这条创新的论文意义是：

- 承认异构数据的最优 budget principle 并不共享单一规律；
- 先做 regime identification，再做 budget calibration；
- 将原本经验性的长度规则，升级为“分布形态感知的 regime routing”。

### 创新点三：Regime-Conditioned Budget Calibration

在识别出 regime 后，不再对所有数据使用完全相同的 budget selection principle，而是在统一框架内启用不同的 calibration path。

这条创新的论文意义是：

- broad-tail / coverage-sensitive 数据需要 coverage-preserving budget path；
- compact-structured 数据需要 compactness-aware budget path；
- budget 选择不再是一个统一 global argmax，而是一个条件化内部模型选择问题。

---

## 5. 最终版算法结构

### 5.1 保持不变的部分

以下部分建议不再大改，作为最终版稳定主干：

1. `PrE-Text` 两阶段总体框架
2. Stage 2 bootstrap 结构
3. `Top-Q support`
4. `importance prior`
5. `genericity penalty`
6. `dynamic redundancy penalty`
7. `boundary-aware negatives`

原因很明确：这部分已经被大量实验验证，继续大改只会放大风险，而不会提升论文主线清晰度。

### 5.2 新增模块一：Shape Descriptor

从 private subset 中提取一个轻量级 `distribution shape descriptor`。推荐初始特征集合为：

1. `mean length`
2. `median length`
3. `p75 length`
4. `tail ratio`（如超过某阈值的比例）
5. `dispersion`（如 `IQR` 或 `std`）

设计原则：

- 不训练复杂模型；
- 只使用容易解释、容易复现实验、容易写进论文的统计量；
- 目的不是精确预测最佳 `k`，而是识别 budget regime。

### 5.3 新增模块二：Regime Router

router 的输出不是具体预算，而是要激活哪种 `budget policy`。

建议先只保留两个主 regime：

1. `compact-structured regime`
2. `broad-tail regime`

原因：

- 当前最强证据正是 `congressional-like` 与 `forums-like` 的两类差异；
- 先保持 regime 数量最小，能显著降低工程与叙事复杂度；
- `jobs / microblog` 可以通过 policy 内部 calibration 或 conservative fallback 自然落位。

### 5.4 新增模块三：Policy-Conditioned Budget Resolver

最终版不建议再做完全自由的统一 global search，而应做“受 policy 约束的 budget 解析”。

#### A. Coverage-Preserving Policy

适用：`broad-tail regime`

核心原则：

1. 避免 budget 被压得过小；
2. coverage sufficiency 优先于 compactness；
3. 允许预算落在已被实验验证有效的较大 budget 区间；
4. 在高预算候选中再做 conservative resolution。

这条 policy 的目标，是把当前 `forums` 上已经验证有效的高预算行为正式纳入统一框架。

#### B. Compactness-Aware Policy

适用：`compact-structured regime`

核心原则：

1. 避免无效扩种；
2. 小而精的 budget 优先；
3. 保持 constrained / utility 风格的 calibration；
4. 在 coverage 不区分 budgets 时，允许 compactness 主导决策。

这条 policy 的目标，是保留当前 `congressional` 上更稳定的小预算路径。

---

## 6. 推荐的统一算法流程

最终版算法建议写成如下统一步骤：

1. 从 `D_init` 构造 prompts，生成 Stage-1 candidates
2. 计算 private sample embeddings
3. 计算 `importance prior`
4. 计算 `Top-Q weighted private support`
5. 计算 `genericity penalty`
6. 计算 `dynamic redundancy penalty`
7. 从 private subset 计算 `shape descriptor`
8. 由 `shape descriptor` 输出 `regime label`
9. 根据 `regime label` 激活对应的 `budget policy`
10. 在该 policy 下解析 `resolved_seed_top_k`
11. 用解析出的 budget 完成 greedy seed selection
12. 输出 `selected seeds / hard negatives / boundary state`
13. 进入保持不变的 Stage-2 bootstrap
14. 生成 synthetic corpus 并进行下游统一评测

关键包装原则：

- 不写成“某数据集对应某个固定 `k`”；
- 写成“某类 private distribution 触发某类 budget policy”；
- 最终 `k` 是 policy-conditioned resolver 的产物。

---

## 7. 论文写法中的关键措辞

最终论文里最重要的是避免把方法写成 heuristic patchwork。

### 7.1 不建议的写法

- `if forums-like then k=22`
- `if mean_len > ... then choose large budget`
- `我们针对每个数据集设置不同规则`

这些写法会显著削弱创新性。

### 7.2 建议的写法

- `distribution-shape-aware regime identification`
- `hierarchical seed-budget selection`
- `regime-conditioned budget calibration`
- `coverage-preserving budget path`
- `compactness-aware budget path`
- `policy-conditioned internal model selection`

也就是说，底层可以保留已有成功行为，但论文表达一定要提升到“机制激活某路径”，而不是“规则直接指定答案”。

---

## 8. 实验设计总规划

为了同时证明“结果有效”和“创新点成立”，实验必须分成 4 层证据。

### 8.1 第一层：主结果实验

目标：

- 最终版 4 数据集统一超过 `PrE-Text`
- 最好多 seed 重复，证明不是单次偶然

建议输出：

1. 主结果表：`jobs / congressional / forums / microblog`
2. 多 seed 稳健性：`mean / std / min / max`
3. 与 `PrE-Text` 最差基准对比
4. 可选：与 `PrE-Text` 的均值基准对比

成功标准：

- 四数据集全部高于 `PrE-Text` 基准下界
- 最好在 `forums` 上保持 `>= 0.2514` 级别

### 8.2 第二层：机制消融实验

目标：

证明 hierarchical 结构是必要的，而不是文档包装。

至少做 3 组对照：

1. 去掉 routing，只保留统一 calibration
2. 保留 routing，但强行让两类 regime 共用同一 policy
3. 完整版 hierarchical 方法

想要看到的结果：

- 单层统一机制在某些数据集掉线；
- 条件化 policy 能修复掉线；
- 说明两层结构是必要机制，而不是装饰。

### 8.3 第三层：Regime Evidence 实验

目标：

证明 `distribution-shape routing` 是有效且合理的。

建议实验：

1. 展示四个数据集的 shape descriptor
2. 展示 router 的 regime 输出
3. 展示不同 regime 下 budget 分布差异
4. 做“错路由”实验：
   - 让 `forums-like` 强行走 compact policy
   - 让 `congressional-like` 强行走 broad-tail policy

如果错路由显著掉分，证据会非常强。

### 8.4 第四层：历史对照实验

目标：

证明最终版不是脱离历史链条的突兀新方案，而是对已有成功线的正式收敛。

建议对照：

1. `PrE-Text`
2. 初始创新算法
3. `Round15`
4. `Round17`
5. `Round18.2 fixed22 forums`
6. 最终版

这会帮助导师看到：最终版不是随意拍脑袋出来的，而是建立在前面所有实验排查基础上的结构性收敛。

---

## 9. 文献依据映射

最终版论文不应声称“完全从某一篇论文直接导出”，而应说明每个模块分别借鉴了哪些成熟思想，并在 `PrE-Text seed-budget selection` 任务上完成了新的组合与问题化。

### 9.1 Selector 主体对应文献线

1. `Top-Q weighted support`
   - weighted kNN
   - local adaptive neighbor weighting

2. `importance prior`
   - active learning 中的 representativeness / density weighting

3. `genericity + redundancy`
   - MMR
   - submodular subset selection
   - diversity-aware data selection

### 9.2 Budget 路径对应文献线

1. hierarchical decision / mixture-of-regimes
2. constrained internal model selection
3. budgeted selection / content-budget disentanglement
4. heterogeneous data / regime-aware routing

### 9.3 建议的论文叙事方式

推荐这样表达：

> 本文不是从零提出一个与既有工作完全断裂的新算法，而是在 `PrE-Text` 两阶段框架上，对 `Stage 1 seed selection` 进行系统性重构，并进一步提出一个分布形态感知的层级式 budget selection 框架，以适配异构 private data 的不同 budget regime。

这样既诚实，又能体现实质创新。

---

## 10. 接下来最该做的实验与开发顺序

### 10.1 第一阶段：把最终版方法原型化

优先开发 3 个新模块：

1. `shape descriptor`
2. `regime router`
3. `policy-conditioned budget resolver`

不要继续横向发散更多 recheck / threshold patch。

### 10.2 第二阶段：先做最小闭环验证

建议先跑以下最关键实验：

1. `forums`：验证 broad-tail policy 是否稳定回到高预算区间，并复现 `>= 0.2514`
2. `congressional`：验证 compactness-aware policy 是否稳定保持高于 `0.2950`

这两点先过，再做四数据集统一回归。

### 10.3 第三阶段：四数据集统一回归

在 `forums` 与 `congressional` 两个关键点都成功后，再跑：

1. `jobs`
2. `congressional`
3. `forums`
4. `microblog`

若 4/4 全过，即进入稳定主线。

### 10.4 第四阶段：补机制消融与错路由实验

主结果一旦成立，优先补：

1. 去 routing 的 ablation
2. 去 policy 分流的 ablation
3. 错路由实验
4. 多 seed 稳健性

这一步是保证论文可发性的关键，而不是可选项。

---

## 11. 最终版方法的成功标准

最终版方法是否成立，不看它是否“更复杂”，只看以下 4 条是否同时满足：

1. 四数据集统一超过 `PrE-Text`
2. `forums` 的高预算优势被稳定保留
3. `congressional` 的小预算优势被稳定保留
4. 层级式 routing + policy-conditioned calibration 的必要性被 ablation 明确证明

如果只满足第 1 条而没有第 4 条，结果能用，但论文创新性偏弱。  
如果只满足第 4 条而没有第 1 条，叙事好看，但不符合当前项目目标。  
因此最终版必须是：

> **结果稳定 + 结构可解释 + 消融可证明 + 文献可映射**

---

## 12. 一句话总结

最终版创新算法不应再被推进为“一个统一公式自动算出所有数据集的最优 budget”，而应被正式收敛为：

> **在保留质量感知 Stage-1 selector 的前提下，引入 distribution-shape-aware regime identification，并在不同 regime 下执行 policy-conditioned budget calibration 的层级式 seed selection 框架。**

这条路线最符合现有实验事实，也最有希望同时满足：

1. 稳定全面超过 `PrE-Text`
2. 向导师展示明确、完整、可解释的创新结构
3. 为论文提供充分的实验支撑与文献依据

