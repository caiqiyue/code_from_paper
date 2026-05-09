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
3. 另一类更短、更集中、更结构化的 private distribution 更适合较小、较紧凑的 budget 路径。

### 2.3 结论

最终版方法**不应再追求“所有数据共用一个完全统一的 budget argmax 公式”**，而应追求：

> 一个统一主框架下的层级式预算决策：先仅根据 private train split 上的可观测分布统计识别 regime，再在该 regime 下运行对应的 budget calibration。

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

1. `median length`
2. `p75 length`
3. `tail ratio`（如 `len >= 350` 的比例）
4. `short ratio`（如 `len <= 120` 的比例）
5. `dispersion`（优先 `IQR`，必要时补 `std`）

记为：

```text
d(x) = [median_len, p75_len, tail_ratio, short_ratio, iqr_len]
```

这些统计量**只允许从 private training split 中提取**，不能看 eval 数据，也不能看 downstream 结果。

设计原则：

- 不训练复杂模型；
- 只使用容易解释、容易复现实验、容易写进论文的统计量；
- 目的不是精确预测最佳 `k`，而是识别 budget regime。
- router 使用的是连续 shape score，而不是单条硬编码长度规则。

### 5.3 新增模块二：Regime Router

router 的输出不是具体预算，而是要激活哪种 `budget policy`。

最终版 router 建议保留 **两个主 regime + 一个 uncertainty fallback**：

1. `compact-structured regime`
2. `broad-tail regime`
3. `uncertain regime`

router 不直接使用数据集名字，而是使用**固定等权**的连续 shape score：

```text
z_shape = z(median_len)
        + z(p75_len)
        + z(iqr_len)
        + tail_ratio
        - short_ratio
```

其中：

- `z(*)` 表示固定 screening-scale z-normalization；
- 不再学习特征权重，默认全部等权；
- 只拟合 `tau_center` 与 `delta_router` 两个 router 参数；
- 一旦开发阶段确定，就在正式 quick comparison 与 final regression 中冻结，不再按目标数据集单独重调。

router 规则为：

```text
if z_shape >= tau_center + delta_router:
    regime = broad-tail
elif z_shape <= tau_center - delta_router:
    regime = compact-structured
else:
    regime = uncertain
```

这样做的意义是：

- 避免“单一长度阈值 = 单一 regime”的硬编码观感；
- 允许 `jobs / microblog` 落入 uncertainty 区间，由统一 fallback resolver 处理；
- 使 router 具备可比较的连续基线形式，后续可直接与简单 length-family rule 做实验对照。

### 5.4 新增模块三：Policy-Conditioned Budget Resolver

最终版不建议再做完全自由的统一 global search，而应做“受 policy 约束的 budget 解析”。

首先固定候选 budget 集合：

```text
K = [18, 19, 20, 21, 22]
```

对任意 `k in K`，都运行一次 budget-conditioned greedy selection，并计算：

```text
M(k) = {
  support_mean(k),
  genericity_mean(k),
  redundancy_mean(k),
  coverage_mean(k),
  coverage_p25(k),
  coverage_min(k),
  budget_cost(k)
}
```

也就是说：

- candidate generation、private support、genericity 是 `k` 无关的前置计算；
- `dynamic redundancy`、最终 greedy selection、boundary-aware negatives 是 `k` 相关的后置计算。

#### A. Coverage-Preserving Policy

适用：`broad-tail regime`

核心原则：

1. 避免 budget 被压得过小；
2. coverage sufficiency 优先于 compactness；
3. 只在高预算带内解析最终 `k`；
4. tie-break 在近边界时偏向更大 budget。

形式化定义建议为：

```text
K_broad = [21, 22]

feasible_broad(k) =
    coverage_p25(k) >= tau_p25_broad * max_j coverage_p25(j)
and coverage_mean(k) >= tau_mean_broad * max_j coverage_mean(j)
```

解析规则：

1. 仅在 `K_broad` 内评估；
2. 若 `feasible_broad` 非空，则按以下字典序排序：
   - 更高 `coverage_p25(k)`
   - 更高 `coverage_mean(k)`
   - 更高 `support_mean(k)`
   - 更大 `k`
3. 若 top-2 的 `coverage_p25` 与 `coverage_mean` 均在 `epsilon_broad` 内，则优先更大 `k`；
4. 若 `feasible_broad` 为空，则回退到 global constrained fallback。

#### B. Compactness-Aware Policy

适用：`compact-structured regime`

核心原则：

1. 避免无效扩种；
2. 小而精的 budget 优先；
3. coverage 只作为最低充分性 guard；
4. tie-break 在近边界时偏向更小 budget。

形式化定义建议为：

```text
K_compact = [18, 19, 20]

feasible_compact(k) =
    coverage_p25(k) >= tau_p25_compact * max_j coverage_p25(j)

U_compact(k) =
    b1 * norm(support_mean(k))
  - b2 * norm(genericity_mean(k))
  - b3 * norm(redundancy_mean(k))
  - b4 * norm(budget_cost(k))
```

解析规则：

1. 仅在 `K_compact` 内评估；
2. 若 `feasible_compact` 非空，则选 `argmax U_compact(k)`；
3. 若 top-2 utility gap `<= epsilon_compact`，优先更小 `k`；
4. 若 `feasible_compact` 为空，则回退到 global constrained fallback。

#### C. Uncertain Fallback Policy

适用：`uncertain regime`

形式化定义：

```text
K_uncertain = K
```

解析规则：

1. 在全部 `K` 上运行已有 `self_calibrated_constrained` 风格的 global fallback；
2. 使用统一 coverage constraint + feasible-set utility；
3. tie-break 仍采用 conservative smaller-budget preference。

这一层的作用是：

- 避免 router 对边界型数据做过强判断；
- 给 `jobs / microblog` 这类中间态分布保留统一 fallback 路径；
- 防止最终方法被解释成“所有数据都必须先被硬分成两类”。

### 5.5 参数拟合与冻结协议

为了避免“参数太多 + 数据集太少”的过拟合风险，最终版只允许拟合最少量的 router 参数：

1. `tau_center`
2. `delta_router`

其余部分全部固定：

1. `z_shape` 的特征权重固定为等权
2. `K = [18, 19, 20, 21, 22]` 固定
3. `K_broad = [21, 22]` 固定
4. `K_compact = [18, 19, 20]` 固定
5. `tau_p25_broad / tau_mean_broad / tau_p25_compact` 固定为开发期一次性选定的全局值
6. `U_compact` 的系数固定为沿用当前 constrained utility 默认值：

```text
support_weight = 1.0
genericity_weight = 0.5
redundancy_weight = 0.3
budget_weight = 0.1
```

router 参数的开发协议必须写死为：

```text
目标：
    maximize mean seed-level improvement over the fixed PrE-Text screening baseline
约束：
    worst-dataset improvement >= 0
搜索：
    leave-one-dataset-out development fitting on historical screening results
平局：
    prefer smaller delta_router, then simpler boundary placement
```

也就是说，最终版不允许针对某个目标数据集单独再调 router。

---

## 6. 推荐的统一算法流程

最终版算法建议写成如下统一步骤：

1. 从 `D_init` 构造 prompts，生成 Stage-1 candidates
2. 计算 private sample embeddings
3. 计算 `importance prior`
4. 计算 `Top-Q weighted private support`
5. 计算 `genericity penalty`
6. 仅从 private train split 计算 `shape descriptor`
7. 由 `shape descriptor` 计算 `z_shape` 并输出 `regime label`
8. 根据 `regime label` 确定 active budget set 与 active policy
9. 对 active budget set 中每个 `k` 运行一次 budget-conditioned greedy selection
10. 计算 `M(k)` 与 policy-specific objective
11. 在该 policy 下解析 `resolved_seed_top_k`
12. 若 policy 内无可行解，则回退到 global constrained fallback
13. 用最终 budget 的 greedy decision 输出 `selected seeds / hard negatives / boundary state`
14. 进入保持不变的 Stage-2 bootstrap
15. 生成 synthetic corpus 并进行下游统一评测

关键包装原则：

- 不写成“某数据集对应某个固定 `k`”；
- 写成“某类 private distribution 触发某类 budget policy”；
- 最终 `k` 是 policy-conditioned resolver 的产物。

可直接写入论文的方法伪代码为：

```text
Input: private train split D_priv, public init pool D_init, candidate budget set K
Output: selected seeds S

1. Generate candidate set C from D_init
2. Compute private support s(c), genericity g(c) for each c in C
3. Compute descriptor d(D_priv) and shape score z_shape
4. Route to regime r = g_router(d)
5. Choose active budget subset K_r and policy P_r
6. For each k in K_r:
     Run greedy selector with budget k
     Obtain M(k)
7. Resolve k* = argmax under P_r with tie-break and feasibility guards
8. If P_r has no feasible solution, run global constrained fallback on K
9. Return the greedy selection result associated with final k*
```

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

### 8.0 Reproducibility 与公平性协议

以下协议在 quick comparison 和 final regression 中固定：

1. 数据口径：`train_limit = 256`, `eval_limit = 256`, `initialization_limit = 1024`
2. Stage 2：`bootstrap.num_prompts = 100`, `bootstrap.max_tokens = 85`
3. 下游评测：`gpt2 small eval`, `epochs = 6`
4. 候选 budget：固定 `K = [18, 19, 20, 21, 22]`
5. router descriptor 仅能使用 private train split
6. quick comparison seed 集合：`{42}`
7. final robustness seed 集合：推荐 `N = 5`，例如 `{42, 123, 456, 789, 1011}`
8. 主统计：`mean / std / min / max`
9. 不确定性：报告 95% bootstrap CI 或 paired seed-level improvement CI
10. 基线公平性：`PrE-Text` 不在 quick comparison 中重新调参，只使用已经固定口径的 screening baseline；可在附录补充均值基线对照

### 8.1 第一层：主结果实验

目标：

- 最终版 4 数据集统一超过 `PrE-Text`
- 最好多 seed 重复，证明不是单次偶然

建议输出：

1. 主结果表：`jobs / congressional / forums / microblog`
2. 多 seed 稳健性：`mean / std / min / max`
3. 与 `PrE-Text` 最差基准对比
4. 可选：与 `PrE-Text` 的均值基准对比

主结果的主判断不应再写成“追某个历史最好点”，而应写成：

- 四数据集的 seed-level mean improvement 全部高于 `PrE-Text` screening 基准下界
- 四数据集的 worst-seed result 尽量不低于基准
- broad-tail family 的最终预算应稳定落在高预算带
- compact-structured family 的最终预算应稳定落在低预算带

历史上的 `0.2514`、`0.2950+` 等数值，只保留为 retrospective sanity check，而不是 primary success criterion。

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

证明 `distribution-shape routing` 是有效且合理的，而不是把 length-threshold 改名。

建议实验：

1. 展示四个数据集的 shape descriptor 与 `z_shape`
2. 展示 router 的 regime 输出
3. 展示不同 regime 下 budget 分布差异
4. 与简单 `length-family` 阈值路由做对照
5. 做 leave-one-dataset-out routing validation
6. 做“错路由”实验：
   - broad-tail family 强行走 compact policy
   - compact-structured family 强行走 broad-tail policy

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

1. broad-tail guard：验证 broad-tail policy 是否稳定回到高预算区间，并优于统一 fallback
2. compact guard：验证 compactness-aware policy 是否稳定落在小预算区间，并优于统一 fallback

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
2. broad-tail family 的高预算优势被稳定保留
3. compact-structured family 的小预算优势被稳定保留
4. 层级式 routing + policy-conditioned calibration 的必要性被 ablation 明确证明

主结论使用 protocol-based 表述：

> 在固定评测协议、固定 baseline 口径与多 seed 设置下，最终版方法在四数据集上取得正的平均改进，并且最差数据集不再掉出 `PrE-Text` 基准线。

历史最好点只作为 sanity check：

- broad-tail family 是否回到历史高预算带；
- compact family 是否保持历史小预算带；
- 最终 `k` 分布是否符合 router 设计预期。

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
