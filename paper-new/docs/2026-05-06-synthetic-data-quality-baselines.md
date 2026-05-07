# 合成数据质量导向的基线设计说明

## 1. 研究目标与基线选择原则

你的主要研究目标不是隐私强度、通信成本或联邦训练效率本身，而是：

- 使用算法生成的 synthetic data 质量如何；
- synthetic data 是否能有效支持下游任务训练；
- 在相同的下游模型、相同的数据预算、相同或相近的生成约束下，哪种方法生成的数据更有用。

因此，论文中的主结论不应写成：

> 我的方法隐私更强，或者通信更省。

而应写成：

> 在相同约束下，我的方法生成的 synthetic data 对下游任务更有用，能够带来更好的下游性能。

这会直接影响 baseline 的选择逻辑。

如果只围绕 synthetic data quality 展开，那么主对比不应再以 `DP-FedAvg`、`DP-FTRL` 这类“直接训练模型”的方法为主，因为它们的核心目标不是生成高质量 synthetic dataset，而是进行联邦差分隐私训练。它们仍然有参考价值，但更适合作为副对比或补充实验。

主对比应该围绕以下问题展开：

- 谁生成的数据更有用；
- 在同样的下游模型、同样的 synthetic data budget 下，谁带来的效果更好；
- 在相同的 federated / no-finetune 约束下，谁更强。

基于这个目标，推荐把实验对比分成两层：

- 主表：只看 synthetic data quality；
- 副表：补充联邦、隐私、效率类方法，防止审稿人质疑你的方法只在“生成类方法”里有效。

## 2. 推荐的主表基线

### 2.1 主表应比较哪些算法

如果你的论文主线是“我提出了一种基于 PrE-Text 思路的改进算法，目标是提升 synthetic data 的下游有效性”，那么主表建议放以下 6 个方法：

- `c4-only`
- `Expand-only`
- `PrE-Text`
- `WASP`
- `DPGA-TextSyn`
- `Expand-private`

这是最紧凑、最干净、最能服务你论文主问题的一组 baseline。

### 2.2 为什么主表选择这 6 个方法

这 6 个方法几乎都在回答同一个核心问题：

> 生成出来的数据，给到下游模型之后，到底有没有用？

其中：

- `PrE-Text` 是你的直接对标方法，必须放；
- `WASP` 和 `DPGA-TextSyn` 是文本 synthetic data generation 路线中比较适合作为“同赛道方法”比较的算法；
- `c4-only`、`Expand-only`、`Expand-private` 则分别构成下界、public-only 对照和非隐私上界。

这组 baseline 的好处是结构非常清晰：

- 有下界；
- 有上界；
- 有直接对标；
- 有同类型强 baseline；
- 所有方法都能被解释为“它们最终都通过下游任务表现来体现 synthetic data 的质量”。

## 3. 主表中的每个 baseline 分别代表什么

### 3.1 `c4-only`：下界

`c4-only` 表示：

- 完全不使用私有数据；
- 只使用与私有分布无关的公共数据；
- 在这个公共数据上微调下游模型。

它的作用是提供一个明确的下界：

> 如果根本不利用私有分布的信息，下游模型能达到什么水平？

保留 `c4-only` 很重要，因为它可以证明：

- 你的 synthetic data 确实携带了来自私有分布的有效信息；
- 你的方法不是仅靠公共预训练或公共数据就能达到同样效果；
- 合成数据确实在帮助下游模型靠近 private distribution。

如果没有这个 baseline，审稿人可能会问：

> 你所谓的性能提升，是不是只是来自公共数据预训练，而不是 synthetic data 本身？

### 3.2 `Expand-only`：public expand 对照

`Expand-only` 表示：

- 仍然不使用私有数据；
- 先从公共数据中获得初始文本；
- 再通过 expand 机制把公共数据扩展成更大规模的数据集；
- 最终用扩展后的 public-only 数据训练下游模型。

它的作用不是给你上界或下界，而是回答一个很关键的问题：

> 是不是只要做 public expand，就已经足够好了？

换句话说，`Expand-only` 用来排除这样一种可能性：

> 你的方法之所以有效，并不是因为它更好地利用了私有分布，而只是因为你把数据量扩得更大了。

所以 `Expand-only` 是一个非常必要的对照组，它能帮助你把“数据规模效应”和“分布贴近效应”区分开。

如果你的方法优于 `Expand-only`，就说明：

- 不是简单地把数据扩充就够了；
- 更关键的是扩充出来的数据是否贴近 private distribution；
- 你的算法在生成“有用数据”方面确实优于单纯的 public expansion。

### 3.3 `Expand-private`：非隐私上界

`Expand-private` 表示：

- 直接利用私有数据；
- 对私有数据进行扩展；
- 再用扩展后的数据训练下游模型；
- 不严格施加差分隐私约束，或者等价地视为“无限隐私预算”的上界。

它的作用是提供一个很重要的上界：

> 如果允许直接利用私有数据扩展，那下游模型能达到什么水平？

这个 baseline 很重要，因为它能告诉审稿人：

- 你的方法和理想状态之间还有多远；
- 你的方法虽然受制于 no-finetune / federated / privacy 等约束，但已经逼近了怎样的上界；
- 你的 synthetic data 是否已经接近“直接扩展私有数据”这种更强条件下的效果。

如果没有 `Expand-private`，你的结果虽然可能好，但缺少一个“天花板”。有了它，你可以更清楚地讨论：

- 方法还有多大提升空间；
- 当前性能提升到底有多有意义；
- synthetic data 是否足够接近真实私有数据的训练价值。

### 3.4 `PrE-Text`：直接对标基线

`PrE-Text` 是你必须比较的核心 baseline。

它的基本思想是：

- 在联邦分布式私有数据上，不直接用私有数据训练大模型；
- 先利用 `Private Evolution (PE)` 机制生成一批差分隐私种子样本；
- 再将这些 seed data 交给服务器侧的公共大模型做 `expand`；
- 最终得到大规模 synthetic dataset；
- 再把 synthetic dataset 给到下游模型做训练或微调。

它和你的方法之间的比较，是整篇论文最重要的一组对比。

你要用它回答的问题是：

> 在相同的 PrE-Text 风格范式下，我的改进是否让 synthetic data 更有用？

如果你的方法比 `PrE-Text` 更强，那么你就已经证明了自己不只是提出了一个“不同方法”，而是提出了一个“更好的 synthetic data generation 方法”。

## 4. 为什么推荐加入 `WASP`

### 4.1 为什么 `WASP` 适合作为主 baseline

`WASP` 非常适合加入主表，因为它和你的目标高度一致：

- 也是文本 synthetic data generation 方法；
- 强调生成数据质量，而不是直接做联邦训练；
- 依赖预训练语言模型生成数据，而不是把大模型作为需要私有微调的对象；
- 最终也可以通过下游任务表现来衡量生成数据的质量。

与 `PrE-Text` 相比，`WASP` 更强调：

- 多个预训练语言模型协作；
- 对比式生成；
- 在低样本、生成噪声较强时提升 synthetic data quality。

这使它成为非常好的“同赛道质量基线”。

### 4.2 `WASP` 的算法流程

根据你的读后感材料，`WASP` 的基本流程可以概括为以下几步：

1. 从有限的私有样本出发，估计私有数据分布。
2. 不直接微调生成器，而是调用一个或多个预训练语言模型来生成候选 synthetic samples。
3. 使用差分隐私的投票或筛选机制，从候选样本中挑出更接近私有分布的样本。
4. 将已识别的高质量样本与低质量样本一起构造成“对比提示”。
5. 再次引导预训练模型生成新样本，使新样本更接近高质量样本、远离低质量样本。
6. 针对多个预训练语言模型，动态分配权重，优先采用更贴近私有分布的模型输出。
7. 最终获得质量更高的 synthetic dataset，并用其支持下游任务训练。

### 4.3 `WASP` 的核心思想

`WASP` 的关键不在于“训练更强的模型”，而在于：

- 更好地挑选候选样本；
- 更好地利用高质量与低质量样本之间的对比信息；
- 更好地融合多个生成模型的能力。

因此，`WASP` 的本质是一种：

> 面向 synthetic data quality 优化的生成与筛选框架。

这正是你需要的主 baseline 类型。

## 5. 为什么推荐加入 `DPGA-TextSyn`

### 5.1 为什么 `DPGA-TextSyn` 适合作为主 baseline

`DPGA-TextSyn` 也很适合作为主表 baseline，原因是：

- 它明确是文本 synthetic data generation 方法；
- 它不是传统的 federated private training 方法；
- 它的核心目标同样是生成高质量的合成文本；
- 它最终的价值也体现在 synthetic data 对下游模型的帮助上。

与 `PrE-Text` 相比，`DPGA-TextSyn` 的亮点不在联邦架构，而在于：

- 用遗传算法思想优化生成过程；
- 让生成出的样本在质量、多样性和分布贴近性上更好；
- 尤其适合做“生成质量路线”的对比。

### 5.2 `DPGA-TextSyn` 的算法流程

根据你的读后感材料，`DPGA-TextSyn` 的大致流程可以写成：

1. 从任务相关信息或公开信息出发，构造初始 prompt 或初始种子描述。
2. 调用 LLM 生成一批候选 synthetic texts。
3. 基于私有数据的差分隐私反馈机制，对候选文本进行打分或筛选。
4. 将筛选结果映射成类似“适应度”的信号。
5. 对候选样本执行遗传算法式的优化操作，例如选择、变异、重组或 prompt 调整。
6. 重新调用 LLM 生成下一代候选文本。
7. 多轮迭代后，保留高质量 synthetic texts，构成最终 synthetic dataset。
8. 用这个 dataset 去支持下游任务训练，从下游表现反映合成数据质量。

### 5.3 `DPGA-TextSyn` 的核心思想

`DPGA-TextSyn` 的核心可以理解为：

> 不把一次生成结果当作终点，而是把 synthetic data generation 视为一个持续搜索最优样本集合的过程。

这与很多简单的“一次生成、一轮筛选”方法不同。它更像是：

- 生成；
- 评估；
- 筛选；
- 变异；
- 再生成。

因此它也是 synthetic data quality 导向论文里很自然的比较对象。

## 6. 为什么主表不以 `DP-FedAvg` 和 `DP-FTRL` 为主

`DP-FedAvg` 和 `DP-FTRL` 当然可以比较，但不应该作为主表中心，因为它们回答的问题不是：

> 谁生成的数据更有用？

而是：

> 在联邦差分隐私训练里，怎样直接训练一个模型？

这和你的主目标不是同一个问题。

如果把它们放进主表，容易导致论文主线变歪：

- 审稿人会以为你主要在做联邦 DP 训练对比；
- synthetic data generation 的主线会变弱；
- 你需要花更多篇幅解释为什么“直接训练模型”和“生成合成数据再训练”可以直接比较。

因此更好的做法是：

- 主表：只看 synthetic data quality；
- 副表：补充 `DP-FedAvg`、`DP-FTRL`、`DP-Prompt`。

这样结构最清楚。

## 7. 推荐的副表基线

### 7.1 副表建议放哪些算法

建议把以下三类方法放在副表：

- `DP-FedAvg`
- `DP-FTRL`
- `DP-Prompt`

### 7.2 这些方法在副表中的作用

它们的作用不是证明你的方法是最好的 synthetic generation 方法，而是补充说明：

- 你的方法不只是比生成类方法强；
- 你的方法相对于传统 federated private training 也有实用价值；
- 你的方法在“数据生成再训练”这条路线下，确实优于直接做端侧隐私训练或隐私改写的方法。

其中：

- `DP-FedAvg`：代表经典的联邦差分隐私训练；
- `DP-FTRL`：代表另一类联邦差分隐私优化路线；
- `DP-Prompt`：代表“客户端隐私改写再上传”的 text-to-text privatization 路线。

特别是 `DP-Prompt`，虽然它也会产出文本，但它更像是：

> 对私有文本做隐私保护改写

而不是：

> 在服务器侧构造一个高质量 synthetic dataset 并用于下游训练。

因此它更适合作为补充方法，而不是主表核心方法。

## 8. 不建议优先放入主表的算法

为了让主表保持干净、范式一致，以下方法不建议一开始塞进主表：

- `PoPri`
- `KnowledgeSG`
- `RewardDS`
- `GRADMM`

原因分别如下。

### 8.1 `PoPri`

`PoPri` 会显式利用策略优化来微调 LLM，使其生成更高质量的合成数据。

如果你的方法强调的是：

- no-finetune generation；
- 基于 PrE-Text 风格的服务器侧生成；
- 尽量不在私有数据上直接微调大模型；

那么 `PoPri` 和你的设定并不完全同类。它可以在 related work 或扩展实验里出现，但不适合作为最干净的主 baseline。

### 8.2 `KnowledgeSG`

`KnowledgeSG` 更偏向：

- client-server 知识蒸馏；
- 本地模型与服务器模型的交互；
- 通过参数或适配器传递知识；
- 再提升生成质量。

它不是最纯粹的 PrE-Text 同类方法，因此如果你把它放进主表，会让范式变杂。

### 8.3 `RewardDS`

`RewardDS` 更强调：

- 合成数据中的噪声过滤；
- 利用奖励机制提升数据质量；
- 面向后续 LLM 微调的质量优化。

它可以作为补充 baseline，但如果你的论文重点是 federated / no-finetune synthetic generation，那么它不是最优先的一组主对比。

### 8.4 `GRADMM`

`GRADMM` 是更偏理论型的文本合成方法。

它适合出现在“扩展比较”里，但如果你的核心约束是：

- federated setting；
- no-finetune generation；
- synthetic data 用于下游任务；

那么它和主线并不完全同构。

## 9. 最推荐的 baseline 组合

### 9.1 主实验

推荐主实验比较以下方法：

- `c4-only`
- `Expand-only`
- `PrE-Text`
- `WASP`
- `DPGA-TextSyn`
- `Expand-private`

### 9.2 补充实验

推荐补充实验比较以下方法：

- `DP-FedAvg`
- `DP-FTRL`
- `DP-Prompt`

### 9.3 这样设计的好处

这种设计的结构非常清楚：

- 主实验回答 synthetic data quality 问题；
- 补充实验回答“和传统联邦/隐私方法相比有没有价值”的问题；
- 审稿人能清楚看出你的论文主线不是隐私工程，而是 synthetic data effectiveness。

## 10. 论文中应如何解释这些 baseline 的角色

你在论文里可以按下面的逻辑解释：

- `c4-only`：不使用私有分布信息时的下界；
- `Expand-only`：测试“仅靠 public expand 是否已经足够”；
- `Expand-private`：直接利用私有数据扩展时的非隐私上界；
- `PrE-Text`：与你方法最接近的直接对标基线；
- `WASP`：强调多模型协作与对比生成的 synthetic data quality 基线；
- `DPGA-TextSyn`：强调遗传优化搜索的 synthetic data generation 基线；
- `DP-FedAvg` / `DP-FTRL`：传统联邦 DP 训练基线；
- `DP-Prompt`：隐私改写型文本生成基线。

这样一来，每个 baseline 的存在都是有明确理由的，不会显得“方法堆砌”。

## 11. 实验设计上必须统一的条件

为了保证比较公平，你的主表最好统一以下几项：

### 11.1 统一下游模型

例如固定为：

- 全部使用 `DistilGPT2` 做 next-word prediction；或者
- 全部使用同一个 `LLaMA-2-7B + LoRA` 设置。

这样你比较的才是 synthetic data 本身，而不是“不同下游模型能力差异”。

### 11.2 统一 synthetic data budget

例如固定比较：

- `50k`
- `200k`
- `1M`

或者只取一两个代表性规模。

否则审稿人可能会质疑某个方法只是因为生成数据更多才更强。

### 11.3 统一下游任务

例如统一为：

- `next-word prediction`；
- 或同一组 `instruction/downstream tasks`。

不要让某些方法在语言建模任务上比较，另一些方法却在分类或问答任务上比较，否则结论会松散。

### 11.4 统一评价指标

主指标建议使用：

- `downstream accuracy`
- `cross-entropy`
- `task score`

次指标再补：

- privacy
- communication
- compute

这样主次关系就很清楚：先看 synthetic data quality，再看代价。

## 12. 最终建议

如果你的论文主线是：

> 我基于 PrE-Text 做了一个更强的 synthetic data generation 算法，重点提升生成数据在下游任务中的有效性。

那么最推荐的 baseline 设计是：

### 主表

- `c4-only`
- `Expand-only`
- `PrE-Text`
- `WASP`
- `DPGA-TextSyn`
- `Expand-private`

### 副表

- `DP-FedAvg`
- `DP-FTRL`
- `DP-Prompt`

这样写的优点是：

- 主线清晰；
- baseline 角色明确；
- 同类方法比较充分；
- 上下界完整；
- 能回答审稿人关于 federated / privacy / efficiency 的疑问；
- 最重要的是，论文结论会始终围绕 synthetic data quality 展开，而不会跑偏到“隐私工程”。

## 13. 一句话总结

你的论文不应该把核心问题定义为：

> 谁的隐私更强，谁的联邦训练更便宜？

而应该定义为：

> 在相同约束下，谁生成的 synthetic data 更能提升下游任务表现？

围绕这个问题，`PrE-Text + WASP + DPGA-TextSyn + c4-only + Expand-only + Expand-private` 是最合适的一组主 baseline。
