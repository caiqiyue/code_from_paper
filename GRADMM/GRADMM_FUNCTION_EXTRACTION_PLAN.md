# GRADMM 功能抽取技术方案

## 1. 目标与依据

本文档依据 [THESIS_PLATFORM_PLAN.md](D:/学习记录/导师项目/研究/caiqiyue_file/docs/THESIS_PLATFORM_PLAN.md) 中与 `datainf`/`gradmm` scorer 相关的设计要求，对 [GRADMM](D:/学习记录/导师项目/研究/caiqiyue_file/GRADMM) 目录下各模块进行审阅，并给出“新创新项目中需要抽取哪些 GRADMM 功能、对应哪些模块、如何改造成统一平台 scorer”的技术方案。

与本方案直接相关的规划结论有 4 个：

- `DataInf/` 和 `GRADMM/` 只抽取为“样本打分器”，不要照搬原仓库完整实验流程。
- 新平台中的 scorer 统一接口是 `score(samples, client_ctx) -> list[ScoredSample]`。
- scorer 需要配置化支持 `score_direction`、`target_module`、`batch_size` 等字段。
- 平台必须保留中间证据，支持公平对比、模块替换和可复现。

因此，本方案的核心判断是：

- 新项目只需要抽取 `GRADMM` 中“梯度匹配打分”相关能力。
- 不需要迁入 `GRADMM` 的 synthetic generation 主流程。
- 不需要迁入 `addax/` 下的微调评测框架。

## 2. 总体结论

GRADMM 原仓库可分成两条主链路：

1. `gradmm/generate.py` 为核心的“梯度匹配生成 synthetic text”链路。
2. `gradmm/filtering.py` 为核心的“对 synthetic text 做清洗、重打分、筛选”链路。

从论文平台视角看，真正可服务于 `gradmm_scorer` 适配器的能力主要来自第 2 条链路，外加第 1 条链路里少量和梯度目标模块控制相关的实现。

一句话概括：

- 必须抽取：梯度计算、参考梯度聚合、样本重打分、坏样本排序、中间分量保存。
- 可选抽取：标签自检清洗、贪心梯度子集选择。
- 明确不抽取：生成优化、ADMM、DP 生成脚本、微调评估基础设施。

## 3. GRADMM 全模块审阅结论

### 3.1 `gradmm/` 子目录

| 模块 | 原始职责 | 在新项目中的处理结论 |
| --- | --- | --- |
| `gradmm/filtering.py` | synthetic data 清洗、重打分、按指标筛选 | 核心抽取来源，需拆成可调用 API |
| `gradmm/utilities.py` | 梯度、相似度、perplexity、重构损失、token 映射等底层函数 | 核心抽取来源，但只保留 scorer 相关子集 |
| `gradmm/generate.py` | 梯度匹配生成 synthetic text 的主流程 | 不整体抽取；仅参考其中的 `last_layer_gradient`/目标层冻结逻辑 |
| `gradmm/data_utils.py` | 原仓库实验数据加载器 | 不直接复用；只参考其数据读取思路 |
| `gradmm/init.py` | 生成阶段的 embedding 初始化 | 不需要 |
| `gradmm/args_factory.py` | 生成脚本 CLI 参数 | 不需要，改成平台 YAML + dataclass |
| `gradmm/constants.py` | 常量定义 | 当前 scorer 方案不需要 |
| `gradmm/Filtering.ipynb` | filtering 实验 notebook | 仅作流程参考，不作为平台代码来源 |
| `gradmm/scripts/admm.sh` | 批量生成脚本 | 不需要 |
| `gradmm/scripts/admm_dp.sh` | DP 生成脚本 | 不需要 |

### 3.2 `addax/` 子目录

| 模块 | 原始职责 | 在新项目中的处理结论 |
| --- | --- | --- |
| `addax/run.py` | 微调训练入口 | 不需要 |
| `addax/trainer.py` | HF Trainer 扩展与结果保存 | 不需要 |
| `addax/tasks.py` | 下游任务与 synthetic 数据适配 | 不直接复用 |
| `addax/templates.py` | prompt/template 定义 | 不需要 |
| `addax/utils.py` | 训练 collator、prompt 编码、指标落盘 | 不需要 |
| `addax/metrics.py` | 微调评测指标 | 不需要 |
| `addax/lora.py` | LoRA 注入 | 不需要 |
| `addax/prefix.py` | Prefix Tuning 注入 | 不需要 |
| `addax/GPUtil.py` | GPU 使用率统计 | 不需要 |
| `addax/sign_converter.py` | 1-bit sign 压缩工具 | 不需要 |
| `addax/test_sign_converter.py` | sign converter 单测 | 不需要 |
| `addax/Finetuning.ipynb` | 微调结果汇总 notebook | 不需要 |
| `addax/scripts/query_ft.sh` | 批量微调脚本 | 不需要 |

结论：`addax/` 整体属于原论文“生成后再微调评估”的实验基础设施，不属于论文平台里的 scorer 抽取范围。

## 4. 新项目必须抽取的功能与模块映射

### 4.1 核心功能映射表

| 新项目需要的功能 | 作用 | 优先来源模块 | 建议抽取函数 |
| --- | --- | --- | --- |
| 参考真实样本梯度聚合 | 为坏样本打分建立 reference gradient | `gradmm/filtering.py` + `gradmm/utilities.py` | `compute_average_grads`, `compute_grads_lm` |
| 单样本梯度计算 | 计算候选样本对目标模块的梯度 | `gradmm/filtering.py` + `gradmm/utilities.py` | `compute_grads`, `compute_grads_lm`, `compute_grads_lm_ids` |
| 梯度距离计算 | 衡量样本与参考梯度的偏离程度 | `gradmm/filtering.py` + `gradmm/utilities.py` | `grad_dist`, `cos_sim` |
| perplexity 计算 | 作为样本流畅度/异常度的辅助项 | `gradmm/utilities.py` | `get_perplexity_loss` |
| 重构损失计算 | 计算 `rec_loss_ids` 这一核心打分分量 | `gradmm/utilities.py` | `get_reconstruction_loss_ids`, `get_reconstruction_loss` |
| 坏样本综合分数 | 形成平台中的 `score` 字段 | `gradmm/filtering.py` | `calculate_recon_loss_ids` 的思路 + `rec_loss_ids + alpha * perplexity` |
| 样本排序与 top-k 选择 | 供上层 pipeline 选坏样本 | `gradmm/filtering.py` | `extract_top_samples_per_label` 的排序思路 |
| 中间证据落盘 | 保存可解释性证据 | `gradmm/filtering.py` | `output_to_jsonl` 的字段组织思路 |
| 目标模块控制 | 对齐 `target_module` 配置 | `gradmm/generate.py` + `gradmm/utilities.py` | `last_layer_gradient` 相关逻辑，需重构 |

### 4.2 建议作为“可选增强项”抽取的功能

| 可选功能 | 价值 | 来源模块 | 建议 |
| --- | --- | --- | --- |
| 标签自检清洗 | 在生成器输出标签可能不稳定时，先做一轮轻量清洗 | `gradmm/filtering.py` | 可做成 `enable_label_sanity_check` 开关 |
| `relabel`/`remove` 策略 | 错标样本重写标签或直接剔除 | `gradmm/filtering.py` | 可保留，但不放进 scorer 主路径 |
| 贪心梯度子集选择 | 选出最能逼近参考梯度的样本子集 | `gradmm/filtering.py` | 适合后续消融实验，不建议做成首版默认 |
| `deterministic_usm` | 对贪心子集做稳定化 | `gradmm/filtering.py` | 作为研究扩展保留即可 |

## 5. 不建议直接复用的部分

以下代码不适合直接搬到新平台中：

### 5.1 生成主流程

`gradmm/generate.py`、`gradmm/init.py`、`gradmm/scripts/*.sh` 是“生成 synthetic data”的实现，而平台规划已经明确 `GRADMM` 只作为 scorer，不作为 generator。因此：

- `ADMM` 优化循环不需要。
- embedding 初始化不需要。
- DP 生成逻辑不需要。
- `args_factory.py` 这套 CLI 也不需要。

### 5.2 原仓库数据加载方式

`gradmm/data_utils.py` 与 `filtering.py/load_real_data` 都强依赖原始数据格式和原始任务设定：

- 数据集被写死为 `sst2`、`rotten_tomatoes`、`TwitterEmotion`、`imdb`、`rtpolarity`。
- 标签假定为二分类 `0/1`。
- prompt 结构被写死成 `"It was"` 或 `"Does the tweet express joy or sadness?"`。

新平台需要统一 schema，因此这些模块只能参考，不能照搬。

### 5.3 `addax/` 微调评估链路

`addax/` 只在原论文里负责“拿筛好的 synthetic data 去微调模型并评测”。新项目中的 scorer 只负责“打分”，不负责后续训练。所以：

- `addax/run.py`
- `addax/trainer.py`
- `addax/tasks.py`
- `addax/templates.py`
- `addax/utils.py`

都不应进入首版 `gradmm_scorer` 抽取范围。

## 6. 面向论文平台的推荐抽取边界

### 6.1 首版必须落地的最小闭环

建议在新平台中只保留下面这条 scorer 闭环：

1. 从统一 schema 的 `samples` 中提取待打分文本与标签。
2. 从 `client_ctx` 中读取真实样本或 public seed 作为 reference set。
3. 计算 reference set 的平均梯度。
4. 为每个候选样本计算：
   - `rec_loss_ids`
   - `perplexity`
   - `score = rec_loss_ids + alpha * perplexity`
5. 返回 `ScoredSample` 列表，并保存中间证据。

这条链路对应的原始代码来源主要是：

- `gradmm/filtering.py`
- `gradmm/utilities.py`
- `gradmm/generate.py` 中的目标层冻结逻辑

### 6.2 推荐的新平台模块划分

建议不要把原始 `filtering.py` 整个导入，而是在新项目中拆成下面几个平台内模块：

| 平台新模块 | 作用 | 主要参考来源 |
| --- | --- | --- |
| `thesis_platform/adapters/scorers/gradmm_scorer.py` | 对外暴露统一 scorer 接口 | `filtering.py` |
| `thesis_platform/adapters/scorers/gradmm_backend.py` | 梯度计算、reference gradient 聚合、分数分解 | `filtering.py`, `utilities.py` |
| `thesis_platform/adapters/scorers/gradmm_formatter.py` | 将统一 schema 样本转成 GRADMM 所需的 prompt/label 形式 | `filtering.py`, `data_utils.py` |
| `thesis_platform/adapters/scorers/gradmm_selector.py` | 可选的 top-k / greedy / per-label 排序策略 | `filtering.py` |

## 7. 必须改造的技术点

### 7.1 `score_direction` 需要反转到“坏样本语义”

原始 GRADMM filtering 的目标是“从 synthetic candidates 中挑出更好的样本”，因此排序逻辑偏向低 `metric` 样本。  
而论文平台里的 scorer 语义是“给坏样本打分”，因此建议在平台侧统一约定：

- `raw_badness = rec_loss_ids + alpha * perplexity`
- `score_direction = larger_is_worse`

也就是说，平台里保留的是“坏度分数”，而不是原 notebook 的“保留分数”。

### 7.2 `target_module` 不能沿用原始默认行为

`gradmm/utilities.py` 中的 `compute_grads_lm`/`compute_grads_lm_ids` 默认对所有 `requires_grad=True` 参数求梯度。  
而平台计划里 scorer 需要显式支持 `target_module`，例如：

- `last_layer`
- `lora`
- `lora_or_last_layer`

因此必须把原来的“依赖 `requires_grad` 隐式选择参数”改成：

- 显式的参数过滤器；
- 或在 scorer 初始化阶段显式冻结无关层。

可直接参考 `gradmm/generate.py` 中 `last_layer_gradient` 对 `lm_head` 的冻结逻辑，但需要抽象成可配置版本。

### 7.3 输入格式必须从“原始数据集”改成“统一 schema”

原始 GRADMM scorer 逻辑默认输入是：

- 固定二分类任务；
- 固定标签词；
- 固定 prompt 后缀。

而论文平台的数据是统一 schema，不同数据源、不同任务格式可能共存。因此需要增加一层格式器，把平台样本映射成 scorer 所需的训练视角。至少要支持：

- 文本分类型样本；
- instruction-response 型样本；
- 可从配置注入标签文本或 answer target。

### 7.4 原始脚本式 I/O 需要改成函数式 API

`filtering.py` 当前是以“扫描目录、读取 JSONL、写回 JSONL”为中心组织的。新平台不应该直接调用这个脚本层，而要改成：

```python
score(samples, client_ctx) -> list[ScoredSample]
```

中间产物再由平台统一落盘，而不是让 scorer 自己决定目录结构。

### 7.5 首版不建议引入 greedy selection 作为默认路径

`greedy_selection`、`grad_similarity_selection`、`deterministic_usm` 都会显著增加梯度重算次数，算力开销较大。  
对论文平台首版而言，建议先做“逐样本打分 + 排序”，把 greedy 子集选择作为后续消融项。

## 8. 推荐保留的中间证据

为了满足 `THESIS_PLATFORM_PLAN.md` 对“中间证据可保存”的要求，建议 `gradmm_scorer` 至少落盘以下字段：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 平台统一样本 ID |
| `raw_text` | 原始样本文本 |
| `label` | 标签或目标答案 |
| `target_module` | 本轮使用的梯度目标模块 |
| `reference_group` | 该样本对应的 reference set 标识 |
| `perplexity` | 流畅度分量 |
| `rec_loss_ids` | 梯度匹配分量 |
| `score` | 最终坏样本分数 |
| `score_direction` | 建议固定为 `larger_is_worse` |
| `rank_in_client` | 当前客户端内排序位置 |

如果后续要做案例分析，还建议额外保存：

- reference gradient 的摘要信息；
- 每轮 scorer 配置快照；
- top-k bad sample 清单。

## 9. 推荐实施顺序

1. 先从 `gradmm/utilities.py` 抽出最小梯度计算内核。
2. 再从 `gradmm/filtering.py` 抽出 reference gradient 聚合与单样本分数计算逻辑。
3. 接着实现 `target_module` 选择器，补齐与平台配置的一致性。
4. 再实现 `score(samples, client_ctx)` 的统一接口。
5. 最后再决定是否加入 `label_sanity_check` 和 greedy selection 作为增强选项。

## 10. 最终抽取结论

### 10.1 新项目中建议直接抽取的模块来源

- 主来源：`gradmm/filtering.py`
- 主来源：`gradmm/utilities.py`
- 辅助参考：`gradmm/generate.py` 中的目标层冻结逻辑
- 参考但不直接复用：`gradmm/data_utils.py`

### 10.2 新项目中不建议抽取的模块来源

- 不抽取：`gradmm/generate.py` 的生成优化主流程
- 不抽取：`gradmm/init.py`
- 不抽取：`gradmm/args_factory.py`
- 不抽取：`gradmm/scripts/*`
- 不抽取：`addax/*`

### 10.3 适合落入平台的功能清单

- 参考样本平均梯度计算
- 候选样本梯度距离计算
- `rec_loss_ids` 计算
- `perplexity` 计算
- 综合坏样本分数计算
- 按分数排序并返回 `ScoredSample`
- 中间证据保存

### 10.4 适合作为后续扩展的功能清单

- 标签自检清洗
- `relabel/remove` 预过滤
- greedy gradient subset selection
- deterministic USM 稳定化选择

## 11. 结论

从论文平台设计要求出发，GRADMM 在新项目中的正确定位不是“生成算法仓库”，而是“梯度匹配 bad-sample scorer 的能力来源”。  
因此最合理的抽取策略是：

- 只保留 `filtering + utilities + 少量 target_module 控制逻辑`；
- 把原始脚本式流程重构为统一 scorer 适配器；
- 明确把分数语义改成 `larger_is_worse`；
- 不引入 `addax` 和生成主流程，避免平台边界失控。
