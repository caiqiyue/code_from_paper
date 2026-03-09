# PrE-Text 功能抽取技术方案

## 1. 目标与总体结论

依据 [docs/THESIS_PLATFORM_PLAN.md](../docs/THESIS_PLATFORM_PLAN.md) 中对 `PrE-Text` 的定位，`PrE-Text` 在新平台里承担两类职责：

1. `generator`
   - 服务端合成数据生成基线。
   - 对应平台里的 `pretext_seed` 和 `pretext_bootstrap`。
2. `scorer`
   - 作为原始筛选基线 `pretext_hist`。
   - 本质是用直方图支持度判断候选样本是否贴近目标数据分布。

平台文档已经明确给出：

1. `PrE-Text/` 是当前最合适的服务端生成基线。
2. `PrE-Text/` 还提供“直方图筛选基线”和“bootstrap 扩展生成”。
3. `generator` 预设参数中已经保留了：
   - `init_population_path`
   - `seq_len`
   - `mask`
   - `lookahead`
   - `multiplier`
   - `t_steps`
   - `bootstrap_enable`
   - `bootstrap_model`
   - `generated_per_round`
4. 阶段 0 和阶段 1 还要求保留 `pretext_hist` 作为 scorer baseline。

结合对 `PrE-Text` 仓库全部模块的审阅，最终结论如下：

1. 新平台必须抽取的核心能力有三类：
   - 基于 mask-fill 的候选文本变异生成。
   - 基于 lookahead embedding + 最近邻直方图的候选筛选/打分。
   - 基于 seed 文本的 LLM bootstrap 扩展生成。
2. 新平台不应直接搬运 `PrE-Text` 的整套实验脚手架：
   - `main.py` 中的完整 DP 实验循环
   - 两个下游评测脚本
   - 原始数据文件命名和输出目录命名约定
3. `PrE-Text` 在新平台里真正要保留的是：
   - `variation.py`
   - `similarity.py`
   - `nn_histogram.py`
   - `llama_bootstrap.py`
   - `main.py` 中的流程编排思想
4. 第一版平台中的 `PrE-Text` 更适合采用“重构式抽取”，而不是把 `main.py` 作为独立子流程直接调用。

一句话总结：

`PrE-Text` 在新平台中应被拆成 `pretext_generator`、`pretext_bootstrap` 和 `pretext_histogram` 三个可插拔部件，而不是保留原仓库的一整段 DP Private Evolution 脚本。

## 2. 平台侧约束

从 `THESIS_PLATFORM_PLAN.md` 可归纳出 6 个直接约束：

1. `PrE-Text` 的平台角色首先是 `generator`。
2. `PrE-Text` 还要被拆出一个 `pretext_hist` scorer baseline。
3. 平台统一接口是按流程阶段拆分，而不是按论文仓库拆分。
4. `generator` 当前显式暴露的配置只有：
   - `init_population_path`
   - `seq_len`
   - `mask`
   - `lookahead`
   - `multiplier`
   - `t_steps`
   - `bootstrap_enable`
   - `bootstrap_model`
   - `generated_per_round`
5. 平台第一阶段主要面向公开数据集工程验证；第二阶段才迁移到联邦私有数据。
6. 阶段 0 和阶段 1 的实验矩阵要求保留：
   - `pretext_seed`
   - `pretext_hist`

这意味着：

1. 新平台需要的是“服务端生成器”和“基线打分器”。
2. 不需要直接继承 `PrE-Text` 的原始数据格式、DP 实验目录和下游 next-token 评测脚本。
3. 原仓库中的 DP 参数应被视为“可选隐私扩展”，而不是第一版平台的最小核心。

## 3. 仓库模块审阅结论

## 3.1 模块总表

| 模块 | 原始职责 | 新平台处理方式 | 结论 |
| --- | --- | --- | --- |
| `main.py` | 第一阶段 Private Evolution 主流程 | 抽取流程与局部函数调用关系，不直接复用整文件 | 部分抽取 |
| `variation.py` | 候选文本 mask-fill 变异器 | 直接抽取核心逻辑 | 必须抽取 |
| `similarity.py` | 候选/私有文本 embedding 与 lookahead embedding | 直接抽取核心逻辑 | 必须抽取 |
| `nn_histogram.py` | 最近邻直方图计数、加噪、阈值筛选 | 直接抽取核心逻辑 | 必须抽取 |
| `llama_bootstrap.py` | 第二阶段 seed 扩增生成 | 抽取为独立 bootstrap 生成器 | 必须抽取 |
| `custom_datasets.py` | 轻量数据集包装器 | 按需借鉴或直接替换 | 可选复用 |
| `eval_distilgpt2.py` | 小模型 downstream 评测 | 不进入平台主链路 | 不抽取 |
| `eval_llama2.py` | 大模型 downstream 评测 | 不进入平台主链路 | 不抽取 |
| `README.md` / `quick_start.md` | 说明文档 | 仅参考参数语义 | 不抽取 |
| `assets/*` | 展示图片 | 与平台实现无关 | 不抽取 |

## 3.2 核心流程模块：`main.py`

这是 `PrE-Text` 的主流程入口，但不应整体搬进新平台。

### 原始职责

`main.py` 的主链路可以概括为：

1. 加载 `roberta-large` 作为 masked LM。
2. 加载 `all-MiniLM-L6-v2` 作为 embedding 模型。
3. 读取私有训练样本。
4. 读取 `initialization.json` 作为初始种群。
5. 计算隐私预算相关量：
   - `epsilon`
   - `sigma`
   - `H`
6. 进行 11 轮 Private Evolution：
   - `NN_Histogram.dp_nn_histogram(...)`
   - 按 histogram 重采样父代
   - `Variation.produce_variation(...)`
   - 保存 `generated_text_it*.json` 与 `surviving_text_it*.json`

### 对新平台的价值

它真正值得抽取的是“调度关系”而不是“脚本本身”：

1. 初始种群采样逻辑。
2. `variation -> similarity -> histogram -> resample -> variation` 的循环。
3. 每轮产物的保存点位设计。

### 对新平台不适合直接复用的地方

1. 路径写死：
   - `./data/{dataset}_train.json`
   - `./data/initialization.json`
2. 输出目录命名绑死在一长串 DP 参数上。
3. 将 11 轮、batch size、num_workers、num_gpus 等大量实验参数写死在脚本里。
4. 直接耦合 Opacus RDP accountant。
5. 直接把 histogram 用于“保留高支持度样本”，而平台中的 `pretext_hist` scorer 要输出的是样本分数。

### 结论

1. 只抽取其流程编排思想。
2. 不直接作为平台入口脚本复用。

## 3.3 核心模块一：`variation.py`

这是必须抽取的生成内核。

### 原始职责

`variation.py` 包含三个核心部分：

1. `top_k_top_p_filtering(...)`
   - 对 MLM logits 做 top-k / top-p 截断。
2. `Variation.collate_fn_tokenizer(...)`
   - 在 batch 内按比例随机选择 token 做 mask。
3. `Variation.sample(...)`
   - 用 masked LM 对 mask 位置逐步采样填回。
4. `Variation.produce_variation(...)`
   - 重复 `t_steps` 次 mask-fill，得到最终变体。

### 新平台中的作用

这是 `pretext_seed` 生成器真正的“变异引擎”。

平台中至少有两处会用到它：

1. `generator/pretext_generator.py`
   - 从当前父代生成新的候选样本。
2. `scorer/pretext_histogram.py`
   - 在做 lookahead embedding 时，模拟未来几步变体。

### 必须保留的能力

1. mask 比例控制
2. 基于 MLM 的采样填充
3. `t_steps` 多次迭代变异
4. 输出 `input_ids` 与 `attention_mask`

### 必须改造的点

1. 去掉对全局 `config` 字典的强耦合。
2. 去掉和 `Accelerator` 的硬绑定，让生成器适配器自己决定 runtime。
3. 让输出直接映射回平台统一 schema，而不是停留在 token tensor。
4. 将 `batch_size`、`top_k`、`top_p` 等参数配置化。

### 结论

必须抽取。

## 3.4 核心模块二：`similarity.py`

这是必须抽取的“分布贴近度”模块。

### 原始职责

1. `Similarity.sentence_embedding(...)`
   - 用 `SentenceTransformer` 计算文本 embedding。
2. `Similarity.concat_embedding(...)`
   - 当前实现只是直接调用句向量编码。
3. `Similarity.lookahead_embedding(...)`
   - 对每个候选样本先做多次未来变异，再平均 embedding。

### 新平台中的作用

它是 `pretext_hist` scorer 的关键前置步骤。

因为直方图计数不是直接对当前样本做，而是对“未来几步可能演化到的位置”做近邻评估。

### 必须保留的能力

1. 基础 embedding 编码
2. lookahead embedding 机制
3. 与 `Variation.produce_variation(...)` 的组合调用关系

### 必须改造的点

1. 去掉对 `config["mpnet"]` 这种全局对象注入方式的耦合。
2. 将 embedding backend 替换为平台统一配置的编码模型。
3. 修正/统一解码接口与文本恢复逻辑，避免和 tokenizer 细节绑死。
4. 让它能接收统一 schema 中的 `text` 字段，而不是默认原始 `PrE-Text` 文本列表。

### 结论

必须抽取。

## 3.5 核心模块三：`nn_histogram.py`

这是必须抽取的直方图筛选/打分核心。

### 原始职责

`NN_Histogram.dp_nn_histogram(...)` 的链路是：

1. 先调用 `Similarity.lookahead_embedding(...)`
2. 用 FAISS 建立候选样本向量索引
3. 对私有样本 embedding 做最近邻搜索
4. 统计每个候选样本被命中的次数
5. 加高斯噪声
6. 再做阈值裁剪，得到 `noised_histogram_thresh`

### 对新平台的重要意义

这正是平台中 `pretext_hist` baseline 的来源。

但需要特别说明一点：

1. 在原始 `PrE-Text` 中，histogram 值越大，表示该候选越贴近目标分布，更适合被保留。
2. 在新平台的 `scorer` 语义中，通常是要找“坏样本”。

因此平台里 `pretext_hist` 必须做一个语义映射：

1. 方案 A：保留原 histogram 作为 `quality_score`，并设定 `score_direction=smaller_is_worse`
2. 方案 B：输出 `bad_score = -histogram`，统一成 `larger_is_worse`

从工程一致性看，更推荐方案 B。

### 必须保留的能力

1. lookahead + nearest neighbor histogram
2. FAISS 最近邻搜索
3. 直方图加噪与阈值处理的逻辑骨架

### 必须改造的点

1. 把 `dp_nn_histogram(...)` 拆成更清晰的两部分：
   - `compute_histogram_scores(...)`
   - `sample_survivors(...)`
2. 平台 `pretext_hist` scorer 只需要前者，不需要后者。
3. 第一版平台可将 DP 噪声设为可选：
   - 公共数据 smoke 阶段：允许不加噪
   - 私有联邦阶段：再开启噪声与阈值
4. 将 `sigma`、`H`、`embed_dim`、`nearest_neighbors_print` 等从脚本配置改成 scorer/generator 显式配置。

### 结论

必须抽取。

## 3.6 核心模块四：`llama_bootstrap.py`

这是必须抽取的第二阶段扩增模块。

### 原始职责

1. 读取 11 轮 `surviving_text_it*.json`
2. 合并为 seed 列表
3. 每次随机抽样若干 seed 样本拼 prompt
4. 用 `Llama-2-7b-hf` 生成更多文本
5. 保存到 `llama7b_text_syn.json`

### 新平台中的作用

它对应平台设计中的：

1. `pretext_bootstrap`
2. `bootstrap_enable`
3. `bootstrap_model`

### 必须保留的能力

1. 基于 seed 样本的服务端扩增
2. 通过本地/远程大模型批量生成文本
3. 生成数控制

### 必须改造的点

1. 不能依赖 `surviving_text_it0..10.json` 这种文件协议，应该直接接收内存中的 seed 样本列表。
2. 不能把目标样本数写死为 `50000`。
3. 不能把 bootstrap prompt 写死，应把模板配置化。
4. 不能把模型后端写死为 `vllm.LLM`。

### 结论

必须抽取，但需要重构为真正的平台 `generator` 子模块。

## 3.7 辅助模块：`custom_datasets.py`

### 原始职责

1. `ListDataset`
   - 包装字符串列表。
2. `MatrixDataset`
   - 包装 `input_ids` 与 `attention_mask`。

### 对新平台的意义

这只是两个很薄的数据包装层。

### 处理建议

1. 如果平台自己的 batch builder 已经足够，就不需要复用。
2. 如果要快速重构 `Variation` 相关逻辑，可以临时保留 `MatrixDataset`。

### 结论

可选复用，不是平台核心。

## 3.8 下游评测模块：`eval_distilgpt2.py` 与 `eval_llama2.py`

这两个文件都不应进入第一版平台主链路。

### `eval_distilgpt2.py`

原职责：

1. 读取合成文本
2. 微调 `distilgpt2`
3. 在 eval 集上计算 cross-entropy 与 top-k accuracy

### `eval_llama2.py`

原职责：

1. 读取合成文本
2. 对 `Llama-2-7b` 做 LoRA 微调
3. 在 eval 集上做 next-token prediction 评测

### 为什么不应抽取

1. 平台当前主线不是 next-token prediction benchmark。
2. 平台已经有自己独立的 evaluator 和实验矩阵。
3. 这两个脚本只服务于原论文复现实验，不服务于新的统一流程。

### 结论

不抽取。

## 4. 新项目真正需要的 PrE-Text 功能

## 4.1 P0 必须落地的功能

| 功能 | 作用 | 对应原模块 |
| --- | --- | --- |
| seed 候选生成 | 从初始种群生成当前轮服务端候选样本 | `main.py`, `variation.py` |
| mask-fill 变异 | 生成新的文本变体 | `variation.py` |
| lookahead embedding | 评估未来几步变异后的候选质量 | `similarity.py`, `variation.py` |
| histogram 打分 | 形成 `pretext_hist` 基线分数 | `nn_histogram.py`, `similarity.py` |
| survivor 重采样 | 用于复现原始 `PrE-Text` 生成链 | `main.py`, `nn_histogram.py` |
| bootstrap 扩增 | 从 seed 样本扩展更大合成语料 | `llama_bootstrap.py` |

## 4.2 P1 建议保留的功能

| 功能 | 作用 | 对应原模块 |
| --- | --- | --- |
| FAISS 近邻检索 | 加速 histogram 构造 | `nn_histogram.py` |
| 轻量 token/matrix dataset 包装 | 支持快速 batch 化 | `custom_datasets.py` |
| DP 噪声与阈值开关 | 为第二阶段私有联邦实验保留 | `main.py`, `nn_histogram.py` |
| 隐私会计逻辑 | 迁移到正式私有实验时使用 | `main.py` |

## 4.3 P2 可以先不做的功能

| 功能 | 原模块 | 暂不纳入原因 |
| --- | --- | --- |
| DistilGPT2 downstream eval | `eval_distilgpt2.py` | 平台已有独立评测链 |
| LLaMA2 downstream eval | `eval_llama2.py` | 平台主线不是原论文 next-token evaluation |
| 原始脚本输出目录协议 | `main.py`, `llama_bootstrap.py` | 平台已有统一输出结构 |

## 5. 功能到模块的抽取映射

## 5.1 推荐抽取映射表

| 新平台能力 | 需要抽取的原模块 | 抽取粒度 | 说明 |
| --- | --- | --- | --- |
| `pretext_generator.generate()` | `main.py`, `variation.py`, `similarity.py`, `nn_histogram.py` | 核心重构 | 复现一轮 Private Evolution 的服务端生成逻辑 |
| `pretext_bootstrap.generate()` | `llama_bootstrap.py` | 核心重构 | 作为第二阶段扩增生成器 |
| `pretext_histogram.score()` | `nn_histogram.py`, `similarity.py`, `variation.py` | 核心重构 | 将 histogram 改造成统一 scorer 输出 |
| `variation_engine.py` | `variation.py` | 直接抽取 + 解耦 | 独立封装 mask-fill 变异 |
| `embedding_engine.py` | `similarity.py` | 直接抽取 + 解耦 | 独立封装句向量和 lookahead embedding |
| `survivor_sampler.py` | `main.py`, `nn_histogram.py` | 逻辑级抽取 | 只供 generator 使用 |
| `batch_utils.py` | `custom_datasets.py` | 可选借鉴 | 仅在平台缺少轻量 dataset 封装时使用 |

## 5.2 推荐的新平台文件拆分

建议新平台不要把 `PrE-Text` 原始文件整体放进 `adapters/`，而是拆成 5 个更清晰的部件：

1. `thesis_platform/adapters/generators/pretext_generator.py`
   - 实现 `generate(round_ctx) -> list[Sample]`
2. `thesis_platform/adapters/generators/pretext_bootstrap.py`
   - 实现 bootstrap 扩增
3. `thesis_platform/adapters/scorers/pretext_histogram.py`
   - 实现 `score(samples, client_ctx) -> list[ScoredSample]`
4. `thesis_platform/adapters/generators/pretext_variation.py`
   - 抽离 `variation.py` 的变异内核
5. `thesis_platform/adapters/generators/pretext_similarity.py`
   - 抽离 embedding 与 lookahead 逻辑

这样做的原因是：

1. `generator` 和 `scorer` 可以共用 `variation` 与 `similarity`。
2. `pretext_hist` 不需要复用完整生成循环。
3. 后续如果替换 embedding backend 或 bootstrap backend，不会影响主生成器外壳。

## 6. 与原仓库的关键差异

## 6.1 原仓库是 DP Private Evolution 脚本，平台里是可插拔生成器

原 `main.py` 的目标是一次性跑完整个 PrE-Text 实验。

而平台里的 `pretext_generator` 只是整个大流程中的一个阶段。

所以：

1. 不能把 `main.py` 原样作为子进程调用。
2. 必须把它拆成若干个可组合函数：
   - 初始化候选池
   - 变异
   - lookahead 打分
   - 重采样
   - 导出样本

## 6.2 原仓库的 histogram 语义与平台 scorer 语义不一致

原始 histogram 越大，说明候选越像目标分布，越应该保留。

但平台的 scorer 阶段通常是找“坏样本”。

因此必须在平台里显式定义：

1. `quality_score = histogram`
2. `bad_score = -quality_score`

然后统一到平台的 `score_direction` 语义中。

这是 `pretext_hist` 抽取时最关键的转换。

## 6.3 平台 generator 配置没有直接暴露 DP 参数

当前 `THESIS_PLATFORM_PLAN.md` 中 generator 配置没有直接包含：

1. `sigma`
2. `delta`
3. `sensitivity`
4. `H_multiplier`

而这些参数在原始 `PrE-Text` 中非常关键。

这说明平台目前更偏向先抽取“生成机制”做工程验证，而不是先把完整 DP 论文设定全盘带入。

因此推荐的实现策略是：

1. 第一版平台：
   - 只保留 `mask/lookahead/multiplier/t_steps/bootstrap_*`
   - 将 DP 噪声路径作为可选开关
2. 第二版迁移到私有联邦数据时：
   - 再新增 `privacy:` 配置块，承接 `sigma/delta/sensitivity/H`

## 6.4 原仓库 bootstrap 依赖文件协议，平台里应改成对象接口

原 `llama_bootstrap.py` 是从 11 轮 `surviving_text_it*.json` 文件反读 seed。

平台里不应保留这种文件耦合，而应该改成：

1. 直接接收 `seed_samples`
2. 输出 `bootstrap_samples`

## 7. 新平台里的实现建议

## 7.1 `pretext_generator` 的推荐流程

建议新平台中的一轮 `pretext_generator` 固定为：

1. 从 `init_population_path` 或上一轮 prompt 派生的候选池中采样初始父代
2. 使用 `variation_engine` 生成候选变体
3. 使用 `pretext_histogram` 或其底层评分函数做质量评估
4. 重采样 surviving parents
5. 输出本轮生成样本

这样做的好处是：

1. 与原论文主逻辑一致
2. 与平台 round-based 编排结构一致
3. 可以复用 `pretext_hist` 作为独立 scorer baseline

## 7.2 `pretext_histogram` scorer 的推荐实现

建议不要让 scorer 再承担“采样 survivors”的职责，而只返回分数：

```python
score(samples, client_ctx) -> list[ScoredSample]
```

推荐输出：

```python
[
  {
    "sample_id": "...",
    "score": -12.4,
    "score_name": "pretext_hist",
    "score_direction": "larger_is_worse",
    "meta": {
      "raw_histogram": 12.4,
      "lookahead": 4,
      "mask": 0.3
    }
  }
]
```

说明：

1. `raw_histogram` 保留原始支持度
2. `score` 则变成平台统一的坏样本分数

## 7.3 `pretext_bootstrap` 的推荐实现

建议实现为平台里的第二生成器模式：

1. 输入：
   - `seed_samples`
   - `bootstrap_model`
   - `generated_per_round`
2. 输出：
   - 新的 synthetic samples

并把下面这些从脚本硬编码改成配置：

1. 提示模板
2. 每条 prompt 拼接多少 seed 样本
3. 一轮生成多少条样本
4. `max_tokens`
5. `temperature` / `top_p`

## 8. 最终抽取清单

## 8.1 必须抽取

1. `variation.py`
2. `similarity.py`
3. `nn_histogram.py`
4. `llama_bootstrap.py`
5. `main.py` 中的生成-筛选-重采样流程关系

## 8.2 可选抽取

1. `custom_datasets.py`
2. `main.py` 中的 Opacus 隐私会计逻辑
3. `main.py` 中的输出结构设计

## 8.3 不抽取

1. `eval_distilgpt2.py`
2. `eval_llama2.py`
3. `README.md`
4. `quick_start.md`
5. `assets/*`

## 9. 推荐实施顺序

1. 先实现 `pretext_variation.py`
2. 再实现 `pretext_similarity.py`
3. 再实现 `pretext_histogram.py`
4. 然后封装 `pretext_generator.py`
5. 再封装 `pretext_bootstrap.py`
6. 最后决定是否补上 `privacy` 配置块，支持正式私有联邦实验

## 10. 最终结论

面向新的创新项目，`PrE-Text` 需要抽取的不是整套原始 DP 实验，而是三块非常明确的核心能力：

1. 候选文本变异生成
2. 基于 lookahead histogram 的分布贴近度打分
3. 基于 seed 样本的 bootstrap 扩增

其中：

1. `variation.py + similarity.py + nn_histogram.py` 共同构成平台中的 `pretext_generator` 和 `pretext_hist` 核心。
2. `llama_bootstrap.py` 对应平台中的 `pretext_bootstrap`。
3. `main.py` 只保留流程编排思想，不直接作为平台运行单元。
4. 两个 eval 脚本不进入平台主链路。

因此，新的平台实现应采用“生成内核抽取 + 打分语义转换 + bootstrap 独立封装”的方案，而不是“把 PrE-Text 原仓库当成一个黑盒生成程序直接调用”。
