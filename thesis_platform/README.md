# thesis_platform

`thesis_platform` 是当前论文创新算法实验平台的首版可运行 MVP。它把你在 `docs/` 中定义的主流程拆成统一接口和可替换模块，目标是先跑通一条稳定、可扩展、可迁移到 Linux 的实验主链，而不是直接把各论文仓库的原始训练脚手架硬拼在一起。

当前平台已经能跑通下面这条闭环：

```text
generator -> scorer -> selector -> retriever -> critic -> aggregator -> prompt update -> evaluator
```

当前已接入并可运行的模块：

- `pretext_seed`：借鉴 `PrE-Text` 的变异生成逻辑。
- `pretext_hist`：借鉴 `PrE-Text` 的直方图支持度打分思路。
- `datainf`：借鉴 `DataInf` 的坏样本效用打分思想，当前是轻量近似实现。
- `gradmm`：借鉴 `GRADMM` 的梯度匹配坏样本筛选思想，当前是轻量近似实现。
- `knn`：基于 embedding 的真实样本召回。
- `fedtextgrad_qwen`：借鉴 `FedTextGrad` 的 textual feedback 设计，当前为启发式对比式 critique 生成。
- `summarization`、`uid`：服务端 critique 聚合器。

当前仅保留接口、默认禁用的模块：

- `ira`
- `dbscan_attn`
- `dbscan_attn_tsgdm`

这些模块已经有类、注册表和配置文件，但会在运行时显式报出“not enabled in MVP”语义，避免误认为已经实现完成。

## 1. 项目目标

本平台解决的是“多篇算法论文能力的统一编排”问题，而不是单论文复现问题。它的设计目标是：

- 用统一 `Sample / ScoredSample / PairedSample / Critique / PromptUpdate` schema 串起整个实验链路。
- 用统一 `generate / score / retrieve / critique / aggregate` 接口替换论文原仓库互不兼容的脚本。
- 保证路径、配置、输出结构、测试入口都适合后续迁移到 Linux 服务器。
- 优先本地运行；本地模型不可用时提供依赖更少的回退实现。

## 2. 当前 MVP 的能力边界

### 2.1 已实现

- 单实验运行：`python -m thesis_platform.scripts.run_experiment --config ...`
- 批量实验运行：`python -m thesis_platform.scripts.run_matrix --config_dir ...`
- 结果汇总：`python -m thesis_platform.scripts.summarize_results --input ...`
- 三个实验配置已经可跑：
  - `smoke_pretext_hist_congressional`
  - `smoke_datainf_uid_congressional`
  - `scorer_gradmm_uid_openreview`
- 单元测试已经迁移到包内目录：`thesis_platform/tests/`

### 2.2 当前限制

- `datainf` 和 `gradmm` 目前是“面向平台联调的轻量近似实现”，保留接口位置和核心思想，但没有把原论文仓库的完整训练流程直接搬进来。
- `fedtextgrad_qwen` 当前默认使用启发式后端，而不是完整的 LLM textual gradient 计算图。
- `SentenceTransformer` 模型只有在本地路径存在且依赖安装完成时才会启用；否则自动回退到 `HashingEmbedder`。
- `TransformersTextBackend` 仍是占位实现，当前不建议作为主实验路径使用。

## 3. 项目结构

```text
thesis_platform/
├─ README.md
├─ requirements.txt
├─ __init__.py
├─ adapters/
│  ├─ aggregators/
│  ├─ critics/
│  ├─ generators/
│  ├─ retrievers/
│  └─ scorers/
├─ algorithms/
│  ├─ aggregators/
│  ├─ critics/
│  ├─ generators/
│  ├─ retrievers/
│  └─ scorers/
├─ configs/
│  ├─ base/
│  ├─ experiments/
│  └─ methods/
├─ core/
├─ data/
├─ evaluation/
├─ models/
├─ prompts/
│  ├─ aggregation/
│  └─ critique/
├─ scripts/
└─ tests/
```

## 4. 各包与模块说明

### 4.1 `thesis_platform/__init__.py`

| 模块 | 作用 |
| --- | --- |
| `__init__.py` | 包入口与版本号定义，当前版本为 `0.1.0`。 |

### 4.2 `thesis_platform/core`

`core` 是平台骨架，负责配置、上下文、注册表、主流程和产物落盘。

| 模块 | 作用 |
| --- | --- |
| `config.py` | 负责加载 YAML 配置、解析 `inherits`、深合并配置、做相对路径解析，并产出 `ExperimentConfig`。 |
| `context.py` | 定义 `RoundContext`、`ClientContext`、`ServerContext`、`EvalContext` 等运行期上下文对象。 |
| `experiment_runner.py` | 平台顶层编排器。负责加载数据、划分客户端、构建适配器、循环执行每一轮实验并写出最终 summary。 |
| `io_utils.py` | 封装输出目录创建、JSON/JSONL/文本写盘等通用 I/O 逻辑。 |
| `logging_utils.py` | 提供统一 logger。 |
| `pipeline.py` | 给命令行入口提供一个简洁的 `run_pipeline(config_path)` 调用。 |
| `prompt_updater.py` | 将聚合得到的 `PromptUpdate` 应用到当前 server prompt。 |
| `registry.py` | 平台适配器注册中心。不同模块通过 `register/create/registered_names` 统一管理。 |
| `round_runner.py` | 负责执行单轮 `generate -> score -> retrieve -> critique -> aggregate` 主链，并写出该轮所有中间产物。 |
| `schemas.py` | 定义统一 schema：`Sample`、`ScoredSample`、`PairedSample`、`Critique`、`PromptUpdate`。 |
| `selector.py` | 从 scorer 输出中选出 top-k 坏样本。 |

### 4.3 `thesis_platform/data`

`data` 负责把原始数据整理成平台统一样本对象。

| 模块 | 作用 |
| --- | --- |
| `loaders.py` | 支持从 JSON 文件或 JSON 目录读取数据，并把不同 shape 的原始 payload 归一化为文本，再构造成 `Sample`。 |
| `partition.py` | 把一个数据集稳定地切分成多个 client bucket，并为每个 client 划分 train/validation。 |

当前 `loaders.py` 支持的数据形态：

- 一个 JSON 文件，顶层是 `list`
- 一个 JSON 文件，顶层是 `dict`
- 一个目录，目录下包含多个 `*.json`
- 元素内部可以是嵌套 `list/dict/str`，会自动扁平化为文本

### 4.4 `thesis_platform/models`

`models` 负责 embedding 后端和文本后端的抽象。

| 模块 | 作用 |
| --- | --- |
| `embedding.py` | 定义 `BaseEmbedder`、`HashingEmbedder`、`SentenceTransformerEmbedder`，并通过 `build_embedder` 自动选择后端。 |
| `backends.py` | 定义 `BaseTextBackend`、`HeuristicTextBackend`、`TransformersTextBackend`，为将来接入本地 LLM 预留统一接口。 |

后端选择逻辑：

- 如果 `retriever.embedding_model` 指向的本地模型目录存在，并且已安装 `sentence-transformers`，则使用 `SentenceTransformerEmbedder`
- 否则自动回退到 `HashingEmbedder`
- `critic.engine=heuristic` 时使用启发式文本后端，不要求本地 LLM

### 4.5 `thesis_platform/algorithms`

`algorithms` 放“算法核心逻辑”，尽量不直接依赖平台上下文对象，方便以后复用和替换。

#### `algorithms/math_utils.py`

| 函数 | 作用 |
| --- | --- |
| `dot` | 向量点积。 |
| `l2_norm` | L2 范数。 |
| `normalize` | 向量归一化。 |
| `mean_vector` | 多向量均值。 |
| `subtract` | 向量减法。 |
| `add` | 向量加法。 |
| `scale` | 向量按标量缩放。 |
| `cosine_similarity` | 计算余弦相似度。 |

#### `algorithms/generators`

| 模块 | 作用 |
| --- | --- |
| `pretext_variation.py` | `VariationEngine`，实现 PrE-Text 风格的 token 级 mask/replace/mutate 变异。 |

#### `algorithms/scorers`

| 模块 | 作用 |
| --- | --- |
| `pretext_histogram.py` | 计算候选样本相对私有样本分布支持度，输出 PrE-Text 风格坏样本分数。 |
| `datainf_core.py` | 计算轻量版 DataInf 分数，模拟“样本对验证参考集的负面影响”大小。 |
| `gradmm_core.py` | 计算轻量版 GRADMM 分数，将近似“梯度匹配差异”与文本稀有度组合起来。 |

#### `algorithms/retrievers`

| 模块 | 作用 |
| --- | --- |
| `knn_core.py` | 给定 query 向量和 corpus 向量，基于余弦相似度返回 top-k 索引。 |

#### `algorithms/critics`

| 模块 | 作用 |
| --- | --- |
| `contrastive_critic_core.py` | 对 bad sample 与 real sample 做 token 级对比，生成 critique 规则，并支持简单脱敏。 |

#### `algorithms/aggregators`

| 模块 | 作用 |
| --- | --- |
| `summarization_core.py` | 将多个 client critique 规则归一、计数、排序，得到服务端 prompt 更新。 |

### 4.6 `thesis_platform/adapters`

`adapters` 把 `algorithms` 层和 `core` 层连接起来。每个 adapter 都遵守统一接口，便于在配置中切换方法。

#### `adapters/generators`

| 模块 | 作用 |
| --- | --- |
| `pretext_generator.py` | `PretextSeedGenerator`。从 public seed 样本池中轮转取样，调用 `VariationEngine` 生成合成样本。 |

#### `adapters/scorers`

| 模块 | 作用 |
| --- | --- |
| `pretext_histogram.py` | `PretextHistogramScorer`。把 synthetic 样本与 client 私有样本 embedding 后送入 histogram scorer。 |
| `datainf_scorer.py` | `DataInfScorer`。以 client validation/train 样本为 anchor 计算 DataInf 风格分数。 |
| `gradmm_scorer.py` | `GradMMScorer`。以 client 训练样本为参考，计算 GRADMM 风格分数。 |
| `ira_scorer.py` | `IRAScorer`。当前为禁用占位接口。 |

#### `adapters/retrievers`

| 模块 | 作用 |
| --- | --- |
| `knn_retriever.py` | `KNNRetriever`。从 client 本地样本中取最相似的真实样本作为 bad sample 的对比锚点。 |
| `label_match.py` | `LabelMatchRetriever`。按 label 召回真实样本，适合后续做分类任务扩展。 |
| `none.py` | `NoRetriever`。关闭检索步骤时使用。 |

#### `adapters/critics`

| 模块 | 作用 |
| --- | --- |
| `fedtextgrad_critic.py` | `FedTextGradCritic`。把 `(x_bad, x_real)` 配对转换为结构化 critique 规则。 |
| `none.py` | `NoCritic`。关闭 critique 步骤时使用。 |

#### `adapters/aggregators`

| 模块 | 作用 |
| --- | --- |
| `summarization.py` | `SummarizationAggregator`。按频率和归一化规则进行聚合。 |
| `uid.py` | `UIDAggregator`。偏向保留跨客户端重复出现的高密度 critique 规则。 |
| `none.py` | `NoAggregator`。关闭 prompt 更新时使用。 |
| `dbscan_attn.py` | `DBSCANAttnAggregator`。当前禁用，占位接口。 |
| `dbscan_attn_tsgdm.py` | `DBSCANAttnTSGDMAggregator`。当前禁用，占位接口。 |

### 4.7 `thesis_platform/evaluation`

| 模块 | 作用 |
| --- | --- |
| `metrics.py` | 计算生成样本统计、critique 统计、系统耗时和 prompt 长度等指标。 |

### 4.8 `thesis_platform/scripts`

| 模块 | 作用 |
| --- | --- |
| `run_experiment.py` | 运行单个实验配置。 |
| `run_matrix.py` | 扫描一个目录下的所有实验 YAML 并逐个执行。 |
| `summarize_results.py` | 汇总多个实验目录下的 `metrics_summary.json`。 |

### 4.9 `thesis_platform/configs`

`configs` 采用“基础配置 + 方法预设 + 实验配置”三层结构。

| 目录 | 作用 |
| --- | --- |
| `configs/base/` | 路径、运行时和基础评估设置。 |
| `configs/methods/` | 生成器、打分器、检索器、批判器、聚合器的预设配置。 |
| `configs/experiments/` | 具体实验组合配置，通常通过 `inherits` 继承多个基础配置。 |

当前内置实验：

| 配置文件 | 说明 |
| --- | --- |
| `configs/experiments/smoke/smoke_pretext_hist_congressional.yaml` | `pretext_seed + pretext_hist` 的基础 smoke 路径。 |
| `configs/experiments/smoke/smoke_datainf_uid_congressional.yaml` | `pretext_seed + datainf + knn + fedtextgrad_qwen + uid` 的 smoke 路径。 |
| `configs/experiments/scorer_selection/scorer_gradmm_uid_openreview.yaml` | `pretext_seed + gradmm + knn + fedtextgrad_qwen + uid` 的 scorer 选择路径。 |

### 4.10 `thesis_platform/prompts`

| 目录 | 作用 |
| --- | --- |
| `prompts/critique/contrastive_critic_v1.txt` | 对比式 critique 提示模板。 |
| `prompts/aggregation/summarize_v1.txt` | 规则总结提示模板。 |
| `prompts/aggregation/uid_v1.txt` | UID 风格规则聚合提示模板。 |

当前 MVP 的 heuristic backend 没有直接强依赖这些模板，但模板目录已经为后续切换真实 LLM 版本预留好位置。

### 4.11 `thesis_platform/tests`

| 模块 | 作用 |
| --- | --- |
| `test_thesis_platform_adapters.py` | 测试 generator、scorer、retriever、critic、aggregator 的基本输出类型。 |
| `test_thesis_platform_config.py` | 测试配置解析、路径解析与默认 smoke 配置读取。 |
| `test_thesis_platform_pipeline.py` | 测试单轮管线是否生成必须的输出文件。 |

## 5. 环境准备

### 5.1 Python 版本

推荐：

- Python `3.10+`
- 已验证开发环境：Python `3.11`

### 5.2 创建虚拟环境

Windows PowerShell:

```powershell
cd D:\学习记录\导师项目\研究\caiqiyue_file
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux / macOS Bash:

```bash
cd /path/to/caiqiyue_file
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 5.3 安装第三方依赖

最小运行依赖：

```bash
pip install -r thesis_platform/requirements.txt
```

如果希望 `knn` 检索器优先使用本地 `SentenceTransformer` 模型，而不是哈希向量回退实现，再额外安装：

```bash
pip install sentence-transformers
```

建议说明：

- `PyYAML`：标准 YAML 解析器。即使不安装，平台也有一个受限的内置 YAML fallback parser，但建议安装。
- `sentence-transformers`：只有在你本地提供了 embedding 模型目录时才有意义。
- 当前 MVP 不强制要求 `torch`、`transformers` 本地推理链路，因为主实验路径默认可用 `heuristic` backend 跑通。

## 6. 数据准备要求

### 6.1 支持的输入路径

`data.train_path` 和 `data.public_seed_path` 支持：

- 单个 JSON 文件
- 一个包含多个 JSON 文件的目录

### 6.2 支持的 JSON 形态

示例 1：顶层是 `list`

```json
[
  "sample a",
  "sample b"
]
```

示例 2：顶层是 `dict`

```json
{
  "0": ["sample a"],
  "1": ["sample b"]
}
```

示例 3：元素内部允许嵌套

```json
[
  {"instruction": "do x", "response": "answer x"},
  ["nested", {"value": "sample"}]
]
```

平台会把这些结构自动扁平化为文本，然后再包装成统一 `Sample` 对象。

## 7. 配置系统说明

### 7.1 配置继承

每个实验 YAML 都可以通过 `inherits` 引入多个上游配置片段。例如：

```yaml
inherits:
  - ../../base/paths.yaml
  - ../../base/runtime.yaml
  - ../../methods/generators/pretext_seed.yaml
  - ../../methods/scorers/datainf.yaml
  - ../../methods/retrievers/knn.yaml
  - ../../methods/critics/fedtextgrad_qwen.yaml
  - ../../methods/aggregators/uid.yaml
```

平台会按继承顺序深合并配置，后出现的字段覆盖前面的字段。

### 7.2 顶层配置段含义

| 配置段 | 说明 |
| --- | --- |
| `meta` | 实验标识与随机种子。 |
| `paths` | 仓库根目录、输出目录、缓存目录等路径配置。 |
| `data` | 数据集路径、任务类型、client 划分方式。 |
| `federation` | 联邦轮数、每个 client 选取多少坏样本。 |
| `generator` | 生成器名称与生成超参数。 |
| `scorer` | 打分器名称与打分超参数。 |
| `retriever` | 真实样本召回模块配置。 |
| `critic` | 批判器配置。 |
| `aggregator` | 聚合器配置。 |
| `runtime` | 设备、日志等级等运行时配置。 |
| `evaluation` | 中间产物保存等评估相关设置。 |

### 7.3 通用参数解释

#### `meta`

| 参数 | 说明 |
| --- | --- |
| `experiment_id` | 实验名称，决定输出目录名。 |
| `stage` | 实验阶段标签，例如 `smoke`、`scorer_selection`。 |
| `seed` | 实验全局随机种子。 |

#### `paths`

| 参数 | 说明 |
| --- | --- |
| `repo_root` | 仓库根目录，其他相对路径默认基于它解析。 |
| `output_root` | 实验输出根目录。 |
| `cache_root` | 缓存目录。 |
| `model_root` | 预留字段，当前代码未直接消费。 |
| `dataset_root` | 预留字段，当前代码未直接消费。 |

#### `data`

| 参数 | 说明 |
| --- | --- |
| `dataset_name` | 数据集名称，写入样本元信息和输出结果。 |
| `task_type` | 任务类型，当前示例为 `instruction_tuning`。 |
| `train_path` | 训练数据路径。 |
| `public_seed_path` | 公共 seed 数据路径，供 generator 生成合成样本。 |
| `num_clients` | 将训练集切分为多少个 client。 |
| `max_samples_per_client` | 每个 client 最多保留多少条样本。 |
| `validation_ratio` | 从每个 client 的样本中抽多少比例作为 validation anchor。 |

#### `federation`

| 参数 | 说明 |
| --- | --- |
| `rounds` | 总轮数。 |
| `top_k_bad` | 每个 client 每轮保留多少条坏样本继续进入下游流程。 |

#### `runtime`

| 参数 | 说明 |
| --- | --- |
| `device` | 运行设备标记，当前 MVP 主要用于记录，不强绑定底层张量库。 |
| `log_level` | 日志等级。 |

#### `evaluation`

| 参数 | 说明 |
| --- | --- |
| `save_round_artifacts` | 是否保留每轮中间产物。当前实现默认都会落盘关键文件。 |

### 7.4 方法级参数解释

#### `generator: pretext_seed`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `pretext_seed`。 |
| `initial_prompt` | 第 0 轮 server prompt。 |
| `generated_per_round` | 每轮生成多少条 synthetic 样本。 |
| `mask` | 变异时 token mask 比例。 |
| `t_steps` | 变异步数。 |
| `seed` | 生成器内部随机种子。 |

#### `scorer: pretext_hist`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `pretext_hist`。 |
| `score_direction` | 坏样本分数方向，当前约定为 `larger_is_worse`。 |

#### `scorer: datainf`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `datainf`。 |
| `score_direction` | 分数方向。 |
| `lambda_const_param` | 轻量版 DataInf 分数中的平滑常数，控制与 validation anchor 的对比强度。 |

#### `scorer: gradmm`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `gradmm`。 |
| `score_direction` | 分数方向。 |
| `alpha` | 稀有度项与匹配项的组合权重。 |

#### `scorer: ira`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `ira`。 |
| 其他参数 | 当前未启用，保留接口位置。 |

#### `retriever: knn`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `knn`。 |
| `embedding_model` | 本地 embedding 模型目录，存在时优先启用 `SentenceTransformerEmbedder`。 |
| `top_k` | 每个坏样本检索多少条真实样本。 |

#### `retriever: none`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `none`，表示跳过检索。 |

#### `critic: fedtextgrad_qwen`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `fedtextgrad_qwen`。 |
| `engine` | 当前主要作为兼容字段保留；MVP 默认使用启发式对比式 critic。 |
| `compress_to_n_rules` | 每个 pair 最多保留多少条 critique 规则。 |
| `redact_enable` | 是否对 critique 文本做简单脱敏。 |

#### `critic: none`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `none`，表示跳过 critique。 |

#### `aggregator: uid`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `uid`。 |
| `max_rules` | 聚合后最多保留多少条全局规则。 |

#### `aggregator: summarization`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `summarization`。 |
| `max_rules` | 聚合后最多保留多少条规则。 |

#### `aggregator: none`

| 参数 | 说明 |
| --- | --- |
| `name` | 固定为 `none`，表示不做 prompt 更新。 |

#### `aggregator: dbscan_attn` / `dbscan_attn_tsgdm`

| 参数 | 说明 |
| --- | --- |
| `name` | 分别固定为 `dbscan_attn` 和 `dbscan_attn_tsgdm`。 |
| 其他参数 | 当前未启用，保留接口位置。 |

## 8. 如何启动实验

### 8.1 运行单个实验

```bash
python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/smoke/smoke_pretext_hist_congressional.yaml
```

也可以运行：

```bash
python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/smoke/smoke_datainf_uid_congressional.yaml
python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/scorer_selection/scorer_gradmm_uid_openreview.yaml
```

### 8.2 批量运行某个实验目录

```bash
python -m thesis_platform.scripts.run_matrix --config_dir thesis_platform/configs/experiments
```

### 8.3 汇总结果

```bash
python -m thesis_platform.scripts.summarize_results --input outputs/thesis_platform
```

## 9. 如何运行测试

测试已经位于包内目录 `thesis_platform/tests/`，推荐命令：

```bash
python -m unittest discover -s thesis_platform/tests -p "test_thesis_platform*.py" -v
```

## 10. 输出目录说明

默认输出目录：

```text
outputs/thesis_platform/<experiment_id>/
```

每个实验目录下会生成：

| 文件 | 说明 |
| --- | --- |
| `metrics_summary.json` | 实验级汇总信息。 |
| `resolved_config.json` | 解析并继承后的最终配置快照。 |
| `config.yaml` | 当前实现写出的配置快照文本。 |
| `round_000/` | 第 0 轮输出目录。多轮实验会继续产生 `round_001/`、`round_002/`。 |

每个轮次目录下会生成：

| 文件 | 说明 |
| --- | --- |
| `server_prompt.txt` | 该轮开始时服务端 prompt。 |
| `generated_samples.jsonl` | generator 输出的 synthetic 样本。 |
| `scored_samples.jsonl` | scorer 对每个 client 打分后的样本。 |
| `selected_bad_samples.jsonl` | 被选中的坏样本。 |
| `retrieved_pairs.jsonl` | bad sample 与召回真实样本的配对结果。 |
| `client_critiques.jsonl` | client 侧生成的 critique。 |
| `prompt_update.json` | 聚合后得到的 prompt update。 |
| `round_metrics.json` | 该轮指标。 |

## 11. 当前推荐的开发与使用方式

- 想先验证平台主链是否正常，优先跑 `smoke_pretext_hist_congressional.yaml`
- 想验证完整主链，优先跑 `smoke_datainf_uid_congressional.yaml`
- 想对 scorer 方案做对比，优先跑 `scorer_gradmm_uid_openreview.yaml`
- 想扩展新方法时，优先新增 `algorithms/` 核心逻辑，再在 `adapters/` 下接统一接口，再补 `configs/methods/` 预设

## 12. 后续建议

如果你接下来要继续把平台从 MVP 推到论文实验版，建议按这个顺序推进：

1. 把 `ira` scorer 从占位接口补成真实实现。
2. 把 `dbscan_attn` 和 `dbscan_attn_tsgdm` 补成真实 server 聚合器。
3. 为 `fedtextgrad_qwen` 接上真实本地 LLM 或 API backend。
4. 把 `datainf`、`gradmm` 从轻量近似打分替换成更贴近原论文的训练式实现。
5. 增加更系统的评估指标，例如 `accuracy / f1 / rougeL / upload budget / latency per client` 的任务化版本。
