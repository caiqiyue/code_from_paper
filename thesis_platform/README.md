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
├─ dataset_downloaders/
├─ evaluation/
├─ model_downloaders/
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

### 4.11 `thesis_platform/dataset_downloaders` 与 `thesis_platform/model_downloaders`

这两个包是资源准备层，不接入现有实验主链，只负责把论文涉及的数据集和开源模型按统一入口下载到平台目录。

数据集下载子系统：

- 入口：`python -m thesis_platform.scripts.download_datasets`
- 默认输出目录：`thesis_platform/datasets/`
- 支持 `--list`、`--names ...`、`--force`
- 目录结构统一为 `datasets/<name>/metadata.json`，以及按需生成的 `raw/` 与 `formatted/`
- `download_datasets` 会先准备 raw，再调用独立 formatter 生成实验消费格式
- 对本来就符合实验格式的数据集，`formatted_path` 会直接复用 `raw_path`
- `imdb` 与 `rt_polarity` 的 formatted 结果以 `../GRADMM/data/...` 中 vendored JSONL 为权威来源
- DataInf 三个生成型数据集没有单独 raw 目录，直接把共享脚本产物收口到各自 `formatted/`
- 总控会显示进度条，单个数据集失败后跳过，最终写出 `datasets/download_report.json`

模型下载子系统：

- 入口：`python -m thesis_platform.scripts.download_models`
- 默认输出目录：`thesis_platform/open_model/`
- 支持 `--list`、`--names ...`、`--force`、`--include-optional`
- 支持重复参数 `--repo-override <model_name>=<huggingface_repo_id>`
- 默认只下载核心模型；显式加 `--include-optional` 或通过 `--names` 指定时才尝试 gated / 超大模型
- 所有 Llama 模块默认都使用社区镜像，不回退到官方 `meta-llama/*`
- Llama 镜像下载前会检查仓库是否为 Transformers 兼容格式；纯 GGUF / Ollama / llamafile 仓库会被拒绝
- `llama_3_1_405b_instruct` 默认指向 FP8 / compressed-tensors 社区镜像，保留为 optional，主要面向推理侧
- 总控会显示进度条，单个模型失败后跳过，最终写出 `open_model/download_report.json`

当前明确纳入下载模块、但不进默认全量模型下载的 optional 模型包括：

- `llama_2_13b_chat_hf`
- `opt_1_3b`
- `llama_3_2_3b_instruct`
- `llama_3_2_11b_vision_instruct`
- `deepseek_r1_distill_llama_70b`
- `llama_3_1_405b_instruct`

当前只在文档中保留说明、**不创建下载模块** 的模型项：

- `GPT-4`
- `GPT-4o`
- `GPT-3.5`
- 只写成模型族名的 `Llama 3`、`Llama 3.1`、`Qwen 2`

### 4.12 `thesis_platform/tests`

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
- `datasets`、`huggingface_hub`、`tqdm`、`numpy`、`pandas`：下载子系统需要的最小依赖。
- `sentence-transformers`：只有在你本地提供了 embedding 模型目录时才有意义。
- 当前 MVP 不强制要求 `torch`、`transformers` 本地推理链路；下载模型本身也不依赖 `transformers`。

### 5.4 下载论文数据集与开源模型

下载论文涉及的数据集：

```bash
python -m thesis_platform.scripts.download_datasets
```

只下载指定数据集：

```bash
python -m thesis_platform.scripts.download_datasets --names glue_sst2 gsm8k
```

强制重下：

```bash
python -m thesis_platform.scripts.download_datasets --force
```

数据集落盘说明：

```text
datasets/
  <dataset_name>/
    metadata.json
    raw/        # 如果存在稳定原始下载源
    formatted/  # 实验实际消费的格式化结果
```

其中：
- `glue_*`、`rotten_tomatoes`、`three_styles_prompted_250_512x512` 会直接复用 raw 作为 formatted
- `twitter_emotion_binary` 会把 `dair-ai/emotion` 过滤成 `label in [0, 1]`
- `gsm8k` 会额外生成 DSPy 风格 `train.jsonl` / `val.jsonl` / `test.jsonl`
- `livebench_*` 会先按任务过滤，再固定 seed=0 生成本地 JSONL 切分
- `imdb`、`rt_polarity` 的 formatted 结果来自 GRADMM 仓库内 vendored JSONL
- `datainf_*` 会把共享生成脚本产物移动到各自 `formatted/train.hf` 与 `formatted/test.hf`

下载默认核心模型：

```bash
python -m thesis_platform.scripts.download_models
```

下载可选模型：

```bash
python -m thesis_platform.scripts.download_models --include-optional
```

只下载指定模型：

```bash
python -m thesis_platform.scripts.download_models --names roberta_large llama_2_13b_chat_hf
```

覆盖某个 Llama 默认镜像：

```bash
python -m thesis_platform.scripts.download_models \
  --names llama_3_1_8b_instruct \
  --repo-override llama_3_1_8b_instruct=custom-user/Llama-3.1-8B-Instruct
```

补充说明：

- 所有下载路径都由包内相对路径推导，不要求你在仓库根目录执行命令。
- 数据集总控默认会覆盖论文中整理出的所有数据集模块，包括 DataInf 的本地生成数据集包装模块。
- 模型总控默认跳过 optional 模型，避免把 gated 或超大模型作为首次准备的硬依赖。
- `download_models --list` 会显示 `default_repo_id`、`optional` 与 `community_mirror_only`。
- 所有 Llama 默认来源都改成社区镜像；如果某个镜像失效，优先用 `--repo-override` 调整。
- 任一数据集或模型下载失败后，总控会继续处理剩余项，并把失败信息写进 `download_report.json`。

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

## 13. 已下载数据集清单

本次数据集整理时间为 `2026-03-11T13:37:42.524942+00:00`，下载根目录为 `datasets/`。共请求 `20` 个数据集，结果为：`downloaded=20`、`skipped=0`、`failed=0`、`total=20`。

| 数据集 | 状态 | 格式化器 | 路径 | 样本统计 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `bbh_multistep_arithmetic_two` | `downloaded` | `bbh` | `root: datasets/bbh_multistep_arithmetic_two`<br>`raw: datasets/bbh_multistep_arithmetic_two/raw`<br>`formatted: datasets/bbh_multistep_arithmetic_two/formatted`<br>`metadata: datasets/bbh_multistep_arithmetic_two/metadata.json` | `raw: multistep_arithmetic_two=250, total=250`<br>`formatted: test=100, train=50, val=100, total=250` | Download the BBH multistep arithmetic two task and recreate train/val/test CSV splits.<br>Created FedTextGrad-compatible train/val/test CSV splits for BBH. |
| `bbh_object_counting` | `downloaded` | `bbh` | `root: datasets/bbh_object_counting`<br>`raw: datasets/bbh_object_counting/raw`<br>`formatted: datasets/bbh_object_counting/formatted`<br>`metadata: datasets/bbh_object_counting/metadata.json` | `raw: object_counting=250, total=250`<br>`formatted: test=100, train=50, val=100, total=250` | Download the BBH object counting task and recreate train/val/test CSV splits.<br>Created FedTextGrad-compatible train/val/test CSV splits for BBH. |
| `datainf_grammars` | `downloaded` | `datainf` | `root: datasets/datainf_grammars`<br>`raw: -`<br>`formatted: datasets/datainf_grammars/formatted`<br>`metadata: datasets/datainf_grammars/metadata.json` | `formatted: test=100, train=900, total=1000` | Generate the DataInf sentence transformation dataset into thesis_platform/datasets.<br>Generated DataInf dataset artifacts are ready in the formatted directory. |
| `datainf_math_with_reason` | `downloaded` | `datainf` | `root: datasets/datainf_math_with_reason`<br>`raw: -`<br>`formatted: datasets/datainf_math_with_reason/formatted`<br>`metadata: datasets/datainf_math_with_reason/metadata.json` | `formatted: test=100, train=900, total=1000` | Generate the DataInf math-with-reason dataset into thesis_platform/datasets.<br>Generated DataInf dataset artifacts are ready in the formatted directory. |
| `datainf_math_without_reason` | `downloaded` | `datainf` | `root: datasets/datainf_math_without_reason`<br>`raw: -`<br>`formatted: datasets/datainf_math_without_reason/formatted`<br>`metadata: datasets/datainf_math_without_reason/metadata.json` | `formatted: test=100, train=900, total=1000` | Generate the DataInf math-without-reason dataset into thesis_platform/datasets.<br>Generated DataInf dataset artifacts are ready in the formatted directory. |
| `glue_mrpc` | `downloaded` | `glue_datainf` | `root: datasets/glue_mrpc`<br>`raw: datasets/glue_mrpc/raw`<br>`formatted: datasets/glue_mrpc/formatted`<br>`metadata: datasets/glue_mrpc/metadata.json` | `raw: train=3668, validation=408, total=4076`<br>`formatted: train=3668, validation=408, total=4076` | Download the GLUE MRPC splits used by DataInf.<br>Created the train/validation GLUE subset used by DataInf classification experiments. |
| `glue_qnli` | `downloaded` | `glue_datainf` | `root: datasets/glue_qnli`<br>`raw: datasets/glue_qnli/raw`<br>`formatted: datasets/glue_qnli/formatted`<br>`metadata: datasets/glue_qnli/metadata.json` | `raw: train=104743, validation=5463, total=110206`<br>`formatted: train=4500, validation=500, total=5000` | Download the GLUE QNLI splits used by DataInf.<br>Created the train/validation GLUE subset used by DataInf classification experiments. |
| `glue_qqp` | `downloaded` | `glue_datainf` | `root: datasets/glue_qqp`<br>`raw: datasets/glue_qqp/raw`<br>`formatted: datasets/glue_qqp/formatted`<br>`metadata: datasets/glue_qqp/metadata.json` | `raw: train=363846, validation=40430, total=404276`<br>`formatted: train=4500, validation=500, total=5000` | Download the GLUE QQP splits used by DataInf.<br>Created the train/validation GLUE subset used by DataInf classification experiments. |
| `glue_sst2` | `downloaded` | `glue_datainf` | `root: datasets/glue_sst2`<br>`raw: datasets/glue_sst2/raw`<br>`formatted: datasets/glue_sst2/formatted`<br>`metadata: datasets/glue_sst2/metadata.json` | `raw: train=67349, validation=872, total=68221`<br>`formatted: train=4500, validation=500, total=5000` | Download the GLUE SST-2 splits used by DataInf and GRADMM.<br>Created the train/validation GLUE subset used by DataInf classification experiments. |
| `glue_wnli` | `downloaded` | `glue_datainf` | `root: datasets/glue_wnli`<br>`raw: datasets/glue_wnli/raw`<br>`formatted: datasets/glue_wnli/formatted`<br>`metadata: datasets/glue_wnli/metadata.json` | `raw: train=635, validation=71, total=706`<br>`formatted: train=635, validation=71, total=706` | Download the GLUE WNLI splits used by DataInf.<br>Created the train/validation GLUE subset used by DataInf classification experiments. |
| `gsm8k` | `downloaded` | `gsm8k` | `root: datasets/gsm8k`<br>`raw: datasets/gsm8k/raw`<br>`formatted: datasets/gsm8k/formatted`<br>`metadata: datasets/gsm8k/metadata.json` | `raw: train=7473, test=1319, total=8792`<br>`formatted: test=100, train=50, val=100, total=250` | Download GSM8K main and record the DSPy split rules used by FedTextGrad.<br>Created deterministic DSPy-style GSM8K JSONL splits. |
| `imdb` | `downloaded` | `imdb` | `root: datasets/imdb`<br>`raw: -`<br>`formatted: datasets/imdb/formatted`<br>`metadata: datasets/imdb/metadata.json` | `formatted: train_len256=408, validation_len256=462, total=870` | Stage the vendored IMDB len256 subset used by GRADMM.<br>Copied vendored GRADMM IMDB len256 JSONL files into the formatted dataset directory. |
| `livebench_math_amps_hard` | `downloaded` | `livebench` | `root: datasets/livebench_math_amps_hard`<br>`raw: datasets/livebench_math_amps_hard/raw`<br>`formatted: datasets/livebench_math_amps_hard/formatted`<br>`metadata: datasets/livebench_math_amps_hard/metadata.json` | `raw: dataset=150, total=150`<br>`formatted: test=30, train=96, valid=24, total=150` | Download the LiveBench math AMPS-Hard subset used in FedTextGrad.<br>Filtered LiveBench task 'AMPS_Hard' and wrote deterministic JSONL splits. |
| `livebench_reasoning_spatial` | `downloaded` | `livebench` | `root: datasets/livebench_reasoning_spatial`<br>`raw: datasets/livebench_reasoning_spatial/raw`<br>`formatted: datasets/livebench_reasoning_spatial/formatted`<br>`metadata: datasets/livebench_reasoning_spatial/metadata.json` | `raw: dataset=50, total=50`<br>`formatted: test=10, train=32, valid=8, total=50` | Download the LiveBench reasoning spatial subset used in FedTextGrad.<br>Filtered LiveBench task 'spatial' and wrote deterministic JSONL splits. |
| `livebench_reasoning_web_of_lies_v2` | `downloaded` | `livebench` | `root: datasets/livebench_reasoning_web_of_lies_v2`<br>`raw: datasets/livebench_reasoning_web_of_lies_v2/raw`<br>`formatted: datasets/livebench_reasoning_web_of_lies_v2/formatted`<br>`metadata: datasets/livebench_reasoning_web_of_lies_v2/metadata.json` | `raw: dataset=50, total=50`<br>`formatted: test=10, train=32, valid=8, total=50` | Download the LiveBench reasoning Web of Lies V2 subset used in FedTextGrad.<br>Filtered LiveBench task 'web_of_lies_v2' and wrote deterministic JSONL splits. |
| `livebench_reasoning_zebra_puzzle` | `downloaded` | `livebench` | `root: datasets/livebench_reasoning_zebra_puzzle`<br>`raw: datasets/livebench_reasoning_zebra_puzzle/raw`<br>`formatted: datasets/livebench_reasoning_zebra_puzzle/formatted`<br>`metadata: datasets/livebench_reasoning_zebra_puzzle/metadata.json` | `raw: dataset=100, total=100`<br>`formatted: test=20, train=64, valid=16, total=100` | Download the LiveBench reasoning zebra puzzle subset used in FedTextGrad.<br>Filtered LiveBench task 'zebra_puzzle' and wrote deterministic JSONL splits. |
| `rotten_tomatoes` | `downloaded` | `identity` | `root: datasets/rotten_tomatoes`<br>`raw: datasets/rotten_tomatoes/raw`<br>`formatted: datasets/rotten_tomatoes/raw`<br>`metadata: datasets/rotten_tomatoes/metadata.json` | `raw: train=8530, validation=1066, total=9596`<br>`formatted: train=8530, validation=1066, total=9596` | Download the Rotten Tomatoes sentiment dataset used in GRADMM.<br>Raw dataset artifacts already match the experiment format. |
| `rt_polarity` | `downloaded` | `rt_polarity` | `root: datasets/rt_polarity`<br>`raw: -`<br>`formatted: datasets/rt_polarity/formatted`<br>`metadata: datasets/rt_polarity/metadata.json` | `formatted: train=1000, validation=1000, total=2000` | Copy the vendored RT-Polarity JSONL files bundled in GRADMM.<br>Copied vendored GRADMM RT-Polarity JSONL files into the formatted dataset directory. |
| `three_styles_prompted_250_512x512` | `downloaded` | `identity` | `root: datasets/three_styles_prompted_250_512x512`<br>`raw: datasets/three_styles_prompted_250_512x512/raw`<br>`formatted: datasets/three_styles_prompted_250_512x512/raw`<br>`metadata: datasets/three_styles_prompted_250_512x512/metadata.json` | `raw: train=600, val=150, total=750`<br>`formatted: train=600, val=150, total=750` | Download the DataInf style-transfer dataset from Hugging Face.<br>Raw dataset artifacts already match the experiment format. |
| `twitter_emotion_binary` | `downloaded` | `twitter_emotion_binary` | `root: datasets/twitter_emotion_binary`<br>`raw: datasets/twitter_emotion_binary/raw`<br>`formatted: datasets/twitter_emotion_binary/formatted`<br>`metadata: datasets/twitter_emotion_binary/metadata.json` | `raw: train=16000, validation=2000, total=18000`<br>`formatted: train=10028, validation=1254, total=11282` | Download the binary sadness/joy subset of dair-ai/emotion used in GRADMM.<br>Filtered dair-ai/emotion down to the sadness/joy binary subset. |
