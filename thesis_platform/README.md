# thesis_platform

`thesis_platform` 是一个面向论文实验整合的统一平台。它把 `PrE-Text`、`DataInf`、`GRADMM`、`FedTextGrad` 等工作的部分思路抽象成同一套 `schema + adapter + config` 体系，目标是先跑通一条稳定、可比较、可扩展的实验主链，而不是把每篇论文的原始训练仓库直接硬拼在一起。

当前主链为：

```text
generator -> scorer -> selector -> retriever -> critic -> aggregator -> prompt update -> evaluation
```

这个平台适合做三类事情：

1. 用统一接口快速比较不同 bad-sample 发现/筛选方法。
2. 在多客户端设置下观察 critique 聚合和 prompt 更新行为。
3. 逐步把论文思路迁移成可复用组件，而不是维护多套互不兼容脚本。

## 1. 当前状态

### 1.1 已可运行

| 类型 | 名称 | 说明 |
| --- | --- | --- |
| generator | `pretext_seed` | 从公共 seed 池出发，做 PrE-Text 风格的轻量变异生成 |
| scorer | `pretext_hist` | PrE-Text 风格直方图/分布支持度打分 |
| scorer | `datainf` | DataInf 风格轻量近似打分 |
| scorer | `gradmm` | GRADMM 风格轻量近似打分 |
| retriever | `knn` | 基于 embedding 的相似样本召回 |
| retriever | `label_match` | 按标签匹配召回，当前仓库未在内置配置中使用 |
| retriever | `none` | 关闭召回步骤 |
| critic | `fedtextgrad_qwen` | 启发式对比式 critique 生成，接口借鉴 FedTextGrad |
| critic | `none` | 关闭 critique 步骤 |
| aggregator | `uid` | 偏向保留高频、高密度规则的聚合器 |
| aggregator | `summarization` | 频次驱动的简单聚合器 |
| aggregator | `none` | 关闭 prompt 更新 |

### 1.2 已注册但明确禁用

这些模块已经保留了接口、注册和配置位置，但运行时会直接抛出 `not enabled in the MVP`：

- `ira`
- `dbscan_attn`
- `dbscan_attn_tsgdm`

### 1.3 重要边界

- `datainf` 和 `gradmm` 目前是“面向平台联调的轻量近似实现”，不是原论文的完整训练式复现。
- `fedtextgrad_qwen` 当前默认走启发式对比规则生成，不是完整 LLM textual gradient。
- `TransformersTextBackend` 仍是占位实现，当前不建议作为主实验路径使用。
- 默认数据加载器会把 JSON 样本压平成 `Sample.text`，`instruction/response/label` 字段在 schema 中已预留，但默认 loader 不会自动填充。

## 2. 核心设计

平台用一组统一数据结构串起全流程：

- `Sample`
- `ScoredSample`
- `PairedSample`
- `Critique`
- `PromptUpdate`

运行时由两层组成：

- `algorithms/`：尽量只放与平台上下文无关的算法核心逻辑
- `adapters/`：把算法核心接到统一的 `generate / score / retrieve / critique / aggregate` 接口上

顶层执行入口是：

```text
python -m thesis_platform.scripts.run_experiment
  -> thesis_platform.core.pipeline.run_pipeline
  -> thesis_platform.core.experiment_runner.ExperimentRunner
  -> thesis_platform.core.round_runner.RoundRunner
```

其中每一轮会执行：

1. 服务端根据公共 seed 生成 synthetic samples。
2. 每个 client 对同一批 synthetic samples 打分。
3. 对每个 client 选出 top-k bad samples。
4. 在 client 本地数据中召回相似真实样本。
5. 把 `(bad sample, real samples)` 翻译成 critique 规则。
6. 服务端聚合所有 critique，得到下一轮 prompt update。
7. 写出 round 级中间产物和 metrics。

## 3. 目录总览

下面的路径以仓库根目录 `caiqiyue_file/` 为基准：

| 路径 | 作用 |
| --- | --- |
| `thesis_platform/core/` | 配置解析、上下文对象、注册表、实验与 round 调度、I/O |
| `thesis_platform/algorithms/` | 生成、打分、召回、critique、聚合等算法核心 |
| `thesis_platform/adapters/` | 统一 adapter 实现与注册 |
| `thesis_platform/data/` | JSON 数据加载与 client 切分 |
| `thesis_platform/evaluation/` | 轻量指标统计 |
| `thesis_platform/configs/` | 基础配置、方法配置、实验配置 |
| `thesis_platform/scripts/` | CLI 入口：实验运行、矩阵运行、结果汇总、数据/模型下载 |
| `thesis_platform/tests/` | `unittest` 测试 |
| `thesis_platform/prompts/` | 预留的 critique / aggregation prompt 模板 |
| `thesis_platform/dataset_downloaders/` | 数据集下载/生成控制器 |
| `thesis_platform/dataset_formatters/` | 下载后格式化逻辑 |
| `thesis_platform/model_downloaders/` | 模型下载控制器 |
| `thesis_platform/open_model/` | 由模型下载器管理的模型快照目录 |
| `thesis_platform/datasets/` | 由数据下载器管理的 benchmark/格式化数据目录 |
| `datasets/` | 仓库根目录下的原始实验输入数据，内置 smoke 配置主要从这里读取 |
| `models/` | 仓库根目录下的历史本地模型目录，部分配置仍可直接引用 |
| `outputs/thesis_platform/` | 实验输出目录 |

## 4. 路径约定

这是当前项目最容易混淆的地方：

- `datasets/` 是仓库根目录下的实验输入目录。
  - 当前仓库已经包含 `congressional_train.json`、`congressional_eval.json`、`bioarxiv_train.json`、`bioarxiv_eval.json`、`initial_set.json`、`openreview_init_data/` 等文件。
  - 内置 smoke 配置主要使用这里的数据，因此很多实验可以直接跑，不必先执行下载器。
- `thesis_platform/datasets/` 是平台自己的数据下载器输出目录。
  - 这里放 GLUE、GSM8K、LiveBench、PrE-Text 近似数据、DataInf 生成数据等。
  - 每个数据集目录下会写 `metadata.json`，汇总报告写在 `thesis_platform/datasets/download_report.json`。
- `models/` 是仓库历史上已有的本地模型目录。
- `thesis_platform/open_model/` 是平台模型下载器的输出目录。
  - 汇总报告写在 `thesis_platform/open_model/download_report.json`。

`retriever.embedding_model` 只要指向一个真实存在的本地句向量模型目录，就会尝试启用 `SentenceTransformerEmbedder`；如果路径不存在、模型不完整或依赖缺失，则会自动退回 `HashingEmbedder`，不会因为缺少本地 embedding 模型而让整条主链失效。

## 5. 环境准备

下面所有命令都默认从仓库根目录 `caiqiyue_file/` 执行，而不是从 `thesis_platform/` 子目录执行。

### 5.1 Python

推荐：

- Python `3.10+`
- 已验证开发环境：Python `3.11`

### 5.2 创建虚拟环境

Windows PowerShell：

```powershell
cd D:\学习记录\导师项目\研究\caiqiyue_file
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r thesis_platform/requirements.txt
```

Linux / macOS：

```bash
cd /path/to/caiqiyue_file
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r thesis_platform/requirements.txt
```

`requirements.txt` 中包含平台当前主链和下载器所需的最小第三方依赖：

- `PyYAML`
- `datasets`
- `huggingface_hub`
- `numpy`
- `pandas`
- `sentence-transformers`
- `tqdm`

补充说明：

- 即使没有 `PyYAML`，项目也带有受限的 YAML fallback parser，但正常开发仍建议安装。
- 即使本地没有可用的 sentence-transformer 模型，平台也会退回哈希 embedding。
- DataInf 真正的上游训练/影响函数依赖没有并入这里；如果要跑其原始仓库流程，请单独处理 `DataInf/requirements.txt`。

## 6. 快速开始

### 6.1 最小 smoke 路径

只测试生成与打分，不做召回/critique/聚合：

```bash
python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/smoke/smoke_pretext_hist_congressional.yaml
```

### 6.2 完整一轮主链 smoke

包含 `datainf + knn + fedtextgrad_qwen + uid`：

```bash
python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/smoke/smoke_datainf_uid_congressional.yaml
```

### 6.3 scorer 对比 smoke

```bash
python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/scorer_selection/scorer_gradmm_uid_openreview.yaml
```

### 6.4 2 轮端到端实验

仓库内还提供了两套更接近完整链路的示例：

- `thesis_platform/configs/experiments/smoke/smoke_pretext_minilm_congressional.yaml`
- `thesis_platform/configs/experiments/smoke/smoke_pretext_minilm_jobs.yaml`

其中 `smoke_pretext_minilm_jobs.yaml` 已显式指向 `thesis_platform/open_model/all_minilm_l6_v2`；如果你要跑 `smoke_pretext_minilm_congressional.yaml` 且希望真正使用 MiniLM，而不是 fallback 到哈希 embedding，请先确认 `retriever.embedding_model` 指向的是一个真实存在的本地模型目录。

### 6.5 批量执行一个配置目录

```bash
python -m thesis_platform.scripts.run_matrix --config_dir thesis_platform/configs/experiments
```

### 6.6 汇总结果

```bash
python -m thesis_platform.scripts.summarize_results --input outputs/thesis_platform
```

## 7. 数据与模型下载

### 7.1 数据集下载

查看已注册数据集：

```bash
python -m thesis_platform.scripts.download_datasets --list
```

下载部分数据集：

```bash
python -m thesis_platform.scripts.download_datasets --names glue_sst2 gsm8k
```

下载可选的 PrE-Text 近似数据：

```bash
python -m thesis_platform.scripts.download_datasets --include-optional --names pretext_initialization_c4_en pretext_jobs
```

强制重新下载：

```bash
python -m thesis_platform.scripts.download_datasets --force --names glue_sst2
```

当前下载器支持三类资源来源：

- 直接从 Hugging Face 或公开数据源下载
- 调用上游脚本生成
  - `datainf_grammars`
  - `datainf_math_without_reason`
  - `datainf_math_with_reason`
  - 这三者会共用 `../DataInf/src/generate_sentence-math_datasets.py`
- 直接整理仓库内 vendored 资源
  - 例如 `imdb`、`rt_polarity`

关于 PrE-Text 数据要注意：

- `pretext_jobs`、`pretext_forums`、`pretext_microblog`、`pretext_code`、`pretext_initialization_c4_en` 都是基于 `c4-en` URL/站点启发式近似重建，不是论文原始私有语料的逐字复刻。
- 下载器会维护 `_pretext_c4_cache/`，只预热本次请求需要的类别。

### 7.2 模型下载

查看已注册模型：

```bash
python -m thesis_platform.scripts.download_models --list
```

下载常用模型：

```bash
python -m thesis_platform.scripts.download_models --names all_minilm_l6_v2 roberta_large
```

下载可选模型：

```bash
python -m thesis_platform.scripts.download_models --include-optional
```

显式启用大模型下载：

```bash
python -m thesis_platform.scripts.download_models --include-large --names llama_3_1_405b_instruct
```

覆盖某个模型的 Hugging Face 仓库：

```bash
python -m thesis_platform.scripts.download_models --names llama_3_1_8b_instruct --repo-override llama_3_1_8b_instruct=custom-user/Llama-3.1-8B-Instruct
```

模型下载器的默认行为：

- 默认不下载 `optional` 模型。
- 默认不下载大于 `15B` 的模型，除非显式传入 `--include-large`。
- 对部分 Llama 类模型，默认使用社区镜像，并在下载前验证该仓库是否是 Transformers 兼容结构。

如果你在 Linux 环境下需要通过 Clash 代理后台下载可选模型，仓库中还提供了：

```text
thesis_platform/scripts/download_models_include_optional_bg.sh
```

它支持 `start | status | stop | logs | clash-logs` 五种动作，并会把状态文件和日志写到 `thesis_platform/open_model/`。

## 8. 配置系统

### 8.1 配置组织方式

`thesis_platform/configs/` 采用三层结构：

- `base/`：路径、runtime 等基础配置
- `methods/`：按 generator / scorer / retriever / critic / aggregator 拆开的预设
- `experiments/`：真正运行的实验组合

实验配置通过 `inherits` 做递归继承与深合并。例如：

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

后出现的字段会覆盖前面的同名字段。

### 8.2 顶层配置段

| 配置段 | 说明 |
| --- | --- |
| `meta` | 实验 ID、stage、随机种子 |
| `paths` | `repo_root`、`output_root`、`cache_root` 等路径 |
| `data` | 数据集路径、任务类型、client 划分参数 |
| `federation` | 轮数、每个 client 选多少 bad samples |
| `generator` | 生成器名称与超参数 |
| `scorer` | 打分器名称与超参数 |
| `retriever` | 召回器配置 |
| `critic` | critique 生成配置 |
| `aggregator` | 规则聚合配置 |
| `runtime` | 设备、日志等级 |
| `evaluation` | 评估/落盘相关设置 |

### 8.3 常用字段

| 字段 | 说明 |
| --- | --- |
| `data.train_path` | 私有训练数据路径，支持 JSON 文件或 JSON 目录 |
| `data.public_seed_path` | 生成器使用的公共 seed 池 |
| `data.num_clients` | client 数量 |
| `data.max_samples_per_client` | 每个 client 保留的最大样本数 |
| `data.validation_ratio` | 每个 client 划给 validation anchor 的比例 |
| `federation.rounds` | 联邦总轮数 |
| `federation.top_k_bad` | 每个 client 每轮保留多少条高分 bad samples |
| `retriever.embedding_model` | 本地 embedding 模型目录；不存在时自动 fallback |
| `critic.compress_to_n_rules` | 每个 pair 最多保留多少条规则 |
| `critic.redact_enable` | 是否做轻量脱敏 |
| `aggregator.max_rules` | 最多保留多少条聚合规则 |

### 8.4 当前 loader 支持的数据形状

`thesis_platform/data/loaders.py` 当前支持：

- 顶层为 `list` 的 JSON 文件
- 顶层为 `dict` 的 JSON 文件
- 包含多个 `*.json` 的目录
- 嵌套的 `list/dict/str`，会自动压平成文本

这意味着平台当前更偏向“文本样本统一化”而不是“严格保留原始结构字段”。如果你后续要做标签任务或 instruction/response 分离，建议补充更明确的 loader 或 formatter。

## 9. 输出结构

默认实验输出目录：

```text
outputs/thesis_platform/<experiment_id>/
```

实验目录下的核心文件：

| 文件 | 说明 |
| --- | --- |
| `metrics_summary.json` | 实验级汇总，包含 `experiment_id`、`round_count`、`final_prompt`、`round_metrics` |
| `resolved_config.json` | 继承和深合并后的最终配置快照 |
| `config.yaml` | 当前实现写出的文本快照，文件名是 `.yaml`，内容实际是 JSON 格式 |
| `round_000/`, `round_001/` | 每轮产物 |

每个 round 目录下会写出：

| 文件 | 说明 |
| --- | --- |
| `server_prompt.txt` | 本轮开始时的 prompt |
| `generated_samples.jsonl` | synthetic 样本 |
| `scored_samples.jsonl` | 每个 client 的打分结果 |
| `selected_bad_samples.jsonl` | 选中的 bad samples |
| `retrieved_pairs.jsonl` | 召回的 `(bad, real)` 配对 |
| `client_critiques.jsonl` | client 侧 critique |
| `prompt_update.json` | 聚合后的 prompt update，没有更新时可能不存在 |
| `round_metrics.json` | 本轮轻量指标 |

`round_metrics.json` 当前会记录：

- `generated_count`
- `avg_length`
- `diversity`
- `critique_count`
- `critique_rule_count`
- `avg_critique_length`
- `client_latency_s`
- `server_latency_s`
- `upload_tokens`
- `prompt_length_tokens`

说明：

- 配置里有 `evaluation.save_round_artifacts`，但当前实现仍会无条件写出关键 round 产物。
- prompt 更新逻辑由 `core/prompt_updater.py` 实现，会把聚合规则追加到 `### Aggregated Rules` 块中；如果该块已存在，则进行替换。

## 10. 测试

运行全部测试：

```bash
python -m unittest discover -s thesis_platform/tests -p "test_thesis_platform*.py" -v
```

当前测试覆盖重点包括：

- 配置继承与路径解析
- 生成器/打分器/召回器/critic/aggregator 的 smoke 行为
- 单轮 pipeline 是否产出必需文件
- loader 支持的 JSON 结构
- sentence-transformer 本地路径解析
- 数据/模型下载器注册、异常处理、报告生成与 repo override 行为

## 11. 扩展一个新方法的推荐流程

如果你要把新的论文方法接进平台，推荐顺序如下：

1. 在 `thesis_platform/algorithms/` 中先实现与平台上下文解耦的算法核心。
2. 在 `thesis_platform/adapters/` 中包成统一接口。
3. 在 `thesis_platform/adapters/__init__.py` 中注册到 adapter registry。
4. 在 `thesis_platform/configs/methods/` 中加入方法预设。
5. 在 `thesis_platform/tests/` 中补 smoke test 或行为测试。

这样做的好处是：

- 算法核心和平台编排解耦；
- 配置切换简单；
- 后续做矩阵实验和结果汇总时不需要改主链代码。

## 12. 已知限制

- 这套平台优先解决“多篇论文能力统一编排”的问题，不是单论文严格复现框架。
- `ira`、`dbscan_attn`、`dbscan_attn_tsgdm` 只是预留接口，不应在 README 或实验结论里当作已完成实现。
- `fedtextgrad_qwen` 当前是启发式 critique，不是完整本地 LLM 后端。
- `TransformersTextBackend` 仍未接入真实推理。
- `LabelMatchRetriever` 已实现，但默认 loader 不填 label，实际使用前需要补齐数据通路。
- `paths.model_root`、`paths.dataset_root` 等字段目前更多是预留位，当前代码实际主要消费的是 `repo_root`、`output_root`、`cache_root` 以及方法本身的路径字段。

## 13. 建议的阅读顺序

如果你是第一次接手这个项目，建议按这个顺序读代码：

1. `thesis_platform/scripts/run_experiment.py`
2. `thesis_platform/core/pipeline.py`
3. `thesis_platform/core/experiment_runner.py`
4. `thesis_platform/core/round_runner.py`
5. `thesis_platform/core/config.py`
6. `thesis_platform/adapters/__init__.py`
7. 你关心的具体 adapter 与 `algorithms/` 实现
8. `thesis_platform/tests/`

这样最容易先看清楚主链，再理解每个方法是怎么接进来的。
