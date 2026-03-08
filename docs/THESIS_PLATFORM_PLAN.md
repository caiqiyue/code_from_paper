# 论文实验平台设计方案

## 1. 文档目标

本文档基于当前 `caiqiyue_file/` 仓库中的已有源码，将你的论文思路落成一份可执行的实验平台方案。

核心目标：

- 保留各篇论文原始仓库，作为可运行的基线方法。
- 在此基础上新增一层你自己的统一实验编排平台。
- 将不同论文中的可复用部分抽象为 `generator / scorer / retriever / critic / aggregator` 五类模块。
- 同时支持：
  - 第一阶段：基于当前仓库中已有公开数据的工程验证与对比实验
  - 第二阶段：迁移到你自己的联邦指令数据或私有数据上做正式实验

## 2. 当前仓库与论文平台的映射关系

当前仓库已经具备论文平台的大部分可复用组件。新平台不应该推倒重写，而应该在现有源码基础上做封装和统一编排。

| 当前路径 | 在论文平台中的角色 | 可直接复用的核心文件 |
| --- | --- | --- |
| `PrE-Text/` | 服务端合成数据生成基线、直方图筛选基线、bootstrap 扩展生成 | `main.py`, `nn_histogram.py`, `similarity.py`, `variation.py`, `llama_bootstrap.py` |
| `DataInf/` | 基于效用驱动的坏样本打分器 | `src/influence.py`, `src/dataloader.py`, `src/lora_model.py` |
| `GRADMM/` | 基于梯度匹配的坏样本打分器 | `gradmm/filtering.py`, `gradmm/utilities.py` |
| `FedTextGrad/` | 文本化批判生成与服务端 Prompt 聚合基线 | `train_hetero_fed.py`, `textgrad/`, `utils/prompt_template.py` |
| `datasets/` | 第一阶段 smoke test 和小规模实验的共享数据池 | `bioarxiv_*`, `congressional_*`, `initial_set.json`, `openreview_init_data/` |
| `models/` | 模型缓存、本地 embedding 模型和语言模型权重 | `all-MiniLM-L6-v2/`, `distilgpt2/`, `Meta-Llama-2-7b-chat-hf/`, `Meta-Llama-3-8B/` |
| `outputs/` | 统一实验输出根目录 | 建议复用为论文平台的输出目录 |

这里需要提前固定几个工程决策：

- `PrE-Text/` 作为当前最合适的服务端生成基线。
- `FedTextGrad/` 作为当前最合适的文本批判与 Prompt 聚合基线。
- `DataInf/` 和 `GRADMM/` 只抽取为“样本打分器”，不要直接照搬其整套实验流程。
- `DBSCAN` 不需要找专门论文仓库源码，直接使用 `sklearn.cluster.DBSCAN` 实现即可。
- `TSGD-M` 不建议作为独立训练框架引入，而是应作为你自定义聚合器中的“历史动量模块”来实现。

## 3. 建议新增的平台目录设计

建议在当前仓库中新增如下目录：

```text
caiqiyue_file/
  thesis_platform/
    README.md
    core/
      registry.py
      schemas.py
      pipeline.py
      round_runner.py
      experiment_runner.py
      metrics.py
      io_utils.py
    adapters/
      generators/
        pretext_generator.py
        pretext_bootstrap.py
        api_generator.py
      scorers/
        pretext_histogram.py
        datainf_scorer.py
        gradmm_scorer.py
        ira_scorer.py
      retrievers/
        label_match.py
        knn_retriever.py
      critics/
        fedtextgrad_critic.py
        redact_ner.py
      aggregators/
        concat.py
        summarization.py
        uid.py
        dbscan_uid.py
        dbscan_uid_tsgdm.py
    prompts/
      critique/
        contrastive_critic_v1.txt
      aggregation/
        summarize_v1.txt
        uid_v1.txt
        dbscan_uid_v1.txt
    configs/
      base/
        paths.yaml
        runtime.yaml
      datasets/
        bioarxiv.yaml
        congressional.yaml
        openreview.yaml
        custom_instruction.yaml
      methods/
        generator/
          pretext_seed.yaml
          pretext_bootstrap.yaml
        scorer/
          pretext_hist.yaml
          datainf.yaml
          gradmm.yaml
          ira.yaml
        retriever/
          label_match.yaml
          knn.yaml
        critic/
          fedtextgrad_qwen.yaml
        aggregator/
          concat.yaml
          summarization.yaml
          uid.yaml
          dbscan_uid.yaml
          dbscan_uid_tsgdm.yaml
      experiments/
        smoke/
        scorer_selection/
        aggregator_selection/
        ablation/
        stability/
    scripts/
      prepare_dataset.py
      run_experiment.py
      run_matrix.py
      summarize_results.py
    workspace/
      cache/
      intermediate/
      logs/
      reports/
```

### 3.1 各目录职责说明

| 目录 | 作用 |
| --- | --- |
| `core/` | 平台无关的统一接口、轮次级调度逻辑、实验主流程 |
| `adapters/generators/` | 封装 `PrE-Text` 和后续自定义服务端生成模块 |
| `adapters/scorers/` | 统一样本打分接口，输出形式统一为 `sample_id -> score` |
| `adapters/retrievers/` | 从客户端本地数据中召回 `x_real` |
| `adapters/critics/` | 基于 `(x_bad, x_real)` 生成文本化 critique |
| `adapters/aggregators/` | 服务端聚合文本反馈并生成新 Prompt |
| `prompts/` | 所有 prompt 模板，单独版本化管理 |
| `configs/` | 所有实验配置、方法配置、数据配置 |
| `scripts/` | 平台入口脚本和批量实验脚本 |
| `workspace/` | 中间产物、缓存、日志、实验报告 |

### 3.2 为什么必须按模块拆，而不是按论文拆

你的方法不是单一算法，而是一个由多阶段组成的系统：

`generator -> scorer -> bad sample selector -> retriever -> critic -> redaction -> aggregator -> prompt updater -> evaluator`

所以平台必须围绕“流程阶段”来模块化，而不是围绕“某篇论文目录”来模块化。

## 4. 统一数据格式设计

所有方法在进入实验平台前，都应先转换成统一的数据 schema。

### 4.1 建议统一样本结构

```yaml
sample_id: str
client_id: str
round_id: int
source: real | synthetic | retrieved_real | critique
dataset_name: str
task_type: instruction_tuning | classification | next_token
instruction: str
response: str
label: str | int | null
text: str
meta:
  original_path: str
  original_index: int
  score: float | null
  retrieved_from: str | null
  cluster_id: int | null
```

### 4.2 为什么统一 schema 是硬要求

- `PrE-Text` 和 `FedTextGrad` 默认任务形式不同。
- `DataInf` 和 `GRADMM` 对梯度输入的组织方式不同。
- 当前 `datasets/` 下的数据格式不统一。
- 如果没有统一 schema，后面无法做可替换模块，也很难保证实验公平。

## 5. 配置文件设计原则

建议统一使用 YAML。

配置设计原则：

- 每个实验一份总配置文件。
- 每个方法模块一份方法预设配置。
- 每个数据源一份数据配置。
- Python 代码中尽量不写死路径和超参数。

## 6. 主实验配置文件设计

推荐路径：

`thesis_platform/configs/experiments/<exp_name>.yaml`

### 6.1 示例配置

```yaml
meta:
  experiment_id: exp_scorer_datainf_uid_openreview
  description: "DataInf + FedTextGrad critic + UID aggregation on OpenReview"
  stage: scorer_selection
  seed: 42

paths:
  repo_root: .
  model_root: ./models
  dataset_root: ./datasets
  output_root: ./outputs/thesis_platform
  cache_root: ./thesis_platform/workspace/cache

data:
  dataset_name: openreview
  task_type: instruction_tuning
  train_path: ./datasets/openreview_init_data
  eval_path: ./datasets/congressional_eval.json
  public_seed_path: ./datasets/initial_set.json
  schema_name: pretext_json
  num_clients: 10
  partition_method: dirichlet
  dirichlet_alpha: 0.3
  max_samples_per_client: 16
  validation_ratio: 0.1

federation:
  rounds: 5
  clients_per_round: 10
  top_k_bad: 10
  retrieved_real_k: 3
  upload_critiques_per_client: 2

generator:
  name: pretext_seed
  source_repo: PrE-Text
  init_population_path: ./datasets/initial_set.json
  seq_len: 64
  mask: 0.3
  lookahead: 4
  multiplier: 4
  t_steps: 2
  bootstrap_enable: true
  bootstrap_model: meta-llama/Llama-2-7b-chat-hf
  generated_per_round: 100

scorer:
  name: datainf
  source_repo: DataInf
  score_direction: larger_is_worse
  target_module: lora_or_last_layer
  lambda_const_param: 10
  batch_size: 8

retriever:
  name: knn
  embedding_model: all-MiniLM-L6-v2
  similarity_metric: cosine
  top_k: 3

critic:
  name: fedtextgrad_qwen
  source_repo: FedTextGrad
  engine: local_vllm
  model_name: Qwen2.5-1.5B-Instruct
  prompt_template: contrastive_critic_v1
  max_new_tokens: 256
  compress_to_n_rules: 2
  redact_enable: true
  redact_mode: ner

aggregator:
  name: uid
  source_repo: FedTextGrad
  summarizer_model: Meta-Llama-3-8B-Instruct
  prompt_template: uid_v1
  max_aggregate_tokens: 2048

evaluation:
  downstream_metrics: [accuracy, rougeL, bleu]
  generation_metrics: [bertscore, mauve]
  system_metrics: [upload_tokens, client_latency_s, server_latency_s]
  save_round_artifacts: true

runtime:
  device: cuda
  mixed_precision: fp16
  log_level: info
```

### 6.2 各配置段字段含义

#### `meta`

| 字段 | 含义 |
| --- | --- |
| `experiment_id` | 实验唯一标识 |
| `description` | 实验说明 |
| `stage` | 当前实验属于哪个阶段 |
| `seed` | 随机种子 |

#### `paths`

| 字段 | 含义 |
| --- | --- |
| `repo_root` | 仓库根目录 |
| `model_root` | 当前 `models/` 目录 |
| `dataset_root` | 当前 `datasets/` 目录 |
| `output_root` | 最终实验输出目录 |
| `cache_root` | embedding 和临时缓存目录 |

#### `data`

| 字段 | 含义 |
| --- | --- |
| `dataset_name` | 数据集逻辑名称 |
| `task_type` | 任务类型，指令微调/分类/下一词预测 |
| `train_path` | 客户端训练数据路径 |
| `eval_path` | 评估集路径 |
| `public_seed_path` | 初始公开 seed 数据路径 |
| `schema_name` | 数据解析格式 |
| `num_clients` | 客户端数量 |
| `partition_method` | 数据划分方式 |
| `dirichlet_alpha` | Non-IID 强度控制 |
| `max_samples_per_client` | 每客户端样本上限 |
| `validation_ratio` | 本地验证集比例 |

#### `federation`

| 字段 | 含义 |
| --- | --- |
| `rounds` | 全局通信轮数 |
| `clients_per_round` | 每轮参与客户端数 |
| `top_k_bad` | 每客户端选出的坏样本数 |
| `retrieved_real_k` | 每个坏样本召回的真实样本数 |
| `upload_critiques_per_client` | 每客户端上传的浓缩 critique 条数 |

#### `generator`

| 字段 | 含义 |
| --- | --- |
| `name` | 生成器适配器名称 |
| `source_repo` | 对应原始论文仓库 |
| `init_population_path` | 初始生成候选池 |
| `seq_len` | 最大生成长度 |
| `mask` | `PrE-Text` 掩码比例 |
| `lookahead` | `PrE-Text` lookahead 参数 |
| `multiplier` | `PrE-Text` 候选扩展倍数 |
| `t_steps` | `PrE-Text` 变异步数 |
| `bootstrap_enable` | 是否启用 bootstrap 扩展 |
| `bootstrap_model` | 服务端扩展生成模型 |
| `generated_per_round` | 每轮生成样本数 |

#### `scorer`

| 字段 | 含义 |
| --- | --- |
| `name` | `pretext_hist`, `datainf`, `gradmm`, `ira` 之一 |
| `source_repo` | 来源论文仓库 |
| `score_direction` | 分数是越大越差还是越小越差 |
| `target_module` | 计算梯度的目标模块 |
| `lambda_const_param` | DataInf 正则参数 |
| `batch_size` | scorer 批大小 |

#### `retriever`

| 字段 | 含义 |
| --- | --- |
| `name` | `label_match` 或 `knn` |
| `embedding_model` | 检索时使用的编码模型 |
| `similarity_metric` | 相似度计算方式 |
| `top_k` | 返回真实样本数量 |

#### `critic`

| 字段 | 含义 |
| --- | --- |
| `name` | critique 生成器名称 |
| `source_repo` | 来源论文仓库 |
| `engine` | 推理后端，如 `local_vllm` |
| `model_name` | 本地或远程 LLM 名称 |
| `prompt_template` | critique prompt 模板版本 |
| `max_new_tokens` | critique 输出长度 |
| `compress_to_n_rules` | 压缩后保留规则数 |
| `redact_enable` | 是否启用脱敏 |
| `redact_mode` | 脱敏方式，如 NER |

#### `aggregator`

| 字段 | 含义 |
| --- | --- |
| `name` | `concat`, `summarization`, `uid`, `dbscan_uid`, `dbscan_uid_tsgdm` 之一 |
| `source_repo` | 来源仓库或自定义实现 |
| `summarizer_model` | 服务端汇总模型 |
| `prompt_template` | 聚合 prompt 模板 |
| `max_aggregate_tokens` | 聚合时上下文 token 上限 |

#### `evaluation`

| 字段 | 含义 |
| --- | --- |
| `downstream_metrics` | 下游任务指标 |
| `generation_metrics` | 合成质量指标 |
| `system_metrics` | 通信与运行开销指标 |
| `save_round_artifacts` | 是否保存每轮中间产物 |

#### `runtime`

| 字段 | 含义 |
| --- | --- |
| `device` | 运行设备 |
| `mixed_precision` | 混合精度模式 |
| `log_level` | 日志级别 |

## 7. 方法配置文件设计

除了总实验配置，还建议给每个方法模块单独准备预设配置，例如：

- `configs/methods/scorer/datainf.yaml`
- `configs/methods/scorer/gradmm.yaml`
- `configs/methods/aggregator/dbscan_uid_tsgdm.yaml`

这样后续不同实验之间只需要替换方法配置，而不需要反复手工改总实验文件。

## 8. 统一适配器接口设计

### 8.1 Generator

```python
generate(round_ctx) -> list[Sample]
```

### 8.2 Scorer

```python
score(samples, client_ctx) -> list[ScoredSample]
```

### 8.3 Retriever

```python
retrieve(bad_samples, client_ctx) -> list[PairedSample]
```

### 8.4 Critic

```python
critique(paired_samples, client_ctx) -> list[Critique]
```

### 8.5 Aggregator

```python
aggregate(client_critiques, server_ctx) -> PromptUpdate
```

统一接口的意义在于：不同论文算法只要适配成同一输入输出形式，就可以稳定地做组合实验。

## 9. 实验矩阵设计

不要做全排列穷举。应该采用“分阶段筛选”的方式。

### 9.1 阶段 0：平台连通性验证

| ID | 目标 | 数据集 | Generator | Scorer | Critic | Aggregator | 轮数 | 客户端数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0-1 | 验证 `PrE-Text` 原始基线能否完整跑通 | congressional | pretext_seed | pretext_hist | none | none | 1 | 3 |
| S0-2 | 验证 `FedTextGrad` 聚合链路能否跑通 | congressional | fixed_stub | none | fedtextgrad | summarization | 1 | 3 |
| S0-3 | 验证你自己的统一主流程是否跑通 | congressional | pretext_seed | datainf | fedtextgrad | uid | 1 | 3 |

退出条件：

- 一轮实验可完整结束
- 中间产物能正确落盘
- 每个实验能生成一份结果 JSON

### 9.2 阶段 1：模块 A 打分器筛选

固定：

- generator = `pretext_seed`
- retriever = `knn`
- critic = `fedtextgrad_qwen`
- aggregator = `uid`
- 数据集先用 `openreview`，再迁移到 `bioarxiv`

| ID | 目标 | Scorer | 坏样本规则 | 轮数 | 客户端 | 随机种子 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 原始筛选基线 | pretext_hist | top_k_bad=10 | 3 | 5 | 1 |
| A2 | 效用驱动筛选 | datainf | top_k_bad=10 | 3 | 5 | 1 |
| A3 | 梯度匹配筛选 | gradmm | top_k_bad=10 | 3 | 5 | 1 |
| A4 | 低算力筛选 | ira | top_k_bad=10 | 3 | 5 | 1 |
| A5 | 正式 scorer 对比 | A1-A4 中表现最好的两个 | top_k_bad=10 | 5 | 10 | 3 |

核心评价指标：

- 下游任务性能
- 坏样本筛选时间
- 客户端显存占用
- critique 上传 token 数量

目标输出：

- 确定一个主 scorer
- 确定一个备选 scorer 用于消融实验

### 9.3 阶段 2：模块 B 检索与批判验证

固定：

- generator = `pretext_seed`
- scorer = 阶段 1 最优 scorer
- aggregator = `uid`

| ID | 目标 | Retriever | Critique 方式 | 轮数 | 客户端 | 随机种子 |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | 无锚点基线 | none | 单样本批判 | 3 | 5 | 1 |
| B2 | 语义检索 | knn | 对比式 critique | 3 | 5 | 1 |
| B3 | 规则匹配 | label_match | 对比式 critique | 3 | 5 | 1 |
| B4 | 正式检索器选择 | B2-B3 中最优者 | 对比式 critique | 5 | 10 | 3 |

核心评价指标：

- critique 对下一轮性能提升的有效性
- critique 冗余率
- 脱敏后的敏感实体残留率

### 9.4 阶段 3：模块 C 聚合器筛选

固定：

- generator = `pretext_seed`
- scorer = 阶段 1 最优 scorer
- retriever = 阶段 2 最优 retriever
- critic = `fedtextgrad_qwen`

| ID | 目标 | Aggregator | 额外机制 | 轮数 | 客户端 | 随机种子 |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 直接拼接基线 | concat | none | 3 | 5 | 1 |
| C2 | LLM 汇总基线 | summarization | none | 3 | 5 | 1 |
| C3 | FedTextGrad 基线 | uid | none | 3 | 5 | 1 |
| C4 | 论文候选 1 | dbscan_uid | 聚类后选代表反馈 | 3 | 5 | 1 |
| C5 | 论文候选 2 | dbscan_uid_tsgdm | 聚类 + 动量历史平滑 | 3 | 5 | 1 |
| C6 | 正式聚合器比较 | C1-C5 中表现最好的三个 | 相同设定 | 5 | 10 | 3 |

核心评价指标：

- Prompt 更新后带来的下游性能提升
- 多轮 Prompt 稳定性
- critique 压缩率
- 服务端聚合耗时

### 9.5 阶段 4：端到端完整方法对比

| ID | 目标 | 方法组合 | 轮数 | 客户端 | 随机种子 |
| --- | --- | --- | --- | --- | --- |
| D1 | 原始论文基线 | `PrE-Text` original | 5 | 10 | 3 |
| D2 | 仅替换 scorer | `PrE-Text + 最优 scorer` | 5 | 10 | 3 |
| D3 | 替换 scorer + critique | `PrE-Text + 最优 scorer + FedTextGrad critic` | 5 | 10 | 3 |
| D4 | 论文主方法 | `Generator + 最优 scorer + 最优 retriever + critic + 最优 aggregator` | 5 | 10 | 3 |
| D5 | 最强增强版本 | `D4 + dbscan_uid_tsgdm` | 5 | 10 | 3 |

这一阶段的结果就是论文主结果表的核心来源。

### 9.6 阶段 5：消融实验

| ID | 目标 | 变化项 |
| --- | --- | --- |
| E1 | 验证 `x_real` 是否必要 | 去掉 retriever，只对 `x_bad` 做批判 |
| E2 | 验证 critique 压缩是否必要 | 直接上传 10 条 critique，不压缩成 2 条规则 |
| E3 | 验证脱敏步骤是否必要 | 关闭 NER 脱敏 |
| E4 | 验证坏样本数敏感性 | `top_k_bad = 5` |
| E5 | 验证坏样本数敏感性 | `top_k_bad = 20` |
| E6 | 验证聚类是否有效 | 用 `uid` 替换 `dbscan_uid` |
| E7 | 验证动量是否有效 | 用 `dbscan_uid` 替换 `dbscan_uid_tsgdm` |

### 9.7 阶段 6：异质性与稳定性实验

| ID | 目标 | 变量 | 取值 |
| --- | --- | --- | --- |
| F1 | Non-IID 敏感性 | `dirichlet_alpha` | `0.1, 0.3, 1.0` |
| F2 | 客户端规模敏感性 | `num_clients` | `5, 10, 20` |
| F3 | 多轮稳定性 | `rounds` | `3, 5, 10` |
| F4 | 动量记忆强度 | `momentum_beta` | `0.5, 0.8, 0.9` |

这一部分用于支撑你论文中“在联邦异质场景下仍然稳定”的论证。

## 10. 最终论文中的指标体系

建议将指标分为四组。

| 指标组 | 示例 | 在论文中的作用 |
| --- | --- | --- |
| 下游任务指标 | accuracy, F1, rougeL, bleu, perplexity | 证明有效性 |
| 合成质量指标 | bertscore, mauve, diversity | 证明生成质量 |
| 系统开销指标 | upload tokens, client latency, server latency, GPU memory | 证明联邦可部署性 |
| 隐私安全指标 | redaction residual rate, raw entity leakage count | 支撑隐私论述 |

## 11. 实验输出目录设计

每个实验建议都输出如下结构：

```text
outputs/thesis_platform/<experiment_id>/
  config.yaml
  metrics_summary.json
  round_000/
    server_prompt.txt
    generated_samples.jsonl
    selected_bad_samples.jsonl
    retrieved_pairs.jsonl
    client_critiques.jsonl
    aggregated_prompt.txt
  round_001/
  ...
```

这一步很重要，因为你的论文不只需要最后的分数，还需要保留每一轮的中间证据，方便后面做可解释性分析和案例分析。

## 12. 推荐实施顺序

1. 先搭建 `thesis_platform/core/` 和统一 schema。
2. 先封装 `PrE-Text`，作为 generator 基线。
3. 再封装 `DataInf`、`GRADMM` 和 `IRA`，作为 scorer 适配器。
4. 再接入 `FedTextGrad` 的 critique 生成逻辑。
5. 先实现 `uid` 聚合器，保证平台可跑通。
6. 再实现 `dbscan_uid`，使用 `sklearn` 聚类。
7. 再实现 `dbscan_uid_tsgdm`，在聚合摘要上增加动量历史记忆。
8. 完成阶段 0 到阶段 3 后，再开始大规模正式实验。
9. 最后再做完整方法对比和消融实验。

## 13. 第一版平台的推荐默认值

建议你在平台初版里使用以下默认参数：

| 项目 | 默认值 |
| --- | --- |
| 数据集 | smoke 阶段用 `congressional`，第一轮正式筛选用 `openreview` |
| 轮数 | 开发阶段 `3`，论文正式对比 `5` |
| 客户端数 | 开发阶段 `5`，论文正式对比 `10` |
| 每轮生成样本数 | `100` |
| `top_k_bad` | `10` |
| 每个坏样本召回真实样本数 | `3` |
| 每客户端上传 critique 规则数 | `2` |
| scorer 候选 | `pretext_hist`, `datainf`, `gradmm`, `ira` |
| aggregator 候选 | `summarization`, `uid`, `dbscan_uid`, `dbscan_uid_tsgdm` |

## 14. 当前最推荐的主方法路线

如果从当前仓库中的现有代码出发，最有希望成为论文主线的方法组合是：

`PrE-Text 风格的服务端生成 + DataInf 或 GRADMM 坏样本筛选 + KNN 真实样本召回 + FedTextGrad 风格对比批判 + DBSCAN-UID 聚合 + TSGD-M 启发式动量平滑`

但在阶段 1 到阶段 3 没跑完之前，不建议提前锁死这一组合。

所以平台设计必须优先满足以下四点：

- 模块可替换
- 配置可复现
- 中间产物可保存
- 与原始论文基线可公平比较

