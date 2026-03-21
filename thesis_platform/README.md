# thesis_platform

`thesis_platform` 是当前分支上的统一实验编排平台，用来承接论文实验而不是只做一个 mock/MVP 演示。

当前仓库内同时存在两条路径：

- `legacy research pipeline`：保留原有轻量对照、probe 风格 scorer、启发式 critic、旧 smoke 配置，便于研究探索和回归验证。
- `v3 real experiment pipeline`：面向创新算法第三版的真实实验主线，当前第一阶段只交付 `Jobs` 数据集的 paper-aligned 端到端流程。

第三版主链为：

```text
global/cluster generation
-> real scoring
-> bad-sample selection
-> anchor retrieval
-> critique
-> prototype extraction
-> clustering + aggregation
-> prompt update
-> downstream large-eval
```

## 1. 本次实现的重点

第三版真实实验路径已经在平台内落地，核心能力包括：

- 实验入口增加 preflight 预检，启动前检查依赖、模型路径和数据资产，缺失时直接报具体名字并终止。
- 配置层新增 `prototype`、`routing`、`privacy`、`downstream_eval` 四个顶层块。
- 运行态上下文新增 `prototype_vector`、`prototype_weight`、`cluster_id`、`cluster_prompt` 等字段。
- 每轮生成从单一池改为 `global pool + cluster pool` 双池。
- 客户端基于真实召回锚点提取原型向量，服务端执行原型聚类、簇内规则冲突解耦、双层 prompt 生成。
- 新增 `datainf_real` 和 `gradmm_real` 两个第三版 scorer。
- 新增 synthetic corpus 导出与 `PrE-Text/pretext_platform` large-eval 的进程内调用。
- 新增 `Jobs` 第三版配置、tiny validation 配置、prototype/routing/privacy/downstream_eval 预设。

需要明确的是：

- `datainf_real` 和 `gradmm_real` 是平台内原生实现的真实特征版 scorer，依赖本地 Transformer 特征编码，不再走 mock 数据或 shell 调 notebook。
- 它们不是对上游训练仓库的逐脚本复刻；如果要做严格的一比一上游 LoRA 训练复现，仍需要在此基础上继续补齐。

## 2. 关键目录

| 路径 | 说明 |
| --- | --- |
| `thesis_platform/core/` | 配置、上下文、预检、实验/轮次调度 |
| `thesis_platform/algorithms/` | 平台无关的算法核心逻辑 |
| `thesis_platform/adapters/` | generator/scorer/retriever/critic/aggregator 统一接口接入 |
| `thesis_platform/evaluation/` | 轮次指标、synthetic corpus 导出、large-eval 包装 |
| `thesis_platform/models/` | embedding/backend/feature encoder |
| `thesis_platform/configs/` | 基础配置、方法预设、实验配置 |
| `thesis_platform/tests/` | `unittest` 测试 |
| `thesis_platform/docs/` | 研究文档与第三版实施说明 |

## 3. 环境准备

推荐：

- Python `3.10+`
- `pip install -r thesis_platform/requirements.txt`

本次第三版主线额外依赖已经加入 `thesis_platform/requirements.txt`：

- `opacus`
- `peft`
- `vllm`（Windows 上按环境条件跳过）

如果本机默认 Python 环境依赖不完整，可以直接用已有虚拟环境运行，例如：

```powershell
GRADMM\.venv\Scripts\python.exe -m unittest thesis_platform.tests.test_thesis_platform_v3 -v
```

## 4. 第三版 Jobs 主线

第一阶段唯一主线配置：

```text
thesis_platform/configs/experiments/v3/jobs_real_datainf_v3.yaml
```

该配置默认启用：

- generator: `pretext_prompt_llm`
- scorer: `datainf_real`
- retriever: `knn`
- critic: `fedtextgrad_llm`
- aggregator: `dbscan_attn_tsgdm`
- prototype: `minilm_mean`
- routing: `personalized_v3`
- privacy: `jobs_eps129`
- downstream_eval: `pretext_large`

默认关键超参：

- `routing.personalized_mix_ratio = 0.7`
- `aggregator.cluster_eps = 0.35`
- `aggregator.cluster_min_samples = 2`
- `privacy.epsilon = 1.29`
- `privacy.delta = 3e-6`
- `data.max_samples_per_client = 8`

### 4.1 启动前必须具备的资产

`Jobs` 第三版主线的 preflight 会检查以下硬依赖：

- 数据集：`pretext_jobs`
- 初始化语料：`pretext_initialization_c4_en`
- embedding / prototype 模型：`all_minilm_l6_v2`
- scorer 特征模型：`roberta_large`
- LLM：`llama_2_7b_hf`

当前第一阶段明确不要求：

- `distilgpt2`
- `c4_checkpoint.pth`

### 4.2 运行命令

真实主线：

```bash
python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/v3/jobs_real_datainf_v3.yaml
```

轻量验证配置：

```bash
python -m thesis_platform.scripts.run_experiment --config thesis_platform/configs/experiments/validation/jobs_v3_tiny_mock.yaml
```

旧版 smoke/legacy 路径仍可用，但不应当作为第三版主结论来源。

## 5. 配置结构

当前实验配置主要由以下顶层块组成：

| 配置块 | 说明 |
| --- | --- |
| `meta` | 实验 ID、stage、seed |
| `paths` | `repo_root`、`output_root`、缓存与模型路径 |
| `data` | 训练集、初始化语料、client 切分参数 |
| `federation` | 轮数、每轮 bad sample 数量 |
| `generator` | synthetic 生成设置 |
| `scorer` | 打分器配置 |
| `retriever` | 锚点召回配置 |
| `critic` | critique 生成配置 |
| `aggregator` | 聚合、聚类、冲突解耦配置 |
| `prototype` | 原型提取方法与模型 |
| `routing` | 全局池/簇池混样与个性化路由 |
| `privacy` | 第三版实验的隐私预算设置 |
| `downstream_eval` | synthetic corpus 导出与 large-eval 配置 |
| `runtime` | 设备、日志、LLM backend |
| `evaluation` | 结果落盘选项 |

## 6. 每轮产物

默认输出目录：

```text
outputs/thesis_platform/<experiment_id>/
```

每轮目录 `round_000/`、`round_001/` 等现在会同时写出旧产物和第三版新产物。

基础产物：

- `server_prompt.txt`
- `generated_samples.jsonl`
- `scored_samples.jsonl`
- `selected_bad_samples.jsonl`
- `retrieved_pairs.jsonl`
- `client_critiques.jsonl`
- `prompt_update.json`
- `round_metrics.json`

第三版新增产物：

- `client_assigned_samples.jsonl`
- `client_prototypes.jsonl`
- `cluster_assignments.json`
- `cluster_prompts.json`
- `routing_summary.json`
- `prototype_clusters.json`
- `probe_metrics.json`

实验级附加产物：

- `metrics_summary.json`
- `resolved_config.json`
- `downstream_eval/stage2/llama7b_text_syn.json`
- `downstream_eval/pretext_large_eval_summary.json`

## 7. 下游评测

第三版主线不再只停留在 round-level 指标，会在实验末尾新增：

1. synthetic corpus 导出
2. 进程内调用 `PrE-Text/pretext_platform` large-eval
3. baseline summary 收集与统一汇总

当前只接：

- `Jobs + large-eval + epsilon=1.29`

当前明确未接：

- `small-eval`
- `Forums / Microblog / Code` 的第三版主线

## 8. 测试

推荐直接跑：

```bash
python -m unittest discover -s thesis_platform/tests -p "test_thesis_platform*.py" -v
```

本次新增测试覆盖：

- 新配置块解析
- preflight 缺失资产报错
- 第三版 routing/prototype/downstream 产物写出
- 现有 pipeline / research 路径回归

## 9. 当前限制

- 第三版主线当前只对 `Jobs` 做了工程化收口。
- `datainf_real` / `gradmm_real` 已经是本地真实特征版实现，但还不是上游仓库逐脚本训练复现。
- `small-eval` 仍然延后到 `c4_checkpoint.pth` 被纳入资产管理之后。
- `legacy research pipeline` 仍会继续保留；做论文结论时请区分 legacy 结果和第三版真实实验结果。

## 10. 相关文档

- `thesis_platform/docs/创新算法第三版实验版本.md`
- `thesis_platform/docs/第二版与第三版算法差距分析及修改方案.md`
- `thesis_platform/docs/创新算法第三版真实实验平台实施方案.md`
