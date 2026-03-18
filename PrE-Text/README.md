# PrE-Text Platform

`PrE-Text` 已从原始论文脚本仓库重构为一个独立的实验平台包 `pretext_platform/`。  
这个平台保留了原论文的主算法链路和关键超参数语义，同时引入了配置驱动、统一输出目录、可分阶段运行、可测试的工程结构。

它支持完整运行以下 4 个阶段：

1. Stage 1: Private Evolution，生成 DP seed synthetic texts
2. Stage 2: 基于 LLaMA-2-7B 的 bootstrap 扩增
3. Small-model downstream evaluation: DistilGPT2
4. Large-model downstream evaluation: LLaMA-2-7B + LoRA

需要特别说明的是：

- 当前仓库可以在“当前工作区已经存在的数据和模型”上跑完整实验链路，这不是 MVP 级别的假跑通。
- 但如果你追求“严格逐项复现原论文”的结果，仍然需要你自己提供原论文对应的数据集和 `c4_checkpoint.pth` 等外部资源。
- 当前仓库内置的完整实验配置默认面向当前工作区已有的 `congressional` / `bioarxiv` 数据，而不是论文中的 `Jobs / Forums / Microblog / Code` 私有语料。

## 1. 项目定位

这个仓库现在有两个目标：

1. 作为一个可以完整执行 PrE-Text 全流程的实验平台
2. 作为一个更容易维护、复用、扩展、记录实验结果的工程化代码库

与原始论文仓库相比，它做了两类变化：

- 算法层面：
  - 保留原论文的核心主序和默认数学行为
  - Stage 1 仍然是 “初始化种群 -> 私有文本 embedding -> lookahead embedding -> FAISS 最近邻直方图 -> Gaussian 噪声和阈值 -> survivor 重采样 -> MLM mask-fill 变异”
  - Stage 2 仍然是基于 surviving seeds 的 few-shot bootstrap
  - downstream eval 仍然保留 DistilGPT2 和 LLaMA2 两条线
- 工程层面：
  - 变成包结构
  - 变成 YAML 配置驱动
  - 支持分阶段执行和全流程执行
  - 输出目录统一
  - 增加了单元测试
  - 旧脚本入口被保留为兼容包装器

## 2. 框架总览

完整实验链路如下：

```text
datasets + local models
        |
        v
  config loader
        |
        v
  Stage 1: Private Evolution
    - initialization pool
    - private train texts
    - RoBERTa-large mask-fill variation
    - MiniLM embeddings
    - FAISS nearest-neighbor histogram
    - DP noise + threshold
    - survivor resampling
        |
        v
  Stage 2: Bootstrap generation
    - surviving_text_it*.json
    - 3-shot prompt construction
    - LLaMA-2-7B generation
        |
        v
  Synthetic corpus
        |
        +--> DistilGPT2 eval
        |
        +--> LLaMA-2-7B + LoRA eval
```

平台推荐入口是：

```bash
python -m pretext_platform.scripts.run_pipeline --config <config.yaml>
```

但对于真实的长时实验，推荐按阶段运行，而不是一次性全打包执行。原因很简单：

- Stage 1 和 Stage 2 都比较重
- Stage 2 和下游评测经常需要独立切换机器或 GPU 资源
- 你通常会想先检查 Stage 1 输出，再决定是否继续 bootstrap 或评测

## 3. 当前工作区可直接复用的资源

平台默认复用当前工作区已有资源，不会在运行时自动重新下载模型。

### 3.1 数据资源

默认数据根目录：

```text
../datasets
```

当前工作区内已存在并可直接用于实验的主要文件：

| 文件 | 用途 | 当前本地规模 |
| --- | --- | --- |
| `../datasets/initial_set.json` | Stage 1 初始化种群 | 87,317 条公共 seed 文本 |
| `../datasets/congressional_train.json` | `congressional` 训练集 | 133,000 个 client，664,958 条原始训练样本 |
| `../datasets/congressional_eval.json` | `congressional` 评测集 | 23,510 条文本 |
| `../datasets/bioarxiv_train.json` | `bioarxiv` 训练集 | 72,000 个 client，474,149 条原始训练样本 |
| `../datasets/bioarxiv_eval.json` | `bioarxiv` 评测集 | 14,307 条文本 |

注意：

- `congressional_train.json` 和 `bioarxiv_train.json` 不是扁平列表，而是按 client 分桶的 JSON 对象。
- 平台的数据加载器会按 `max_samples_per_client` 做确定性子采样，再展平成原始 PrE-Text 算法所需的私有样本列表。
- 默认 `max_samples_per_client=8`，这是为了与当前 Stage 1 默认的 `sensitivity=8` 对齐。

### 3.2 模型资源

默认模型根目录：

```text
../thesis_platform/open_model
```

当前平台默认会用到下面这些本地模型：

| 逻辑名 | 默认路径 | 用途 |
| --- | --- | --- |
| MiniLM | `../thesis_platform/open_model/all_minilm_l6_v2` | 文本 embedding |
| RoBERTa-large | `../thesis_platform/open_model/roberta_large` | Stage 1 mask-fill 变异 |
| LLaMA-2-7B | `../thesis_platform/open_model/llama_2_7b_hf` | Stage 2 bootstrap、large-model eval |
| DistilGPT2 | `../thesis_platform/open_model/distilgpt2` | small-model eval |

额外说明：

- 平台使用 `local_files_only=True` 从本地模型目录加载，不依赖运行时联网下载。
- `LLaMA-2-7B` 必须事先存在于本地目录中。
- `eval_small` 还需要额外的 `c4_checkpoint.pth`，它不是模型目录里的 Hugging Face 权重，而是原论文小模型评测所需的 warm-start checkpoint。

## 4. 当前仓库结构

```text
PrE-Text/
  pretext_platform/
    algorithms/
    core/
    data/
    evaluation/
    scripts/
  configs/
    base/
    experiments/
    templates/
  tests/
  main.py
  llama_bootstrap.py
  eval_distilgpt2.py
  eval_llama2.py
  variation.py
  similarity.py
  nn_histogram.py
  custom_datasets.py
  requirements.txt
  README.md
```

## 5. 每个模块的功能

### 5.1 `pretext_platform/core/`

这一层负责“平台基础设施”，不直接写算法细节。

| 文件 | 功能 |
| --- | --- |
| `core/config.py` | 解析 YAML 配置，支持 `inherits` 继承、相对路径解析、类型化访问 |
| `core/models.py` | 把逻辑模型名解析成稳定的本地路径 |
| `core/types.py` | 定义 `ModelPaths`、`DatasetBundle`、`StageSummary` 等公共类型 |
| `core/io_utils.py` | 统一目录创建、JSON/JSONL/文本写出 |
| `core/pipeline.py` | 统一 orchestration 入口，负责 Stage 1 / Stage 2 / eval_small / eval_large 的调用 |
| `core/legacy.py` | 把旧 CLI 参数映射到新配置结构，供兼容包装器使用 |

### 5.2 `pretext_platform/data/`

这一层负责把不同形状的数据文件统一转换成实验输入。

| 文件 | 功能 |
| --- | --- |
| `data/loaders.py` | 加载 train/eval/init 数据，支持 flat list、client buckets、`{"1": [...]}` 等不同 JSON 形状 |

数据加载器的关键行为：

1. `train` 支持两种形状：
   - 原版扁平列表：`["text1", "text2", ...]`
   - thesis_platform 风格 client bucket：`{"0": [...], "1": [...], ...}`
2. `eval` 支持两种形状：
   - 扁平列表
   - `{"1": [...]}`
3. `initialization` 会做词数过滤，默认只保留词数大于 20 的文本
4. client bucket 训练集会根据 `max_samples_per_client` 做确定性子采样

### 5.3 `pretext_platform/algorithms/`

这一层负责保留原始论文算法内核。

| 文件 | 功能 |
| --- | --- |
| `algorithms/datasets.py` | `ListDataset` 和 `MatrixDataset`，供变异和批处理使用 |
| `algorithms/variation.py` | Stage 1 的 mask-fill 变异算子；使用 RoBERTa-large 逐 token 填充 |
| `algorithms/similarity.py` | MiniLM embedding 计算和 lookahead embedding 逻辑 |
| `algorithms/histogram.py` | FAISS 最近邻直方图、Gaussian 噪声注入、阈值裁剪 |
| `algorithms/stage1.py` | Stage 1 整体调度，完整实现原始 Private Evolution 主链 |
| `algorithms/bootstrap.py` | Stage 2 bootstrap 调度，负责 few-shot prompt 构造和 LLaMA2 生成 |

#### `algorithms/variation.py`

这是 Stage 1 的核心变异器。

它做的事情：

1. 选定一批 parent texts
2. 对每个样本随机 mask 一部分 token
3. 用 RoBERTa-large 在 masked 位置逐步采样填充
4. 重复 `t_steps` 次 mask-fill，得到变异后的 offspring

#### `algorithms/similarity.py`

它负责两个层次的 embedding：

1. 私有文本的直接 embedding
2. synthetic candidate 的 lookahead embedding

lookahead 的含义是：

- 不是直接对当前 candidate 算 embedding
- 而是先对当前 candidate 进行若干次未来变异
- 再对未来变异结果求平均 embedding

这与原始 PrE-Text 中“用未来演化位置来判断当前候选质量”的思路一致。

#### `algorithms/histogram.py`

它负责 Stage 1 中最关键的 DP 直方图逻辑：

1. 用 FAISS 为每个私有样本找到最近 synthetic candidate
2. 统计每个 candidate 被命中的次数
3. 对计数加入 Gaussian 噪声
4. 用阈值 `H` 做裁剪
5. 得到 survivor resampling 用的 noised histogram

### 5.4 `pretext_platform/evaluation/`

这一层负责下游评测。

| 文件 | 功能 |
| --- | --- |
| `evaluation/distilgpt2_eval.py` | 基于 synthetic corpus 微调 DistilGPT2，并做 next-token evaluation |
| `evaluation/llama2_eval.py` | 基于 synthetic corpus 对 LLaMA2 做 LoRA 微调并评测 |

#### `distilgpt2_eval.py`

它严格保留原项目中的几个关键设定：

- 使用 `c4_checkpoint.pth` 作为 warm-start
- 目标任务是 next-token prediction
- 使用 `AdamW`
- 默认学习率 `0.0002`
- 默认训练 `20` 个 epoch

#### `llama2_eval.py`

它保留原 large-model 设置：

- 底座模型：LLaMA-2-7B
- 微调方式：LoRA
- 默认 `rank=4`，`alpha=8`
- 默认 `1` 个 epoch

### 5.5 `pretext_platform/scripts/`

这一层是新的 CLI 入口。

| 文件 | 功能 |
| --- | --- |
| `scripts/run_pipeline.py` | 运行配置中启用的所有阶段 |
| `scripts/run_stage1.py` | 只跑 Stage 1 |
| `scripts/run_bootstrap.py` | 只跑 Stage 2 |
| `scripts/run_eval_small.py` | 只跑 DistilGPT2 评测 |
| `scripts/run_eval_large.py` | 只跑 LLaMA2 评测 |

### 5.6 根目录兼容包装器

这些文件还在，但不再承载主实现：

| 文件 | 当前角色 |
| --- | --- |
| `main.py` | 兼容旧版 Stage 1 CLI |
| `llama_bootstrap.py` | 兼容旧版 Stage 2 CLI |
| `eval_distilgpt2.py` | 兼容旧版 small-model eval CLI |
| `eval_llama2.py` | 兼容旧版 large-model eval CLI |
| `variation.py` | 兼容导出 `pretext_platform.algorithms.variation` |
| `similarity.py` | 兼容导出 `pretext_platform.algorithms.similarity` |
| `nn_histogram.py` | 兼容导出 `pretext_platform.algorithms.histogram` |
| `custom_datasets.py` | 兼容导出 `pretext_platform.algorithms.datasets` |

## 6. 配置系统

平台使用 YAML 配置，支持 `inherits`。

配置的核心 section 固定为：

| section | 用途 |
| --- | --- |
| `meta` | 实验 ID、随机种子等元信息 |
| `paths` | 仓库根、数据根、模型根、输出根 |
| `data` | 数据集名、train/eval/init 路径、client 样本上限 |
| `models` | MiniLM、RoBERTa、LLaMA2、DistilGPT2、C4 checkpoint 路径 |
| `stage1` | Private Evolution 超参数 |
| `bootstrap` | bootstrap 扩增超参数 |
| `eval_small` | DistilGPT2 评测超参数 |
| `eval_large` | LLaMA2 评测超参数 |
| `runtime` | 设备设置等运行参数 |

### 6.1 基础配置文件

| 文件 | 用途 |
| --- | --- |
| `configs/base/paths.yaml` | 默认路径 |
| `configs/base/models.yaml` | 默认模型路径 |
| `configs/base/runtime.yaml` | 默认 Stage 1 / bootstrap / eval 超参数 |
| `configs/templates/noise_eps129.yaml` | `sigma=11.3` 模板 |
| `configs/templates/noise_eps758.yaml` | `sigma=2.31` 模板 |

### 6.2 当前内置实验配置

| 配置文件 | 说明 |
| --- | --- |
| `configs/experiments/smoke_congressional_eps758.yaml` | 轻量 smoke，数据集为 `congressional` |
| `configs/experiments/full_congressional_eps129.yaml` | 更接近完整实验，数据集为 `congressional` |
| `configs/experiments/smoke_bioarxiv_eps758.yaml` | 轻量 smoke，数据集为 `bioarxiv` |
| `configs/experiments/full_bioarxiv_eps129.yaml` | 更接近完整实验，数据集为 `bioarxiv` |

需要注意：

- `smoke_*` 是轻量验证配置，不代表论文式完整长时实验。
- `full_*` 代表平台默认的全流程配置，但其中 `bootstrap.num_prompts` 默认仍是 `50,000`。
- 对 large-model 路线，`50,000` 是合理的真实完整实验配置。
- 对 small-model 论文式设置，通常需要把 `bootstrap.num_prompts` 改成 `2,000,000`。

## 7. 环境要求

### 7.1 操作系统

推荐：

- Linux
- NVIDIA GPU 驱动与 CUDA 环境正确可用

不推荐把完整真实实验直接跑在 Windows 上，原因主要有：

- `vllm==0.3.3` 在 Linux 上最稳妥
- `bitsandbytes`
- `xformers`
- 多数 GPU 依赖栈都更适合 Linux

Windows 可以做：

- 代码阅读
- 配置检查
- 轻量级测试
- 部分不依赖 `vllm` 的步骤

但如果你要跑完整 Stage 2 和大模型评测，建议使用 Linux GPU 机器。

### 7.2 Python 版本

建议：

- `Python 3.10.x`

理由：

- 原始论文仓库使用 `Python 3.10`
- 当前锁定依赖与 `torch==2.1.2`、`transformers==4.38.2`、`vllm==0.3.3` 的兼容性更接近 Python 3.10

说明：

- 轻量级的配置和单元测试在 `Python 3.11` 下也能工作
- 但如果你的目标是完整 GPU 实验，优先使用 Python 3.10

### 7.3 硬件建议

参考原始项目与当前实现，建议如下：

| 阶段 | 推荐硬件 |
| --- | --- |
| Stage 1 | 单卡 V100 32GB 或 A40 48GB 级别 |
| Stage 2 | 单卡 A40 48GB 或同等级显存 |
| DistilGPT2 eval | 多卡更稳妥，原始经验是 4 x A40 48GB |
| LLaMA2 eval | 多卡更稳妥，原始经验是 4 x A40 48GB |

这些只是经验值，不是硬性限制。

## 8. 第三方包与依赖栈

安装命令：

```bash
pip install -r requirements.txt
```

### 8.1 核心运行依赖

这些是当前项目最关键、最直接参与运行的依赖：

| 包 | 版本 | 作用 |
| --- | --- | --- |
| `torch` | `2.1.2` | 张量运算、模型训练与推理 |
| `transformers` | `4.38.2` | RoBERTa、DistilGPT2、LLaMA2 加载与推理 |
| `accelerate` | `0.28.0` | 分布式 / 多 GPU 包装 |
| `sentence-transformers` | `2.5.1` | MiniLM embedding |
| `faiss-cpu` | `1.8.0` | 最近邻搜索 |
| `opacus` | `1.4.1` | 隐私会计 |
| `datasets` | `2.18.0` | 数据集与 tokenization pipeline |
| `peft` | `0.10.0` | LoRA |
| `vllm` | `0.3.3` | Stage 2 bootstrap 大模型推理 |
| `bitsandbytes` | `0.43.1` | 量化与大模型运行常用依赖 |
| `xformers` | `0.0.23.post1` | Transformer 加速 |
| `tokenizers` | `0.15.2` | tokenizer 后端 |
| `safetensors` | `0.4.2` | safetensors 权重读取 |
| `sentencepiece` | `0.2.0` | LLaMA tokenizer 依赖 |
| `PyYAML` | `6.0.1` | YAML 配置加载 |
| `numpy` | `1.26.4` | 数值计算 |
| `scipy` | `1.12.0` | 科学计算 |
| `pandas` | `2.2.1` | 数据处理 |
| `scikit-learn` | `1.4.1.post1` | 部分算法工具与兼容依赖 |
| `tqdm` | `4.66.2` | 进度显示 |
| `huggingface-hub` | `0.21.4` | Hugging Face 资源管理 |

### 8.2 CUDA / GPU 相关依赖

当前锁定文件中还包含一组与 CUDA 12.x 对齐的 GPU 包，例如：

- `cupy-cuda12x==12.1.0`
- `nvidia-cublas-cu12==12.1.3.1`
- `nvidia-cudnn-cu12==8.9.2.26`
- `nvidia-cusolver-cu12==11.4.5.107`
- `nvidia-cusparse-cu12==12.1.0.106`
- `nvidia-nccl-cu12==2.18.1`

这意味着：

- 当前锁定依赖更偏向 CUDA 12.x 环境
- 如果你的机器不是这个 CUDA 代际，可能需要你自行调整环境

### 8.3 其它被锁定但不是主链核心的包

`requirements.txt` 中还保留了一些历史/兼容/服务侧依赖，例如：

- `fastapi`
- `uvicorn`
- `ray`
- `wandb`
- `rich`
- `GitPython`
- `markdown-it-py`
- `prometheus_client`

这些包不一定在每次运行中都直接参与 Stage 1 / Stage 2 / eval，但它们仍是当前仓库锁定环境的一部分。

如果你要查看完整、精确、带版本号的所有依赖，请直接以 [requirements.txt](./requirements.txt) 为准。

## 9. 环境安装

### 9.1 Linux / 推荐方式

```bash
cd /path/to/PrE-Text
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
accelerate config
```

### 9.2 Windows PowerShell

如果你只做配置检查、单测或轻量试验，可以：

```powershell
cd D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

但如果你要跑完整真实实验，仍建议迁移到 Linux GPU 环境。

## 10. 启动实验前必须确认的前提

### 10.1 必须存在的本地模型

至少确认下面目录真实存在：

```text
../thesis_platform/open_model/all_minilm_l6_v2
../thesis_platform/open_model/roberta_large
../thesis_platform/open_model/llama_2_7b_hf
../thesis_platform/open_model/distilgpt2
```

### 10.2 如果要跑 `eval_small`

必须额外准备：

```text
./c4_checkpoint.pth
```

或者在配置里显式指定：

```yaml
models:
  c4_checkpoint_path: /absolute/path/to/c4_checkpoint.pth
```

当前仓库没有提供训练这个 checkpoint 的脚本。  
如果你要完整跑 DistilGPT2 评测，必须自己按论文要求提前准备好该 checkpoint。

### 10.3 如果要严格贴近论文

请区分两件事：

1. 完整运行当前仓库已有数据和模型的实验链路
2. 严格复现原论文数据条件

当前仓库已经具备第 1 点。  
但第 2 点仍需你自己提供论文对应的原始数据集，例如 `Jobs / Forums / Microblog / Code` 等。

## 11. 如何启动“完整真实实验”

下面讲的不是 smoke 跑通，而是完整实验链路。

### 11.1 推荐运行方式：分阶段执行

推荐的真实运行顺序：

1. 跑 Stage 1
2. 检查 Stage 1 输出
3. 跑 Stage 2
4. 跑 LLaMA2 评测
5. 准备好 `c4_checkpoint.pth` 后再跑 DistilGPT2 评测

### 11.2 真实完整的 `congressional` 大模型实验

这是当前仓库最容易直接落地的完整实验之一。

#### 第一步：运行 Stage 1

```bash
python -m pretext_platform.scripts.run_stage1 --config configs/experiments/full_congressional_eps129.yaml
```

这一步会：

- 加载 `initial_set.json`
- 加载 `congressional_train.json`
- 用默认 `max_samples_per_client=8` 做 client 子采样
- 计算私有样本 embedding
- 连续执行 11 轮 Private Evolution
- 产出 `generated_text_it*.json` 和 `surviving_text_it*.json`

#### 第二步：运行 Stage 2

```bash
python -m pretext_platform.scripts.run_bootstrap --config configs/experiments/full_congressional_eps129.yaml
```

这一步会：

- 读取 `stage1/surviving_text_it*.json`
- 为每个 prompt 随机采样 3 条 surviving seeds
- 用 LLaMA-2-7B 生成 bootstrap 文本
- 写出 `stage2/llama7b_text_syn.json`

#### 第三步：运行 large-model evaluation

```bash
python -m pretext_platform.scripts.run_eval_large --config configs/experiments/full_congressional_eps129.yaml
```

这一步会：

- 读取 `llama7b_text_syn.json`
- 对 LLaMA-2-7B 添加 LoRA
- 在 synthetic corpus 上微调
- 在 `congressional_eval.json` 上做 next-token evaluation

#### 第四步：如果你已经准备好 `c4_checkpoint.pth`，再运行 small-model evaluation

```bash
python -m pretext_platform.scripts.run_eval_small --config configs/experiments/full_congressional_eps129.yaml
```

### 11.3 真实完整的 `bioarxiv` 大模型实验

只需要换配置：

```bash
python -m pretext_platform.scripts.run_stage1 --config configs/experiments/full_bioarxiv_eps129.yaml
python -m pretext_platform.scripts.run_bootstrap --config configs/experiments/full_bioarxiv_eps129.yaml
python -m pretext_platform.scripts.run_eval_large --config configs/experiments/full_bioarxiv_eps129.yaml
```

如果你已经准备好 `c4_checkpoint.pth`，再执行：

```bash
python -m pretext_platform.scripts.run_eval_small --config configs/experiments/full_bioarxiv_eps129.yaml
```

### 11.4 一次性全流程执行

如果你的机器和资源都已经准备好，也可以：

```bash
python -m pretext_platform.scripts.run_pipeline --config configs/experiments/full_congressional_eps129.yaml
```

但真实长时实验不推荐直接这样跑，原因有三点：

1. 难以在阶段之间检查中间产物
2. 某一阶段失败时不方便单独重跑
3. bootstrap 和 eval 经常需要不同 GPU 条件

## 12. 如何做“更贴近原论文”的完整实验

### 12.1 大模型路线

对大模型路线，`50,000` 条 bootstrap synthetic samples 本身就是合理的论文式规模之一。  
因此当前 `full_*` 配置已经比较贴近 large-model 评测路径。

### 12.2 小模型路线

如果你想更贴近论文里的 DistilGPT2 small-model 评测，需要满足两件事：

1. 准备 `c4_checkpoint.pth`
2. 把 bootstrap synthetic corpus 从 `50,000` 提升到 `2,000,000`

也就是说，在跑 `eval_small` 之前，建议把配置中的：

```yaml
bootstrap:
  num_prompts: 50000
```

改成：

```yaml
bootstrap:
  num_prompts: 2000000
```

同时确认：

```yaml
models:
  c4_checkpoint_path: /absolute/path/to/c4_checkpoint.pth
```

### 12.3 如果你要对齐原论文数据集

你需要把配置里的数据路径切换到你自己准备好的：

- `Jobs`
- `Forums`
- `Microblog`
- `Code`

并确保：

- 训练集是 flat list 或 client bucket 之一
- 初始化集是公共 seed 文本列表
- `sensitivity` 与 `max_samples_per_client` 一致

例如：

- `Jobs / Forums / Microblog` 常用 `sensitivity=8`
- `Code` 可能需要 `sensitivity=16`

## 13. 关键超参数说明

### 13.1 Stage 1

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `rounds` | `11` | Private Evolution 轮数 |
| `mask` | `0.3` | 每轮 mask 比例 |
| `lookahead` | `4` | lookahead 次数 |
| `multiplier` | `4` | synthetic population multiplier |
| `seq_len` | `64` | 序列长度 |
| `t_steps` | `2` | 每个样本的 mask-fill 次数 |
| `temperature` | `1.0` | MLM 采样温度 |
| `top_p` | `1.0` | nucleus sampling |
| `top_k` | `0` | 不启用 top-k 截断 |
| `delta` | `3e-6` | DP 的 `delta` |
| `sigma` | `11.3` 或 `2.31` | 对应论文常见的两组噪声设置 |
| `H_multiplier` | `0.25` | 阈值系数 |
| `sensitivity` | 通常 `8` | 每 client 最大样本数，也是 DP sensitivity |

在代码中：

- 直方图噪声尺度使用 `sensitivity * sigma * 1.541 * sqrt(2)`
- 阈值 `H` 使用 `sensitivity * sigma * 4.0 * H_multiplier`

### 13.2 Stage 2

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `num_prompts` | `50000` | 生成多少个 bootstrap prompt |
| `temperature` | `1.0` | LLaMA2 采样温度 |
| `top_p` | `1.0` | top-p |
| `max_tokens` | `85` | 每条生成的最大 token 数 |
| `max_model_len` | `1000` | vLLM 最大上下文长度 |

### 13.3 DistilGPT2 eval

| 参数 | 默认值 |
| --- | --- |
| `epochs` | `20` |
| `batch_size` | `256` |
| `eval_batch_size` | `8` |
| `grad_accum_steps` | `64` |
| `learning_rate` | `0.0002` |
| `cutoff_len` | `64` |

### 13.4 LLaMA2 eval

| 参数 | 默认值 |
| --- | --- |
| `epochs` | `1` |
| `batch_size` | `8` |
| `eval_batch_size` | `2` |
| `grad_accum_steps` | `16` |
| `learning_rate` | `0.0002` |
| `lora_rank` | `4` |
| `lora_alpha` | `8` |
| `lora_dropout` | `0.0` |

## 14. 输出结构

所有实验结果会写到：

```text
outputs/pretext_platform/<experiment_id>/
```

目录结构如下：

```text
outputs/pretext_platform/<experiment_id>/
  resolved_config.json
  metrics_summary.json
  stage1/
    private_embeds.npy
    generated_text_it0.json
    ...
    surviving_text_it0.json
    ...
  stage1_summary.json
  stage2/
    llama7b_text_syn.json
  stage2_summary.json
  eval_small/
    log_models_and_accuracies/
  eval_small_summary.json
  eval_large/
    llama2_models_and_accuracies/
  eval_large_summary.json
```

### 14.1 Stage 1 关键产物

| 文件 | 用途 |
| --- | --- |
| `private_embeds.npy` | 私有训练文本 embedding 缓存 |
| `generated_text_it*.json` | 每轮的 synthetic population |
| `surviving_text_it*.json` | 每轮 survivor seeds |

### 14.2 Stage 2 关键产物

| 文件 | 用途 |
| --- | --- |
| `llama7b_text_syn.json` | bootstrap 生成的大规模 synthetic corpus |

### 14.3 下游评测产物

| 目录 | 用途 |
| --- | --- |
| `eval_small/log_models_and_accuracies/` | DistilGPT2 checkpoint 和统计结果 |
| `eval_large/llama2_models_and_accuracies/` | LLaMA2-LoRA checkpoint 和统计结果 |

## 15. 测试与验证

运行单元测试：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

当前测试覆盖：

- YAML 继承与路径解析
- 训练/评测数据多形状加载
- Stage 1 编排 mock
- pipeline 摘要写出
- legacy CLI 映射
- `eval_small` 缺失 checkpoint 的显式失败

测试通过不代表：

- 你的 GPU 环境已经可跑
- `vllm` 已经能工作
- 本地 LLaMA2 权重已存在
- `c4_checkpoint.pth` 已经准备好

它只代表：

- 配置系统
- 路径解析
- 关键 orchestration 逻辑
- 兼容包装层

这些部分是正常的。

## 16. 常见问题

### 16.1 `eval_small` 一启动就报缺少 checkpoint

这是预期行为。  
你没有提供 `c4_checkpoint.pth`，当前实现不会自动降级到无 warm-start 模式。

### 16.2 `run_bootstrap` 在 Windows 上不稳定或不可用

优先迁移到 Linux GPU 环境。  
完整真实实验不建议在 Windows 上跑 `vllm`。

### 16.3 Stage 1 报 `DP histogram sum is zero after thresholding`

说明当前参数组合太苛刻，通常可以尝试：

- 增大 `stage1.multiplier`
- 降低 `stage1.H_multiplier`
- 调整 `stage1.sigma`

当前实现不会自动改用其他采样策略。

### 16.4 为什么当前默认数据不是原论文里的 Jobs / Forums / Microblog / Code

因为当前工作区中现成可用的是 `congressional` / `bioarxiv`，平台默认首先面向“真实可运行的当前本地资源”。  
如果你自己准备好了原论文数据，只需要在配置里替换数据路径即可。

## 17. 兼容旧命令

以下旧命令仍可使用：

```bash
python main.py ...
python llama_bootstrap.py ...
python eval_distilgpt2.py ...
python eval_llama2.py ...
```

它们当前的作用只是：

1. 解析旧参数
2. 映射为新配置
3. 调用 `pretext_platform` 的对应阶段

不建议在新实验里继续以它们作为主入口，推荐直接使用：

```bash
python -m pretext_platform.scripts.run_stage1 --config ...
python -m pretext_platform.scripts.run_bootstrap --config ...
python -m pretext_platform.scripts.run_eval_small --config ...
python -m pretext_platform.scripts.run_eval_large --config ...
python -m pretext_platform.scripts.run_pipeline --config ...
```

## 18. 引用

如果你在研究中使用 PrE-Text，请引用原论文：

```bibtex
@misc{hou2024pretext,
      title={PrE-Text: Training Language Models on Private Federated Data in the Age of LLMs},
      author={Charlie Hou and Akshat Shrivastava and Hongyuan Zhan and Rylan Conway and Trang Le and Adithya Sagar and Giulia Fanti and Daniel Lazar},
      year={2024},
      eprint={2406.02958},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
