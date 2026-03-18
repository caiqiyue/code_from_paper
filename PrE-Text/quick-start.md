# PrE-Text Quick Start

本文档只解决两件事：

1. 如何用极少数据量做一次验证性实验，确认 `Stage 1 -> Stage 2 -> downstream eval` 的主链是通的。
2. 如何启动当前仓库支持的所有真实实验，包括当前工作区现成数据上的完整实验，以及尽量贴近 `PrE-Text.pdf` 的真实实验。

本文档以仓库根目录 `PrE-Text/` 为工作目录。

---

## 1. 先看结论

当前仓库里最重要的配置文件有 5 份：

| 配置文件 | 用途 | 数据来源 | 模型来源 | 是否适合长时真实实验 |
| --- | --- | --- | --- | --- |
| `configs/experiments/validate_jobs_tiny_thesis_platform.yaml` | 极小样本验证性实验 | `../thesis_platform/datasets` | `../thesis_platform/open_model` | 否 |
| `configs/experiments/full_congressional_eps129.yaml` | 当前工作区 `congressional` 完整实验 | `../datasets` | `../thesis_platform/open_model` | 是 |
| `configs/experiments/full_bioarxiv_eps129.yaml` | 当前工作区 `bioarxiv` 完整实验 | `../datasets` | `../thesis_platform/open_model` | 是 |
| `configs/experiments/paper_jobs_real_large_eval_eps129.yaml` | 贴近论文 `Jobs` 的 large-model 真实实验 | `../thesis_platform/datasets` | `../thesis_platform/open_model` | 是 |
| `configs/experiments/smoke_congressional_eps758.yaml` / `smoke_bioarxiv_eps758.yaml` | 旧的 smoke 级调试配置 | `../datasets` | `../thesis_platform/open_model` | 否 |

建议你按下面顺序使用：

1. 先跑 `validate_jobs_tiny_thesis_platform.yaml`
2. 再跑 `full_congressional_eps129.yaml` 或 `full_bioarxiv_eps129.yaml`
3. 如果要尽量贴近论文 `Jobs` 路线，再跑 `paper_jobs_real_large_eval_eps129.yaml`
4. 如果还要走论文 DistilGPT2 小模型分支，再额外准备 `c4_checkpoint.pth`

---

## 2. 运行前检查

### 2.1 推荐环境

| 项目 | 建议 |
| --- | --- |
| 操作系统 | Linux 优先 |
| Python | 3.10.x |
| CUDA | 与 `requirements.txt` 中的 `torch==2.1.2` / CUDA 12.x 依赖一致 |
| GPU | Stage 1 建议 32GB 以上；Stage 2 和 LLaMA2 eval 建议 48GB 级别显存 |

说明：

- Windows 可以做配置检查、单元测试、极小验证实验。
- 完整真实实验，尤其是 `vllm` 的 Stage 2，更推荐 Linux GPU 环境。

### 2.2 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 关键第三方包

这些包直接参与实验主链：

| 包 | 版本 | 用途 |
| --- | --- | --- |
| `torch` | `2.1.2` | 训练与推理 |
| `transformers` | `4.38.2` | RoBERTa / DistilGPT2 / LLaMA2 |
| `accelerate` | `0.28.0` | 单卡/多卡封装 |
| `sentence-transformers` | `2.5.1` | MiniLM embedding |
| `faiss-cpu` | `1.8.0` | 最近邻搜索 |
| `opacus` | `1.4.1` | DP 会计 |
| `datasets` | `2.18.0` | 数据处理 |
| `peft` | `0.10.0` | LoRA |
| `vllm` | `0.3.3` | Stage 2 bootstrap |
| `PyYAML` | `6.0.1` | YAML 配置 |

完整依赖以 `requirements.txt` 为准。

### 2.4 必须提前确认的本地资源

当前平台默认依赖这些模型目录：

```text
../thesis_platform/open_model/all_minilm_l6_v2
../thesis_platform/open_model/roberta_large
../thesis_platform/open_model/llama_2_7b_hf
../thesis_platform/open_model/distilgpt2
```

如果要跑论文 `Jobs` 路线，还需要这些数据文件存在：

```text
../thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
../thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
../thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

如果要跑当前工作区 `congressional` / `bioarxiv` 路线，还需要这些数据文件存在：

```text
../datasets/congressional_train.json
../datasets/congressional_eval.json
../datasets/bioarxiv_train.json
../datasets/bioarxiv_eval.json
../datasets/initial_set.json
```

### 2.5 关于 `c4_checkpoint.pth`

`eval_small` 不是可选降级逻辑，而是强依赖 warm-start checkpoint。

也就是说：

- 没有 `c4_checkpoint.pth` 时，不要启动 `eval_small`
- 当前仓库不会自动从原始 DistilGPT2 开始训练
- 没有 checkpoint 时，会直接报错

默认配置位置是：

```text
./c4_checkpoint.pth
```

也可以在配置里改成绝对路径：

```yaml
models:
  c4_checkpoint_path: /absolute/path/to/c4_checkpoint.pth
```

---

## 3. 统一启动入口

平台化后的主入口只有 5 个：

| 命令 | 作用 |
| --- | --- |
| `python -m pretext_platform.scripts.run_stage1 --config <yaml>` | 只跑 Stage 1 |
| `python -m pretext_platform.scripts.run_bootstrap --config <yaml>` | 只跑 Stage 2 |
| `python -m pretext_platform.scripts.run_eval_small --config <yaml>` | 只跑 DistilGPT2 eval |
| `python -m pretext_platform.scripts.run_eval_large --config <yaml>` | 只跑 LLaMA2 eval |
| `python -m pretext_platform.scripts.run_pipeline --config <yaml>` | 按配置里启用的阶段顺序全跑 |

### 3.1 CLI 参数说明

所有新脚本都只有一个核心参数：

| 参数 | 是否必填 | 说明 |
| --- | --- | --- |
| `--config` | 是 | 指向某个实验 YAML 配置文件 |

示例：

```bash
python -m pretext_platform.scripts.run_stage1 --config configs/experiments/full_congressional_eps129.yaml
```

### 3.2 为什么真实实验推荐分阶段执行

真实实验不推荐一开始就只用 `run_pipeline`，原因很直接：

1. Stage 1 很重，最好先确认 `surviving_text_it*.json` 正常生成
2. Stage 2 用 `vllm`，经常和 Stage 1 的硬件需求不同
3. `eval_small` 还依赖外部 `c4_checkpoint.pth`
4. 任一阶段失败时，分阶段执行更方便续跑

---

## 4. 验证性实验的启动方式

### 4.1 目标

这套配置不是为了看最终指标，而是为了确认整条主链在你当前机器和当前本地资源上是通的。

使用配置：

```text
configs/experiments/validate_jobs_tiny_thesis_platform.yaml
```

这份配置的关键特征：

- 数据直接来自 `../thesis_platform/datasets`
- 模型直接来自 `../thesis_platform/open_model`
- `train_limit=64`
- `eval_limit=32`
- `initialization_limit=512`
- `stage1.rounds=2`
- `bootstrap.num_prompts=32`
- `eval_large.enabled=true`
- `eval_small.enabled=false`

这意味着它会走通：

1. Stage 1 Private Evolution
2. Stage 2 bootstrap
3. LLaMA2 downstream eval

### 4.2 分阶段启动

#### 第一步：Stage 1

```bash
python -m pretext_platform.scripts.run_stage1 --config configs/experiments/validate_jobs_tiny_thesis_platform.yaml
```

成功后应出现：

```text
outputs/pretext_platform/validate_jobs_tiny_thesis_platform_eps758/stage1/
```

其中至少应有：

```text
private_embeds.npy
generated_text_it0.json
generated_text_it1.json
surviving_text_it0.json
surviving_text_it1.json
```

#### 第二步：Stage 2

```bash
python -m pretext_platform.scripts.run_bootstrap --config configs/experiments/validate_jobs_tiny_thesis_platform.yaml
```

成功后应出现：

```text
outputs/pretext_platform/validate_jobs_tiny_thesis_platform_eps758/stage2/llama7b_text_syn.json
```

#### 第三步：Large-model eval

```bash
python -m pretext_platform.scripts.run_eval_large --config configs/experiments/validate_jobs_tiny_thesis_platform.yaml
```

成功后应出现：

```text
outputs/pretext_platform/validate_jobs_tiny_thesis_platform_eps758/eval_large/llama2_models_and_accuracies/
```

### 4.3 一次性启动

```bash
python -m pretext_platform.scripts.run_pipeline --config configs/experiments/validate_jobs_tiny_thesis_platform.yaml
```

### 4.4 这套验证性实验算“跑通”的标准

满足下面 4 条即可认为主链跑通：

1. `stage1/private_embeds.npy` 正常生成
2. 至少有 `surviving_text_it0.json`
3. `stage2/llama7b_text_syn.json` 正常生成
4. `eval_large/llama2_models_and_accuracies/` 下生成 `baseline_stats.json` 或 `epoch0_stats.json`

---

## 5. 当前工作区现成真实实验的启动方式

这里的“真实实验”指的是：

- 使用当前工作区已经准备好的真实规模数据
- 使用平台默认的完整 Stage 1 主链
- 使用真实的 Stage 2 bootstrap
- 使用真实的 downstream evaluation

它们不是 tiny smoke，也不是只跑一两个 batch。

### 5.1 `congressional` 完整实验

配置文件：

```text
configs/experiments/full_congressional_eps129.yaml
```

#### 推荐启动顺序

Stage 1:

```bash
python -m pretext_platform.scripts.run_stage1 --config configs/experiments/full_congressional_eps129.yaml
```

Stage 2:

```bash
python -m pretext_platform.scripts.run_bootstrap --config configs/experiments/full_congressional_eps129.yaml
```

Large-model eval:

```bash
python -m pretext_platform.scripts.run_eval_large --config configs/experiments/full_congressional_eps129.yaml
```

Small-model eval，前提是已经准备好 `c4_checkpoint.pth`:

```bash
python -m pretext_platform.scripts.run_eval_small --config configs/experiments/full_congressional_eps129.yaml
```

#### 一次性全流程启动

只有在你已经准备好 `c4_checkpoint.pth` 时，才建议这样跑：

```bash
python -m pretext_platform.scripts.run_pipeline --config configs/experiments/full_congressional_eps129.yaml
```

#### 这套实验的真实含义

- 数据：`../datasets/congressional_train.json`、`../datasets/congressional_eval.json`
- 初始化种群：`../datasets/initial_set.json`
- 噪声模板：`sigma=11.3`
- `max_samples_per_client=8`
- Stage 1 默认 11 轮
- Stage 2 默认 50000 prompts

### 5.2 `bioarxiv` 完整实验

配置文件：

```text
configs/experiments/full_bioarxiv_eps129.yaml
```

命令与 `congressional` 相同，只是替换配置文件：

Stage 1:

```bash
python -m pretext_platform.scripts.run_stage1 --config configs/experiments/full_bioarxiv_eps129.yaml
```

Stage 2:

```bash
python -m pretext_platform.scripts.run_bootstrap --config configs/experiments/full_bioarxiv_eps129.yaml
```

Large-model eval:

```bash
python -m pretext_platform.scripts.run_eval_large --config configs/experiments/full_bioarxiv_eps129.yaml
```

Small-model eval，前提是已经准备好 `c4_checkpoint.pth`:

```bash
python -m pretext_platform.scripts.run_eval_small --config configs/experiments/full_bioarxiv_eps129.yaml
```

全流程：

```bash
python -m pretext_platform.scripts.run_pipeline --config configs/experiments/full_bioarxiv_eps129.yaml
```

---

## 6. 贴近论文 `PrE-Text.pdf` 的真实实验启动方式

### 6.1 论文 `Jobs` large-model 真实实验

配置文件：

```text
configs/experiments/paper_jobs_real_large_eval_eps129.yaml
```

这份配置尽量贴近论文 `Jobs` 路线，并直接复用你当前工作区已有的数据和模型：

- 训练集：`../thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json`
- 评测集：`../thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json`
- 初始化池：`../thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json`
- `sigma=11.3`
- `max_samples_per_client=8`
- Stage 1 保持真实默认主链
- Stage 2 为 `50000` prompts
- 只启用 `eval_large`

#### 推荐启动顺序

Stage 1:

```bash
python -m pretext_platform.scripts.run_stage1 --config configs/experiments/paper_jobs_real_large_eval_eps129.yaml
```

Stage 2:

```bash
python -m pretext_platform.scripts.run_bootstrap --config configs/experiments/paper_jobs_real_large_eval_eps129.yaml
```

Large-model eval:

```bash
python -m pretext_platform.scripts.run_eval_large --config configs/experiments/paper_jobs_real_large_eval_eps129.yaml
```

也可以一次性全跑：

```bash
python -m pretext_platform.scripts.run_pipeline --config configs/experiments/paper_jobs_real_large_eval_eps129.yaml
```

### 6.2 论文 `Jobs` small-model 真实实验

这条路线不能直接用当前仓库已有的配置一键启动，因为它还缺一个外部条件：

```text
c4_checkpoint.pth
```

此外，论文 small-model 分支通常要求：

- `bootstrap.num_prompts = 2000000`
- `eval_small.enabled = true`
- `eval_large.enabled = false`

#### 正确做法

在仓库里新建一个你自己的配置，例如：

```text
configs/experiments/paper_jobs_real_small_eval_eps129.local.yaml
```

内容可以基于 `paper_jobs_real_large_eval_eps129.yaml` 改成：

```yaml
inherits:
  - ../base/paths.yaml
  - ../base/models.yaml
  - ../base/runtime.yaml
  - ../templates/noise_eps129.yaml

meta:
  experiment_id: paper_jobs_real_small_eval_eps129
  seed: 42

paths:
  dataset_root: ../thesis_platform/datasets
  model_root: ../thesis_platform/open_model

data:
  dataset_name: jobs
  train_path: ../thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: ../thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: ../thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
  max_samples_per_client: 8
  initialization_min_words: 20

models:
  c4_checkpoint_path: /absolute/path/to/c4_checkpoint.pth

stage1:
  sensitivity: 8

bootstrap:
  num_prompts: 2000000

eval_small:
  enabled: true

eval_large:
  enabled: false
```

#### 启动顺序

Stage 1:

```bash
python -m pretext_platform.scripts.run_stage1 --config configs/experiments/paper_jobs_real_small_eval_eps129.local.yaml
```

Stage 2:

```bash
python -m pretext_platform.scripts.run_bootstrap --config configs/experiments/paper_jobs_real_small_eval_eps129.local.yaml
```

Small-model eval:

```bash
python -m pretext_platform.scripts.run_eval_small --config configs/experiments/paper_jobs_real_small_eval_eps129.local.yaml
```

如果 `c4_checkpoint.pth` 已经准备好，也可以全流程：

```bash
python -m pretext_platform.scripts.run_pipeline --config configs/experiments/paper_jobs_real_small_eval_eps129.local.yaml
```

### 6.3 如果要切换到 `eps=7.58`

做法不是改脚本，而是改配置继承：

把：

```yaml
- ../templates/noise_eps129.yaml
```

换成：

```yaml
- ../templates/noise_eps758.yaml
```

对应关系：

| 模板 | `sigma` | 常见对应 |
| --- | --- | --- |
| `noise_eps129.yaml` | `11.3` | `eps=1.29` |
| `noise_eps758.yaml` | `2.31` | `eps=7.58` |

---

## 7. 配置项说明

这一节不讲命令，而是讲你真正会改的参数。

### 7.1 `meta`

| 参数 | 类型 | 作用 |
| --- | --- | --- |
| `experiment_id` | `str` | 输出目录名 |
| `seed` | `int` | 全流程随机种子 |

### 7.2 `paths`

| 参数 | 类型 | 作用 |
| --- | --- | --- |
| `repo_root` | `str` | 仓库根目录，相对当前配置文件解析 |
| `dataset_root` | `str` | 数据根目录；未显式写 `train_path/eval_path/initialization_path` 时用它回退 |
| `model_root` | `str` | 模型根目录 |
| `output_root` | `str` | 实验输出根目录 |

### 7.3 `data`

| 参数 | 类型 | 作用 |
| --- | --- | --- |
| `dataset_name` | `str` | 数据集逻辑名 |
| `train_path` | `str` | 训练集显式路径，优先级高于 `dataset_root` 回退 |
| `eval_path` | `str` | 评测集显式路径 |
| `initialization_path` | `str` | 初始化种群显式路径 |
| `max_samples_per_client` | `int` | client bucket 数据集每个 client 最多抽多少条 |
| `initialization_min_words` | `int` | 初始化池里保留样本的最小词数阈值 |
| `train_limit` | `int` | 只取前多少条训练文本；主要给验证性实验用 |
| `eval_limit` | `int` | 只取前多少条评测文本；主要给验证性实验用 |
| `initialization_limit` | `int` | 只取前多少条初始化文本；主要给验证性实验用 |

说明：

- `train_limit/eval_limit/initialization_limit` 不是论文参数，而是平台为了验证性实验加入的工程参数。
- 真实实验一般不要设置这 3 个参数。

### 7.4 `models`

| 参数 | 类型 | 作用 |
| --- | --- | --- |
| `minilm_path` | `str` | Stage 1 embedding 模型路径 |
| `roberta_large_path` | `str` | Stage 1 mask-fill 模型路径 |
| `llama2_7b_path` | `str` | Stage 2 和 large-model eval 的 LLaMA2 路径 |
| `distilgpt2_path` | `str` | small-model eval 的 DistilGPT2 路径 |
| `c4_checkpoint_path` | `str` | DistilGPT2 warm-start checkpoint 路径 |

### 7.5 `stage1`

| 参数 | 类型 | 作用 |
| --- | --- | --- |
| `enabled` | `bool` | `run_pipeline` 时是否启用 Stage 1 |
| `rounds` | `int` | Private Evolution 轮数；真实默认值是 `11` |
| `mask` | `float` | 每轮 mask 比例；真实默认值是 `0.3` |
| `lookahead` | `int` | lookahead 次数；真实默认值是 `4` |
| `multiplier` | `int` | synthetic population multiplier；真实默认值是 `4` |
| `seq_len` | `int` | token 截断长度；默认 `64` |
| `t_steps` | `int` | 每个样本做几次 mask-fill 变异；默认 `2` |
| `batch_size` | `int` | 变异主 batch size |
| `embed_batch_size` | `int` | embedding batch size |
| `temperature` | `float` | MLM 采样温度 |
| `top_p` | `float` | nucleus sampling 参数 |
| `top_k` | `int` | top-k 截断；默认 `0` 表示不截断 |
| `num_workers` | `int` | DataLoader worker 数量 |
| `nearest_neighbors_print` | `int` | 每轮打印多少个最近邻样本 |
| `H_multiplier` | `float` | 直方图阈值缩放因子 |
| `delta` | `float` | DP 的 `delta` |
| `sigma` | `float` | 噪声倍率 |
| `sensitivity` | `int` | DP sensitivity；一般与 `max_samples_per_client` 对齐 |

### 7.6 `bootstrap`

| 参数 | 类型 | 作用 |
| --- | --- | --- |
| `enabled` | `bool` | `run_pipeline` 时是否启用 Stage 2 |
| `num_prompts` | `int` | 要生成多少个 bootstrap prompts |
| `temperature` | `float` | LLaMA2 生成温度 |
| `top_p` | `float` | top-p |
| `max_tokens` | `int` | 每次生成的最大 token 数 |
| `max_model_len` | `int` | `vllm` 最大上下文长度 |

经验解释：

- `32` 适合验证性实验
- `50000` 适合 large-model 真实实验
- `2000000` 更接近论文 small-model 真实实验

### 7.7 `eval_small`

| 参数 | 类型 | 作用 |
| --- | --- | --- |
| `enabled` | `bool` | 是否启用 DistilGPT2 eval |
| `cutoff_len` | `int` | token 截断长度 |
| `grad_accum_steps` | `int` | 梯度累积步数 |
| `epochs` | `int` | 训练 epoch 数 |
| `batch_size` | `int` | 训练 batch size |
| `eval_batch_size` | `int` | 验证 batch size |
| `learning_rate` | `float` | 学习率 |
| `num_proc` | `int` | `datasets.map` 的进程数 |

### 7.8 `eval_large`

| 参数 | 类型 | 作用 |
| --- | --- | --- |
| `enabled` | `bool` | 是否启用 LLaMA2 eval |
| `cutoff_len` | `int` | token 截断长度 |
| `grad_accum_steps` | `int` | 梯度累积步数 |
| `epochs` | `int` | epoch 数 |
| `batch_size` | `int` | 训练 batch size |
| `eval_batch_size` | `int` | 验证 batch size |
| `learning_rate` | `float` | 学习率 |
| `num_proc` | `int` | `datasets.map` 的进程数 |
| `lora_rank` | `int` | LoRA rank |
| `lora_alpha` | `int` | LoRA alpha |
| `lora_dropout` | `float` | LoRA dropout |

### 7.9 `runtime`

| 参数 | 类型 | 作用 |
| --- | --- | --- |
| `device` | `str` | 运行设备，通常是 `cuda` |

---

## 8. 输出目录说明

所有配置最终都会写到：

```text
outputs/pretext_platform/<experiment_id>/
```

典型结构：

```text
outputs/pretext_platform/<experiment_id>/
  resolved_config.json
  metrics_summary.json
  stage1/
    private_embeds.npy
    generated_text_it*.json
    surviving_text_it*.json
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

---

## 9. 常见启动问题

### 9.1 `eval_small` 一启动就报 checkpoint 不存在

这是正常保护逻辑，不是 bug。  
处理方式：

1. 准备 `c4_checkpoint.pth`
2. 在配置里设置 `models.c4_checkpoint_path`
3. 或者先把 `eval_small.enabled` 设成 `false`

### 9.2 `run_bootstrap` 在 Windows 上不稳定

优先切到 Linux GPU 环境。  
尤其是 `vllm==0.3.3`，不要把 Windows 当作完整真实实验的主环境。

### 9.3 Stage 1 报 `DP histogram sum is zero after thresholding`

说明噪声、阈值和种群规模的组合不合适。优先尝试：

1. 增大 `stage1.multiplier`
2. 减小 `stage1.H_multiplier`
3. 调整 `stage1.sigma`

### 9.4 为什么 `congressional` / `bioarxiv` 不用 `thesis_platform/datasets`

因为当前工作区现成这两套数据就在仓库外层的 `../datasets`。  
平台没有强行把所有数据都迁到 `thesis_platform/datasets`，而是按现有本地资源直接复用。

### 9.5 为什么 `validate_jobs_tiny_thesis_platform.yaml` 可以直接走 `thesis_platform/datasets`

因为这份验证配置显式写了：

- `paths.dataset_root`
- `data.train_path`
- `data.eval_path`
- `data.initialization_path`

并且通过 `train_limit/eval_limit/initialization_limit` 把真实数据裁成极小子集。

---

## 10. 推荐使用顺序

如果你刚接手这个平台，最稳妥的顺序是：

1. `validate_jobs_tiny_thesis_platform.yaml`
2. `full_congressional_eps129.yaml`
3. `full_bioarxiv_eps129.yaml`
4. `paper_jobs_real_large_eval_eps129.yaml`
5. 准备好 `c4_checkpoint.pth` 后，再跑论文 `Jobs` small-model 分支

这个顺序的好处是：

- 先验证平台是通的
- 再跑当前工作区已经准备好的完整实验
- 最后再推进到更贴近论文的真实复现
