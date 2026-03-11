# PrE-Text 论文模型、数据集、超参数与当前项目数据集现状总结

## 1. 说明

本总结基于以下材料整理：

- `PrE-Text.pdf` 对应论文内容
- `PRETEXT_FEATURE_EXTRACTION_PLAN.md`
- 当前仓库中的 `README.md`、`project_manual.md`、`main.py`、`llama_bootstrap.py`、`eval_distilgpt2.py`、`eval_llama2.py`

其中，论文 PDF 的实验细节我额外对照了对应的 arXiv 源文件，以补全网页/终端文本提取时丢失的公式和参数。

## 2. PrE-Text 论文中使用了哪些模型

### 2.1 PrE-Text 主方法直接使用的模型

| 用途 | 模型 |
| --- | --- |
| `Variation` 阶段的 mask-fill 变异模型 | `RoBERTa-large` |
| 文本 embedding / 最近邻直方图打分 | `all-MiniLM-L6-v2` |
| `Expand` 阶段的 seed 扩增大模型 | `LLaMA-2-7B` |

### 2.2 论文中的下游评测模型

| 场景 | 模型 |
| --- | --- |
| 小模型、端侧 next-token prediction 评测 | `DistilGPT2` |
| 大模型、服务端微调评测 | `LLaMA-2-7B` |

### 2.3 论文中的对比基线还涉及的模型/方法

| 类型 | 名称 |
| --- | --- |
| 端侧 DP 训练基线 | `DP-FedAvg` |
| 端侧 DP 训练基线 | `DP-FTRL`（论文中使用 `DP-FTRL-TreeRestart` 变体） |
| 文本到文本的 DP 合成基线 | `DP-Prompt` |
| `DP-Prompt` 所用模型 | `flan-t5-3b` |

## 3. PrE-Text 论文中使用了哪些数据集

### 3.1 私有联邦数据集

论文构造并评测了 4 个私有联邦数据集：

| 数据集 | 来源 | 说明 |
| --- | --- | --- |
| `Jobs` | `c4-English (c4-en)` 子集 | jobs 站点文本子集 |
| `Forums` | `c4-English (c4-en)` 子集 | forum 站点文本子集 |
| `Microblog` | `c4-English (c4-en)` 子集 | microblog 站点文本子集 |
| `Code` | 编码/技术问答数据集 | 按用户划分的评论/问答文本 |

### 3.2 数据集构造细节

- `Jobs` / `Forums` / `Microblog`
  - 各自从 `c4-en` 中取对应站点来源的前 `11,000` 条样本。
  - 其中 `10,000` 条作为 private train，`1,000` 条作为 eval。
  - 训练集被均匀随机拆成 `1250` 个客户端，每个客户端 `8` 条样本。
- `Code`
  - 是面向 coding / technical topics 的问答数据集。
  - 构造 `1250` 个用户客户端。
  - 每个用户训练时最多保留 `128` 条 comments。
  - eval 集来自后续 `100` 个用户中的前 `2000` 条样本。

### 3.3 公共/初始化数据

- `c4-English (c4-en)` 也被用作公共数据来源。
- PrE-Text 的初始种群 `S1` 使用的是一个不属于任何私有数据集、且不来自相同网站来源的 `c4-en` 子集。
- 论文中 `c4-only` / `Expand-only` 基线使用了一个约 `87k` 样本的公共 `c4-en` 子集。

### 3.4 合成数据规模

| 场景 | 合成数据规模 |
| --- | --- |
| 小模型 on-device 评测 | 扩增到 `2,000,000` 条 synthetic samples |
| 大模型 on-server 评测 | 由于算力限制，仅扩增到 `50,000` 条 synthetic samples |
| 扩展性实验 | 在 `50,000` 到 `2,000,000` 之间变化 |

## 4. PrE-Text 论文中使用了哪些超参数

## 4.1 论文中明确写出的 PrE-Text 关键超参数

| 类别 | 超参数 | 论文值/说明 |
| --- | --- | --- |
| 隐私 | `delta` | `3e-6` |
| 隐私 | `epsilon` | 主要报告 `1.29` 和 `7.58` |
| 隐私会计 | 迭代轮数 `T` | `11` |
| 隐私会计 | 采样率 `q` | `1.0` |
| 变异 | `Mask%` | `30%` |
| 变异 | `W_steps` | `2` |
| 变异采样 | `top_p` | `1.0` |
| 变异采样 | `temperature` | `1.0` |
| embedding | 维度 | `384`（`all-MiniLM-L6-v2`） |
| Expand 采样 | `top_p` | `1.0` |
| Expand 采样 | `temperature` | `1.0` |
| 候选规模 | `N_syn` at `epsilon=1.29` | `1024` |
| 候选规模 | `N_syn` at `epsilon=7.58` | `2048` |
| 阈值 | `H` at `epsilon=1.29` | `5.9 * 8.0 * 1.541 * sqrt(2)` |
| 阈值 | `H` at `epsilon=7.58` | `8.0 * 1.541 * sqrt(2)` |

### 4.2 与敏感度相关的设置

- `Jobs` / `Forums` / `Microblog`
  - 每客户端最多 `8` 条样本，因此 sensitivity 上限为 `8`。
- `Code`
  - 论文实验中额外把每客户端样本数裁到 `16`，因此 sensitivity 上限为 `16`。
  - 因为 sensitivity 变大，`Code` 上使用的噪声和阈值 `H` 都相对翻倍。

### 4.3 下游训练超参数

#### 小模型 `DistilGPT2`

- 训练目标：next-token prediction
- 优化器：`AdamW`
- 学习率：`0.0002`
- 训练 epoch：`20`
- `c4-only` 微调 batch size：`256`
- `Expand-only` / `Expand-private` 扩增数据微调 batch size：`65536`

#### 大模型 `LLaMA-2-7B`

- 训练方式：`LoRA` 微调
- epoch：`1`
- LoRA rank：`4`
- `alpha=8`
- 作用位置：应用到 `LLaMA-2-7B` 的所有 projection matrices
- 优化器：`AdamW`
- 学习率：`0.0002`
- batch size：`512`

### 4.4 对比基线中的超参数

#### `DP-FedAvg` / `DP-FTRL`

- client learning rate 网格：`{0.1, 0.01, 0.001}`
- communication rounds 网格：`{10, 20, 100}`
- clipping 网格：`{1.0, 0.1, 0.01, 0.001}`
- batch size：`4`
- server momentum：`0.9`

#### `DP-Prompt`

- 使用 `flan-t5-3b`
- 下游 `DistilGPT2` 微调使用：
  - `AdamW`
  - learning rate `0.0002`
  - batch size `256`
  - 训练 `20` epochs

### 4.5 论文复现脚本和计划文档中保留的默认参数

下面这些值在论文正文里不是全部逐项展开，但在当前仓库的复现脚本、README 和计划文档中是明确出现的，可视为当前项目对论文实验设置的直接继承：

| 参数 | 当前仓库默认值 | 说明 |
| --- | --- | --- |
| `mask` | `0.3` | 与论文中的 `Mask%=30%` 一致 |
| `lookahead` | `4` | lookahead 次数 |
| `multiplier` | `4` | 每轮候选扩增倍数 |
| `t_steps` | `2` | 与论文中的 `W_steps=2` 一致 |
| `H_multiplier` | `0.25` | 用于构造阈值 `H` |
| `top_k` | `0` | 只使用 `top_p` 采样截断 |
| `top_p` | `1.0` | 与论文一致 |
| `batch_size` | `256` | stage-1 候选生成 batch size |
| `embed_batch_size` | `512` | embedding 计算 batch size |

另外，README 中给出了论文复现时常用的噪声比值对应关系：

- `sigma/noise = 11.3` 对应 `epsilon = 1.29`
- `sigma/noise = 2.31` 对应 `epsilon = 7.58`

## 5. 从 `PRETEXT_FEATURE_EXTRACTION_PLAN.md` 可直接提炼出的参数/能力

计划文档没有改变论文实验结论，但明确指出当前项目如果抽取 PrE-Text 能力，核心保留的是下面这些参数和模块：

### 5.1 计划文档点名保留的参数

- `init_population_path`
- `seq_len`
- `mask`
- `lookahead`
- `multiplier`
- `t_steps`
- `bootstrap_enable`
- `bootstrap_model`
- `generated_per_round`

### 5.2 计划文档点名保留的核心模块

- `variation.py`
- `similarity.py`
- `nn_histogram.py`
- `llama_bootstrap.py`
- `main.py` 中的生成-打分-重采样流程思想

### 5.3 计划文档对参数的补充解释

- `mask`：mask-fill 变异比例
- `lookahead`：未来若干步变异后的平均 embedding 次数
- `multiplier`：每轮候选扩增倍数
- `t_steps`：每个样本重复执行 mask-fill 的次数
- `bootstrap_model`：第二阶段扩增所用大模型

## 6. 当前项目中有哪些数据集

### 6.1 当前仓库里实际存在的数据集

结论：**当前项目根目录下没有实际提供任何实验数据集文件。**

我已检查当前仓库，结果如下：

- 没有 `data/` 目录
- 没有 `initialization.json`
- 没有 `*_train.json`
- 没有 `*_eval.json`
- 没有 `*.jsonl / *.csv / *.parquet` 等实际数据文件

也就是说，当前项目是“代码与文档在、数据不随仓库分发”的状态。

### 6.2 当前项目代码所约定的数据集接口

虽然仓库里没有真实数据，但代码和文档约定了应当准备以下输入：

| 路径模式 | 用途 |
| --- | --- |
| `data/initialization.json` | 初始种群 `S1`，至少约 `10000` 条公共样本 |
| `data/<dataset_name>_train.json` | 训练集，聚合后的私有训练样本列表 |
| `data/<dataset_name>_eval.json` | 评测集，格式类似 `{"1": ["text1", "text2"]}` |

### 6.3 当前项目里与数据集相关、但不是数据集本身的文件

- `PrE-Text.pdf`：论文
- `pretext_arxiv_source.tar` / `arxiv_src/`：我为核对论文细节临时下载/解压的论文源码
- `c4_checkpoint.pth`：文档中提到的小模型 warm-start checkpoint，但当前仓库里也不存在

## 7. 一句话结论

PrE-Text 论文的核心模型是 `RoBERTa-large + all-MiniLM-L6-v2 + LLaMA-2-7B`，主要数据集是从 `c4-en` 构造出的 `Jobs / Forums / Microblog` 以及独立的 `Code` 数据集，核心实验围绕 `Mask%=30%`、`W_steps=2`、`delta=3e-6`、`epsilon in {1.29, 7.58}`、`N_syn in {1024, 2048}` 展开；而**当前这个项目仓库本身并没有附带任何真实数据集文件，只定义了数据输入格式和路径规范。**
