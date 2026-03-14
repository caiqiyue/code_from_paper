# PrE-Text 快速阅读与上手说明

## 1. 项目是做什么的

PrE-Text 用来在**不做端上训练**的前提下，对**私有联邦文本数据**进行差分隐私建模。整体流程分两段：

1. `main.py` 在服务端根据私有数据生成一小批 **DP seed synthetic data**。
2. `llama_bootstrap.py` 再基于这些 seed 文本扩增出更大的合成数据集。

之后，项目还提供两类下游评测：

1. `eval_distilgpt2.py`：用小模型 `DistilGPT2` 做 next-token prediction 评测。
2. `eval_llama2.py`：用 `LLaMA-2-7b` + LoRA 做大模型 next-token prediction 评测。

## 2. 当前仓库目录结构

这个仓库本身非常精简，**已提交到仓库中的文件夹只有 `assets/`**，其余核心逻辑都在项目根目录的 Python 脚本里。

| 路径 | 作用 |
| --- | --- |
| `assets/` | README 中展示论文结果和对比图的图片资源。 |
| `README.md` | 官方说明文档，包含环境安装、数据格式、运行命令和超参数解释。 |
| `requirements.txt` | Python 依赖列表。 |
| `main.py` | PrE-Text 第一阶段入口：Private Evolution，生成 DP seed 文本。 |
| `variation.py` | 文本变异模块：对候选文本做 mask，然后用 `roberta-large` 逐步填充生成新变体。 |
| `similarity.py` | 相似度模块：调用 `all-MiniLM-L6-v2` 生成句向量，并做 lookahead embedding。 |
| `nn_histogram.py` | DP 直方图模块：用 FAISS 做最近邻检索，对计数加高斯噪声并阈值裁剪。 |
| `custom_datasets.py` | 两个轻量数据集封装：文本列表数据集、token 矩阵数据集。 |
| `llama_bootstrap.py` | 第二阶段入口：读取 seed 文本，用 `Llama-2-7b` 扩增成更大合成语料。 |
| `eval_distilgpt2.py` | 小模型评测入口：加载合成数据微调 `DistilGPT2`，在 eval 集上做 next-token prediction。 |
| `eval_llama2.py` | 大模型评测入口：加载合成数据，用 LoRA 微调 `Llama-2-7b` 并评测。 |

## 3. 运行时还需要的目录

这些目录**没有随仓库提交**，但代码运行时默认会使用它们：

| 路径 | 是否仓库自带 | 作用 |
| --- | --- | --- |
| `data/` | 否 | 放数据集输入文件。 |
| 自定义 `OUTPUT_DIR/` | 否 | 保存每次实验产物。 |
| 自定义 `MODEL_DIR/` | 否 | 保存 Hugging Face 模型下载缓存。 |

## 4. 每个模块的功能细化

### `main.py`

第一阶段主程序，负责 Private Evolution。

核心工作：

1. 加载 `roberta-large` 作为 masked LM。
2. 加载 `all-MiniLM-L6-v2` 作为句向量模型。
3. 读取 `data/<dataset_name>_train.json` 作为私有训练文本。
4. 读取 `data/initialization.json` 作为初始化种群，并过滤掉词数不超过 20 的样本。
5. 根据 Opacus 的 RDP accountant 计算本次配置对应的 `epsilon`。
6. 反复执行 11 轮：
   - 对当前候选文本做 lookahead；
   - 计算和私有文本最近邻关系；
   - 形成带噪声 DP histogram；
   - 采样存活父代；
   - 再用 `variation.py` 生成下一代文本。
7. 将每一轮结果输出到实验目录。

主要输出：

1. `private_embeds.np`
2. `generated_text_it0.json` 到 `generated_text_it10.json`
3. `surviving_text_it0.json` 到 `surviving_text_it10.json`

### `variation.py`

文本“变异器”。

核心工作：

1. 从父代 token 序列中随机选若干 token 进行 mask。
2. 用 `roberta-large` 对这些 mask 位置逐个采样填回。
3. 连续执行 `t_steps` 次 mask-fill，得到新的文本变体。

这个模块决定了 PrE-Text 中候选样本如何被扰动和扩展。

### `similarity.py`

相似度与 embedding 模块。

核心工作：

1. 使用 `SentenceTransformer` 生成句向量。
2. 对私有样本直接生成 embedding。
3. 对候选样本先做多次 lookahead 变异，再对这些未来候选求平均 embedding。

它为 `nn_histogram.py` 提供最近邻检索所需的向量表示。

### `nn_histogram.py`

差分隐私最近邻直方图模块。

核心工作：

1. 使用 FAISS 对私有样本 embedding 和候选样本 embedding 做最近邻搜索。
2. 统计每个候选样本被多少私有样本“投票”。
3. 对计数加入高斯噪声。
4. 做阈值裁剪，生成最终可用于重采样的 noised histogram。

这个模块是 PrE-Text 的 DP 关键步骤。

### `custom_datasets.py`

两个简单的数据集包装器：

1. `ListDataset`：包装字符串列表。
2. `MatrixDataset`：包装 tokenized 的 `input_ids` 和 `attention_mask`。

### `llama_bootstrap.py`

第二阶段扩增脚本。

核心工作：

1. 读取 11 轮 `surviving_text_it*.json` 中的 seed 文本。
2. 每次随机抽 3 条 seed 文本拼成提示词。
3. 用 `meta-llama/Llama-2-7b-hf` 生成更多文本。
4. 保存到 `llama7b_text_syn.json`。

当前脚本默认生成 **50000** 条样本。README 提到论文实验扩增到 200 万条，但当前仓库脚本里写死的是 50000，需要自行改脚本常量。

### `eval_distilgpt2.py`

小模型评测脚本。

核心工作：

1. 读取 `llama7b_text_syn.json` 作为训练语料。
2. 读取 `data/<dataset_name>_eval.json` 作为测试集。
3. 加载 `distilgpt2`。
4. 加载根目录下的 `c4_checkpoint.pth` 作为 warm start。
5. 微调并输出 cross-entropy、Top-k accuracy、checkpoint。

注意：

1. 这个脚本依赖一个**额外准备**的 `c4_checkpoint.pth`。
2. README 说明该 checkpoint 是“先在 C4 子集上训练过的 DistilGPT2”。

### `eval_llama2.py`

大模型评测脚本。

核心工作：

1. 读取 `llama7b_text_syn.json` 作为训练语料。
2. 读取 `data/<dataset_name>_eval.json` 作为测试集。
3. 加载 `meta-llama/Llama-2-7b-hf`。
4. 用 LoRA 方式微调。
5. 输出 cross-entropy、Top-k accuracy、checkpoint。

## 5. 实验用到的数据集是什么

基于当前仓库代码和 `README.md`，可以明确得到下面几点：

### 5.1 仓库**没有自带真实实验数据**

仓库里没有 `data/` 目录，也没有任何现成数据文件。你需要自己准备数据。

### 5.2 PrE-Text 主实验使用的是“自定义私有联邦文本数据”

代码和 README 要求你至少准备下面 3 个文件：

1. `data/initialization.json`
2. `data/<dataset_name>_train.json`
3. `data/<dataset_name>_eval.json`

含义分别是：

| 文件 | 作用 | 格式 |
| --- | --- | --- |
| `data/initialization.json` | 初始化种群 `S1`，至少 10000 条文本 | JSON 数组，如 `["text1", "text2"]` |
| `data/<dataset_name>_train.json` | 所有联邦客户端私有样本汇总后的训练集 | JSON 数组 |
| `data/<dataset_name>_eval.json` | 评测集 | JSON 对象，形如 `{"1": ["text1", "text2"]}` |

README 还强调：

1. 每个客户端样本上限建议不超过 16。
2. `sensitivity` 就是每个客户端最大样本数，也是 DP 敏感度。
3. 如果单客户端样本数太大，需要先做子采样再汇总。

### 5.3 小模型评测还额外依赖 C4

`eval_distilgpt2.py` 不直接下载 C4 数据，但 README 明确要求你先按论文流程准备一个 **在 C4 子集上预训练/微调过的 DistilGPT2 checkpoint**，并放到项目根目录：

`./c4_checkpoint.pth`

所以从实验视角看：

1. **PrE-Text 主流程数据**：你自己的私有联邦文本数据。
2. **小模型评测 warm start 数据**：C4 数据集的一个子集。

### 5.4 README 没有给出论文原始私有数据的公开下载方式

也就是说，单靠当前仓库和 README，无法直接获得论文中的真实私有联邦数据，只能按它要求准备同格式数据复现实验流程。

## 6. 环境准备

README 给出的基础安装方式如下：

```bash
git clone https://github.com/houcharlie/PrE-Text.git
cd PrE-Text
conda create -n pretext python=3.10 -y
conda activate pretext
pip install -r requirements.txt
```

建议额外做两件事：

1. 先运行 `accelerate config` 配好 GPU/多卡。
2. 确认你有权限下载 `meta-llama/Llama-2-7b-hf`，因为这通常要求先在 Hugging Face 接受模型许可。

## 7. 数据准备格式示例

### `data/initialization.json`

```json
[
  "This is a public or seed text sample used to initialize the population.",
  "Another sufficiently long text sample used as part of S1."
]
```

### `data/<dataset_name>_train.json`

```json
[
  "private federated sample 1",
  "private federated sample 2"
]
```

### `data/<dataset_name>_eval.json`

```json
{
  "1": [
    "evaluation sample 1",
    "evaluation sample 2"
  ]
}
```

## 8. 如何启动实验

下面用 PowerShell 写法给出最直接的启动方式，因为你当前环境就是 PowerShell。

### 8.1 设置公共环境变量

```powershell
$env:OUTPUT_DIR = "D:\results\pretext"
$env:MODEL_DIR = "D:\models\hf_cache"
$env:DATASET_NAME = "my_dataset"
$env:MAX_SAMPLES = "16"
$env:NOISE = "11.3"
$env:DELTA = "3e-6"
```

说明：

1. `DATASET_NAME` 对应 `data/<dataset_name>_train.json` 和 `data/<dataset_name>_eval.json` 的前缀。
2. README 中给出的典型设置是：
   - `DELTA = 3e-6`
   - `NOISE = 11.3` 对应 `epsilon = 1.29`
   - `NOISE = 2.31` 对应 `epsilon = 7.58`

### 8.2 第一阶段：生成 DP seed 文本

```powershell
$env:TOKENIZERS_PARALLELISM = "false"
accelerate launch main.py `
  -outputdir $env:OUTPUT_DIR `
  -cachedir $env:MODEL_DIR `
  -datadir $env:DATASET_NAME `
  -sensitivity $env:MAX_SAMPLES `
  -sigma $env:NOISE `
  -delta $env:DELTA
```

作用：

1. 下载并缓存 `roberta-large` 和 `all-MiniLM-L6-v2`。
2. 运行 11 轮 Private Evolution。
3. 在输出目录下生成 seed 文本和中间文件。

### 8.3 第二阶段：扩增合成数据

```powershell
python .\llama_bootstrap.py `
  -outputdir $env:OUTPUT_DIR `
  -cachedir $env:MODEL_DIR `
  -datadir $env:DATASET_NAME `
  -sensitivity $env:MAX_SAMPLES `
  -sigma $env:NOISE `
  -delta $env:DELTA
```

作用：

1. 读取第一阶段的 `surviving_text_it*.json`。
2. 调用 `Llama-2-7b-hf` 扩增出 `llama7b_text_syn.json`。

### 8.4 小模型评测：DistilGPT2

前提：根目录必须存在 `c4_checkpoint.pth`。

```powershell
accelerate launch eval_distilgpt2.py `
  -outputdir $env:OUTPUT_DIR `
  -cachedir $env:MODEL_DIR `
  -datadir $env:DATASET_NAME `
  -sensitivity $env:MAX_SAMPLES `
  -sigma $env:NOISE `
  -delta $env:DELTA
```

### 8.5 大模型评测：LLaMA-2-7b + LoRA

```powershell
accelerate launch eval_llama2.py `
  -outputdir $env:OUTPUT_DIR `
  -cachedir $env:MODEL_DIR `
  -datadir $env:DATASET_NAME `
  -sensitivity $env:MAX_SAMPLES `
  -sigma $env:NOISE `
  -delta $env:DELTA
```

## 9. 输出目录会生成什么

实验输出目录格式由脚本自动拼接，形如：

```text
<OUTPUT_DIR>/<dataset>_<mask>_<lookahead>_<nsyn>_<t_steps>_<H_multiplier>_<sensitivity>_<sigma>_<delta>_<trial>/
```

例如第一阶段和第二阶段共用同一个实验目录。

常见产物如下：

| 文件/目录 | 来源脚本 | 作用 |
| --- | --- | --- |
| `private_embeds.np` | `main.py` | 私有训练文本的 embedding 缓存。 |
| `generated_text_it*.json` | `main.py` | 每一轮生成的候选文本。 |
| `surviving_text_it*.json` | `main.py` | 每一轮筛选后留下的 seed 文本。 |
| `llama7b_text_syn.json` | `llama_bootstrap.py` | 第二阶段扩增得到的大规模合成语料。 |
| `log_models_and_accuracies/` | `eval_distilgpt2.py` | DistilGPT2 训练日志、每轮指标、checkpoint。 |
| `llama2_models_and_accuracies/` | `eval_llama2.py` | LLaMA2-LoRA 训练日志、每轮指标、checkpoint。 |

## 10. 硬件与运行注意事项

README 提供了作者的参考硬件：

1. `main.py` 和 `llama_bootstrap.py`：单张 V100 32GB 或 A40 48GB。
2. `eval_distilgpt2.py`：4 张 A40 48GB。
3. `eval_llama2.py`：4 张 A40 48GB。

额外注意：

1. `llama_bootstrap.py` 和 `eval_llama2.py` 都依赖 `meta-llama/Llama-2-7b-hf`。
2. `eval_distilgpt2.py` 依赖你自行准备的 `c4_checkpoint.pth`。
3. `main.py` 会过滤 `initialization.json` 中词数不超过 20 的样本，因此初始化集不能只准备很短文本。

## 11. 最简复现顺序

如果只想把流程先跑通，顺序就是：

1. 安装依赖。
2. 创建 `data/initialization.json`、`data/<dataset_name>_train.json`、`data/<dataset_name>_eval.json`。
3. 设置 `OUTPUT_DIR`、`MODEL_DIR`、`DATASET_NAME`、`MAX_SAMPLES`、`NOISE`、`DELTA`。
4. 运行 `accelerate launch main.py`。
5. 运行 `python llama_bootstrap.py`。
6. 如果要做评测，再准备 `c4_checkpoint.pth` 并运行 `eval_distilgpt2.py`。
7. 如果要做大模型评测，再运行 `eval_llama2.py`。
> Note
> 当前仓库已经平台化重构为 `pretext_platform/`。
> 本文件主要保留旧版脚本化实现的阅读说明；实际运行方式、目录结构和配置入口请优先参考根目录 `README.md`。
