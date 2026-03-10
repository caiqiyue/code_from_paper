# DataInf 项目结构与实验使用说明

## 1. 项目定位

这个仓库对应论文 **DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models**。  
它的核心目标是：

- 用 LoRA 微调模型，而不是全量微调。
- 提取训练样本与验证样本的梯度。
- 用 DataInf 近似公式估计 influence function。
- 将 influence 分数用于两类任务：
  - 错标数据检测
  - 关键训练样本识别

当前仓库不是一个标准的可安装 Python 包工程。`src/` 中是**扁平化脚本模块**，没有 `__init__.py`，因此更适合按脚本/Notebook 方式运行。

---

## 2. 顶层目录结构

```text
DataInf/
├── datasets/                         # 本地 HuggingFace 数据集目录
├── figures/                          # README/论文配图
├── notebooks/                        # 交互式实验入口
├── src/                              # 核心源码
├── DATAINF_FEATURE_EXTRACTION_PLAN.md
├── quick_start.md
├── README.md
├── requirements.txt
└── DATAINF_PROJECT_GUIDE.md          # 当前文档
```

### 2.1 各目录职责

| 路径 | 作用 |
| --- | --- |
| `datasets/` | 保存 `Dataset.save_to_disk()` 生成的本地 HuggingFace 数据集，例如 `grammars_train.hf`、`math_with_reason_test.hf`。 |
| `figures/` | 项目展示图片，目前主要用于 README。 |
| `notebooks/` | 论文实验的可视化/交互式入口。适合快速理解流程。 |
| `src/` | 实验主逻辑，包含配置、数据加载、LoRA 微调、梯度提取、Influence 计算和扩散模型训练脚本。 |
| `README.md` | 官方简介与最简使用方式。 |
| `requirements.txt` | 基础依赖列表，但**并不覆盖所有脚本需要的扩展依赖**。 |

### 2.2 `notebooks/` 下各文件作用

| Notebook | 作用 |
| --- | --- |
| `notebooks/Mislabeled_Data_Detection-RoBERTa-MRPC.ipynb` | 用 RoBERTa-large + LoRA 在 MRPC 上做错标样本检测。 |
| `notebooks/Influential_Data_Identification-Llama2-Sentence.ipynb` | 用 Llama2 + LoRA 在句子变换任务上做关键样本识别。 |
| `notebooks/Influential_Data_Identification-Llama2-Math-Without-Reason.ipynb` | 用 Llama2 + LoRA 在无推理数学题上做关键样本识别。 |
| `notebooks/Influential_Data_Identification-Llama2-Math-Reason.ipynb` | 用 Llama2 + LoRA 在带推理数学题上做关键样本识别。 |
| `notebooks/Influential_Data_Identification-Stable_Diffusion-Style_Transfer.ipynb` | 用 Stable Diffusion + LoRA 做风格迁移场景下的影响样本分析。 |

---

## 3. `src/` 源码结构说明

### 3.1 模块总览

| 模块 | 主要职责 |
| --- | --- |
| `src/configs.py` | 生成 GLUE 文本分类实验配置，定义不同数据集和不同 LoRA rank 的配置入口。 |
| `src/dataloader.py` | 下载/裁剪 GLUE 数据集，注入标签噪声，并构建 `DataLoader`。 |
| `src/generate_sentence-math_datasets.py` | 构造句子变换与数学题本地数据集，并保存到 `datasets/`。 |
| `src/influence.py` | 实现 DataInf、Identity、LiSSA、Accurate 等 influence/HVP 计算逻辑。 |
| `src/launcher.py` | 文本分类 CLI 入口。 |
| `src/lora_model.py` | 定义分类模型和生成模型的 LoRA 加载、训练与梯度提取流程。 |
| `src/run_experiment.py` | 串联数据加载、LoRA 微调、梯度提取和 influence 计算。 |
| `src/sft_trainer.py` | 基于 TRL `SFTTrainer` 的 LLaMA/LM LoRA SFT 训练脚本。 |
| `src/simulator.py` | 设置随机种子并调用实验主流程。 |
| `src/train_text_to_image_lora.py` | 基于 Diffusers 的 Stable Diffusion LoRA 训练脚本。 |

### 3.2 逐模块详细说明

#### `src/configs.py`

- 用途：为文本分类实验生成配置字典。
- 核心函数：
  - `_setup_env()`：记录当前主机名与用户名。
  - `generate_config(...)`：生成实验级配置和多次运行配置列表。
  - `config_qnli1()` ~ `config_qnli5()`：QNLI，不同 LoRA rank。
  - `config_qqp1()` ~ `config_qqp5()`：QQP，不同 LoRA rank。
  - `config_sst21()` ~ `config_sst25()`：SST-2，不同 LoRA rank。
  - `config_mrpc1()` ~ `config_mrpc5()`：MRPC，不同 LoRA rank。
  - `config_wnli1()` ~ `config_wnli5()`：WNLI，不同 LoRA rank。

配置规律：

| 配置后缀 | LoRA rank |
| --- | --- |
| `1` | 1 |
| `2` | 2 |
| `3` | 4 |
| `4` | 8 |
| `5` | 16 |

#### `src/dataloader.py`

- 用途：构建带噪声的 GLUE 数据集与 PyTorch dataloader。
- 主要逻辑：
  - `load_dataset("glue", task)` 下载指定 GLUE 子任务。
  - 若训练集过大，则裁剪为训练最多 4500 条、验证最多 500 条。
  - 按 `noise_ratio` 随机翻转部分训练样本标签。
  - 用 `AutoTokenizer` 做 tokenization。
  - 返回训练/验证 dataloader、噪声样本索引、tokenized dataset、collate 函数。

#### `src/generate_sentence-math_datasets.py`

- 用途：生成本地文本生成实验所需数据集。
- 生成两类数据：
  - 句子变换数据集 `grammars_*`
  - 数学题数据集 `math_without_reason_*` 和 `math_with_reason_*`
- 数据特点：
  - 句子变换数据集：10 种 transformation，每种 transformation 基于固定短句集合构造训练/测试样本。
  - 数学题数据集：10 种小学应用题模板，每种模板生成 100 个样本，默认 90/10 划分 train/test。

#### `src/influence.py`

- 用途：计算 influence function。
- 包含两个核心类：
  - `IFEngine`
    - 面向分类任务。
    - 先对验证集梯度求均值，再计算 HVP，再计算 influence。
  - `IFEngineGeneration`
    - 面向生成任务。
    - 为每一个验证样本分别计算 influence，最终得到 `val_id x train_id` 矩阵。
- 支持的方法：
  - `identity`
  - `proposed`（论文里的 DataInf 闭式近似）
  - `LiSSA`
  - `accurate`（显式矩阵分解，更耗时）

#### `src/launcher.py`

- 用途：GLUE 文本分类实验 CLI 入口。
- 主要行为：
  - 根据 `exp_id` 找到 `configs.py` 中对应配置函数。
  - 取出指定 `run_id` 的配置。
  - 把配置保存为 `config.pickle`。
  - 调用 `simulator.main(config)` 启动实验。

#### `src/lora_model.py`

- 用途：LoRA 模型构建、训练和梯度提取。
- 包含两个核心类：
  - `LORAEngine`
    - 面向 RoBERTa/GLUE 分类任务。
    - 使用 `AutoModelForSequenceClassification + PEFT LoRA`。
    - 训练完成后逐样本提取 LoRA 参数梯度。
  - `LORAEngineGeneration`
    - 面向 Llama2 等生成任务。
    - 加载 base model + 已训练好的 LoRA adapter。
    - 逐样本计算 causal LM loss，并提取 LoRA 参数梯度。

#### `src/run_experiment.py`

- 用途：分类实验总控。
- 流程：
  - 构建 dataloader
  - 微调 LoRA 分类模型
  - 提取 train/validation 梯度
  - 计算 influence
  - 保存结果

#### `src/sft_trainer.py`

- 用途：Llama2 等生成模型的 SFT 训练脚本。
- 基于：
  - `transformers`
  - `peft`
  - `trl.SFTTrainer`
- 支持：
  - 从 HuggingFace Hub 加载数据集
  - 从本地 `Dataset.load_from_disk()` 加载数据集
  - 8bit/4bit 模型加载
  - PEFT/LoRA 微调

#### `src/simulator.py`

- 用途：设置随机种子并调用 `run_experiment_core(config)`。
- 作用比较薄，主要是保证多次运行时的种子一致性。

#### `src/train_text_to_image_lora.py`

- 用途：Stable Diffusion 文生图 LoRA 训练。
- 来源：基于 HuggingFace Diffusers 官方示例改写。
- 主要功能：
  - 加载 `scheduler / tokenizer / text_encoder / vae / unet`
  - 冻结底座模型
  - 向 UNet 注入 LoRA adapter
  - 构建图像+文本 dataloader
  - 执行扩散训练
  - 保存 checkpoint 与最终 LoRA 权重
  - 可选推送到 HuggingFace Hub

---

## 4. 当前项目依赖的第三方 Python 包及职责

### 4.1 基础依赖

| 包 | 在本项目中的作用 |
| --- | --- |
| `argh` | 给 `launcher.py`、`simulator.py` 提供简单 CLI。 |
| `datasets` | 下载 GLUE、加载 HF Hub 数据集、保存/读取本地 `.hf` 数据集。 |
| `evaluate` | 计算 GLUE 指标。 |
| `numpy` | 随机采样、数值处理、数据生成。 |
| `pandas` | 组装中间表格，并转成 HuggingFace Dataset。 |
| `peft` | LoRA/PEFT 配置、adapter 加载与权重提取。 |
| `torch` | 训练、梯度计算、DataLoader、优化器、GPU 张量计算。 |
| `tqdm` | 进度条显示。 |
| `transformers` | RoBERTa、Llama、CLIP 等模型与 tokenizer。 |

### 4.2 扩展依赖

这些包在源码里被直接使用，但 `requirements.txt` 中没有完整列出：

| 包 | 作用 | 主要使用位置 |
| --- | --- | --- |
| `accelerate` | 多卡/混合精度/分布式训练封装 | `src/sft_trainer.py`、`src/train_text_to_image_lora.py` |
| `trl` | SFTTrainer，用于 LLM 的监督微调 | `src/sft_trainer.py` |
| `diffusers` | Stable Diffusion 模型与训练工具 | `src/train_text_to_image_lora.py` |
| `torchvision` | 图像变换与预处理 | `src/train_text_to_image_lora.py` |
| `huggingface_hub` | 创建 repo、上传 LoRA 权重到 HF Hub | `src/train_text_to_image_lora.py` |
| `packaging` | 版本比较 | `src/train_text_to_image_lora.py` |
| `sentencepiece` | LLaMA tokenizer 常见必需依赖 | `src/sft_trainer.py`、`src/lora_model.py` |

### 4.3 可选依赖

| 包 | 是否必须 | 用途 |
| --- | --- | --- |
| `bitsandbytes` | 可选但常用 | 8bit/4bit 加载与 8-bit Adam。 |
| `xformers` | 可选 | 提供更省显存的 attention。 |
| `wandb` | 可选 | 训练日志上报。 |
| `jupyter` / `notebook` | 如果跑 notebook 则需要 | 打开 `notebooks/` 中的实验。 |

---

## 5. 环境准备与安装方式

## 5.1 创建虚拟环境

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 5.2 安装第三方包

### 方案 A：先装基础依赖，再补充扩展依赖

```powershell
pip install -r requirements.txt
pip install accelerate trl diffusers torchvision huggingface_hub packaging sentencepiece jupyter notebook
```

### 方案 B：如果你需要 8bit/4bit、xformers 或日志平台

```powershell
pip install bitsandbytes xformers wandb
```

### 5.3 安装时需要注意的事项

1. `requirements.txt` 里同时出现了两个 `numpy` 版本行，因此它更像“最小参考依赖”，不是完全可复现实验环境清单。
2. `src/sft_trainer.py` 和 `src/train_text_to_image_lora.py` 还依赖 `accelerate`、`trl`、`diffusers`、`torchvision`、`huggingface_hub` 等扩展包。
3. Llama2 相关实验通常需要：
   - 有 CUDA 的 Linux 环境更稳妥
   - 已登录 HuggingFace
   - 已获得 `meta-llama/Llama-2-13b-chat-hf` 访问权限
4. Stable Diffusion / Llama2 / bitsandbytes 在 Windows 下可能遇到兼容性问题，建议优先 Linux + CUDA。

### 5.4 HuggingFace 登录

如果要访问 gated 模型或上传模型：

```powershell
huggingface-cli login
```

---

## 6. 数据集与模型资源说明

### 6.1 仓库内已有本地数据集

| 数据集目录 | 含义 |
| --- | --- |
| `datasets/grammars_train.hf` | 句子变换训练集 |
| `datasets/grammars_test.hf` | 句子变换测试集 |
| `datasets/math_without_reason_train.hf` | 无推理数学题训练集 |
| `datasets/math_without_reason_test.hf` | 无推理数学题测试集 |
| `datasets/math_with_reason_train.hf` | 带推理数学题训练集 |
| `datasets/math_with_reason_test.hf` | 带推理数学题测试集 |

### 6.2 主要底座模型

| 模型 | 用途 |
| --- | --- |
| `roberta-large` | 文本分类错标检测实验 |
| `meta-llama/Llama-2-13b-chat-hf` | 句子变换 / 数学题生成实验 |
| `runwayml/stable-diffusion-v1-5` | 文生图 LoRA 实验 |

### 6.3 Stable Diffusion 示例数据集

Notebook 中使用的风格迁移数据集为：

- `kewu93/three_styles_prompted_250_512x512`

---

## 7. 实验启动方式

## 7.1 文本分类：错标数据检测 CLI

命令示例：

```powershell
python src/launcher.py run --exp_id=mrpc4 --run-id=0 --runpath=.
```

含义：

- `exp_id=mrpc4`：MRPC 数据集，LoRA rank = 8。
- `run-id=0`：取该配置的第 0 个 run。
- `runpath=.`：在当前项目根目录执行并保存结果。

执行后流程：

1. 读取 `config_mrpc4()`
2. 下载/采样 GLUE MRPC
3. 注入标签噪声
4. 微调 RoBERTa + LoRA
5. 提取梯度
6. 计算 influence
7. 保存 `config.pickle` 和 `results_0.pkl`

## 7.2 重新生成本地句子/数学数据集

```powershell
python src/generate_sentence-math_datasets.py
```

这个脚本会生成：

- `datasets/grammars_train.hf`
- `datasets/grammars_test.hf`
- `datasets/math_without_reason_train.hf`
- `datasets/math_without_reason_test.hf`
- `datasets/math_with_reason_train.hf`
- `datasets/math_with_reason_test.hf`

## 7.3 Llama2 SFT 训练

以带推理数学题为例：

```powershell
python src/sft_trainer.py `
  --model_name meta-llama/Llama-2-13b-chat-hf `
  --dataset_name .\datasets\math_with_reason_train.hf `
  --output_dir .\models\math_with_reason_13bf `
  --dataset_text_field text `
  --load_in_8bit `
  --use_peft
```

其它两个本地数据集只需要替换：

- `dataset_name`
- `output_dir`

对应关系：

| 训练集 | 建议输出目录 |
| --- | --- |
| `datasets/grammars_train.hf` | `models/grammars_13bf` |
| `datasets/math_without_reason_train.hf` | `models/math_without_reason_13bf` |
| `datasets/math_with_reason_train.hf` | `models/math_with_reason_13bf` |

训练完成后，再打开对应 notebook 做 influence 分析。

## 7.4 Stable Diffusion LoRA 训练

```powershell
accelerate launch src/train_text_to_image_lora.py `
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 `
  --dataset_name kewu93/three_styles_prompted_250_512x512 `
  --caption_column text `
  --resolution 512 `
  --train_batch_size 1 `
  --gradient_accumulation_steps 4 `
  --learning_rate 1e-4 `
  --num_train_epochs 100 `
  --checkpointing_steps 500 `
  --output_dir .\models\three_styles_lora `
  --validation_prompt "cartoon style cat"
```

训练完成后，再进入：

- `notebooks/Influential_Data_Identification-Stable_Diffusion-Style_Transfer.ipynb`

## 7.5 直接跑 Notebook

如果只是先理解流程，推荐先启动 Jupyter：

```powershell
jupyter notebook
```

建议阅读顺序：

1. `README.md`
2. `notebooks/Mislabeled_Data_Detection-RoBERTa-MRPC.ipynb`
3. `src/influence.py`
4. `src/lora_model.py`
5. Llama2 / Stable Diffusion 相关 notebook

---

## 8. 核心配置参数说明

## 8.1 `configs.py -> generate_config(...)` 参数

| 参数 | 含义 |
| --- | --- |
| `expno_name` | 实验名，通常对应 `mrpc4`、`qnli2` 这种预设 id。 |
| `task` | GLUE 子任务名，例如 `mrpc`、`qnli`、`qqp`、`sst2`、`wnli`。 |
| `model` | 模型族标识，当前分类实验里实际使用的是 `roberta`。 |
| `low_rank` | LoRA rank。 |
| `n_runs` | 该实验预设下总共有多少个 `run_id`。 |

## 8.2 文本分类 run config 中的字段含义

| 字段 | 含义 |
| --- | --- |
| `task` | 使用哪个 GLUE 子任务。 |
| `model` | 模型族名，主要用于标识。 |
| `noise_ratio` | 训练集标签翻转比例，默认 0.2。 |
| `device` | 训练设备，默认 `cuda`。 |
| `lr` | 学习率。 |
| `model_name_or_path` | 实际加载的 HuggingFace 模型名。 |
| `batch_size` | dataloader batch size。 |
| `num_epochs` | 训练轮数。 |
| `target_modules` | LoRA 注入的目标模块名。 |
| `N_repeat` | 在一次 run 中重复构造 noisy 数据并重复实验的次数。 |
| `low_rank` | LoRA rank。 |
| `compute_accurate` | 是否尝试精确 HVP；当 rank > 4 时主流程会自动关闭。 |
| `run_id` | 当前 run 编号，也用作随机种子。 |
| `runpath` | 输出目录和执行目录。 |

## 8.3 `src/launcher.py run` 参数

| 参数 | 含义 |
| --- | --- |
| `exp_id` | 预设实验 id，例如 `mrpc4`、`qnli1`。 |
| `run_id` | 选择配置列表中的第几个 run。 |
| `runpath` | 切换到哪个目录执行实验。 |

## 8.4 `src/sft_trainer.py` 参数

| 参数 | 含义 |
| --- | --- |
| `model_name` | 底座语言模型名或本地路径。 |
| `dataset_name` | HF Hub 数据集名，或本地 `load_from_disk()` 数据集路径。 |
| `dataset_text_field` | 训练时用于监督学习的文本字段名，默认 `text`。 |
| `log_with` | 日志平台，例如 `wandb`。 |
| `learning_rate` | 学习率。 |
| `batch_size` | 每设备 batch size。 |
| `seq_length` | 最大输入序列长度。 |
| `gradient_accumulation_steps` | 梯度累积步数。 |
| `load_in_8bit` | 是否以 8bit 方式加载模型。 |
| `load_in_4bit` | 是否以 4bit 方式加载模型。 |
| `use_peft` | 是否启用 LoRA/PEFT。 |
| `trust_remote_code` | 是否允许加载远程自定义模型代码。 |
| `output_dir` | 输出目录。 |
| `peft_lora_r` | LoRA rank。 |
| `peft_lora_alpha` | LoRA alpha。 |
| `logging_steps` | 每隔多少步打印/记录一次日志。 |
| `use_auth_token` | 是否使用 HF 登录 token 访问模型。 |
| `num_train_epochs` | 训练 epoch 数。 |
| `max_steps` | 最大训练步数，`-1` 表示主要按 epoch 控制。 |
| `save_steps` | checkpoint 保存间隔。 |
| `save_total_limit` | 最多保留多少 checkpoint。 |
| `push_to_hub` | 是否上传到 HuggingFace Hub。 |
| `hub_model_id` | 上传到 HF Hub 时的目标仓库名。 |

## 8.5 `src/train_text_to_image_lora.py` 参数

### 模型与数据参数

| 参数 | 含义 |
| --- | --- |
| `pretrained_model_name_or_path` | 必填。Stable Diffusion 底座模型名或路径。 |
| `revision` | 指定模型 revision。 |
| `variant` | 指定模型变体，如 `fp16`。 |
| `dataset_name` | HF Hub 数据集名。 |
| `dataset_config_name` | 数据集配置名。 |
| `train_data_dir` | 本地图像数据目录，和 `dataset_name` 二选一。 |
| `image_column` | 数据集中图像列名。 |
| `caption_column` | 数据集中 caption 列名。 |
| `cache_dir` | 模型和数据缓存目录。 |
| `max_train_samples` | 调试用，限制训练样本数。 |

### 验证与输出参数

| 参数 | 含义 |
| --- | --- |
| `validation_prompt` | 验证阶段生成图像时使用的文本提示。 |
| `num_validation_images` | 每次验证生成多少张图。 |
| `validation_epochs` | 每多少个 epoch 跑一次验证。 |
| `output_dir` | 输出目录。 |
| `logging_dir` | 日志目录名，实际位于 `output_dir/logging_dir`。 |
| `checkpointing_steps` | 每多少个优化步保存一次 checkpoint。 |
| `checkpoints_total_limit` | 最多保留多少个 checkpoint。 |
| `resume_from_checkpoint` | 从哪个 checkpoint 恢复，或使用 `latest`。 |
| `push_to_hub` | 是否把结果同步到 HuggingFace Hub。 |
| `hub_token` | 上传模型时使用的 token。 |
| `hub_model_id` | 上传目标 repo 名。 |

### 图像预处理参数

| 参数 | 含义 |
| --- | --- |
| `resolution` | 输入图像分辨率。 |
| `center_crop` | 是否做中心裁剪。 |
| `random_flip` | 是否随机水平翻转。 |

### 训练参数

| 参数 | 含义 |
| --- | --- |
| `seed` | 随机种子。 |
| `train_batch_size` | 每设备 batch size。 |
| `num_train_epochs` | 训练 epoch 数。 |
| `max_train_steps` | 最大训练步数，设置后会覆盖 `num_train_epochs`。 |
| `gradient_accumulation_steps` | 梯度累积步数。 |
| `gradient_checkpointing` | 是否启用梯度检查点。 |
| `learning_rate` | 学习率。 |
| `scale_lr` | 是否按设备数和 batch 自动缩放学习率。 |
| `lr_scheduler` | 学习率调度器类型。 |
| `lr_warmup_steps` | warmup 步数。 |
| `snr_gamma` | SNR reweighting 参数。 |
| `use_8bit_adam` | 是否使用 8-bit Adam。 |
| `allow_tf32` | 是否允许 TF32。 |
| `dataloader_num_workers` | dataloader worker 数。 |
| `adam_beta1` | Adam beta1。 |
| `adam_beta2` | Adam beta2。 |
| `adam_weight_decay` | Adam 权重衰减。 |
| `adam_epsilon` | Adam epsilon。 |
| `max_grad_norm` | 梯度裁剪上限。 |
| `prediction_type` | 噪声预测目标类型，如 `epsilon` 或 `v_prediction`。 |
| `mixed_precision` | 混合精度模式：`no`、`fp16`、`bf16`。 |
| `report_to` | 日志后端，如 `tensorboard`、`wandb`。 |
| `local_rank` | 分布式训练 local rank。 |
| `enable_xformers_memory_efficient_attention` | 是否启用 xformers memory-efficient attention。 |
| `noise_offset` | 额外噪声偏移量。 |
| `rank` | Stable Diffusion LoRA rank。 |

---

## 9. 结果输出位置

| 产物 | 默认位置 |
| --- | --- |
| 文本分类实验配置快照 | `config.pickle` |
| 文本分类 influence 结果 | `results_<run_id>.pkl` |
| 本地句子/数学数据集 | `datasets/*.hf` |
| Llama2 LoRA adapter | `models/<dataset_name>_13bf` |
| Stable Diffusion LoRA 权重 | `models/three_styles_lora` 或你自定义的 `output_dir` |
| Stable Diffusion checkpoint | `output_dir/checkpoint-*` |

---

## 10. 推荐理解顺序

如果你想尽快看懂整个实验平台，建议顺序如下：

1. 先读 `README.md`
2. 再看 `src/configs.py`、`src/dataloader.py`
3. 再看 `src/lora_model.py`
4. 再看 `src/influence.py`
5. 再跑 `notebooks/Mislabeled_Data_Detection-RoBERTa-MRPC.ipynb`
6. 然后再看 Llama2 和 Stable Diffusion 部分

这样最容易把下面这条主线串起来：

`数据准备 -> LoRA 微调 -> 逐样本梯度 -> HVP 近似 -> Influence 计算 -> 分析结果`

---

## 11. 当前仓库的几个实现注意点

1. `src/` 是脚本目录，不是标准 package；导入方式依赖“从项目根目录启动脚本”。
2. `requirements.txt` 不是完整依赖清单，尤其缺少 LLM 和 Diffusion 相关扩展依赖。
3. `Llama2` 与 `Stable Diffusion` 实验对 GPU、CUDA 和 HF 权限要求明显更高。
4. `run_experiment.py` 中当 `low_rank > 4` 时，会自动关闭 `compute_accurate`，避免精确 HVP 的开销过大。
5. 训练生成模型前，通常需要先准备对应的 `models/<dataset>_13bf` LoRA adapter 目录，否则后续 influence notebook 无法直接加载。
