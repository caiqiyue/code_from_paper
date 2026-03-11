# DataInf 论文实验整理与当前项目数据集观察

## 1. 说明

本文档依据以下材料整理：

- `DataInf.pdf`
- `DATAINF_PROJECT_GUIDE.md`
- `DATAINF_FEATURE_EXTRACTION_PLAN.md`
- `README.md`
- `src/configs.py`
- `src/dataloader.py`
- `src/lora_model.py`
- `src/sft_trainer.py`
- `src/train_text_to_image_lora.py`
- `src/generate_sentence-math_datasets.py`
- `notebooks/*.ipynb`

整理目标：

1. 汇总 DataInf 论文实验中实际使用的数据集、模型和参数设置。
2. 观察当前项目根目录下已经存在的本地数据集。
3. 明确论文描述与仓库实现之间的几个关键差异。

---

## 2. 论文实验用了哪些数据集、模型、参数

### 2.1 实验总览

| 实验类型 | 论文任务 | 数据集 | 基础模型 | LoRA / 训练对象 | 主要用途 |
| --- | --- | --- | --- | --- | --- |
| 错标数据检测 | Binary text classification | GLUE 二分类子任务：`MRPC`、`QNLI`、`QQP`、`SST2`、`WNLI` | `roberta-large` | LoRA 分类模型 | 检测被人为翻转标签的训练样本 |
| 关键样本识别 | Text generation | Synthetic Sentence Transformation | `meta-llama/Llama-2-13b-chat-hf` | LoRA Causal LM | 找出对验证样本最有影响的训练样本 |
| 关键样本识别 | Text generation | Synthetic Math Problem (without reasoning) | `meta-llama/Llama-2-13b-chat-hf` | LoRA Causal LM | 同上 |
| 关键样本识别 | Text generation | Synthetic Math Problem (with reasoning) | `meta-llama/Llama-2-13b-chat-hf` | LoRA Causal LM | 同上 |
| 关键样本识别 | Text-to-image generation | `kewu93/three_styles_prompted_250_512x512` | `runwayml/stable-diffusion-v1-5` | LoRA UNet | 找出对目标风格生成最有影响的训练图片/提示词样本 |

### 2.2 各实验的关键参数

| 实验类型 | 数据集 / 任务 | 论文或仓库中能确认的参数 |
| --- | --- | --- |
| 错标数据检测 | GLUE 二分类任务 | 训练标签噪声比 `0.2`；学习率 `3e-4`；batch size `32`；epoch `10`；LoRA rank 取 `1 / 2 / 4 / 8 / 16`；平均 `10` 次随机运行；当数据集过大时，代码会截断为最多 `4500` 条 train、`500` 条 validation |
| 错标数据检测 | GLUE 二分类任务 | Notebook 演示任务是 `MRPC`，示例 LoRA rank 为 `8` |
| 文本生成关键样本识别 | Sentence / Math 三类生成任务 | 基础模型 `Llama-2-13b-chat-hf`；学习率 `3e-4`；batch size `64`；gradient accumulation steps `16`；sequence length `128`；epoch `10`；LoRA rank `8`；LoRA alpha `32`；LoRA dropout `0.05`；`--load_in_8bit`；LoRA target modules 为 `q_proj`、`v_proj` |
| 图像生成关键样本识别 | Style transfer | 基础模型 `Stable Diffusion v1-5`；学习率 `1e-4`；LoRA rank `2`；LoRA trainable modules 为 `to_k`、`to_q`、`to_v`、`to_out.0`；论文附录写的是训练 `3000` gradient steps |

### 2.3 影响函数 / HVP 估计方法

| 方法 | 论文/代码名称 | 作用 | 当前仓库能确认的默认参数 |
| --- | --- | --- | --- |
| DataInf | `proposed` | 论文主方法，闭式近似 HVP | `lambda_const_param=10` |
| Identity baseline | `identity` | 不显式近似 Hessian，仅作对照 | 无额外 HVP 超参数 |
| LiSSA | `LiSSA` | 迭代式 HVP 近似 | `lambda_const_param=10`，`n_iteration=10`，`alpha_const=1.0` |
| Accurate | `accurate` | 显式矩阵分解的精确 HVP | `lambda_const_param=10`；代码中当 `low_rank > 4` 时会自动关闭 |

---

## 3. 各实验数据集与模型的具体整理

### 3.1 错标数据检测

| 项目 | 内容 |
| --- | --- |
| 数据集 | GLUE 二分类子任务：`MRPC`、`QNLI`、`QQP`、`SST2`、`WNLI` |
| 模型 | `roberta-large` |
| 训练方式 | LoRA 微调 |
| 目标 | 识别训练集中被翻转标签的坏样本 |
| 噪声设置 | 随机翻转 `20%` 训练样本标签 |
| 训练参数 | `lr=3e-4`，`batch_size=32`，`num_epochs=10` |
| LoRA rank | `1 / 2 / 4 / 8 / 16` |
| 运行次数 | 每个配置 `10` 次随机运行 |
| 代码补充 | `src/dataloader.py` 会在大数据集上裁到 `train<=4500`、`validation<=500` |

### 3.2 文本生成关键样本识别

| 项目 | Sentence Transformation | Math Without Reason | Math With Reason |
| --- | --- | --- | --- |
| 数据集 | synthetic sentence transformation | synthetic math problem | synthetic math problem with reasoning |
| 模型 | `meta-llama/Llama-2-13b-chat-hf` | `meta-llama/Llama-2-13b-chat-hf` | `meta-llama/Llama-2-13b-chat-hf` |
| 训练方式 | LoRA SFT | LoRA SFT | LoRA SFT |
| 训练参数 | `lr=3e-4`，`batch_size=64`，`gradient_accumulation_steps=16`，`seq_length=128`，`epochs=10` | 同左 | 同左 |
| LoRA 参数 | `r=8`，`alpha=32`，`dropout=0.05` | 同左 | 同左 |
| LoRA 模块 | `q_proj`、`v_proj` | `q_proj`、`v_proj` | `q_proj`、`v_proj` |
| 量化/加载 | `load_in_8bit=True` | `load_in_8bit=True` | `load_in_8bit=True` |
| 用途 | 识别对验证句子变换最关键的训练样本 | 识别对数学答案生成最关键的训练样本 | 识别对带推理数学答案生成最关键的训练样本 |

### 3.3 图像生成关键样本识别

| 项目 | 内容 |
| --- | --- |
| 数据集 | `kewu93/three_styles_prompted_250_512x512` |
| 数据规模 | 论文文字说明为 3 种风格、共 `250` 张图片 |
| 模型 | `runwayml/stable-diffusion-v1-5` |
| 训练对象 | UNet 上的 LoRA adapter |
| LoRA 模块 | `to_k`、`to_q`、`to_v`、`to_out.0` |
| LoRA rank | `2` |
| 学习率 | `1e-4` |
| 论文附录设置 | 训练 `3000` gradient steps |
| Notebook 示例命令 | `resolution=512`、`center_crop`、`random_flip`、`train_batch_size=1`、`gradient_accumulation_steps=4`、`max_train_steps=10000`、`lr_scheduler=cosine`、`lr_warmup_steps=0`、`seed=1337`、`checkpointing_steps=1000` |

---

## 4. 当前项目里实际存在的数据集

### 4.1 本地 `datasets/` 目录中已存在的 HuggingFace 数据集

| 本地目录 | 任务类型 | split | 字段 | 样本数观察 |
| --- | --- | --- | --- | --- |
| `datasets/grammars_train.hf` | Sentence Transformation | train | `prompt`、`text`、`answer`、`variation` | 按 `src/generate_sentence-math_datasets.py` 推断为 `900` |
| `datasets/grammars_test.hf` | Sentence Transformation | test | `prompt`、`text`、`answer`、`variation` | 推断为 `100` |
| `datasets/math_without_reason_train.hf` | Math without reasoning | train | `prompt`、`text`、`answer`、`variation` | 推断为 `900` |
| `datasets/math_without_reason_test.hf` | Math without reasoning | test | `prompt`、`text`、`answer`、`variation` | 推断为 `100` |
| `datasets/math_with_reason_train.hf` | Math with reasoning | train | `prompt`、`text`、`answer`、`reason`、`variation` | 推断为 `900` |
| `datasets/math_with_reason_test.hf` | Math with reasoning | test | `prompt`、`text`、`answer`、`reason`、`variation` | 推断为 `100` |

### 4.2 当前项目中未本地落盘、但论文实验会用到的数据集

| 数据集 | 当前项目状态 | 说明 |
| --- | --- | --- |
| GLUE `MRPC / QNLI / QQP / SST2 / WNLI` | 未在仓库中本地保存 | 通过 `datasets.load_dataset("glue", task)` 运行时下载 |
| `kewu93/three_styles_prompted_250_512x512` | 未在仓库中本地保存 | Stable Diffusion 实验在 HuggingFace Hub 上使用 |

---

## 5. 论文描述与当前仓库实现的差异

这些差异值得单独记录，避免后续复现实验时混淆：

| 差异点 | 论文 / 附录描述 | 当前仓库实现 / 示例 |
| --- | --- | --- |
| 分类任务 LoRA 目标模块 | 附录文字写成 query/value 矩阵和输出分类层 | `src/configs.py`、Notebook、`src/lora_model.py` 实际使用的是 `target_modules=["value"]`，并额外保存 `out_proj.weight` 梯度 |
| Sentence / Math 数据规模 | 论文正文写成 100 train + 100 validation 的描述 | `src/generate_sentence-math_datasets.py` 实际会生成每类任务 `900 train + 100 test` 的本地数据集 |
| Stable Diffusion 训练步数 | 论文附录写 `3000` gradient steps | Notebook 示例命令写 `max_train_steps=10000`；`DATAINF_PROJECT_GUIDE.md` 里的演示命令写 `num_train_epochs=100` |

---

## 6. 最终结论

可以把 DataInf 论文实验分成三组：

1. `RoBERTa-large + LoRA` 做 GLUE 二分类错标检测。
2. `Llama-2-13b-chat-hf + LoRA` 做三类文本生成任务的关键样本识别。
3. `Stable Diffusion v1-5 + LoRA` 做 style-transfer 图像生成任务的关键样本识别。

当前项目本地已经准备好的数据集只有三类合成文本任务的 `train/test` 六个 `.hf` 数据集；GLUE 和 style-transfer 数据集仍然是运行时外部加载，而不是仓库内置。
