# DataInf 项目阅读与快速启动说明

## 1. 项目是做什么的

这个仓库对应论文 **DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models**。核心目标是：

- 先用 LoRA 微调模型。
- 再为每个训练样本和验证样本计算梯度。
- 用 `DataInf` 近似 Hessian-vector product，再估计 influence function。
- 用这些影响值做两类任务：
  - **错标数据检测**：找出最可能被错误标注的训练样本。
  - **关键样本识别**：找出对某个验证样本最有帮助或最有害的训练样本。

---

## 2. 项目目录说明

### 根目录

- `README.md`
  - 官方简介与最简示例，覆盖了 MRPC/QNLI 的文本分类实验，以及 LLaMA2 数学数据集实验。
- `requirements.txt`
  - 基础依赖列表，但**并不完整**，只能覆盖一部分实验。
- `quick_start.md`
  - 本文件，整理后的中文快速上手说明。

### `datasets/`

本地已经存在 6 个 HuggingFace `Dataset.save_to_disk()` 产物：

- `grammars_train.hf` / `grammars_test.hf`
  - 句子变换任务数据集。
- `math_without_reason_train.hf` / `math_without_reason_test.hf`
  - 不带推理过程的数学题数据集。
- `math_with_reason_train.hf` / `math_with_reason_test.hf`
  - 带推理过程的数学题数据集。

这些目录里保存的是 `.arrow` 数据文件和元信息，不是原始 CSV。

### `figures/`

- 论文/README 用图，目前只有 `llama-diffusion-new.png`。

### `notebooks/`

交互式实验入口，基本覆盖论文里的主要示例：

- `Mislabeled_Data_Detection-RoBERTa-MRPC.ipynb`
  - RoBERTa-large + LoRA，在 GLUE-MRPC 上做错标样本检测。
- `Influential_Data_Identification-Llama2-Sentence.ipynb`
  - Llama2 + LoRA，在句子变换数据集上做 influential data identification。
- `Influential_Data_Identification-Llama2-Math-Without-Reason.ipynb`
  - Llama2 + LoRA，在不带推理过程的数学题数据集上做 influential data identification。
- `Influential_Data_Identification-Llama2-Math-Reason.ipynb`
  - Llama2 + LoRA，在带推理过程的数学题数据集上做 influential data identification。
- `Influential_Data_Identification-Stable_Diffusion-Style_Transfer.ipynb`
  - Stable Diffusion v1.5 + LoRA，在风格迁移图像数据集上做 influential data identification。

### `src/`

核心源码目录，训练、数据生成、影响函数计算都在这里。

---

## 3. 每个源码模块的功能

### `src/configs.py`

文本分类 CLI 实验的配置工厂。

- `generate_config(...)`
  - 生成一组实验配置，默认使用 `roberta-large`、LoRA、`noise_ratio=0.2`、`batch_size=32`、`num_epochs=10`。
- `config_qnli1` 到 `config_qnli5`
  - QNLI 任务，不同 LoRA rank。
- `config_qqp1` 到 `config_qqp5`
  - QQP 任务，不同 LoRA rank。
- `config_sst21` 到 `config_sst25`
  - SST-2 任务，不同 LoRA rank。
- `config_mrpc1` 到 `config_mrpc5`
  - MRPC 任务，不同 LoRA rank。
- `config_wnli1` 到 `config_wnli5`
  - WNLI 任务，不同 LoRA rank。

我已补上 `_setup_env()`，否则 `launcher.py` 调配置时会直接报错。

### `src/dataloader.py`

文本分类任务的数据加载与噪声注入。

- 从 HuggingFace `glue` 数据集下载 `mrpc/qnli/qqp/sst2/wnli/...`。
- 对较大的 GLUE 子集自动裁剪到：
  - 训练集最多 4500 条
  - 验证集最多 500 条
- 在训练集上按比例翻转标签，构造错标样本。
- 对句对/单句任务做 tokenization。
- 返回：
  - 训练 `DataLoader`
  - 验证 `DataLoader`
  - 被翻转的噪声样本索引
  - tokenized dataset
  - `collate_fn`

### `src/generate_sentence-math_datasets.py`

本地合成数据集生成器。

- 句子变换任务：
  - 定义了 10 种 transformation，如倒序、隔位大写、去元音、重复单词等。
  - 基于固定短句列表生成 `grammars_train.hf` 和 `grammars_test.hf`。
- 数学题任务：
  - 定义了 10 类小学应用题模板，如披萨、郊游、折扣、面积、利息等。
  - 可生成：
    - `math_without_reason_*`
    - `math_with_reason_*`
- 运行脚本后会一次性生成三套数据集。

从代码逻辑看，三套本地数据集都是 **1000 条总样本，按 900/100 划分 train/test**。

### `src/influence.py`

影响函数计算核心。

- `IFEngine`
  - 面向文本分类任务。
  - 先对验证集梯度取平均，再计算 HVP 和 influence。
  - 支持 4 种 HVP 方式：
    - `identity`
    - `proposed`，即论文中的 DataInf 近似
    - `LiSSA`
    - `accurate`，显式分解版本，代价高
  - 结果保存为 `results_<run_id>.pkl`。
- `IFEngineGeneration`
  - 面向生成任务。
  - 为每个验证样本分别算 influence，而不是只算验证集平均梯度。
  - 输出是一个按 `val_id x train_id` 组织的影响矩阵。

### `src/launcher.py`

文本分类 CLI 入口。

- 命令形式：`python src/launcher.py run ...`
- 行为：
  - 根据 `exp_id` 读取 `configs.py` 中的配置函数。
  - 把单次运行配置写到 `config.pickle`。
  - 调 `simulator.main(config)` 执行实验。

### `src/lora_model.py`

LoRA 微调与逐样本梯度提取。

- `LORAEngine`
  - 面向 `roberta-large` 这类序列分类模型。
  - 用 PEFT/LoRA 进行 GLUE 微调。
  - 训练完成后，逐条扫描 train/validation 样本，提取 LoRA 参数梯度。
- `LORAEngineGeneration`
  - 面向 Llama2 这类生成模型。
  - 从 base model + LoRA adapter 加载模型。
  - 从本地 `.hf` 数据集读 `prompt/text/answer/...`。
  - 对每个样本计算 causal LM loss 梯度，并抽取 LoRA 参数梯度。

我已把 adapter 路径改为随 `dataset_name` 自动切换：

- `grammars` -> `models/grammars_13bf`
- `math_without_reason` -> `models/math_without_reason_13bf`
- `math_with_reason` -> `models/math_with_reason_13bf`

否则仓库原始代码只会固定去找 `models/math_with_reason_13bf`，与另外两个 notebook 不一致。

### `src/run_experiment.py`

文本分类实验总控逻辑。

- 创建 dataloader。
- 训练 LoRA 分类模型。
- 提取 train/validation 梯度。
- 调 `IFEngine` 计算 influence。
- 保存结果并清理 CUDA cache。

### `src/sft_trainer.py`

Llama2 生成实验的 SFT 训练脚本。

- 基于 HuggingFace TRL 的 `SFTTrainer`。
- 支持：
  - 本地 `load_from_disk()` 数据集
  - 8bit/4bit 加载
  - PEFT/LoRA 微调
- 主要用于：
  - `grammars_train.hf`
  - `math_without_reason_train.hf`
  - `math_with_reason_train.hf`

输出是训练后的 LoRA adapter 目录，一般放到 `models/<dataset_name>_13bf`。

### `src/simulator.py`

非常薄的一层封装。

- 根据 `run_id` 固定随机种子。
- 调 `run_experiment_core(config)`。

### `src/train_text_to_image_lora.py`

Stable Diffusion 文生图 LoRA 训练脚本。

- 基于 HuggingFace Diffusers 官方示例改造。
- 支持：
  - HF Hub 数据集或本地 imagefolder
  - `accelerate launch`
  - checkpoint 保存与恢复
  - 验证 prompt 生成图片
  - LoRA 权重导出
- 这是 `Stable_Diffusion-Style_Transfer.ipynb` 的训练入口。

---

## 4. 实验用到的数据集是什么

### A. 文本分类错标检测实验

数据来自 HuggingFace `glue`：

- README 和 notebook 主示例：`GLUE-MRPC`
- CLI 配置还支持：
  - `QNLI`
  - `QQP`
  - `SST-2`
  - `WNLI`

其中错标数据是**合成噪声**：

- `src/dataloader.py` 会随机翻转训练集标签。
- 默认翻转比例是 `0.2`，也就是 **20%**。

### B. Llama2 关键样本识别实验

有 3 套本地合成数据：

1. `grammars_*`
   - 句子变换任务。
   - 字段：`prompt`, `text`, `answer`, `variation`
2. `math_without_reason_*`
   - 不带推理过程的数学题。
   - 字段：`prompt`, `text`, `answer`, `variation`
3. `math_with_reason_*`
   - 带推理过程的数学题。
   - 字段：`prompt`, `text`, `answer`, `reason`, `variation`

这些数据都由 `src/generate_sentence-math_datasets.py` 生成，仓库里已经附带了一份生成结果。

### C. Stable Diffusion 风格迁移实验

数据集不是本地自带，而是 notebook 中在线加载的 HuggingFace 数据集：

- `kewu93/three_styles_prompted_250_512x512`

这个数据集混合了 3 种风格：

- cartoon
- sketch
- pixel-art

### D. 预训练底座模型

项目实际依赖的底座模型有 3 类：

- 文本分类：`roberta-large`
- 文本生成：`meta-llama/Llama-2-13b-chat-hf`
- 文生图：`runwayml/stable-diffusion-v1-5`

---

## 5. 启动前准备

### 基础环境

建议至少准备：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install jupyter notebook accelerate trl diffusers torchvision huggingface_hub packaging
```

说明：

- `requirements.txt` 没有把 `accelerate`、`trl`、`diffusers`、`torchvision`、`huggingface_hub` 等依赖补全。
- 如果跑 8bit/4bit LoRA，通常还需要 `bitsandbytes`。
- `Llama-2-13b-chat-hf` 是 gated model，需要先完成 HuggingFace 登录和权限申请。
- Llama2 与 Stable Diffusion 相关实验更适合在 **Linux + CUDA** 环境运行；Windows 下尤其是 `bitsandbytes` 兼容性需要额外确认。

### 启动 Jupyter

```powershell
jupyter notebook
```

---

## 6. 如何启动实验

## 6.1 最快的入口：直接运行 notebook

按任务打开对应 notebook 即可：

- 错标检测：
  - `notebooks/Mislabeled_Data_Detection-RoBERTa-MRPC.ipynb`
- Llama2 句子变换：
  - `notebooks/Influential_Data_Identification-Llama2-Sentence.ipynb`
- Llama2 数学题（无推理）：
  - `notebooks/Influential_Data_Identification-Llama2-Math-Without-Reason.ipynb`
- Llama2 数学题（有推理）：
  - `notebooks/Influential_Data_Identification-Llama2-Math-Reason.ipynb`
- Stable Diffusion 风格迁移：
  - `notebooks/Influential_Data_Identification-Stable_Diffusion-Style_Transfer.ipynb`

如果你只是想先看最容易跑通的例子，优先从 **MRPC notebook** 开始。

## 6.2 生成本地合成数据集

如果你想重新生成本地 `.hf` 数据集，执行：

```powershell
python src/generate_sentence-math_datasets.py
```

输出会覆盖/生成：

- `datasets/grammars_train.hf`
- `datasets/grammars_test.hf`
- `datasets/math_without_reason_train.hf`
- `datasets/math_without_reason_test.hf`
- `datasets/math_with_reason_train.hf`
- `datasets/math_with_reason_test.hf`

## 6.3 启动文本分类 CLI 实验

README 里给出的命令少了 `src/` 路径。当前仓库正确入口应写成：

```powershell
python src/launcher.py run --exp_id=qnli4 --run-id=0 --runpath=.
```

可替换的 `exp_id` 包括：

- `qnli1` ~ `qnli5`
- `qqp1` ~ `qqp5`
- `sst21` ~ `sst25`
- `mrpc1` ~ `mrpc5`
- `wnli1` ~ `wnli5`

其中数字越大，LoRA rank 越高：

- `1 -> r=1`
- `2 -> r=2`
- `3 -> r=4`
- `4 -> r=8`
- `5 -> r=16`

这个 CLI 流程会：

- 在线下载对应 GLUE 数据集
- 注入 20% 标签噪声
- 微调 `roberta-large`
- 计算 influence
- 在项目根目录写出：
  - `config.pickle`
  - `results_0.pkl`
  - `results_1.pkl`
  - ...

## 6.4 启动 Llama2 实验

### 第一步：先训练 LoRA adapter

以 `math_with_reason` 为例：

```powershell
python src/sft_trainer.py `
  --model_name meta-llama/Llama-2-13b-chat-hf `
  --dataset_name .\datasets\math_with_reason_train.hf `
  --output_dir .\models\math_with_reason_13bf `
  --dataset_text_field text `
  --load_in_8bit `
  --use_peft
```

另外两套数据只需要替换数据路径和输出目录：

- `grammars_train.hf` -> `models/grammars_13bf`
- `math_without_reason_train.hf` -> `models/math_without_reason_13bf`

### 第二步：进入对应 notebook 计算 influence

训练完 adapter 后，打开以下 notebook 继续执行梯度提取和 influence 计算：

- `Influential_Data_Identification-Llama2-Sentence.ipynb`
- `Influential_Data_Identification-Llama2-Math-Without-Reason.ipynb`
- `Influential_Data_Identification-Llama2-Math-Reason.ipynb`

你需要在 notebook 里把路径改成你自己的环境，例如：

- `base_path`
  - 指向 Llama2 base model
- `project_path`
  - 指向当前项目根目录

### 产物

Llama2 训练阶段的主要输出是：

- `models/grammars_13bf`
- `models/math_without_reason_13bf`
- `models/math_with_reason_13bf`

influence 结果主要在 notebook 运行态中分析，如需落盘，需要你在 notebook 中额外保存。

## 6.5 启动 Stable Diffusion 风格迁移实验

训练入口是：

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

训练完成后，再打开：

- `notebooks/Influential_Data_Identification-Stable_Diffusion-Style_Transfer.ipynb`

继续做：

- LoRA 权重加载
- 图像生成
- 训练/验证样本梯度提取
- influence 计算与可视化

如果你不想自己训练，notebook 里也提供了公开权重：

- `kewu93/three_styles_lora`

---

## 7. 主要输出文件会保存到哪里

### 文本分类 CLI

保存在项目根目录：

- `config.pickle`
- `results_<run_id>.pkl`

### 本地数据集生成

保存在 `datasets/`：

- `*_train.hf`
- `*_test.hf`

### Llama2 LoRA 微调

通常保存在 `models/`：

- `models/grammars_13bf`
- `models/math_without_reason_13bf`
- `models/math_with_reason_13bf`

### Stable Diffusion LoRA 微调

通常保存在 `models/three_styles_lora`：

- LoRA 权重
- checkpoint
- 日志

---

## 8. 推荐的阅读/运行顺序

如果你是第一次接触这个仓库，建议顺序如下：

1. 先看 `README.md`
2. 再跑 `notebooks/Mislabeled_Data_Detection-RoBERTa-MRPC.ipynb`
3. 再看 `src/influence.py` 和 `src/lora_model.py`
4. 再运行 `python src/generate_sentence-math_datasets.py`
5. 最后再尝试 Llama2 或 Stable Diffusion notebook

这样能最快理解这个项目是怎么把 **LoRA 微调**、**逐样本梯度** 和 **DataInf influence approximation** 串起来的。
