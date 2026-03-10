# GRADMM Project Guide

## 1. 项目定位

这个仓库可以分成两条主线：

1. `gradmm/`：根据真实样本的梯度信息生成 synthetic text，并做后续过滤。
2. `addax/`：把过滤后的 synthetic data 当作训练集，对语言模型做微调和评估。

完整实验链路是：

1. 从真实数据集中抽样。
2. 用 `gradmm/generate.py` 进行梯度匹配生成。
3. 用 `gradmm/filtering.py` 或 `gradmm/Filtering.ipynb` 清洗/筛选 synthetic data。
4. 用 `addax/run.py` 或 `addax/scripts/query_ft.sh` 对 synthetic data 做微调。
5. 用 `addax/Finetuning.ipynb` 汇总微调结果。

说明：

- `gradmm/init.py` 是“初始化策略模块”，不是 Python 包初始化文件；仓库里没有 `__init__.py`。
- README 里提到的 `gradmm/Finetuning.ipynb`，仓库实际文件在 `addax/Finetuning.ipynb`。

## 2. 项目结构

```text
GRADMM/
├─ addax/
│  ├─ Finetuning.ipynb
│  ├─ GPUtil.py
│  ├─ lora.py
│  ├─ metrics.py
│  ├─ prefix.py
│  ├─ run.py
│  ├─ sign_converter.py
│  ├─ tasks.py
│  ├─ templates.py
│  ├─ test_sign_converter.py
│  ├─ trainer.py
│  ├─ utils.py
│  └─ scripts/
│     └─ query_ft.sh
├─ data/
│  ├─ imdb/
│  └─ rtpolarity/
├─ gradmm/
│  ├─ Filtering.ipynb
│  ├─ args_factory.py
│  ├─ constants.py
│  ├─ data_utils.py
│  ├─ filtering.py
│  ├─ generate.py
│  ├─ init.py
│  ├─ utilities.py
│  └─ scripts/
│     ├─ admm.sh
│     └─ admm_dp.sh
├─ README.md
├─ requirements.txt
└─ GRADMM_PROJECT_GUIDE.md
```

## 3. 本地代码包与模块说明

### 3.1 `gradmm/`

`gradmm/` 是 synthetic text 生成与筛选主包。

| 模块 | 作用 |
| --- | --- |
| `args_factory.py` | 定义 `generate.py` 的命令行参数，主要是数据集、优化算法、ADMM、DP、输出目录等。 |
| `constants.py` | 保存少量 BERT 特殊 token 常量，当前主流程里不是核心依赖。 |
| `data_utils.py` | 负责真实数据集加载、分层抽样、batch/cluster 迭代。 |
| `filtering.py` | synthetic data 过滤入口，支持分类校验、按分数筛样、部分梯度选择。 |
| `generate.py` | synthetic generation 主入口，完成真实样本抽取、真实梯度估计、embedding 优化、离散 token 投影和结果落盘。 |
| `init.py` | 生成初始 embedding 的策略模块，支持随机初始化、随机词向量初始化和基于真实样本初始化。 |
| `utilities.py` | 公共工具函数，包含梯度计算、梯度距离、embedding 到 token 投影、perplexity、few-shot prefix 构造等。 |
| `Filtering.ipynb` | 过滤实验的 notebook 入口，更适合交互式筛选。 |
| `scripts/admm.sh` | 非 DP 的批量生成脚本，会按多个 `rho` 并行起任务。 |
| `scripts/admm_dp.sh` | 带差分隐私噪声的批量生成脚本。 |

### 3.2 `addax/`

`addax/` 是微调、推理和结果汇总主包，基于 Addax 改造。

| 模块 | 作用 |
| --- | --- |
| `run.py` | 微调和 ICL 入口，负责解析参数、加载任务、加载模型、组装 Trainer、启动训练和评估。 |
| `trainer.py` | 自定义 HuggingFace `Trainer`，扩展了日志、评估、checkpoint 和 `main_results.json` 的保存逻辑。 |
| `tasks.py` | 任务适配层，把真实数据或 synthetic data 转成统一的 `Sample`/`Dataset` 抽象。 |
| `templates.py` | prompt/template 层，定义每个任务怎样把样本拼成 LM 输入。 |
| `utils.py` | collator、前向包装、时间统计、GPU 统计、JSON 序列化等辅助能力。 |
| `metrics.py` | accuracy、per-class accuracy、EM、F1 等指标实现。 |
| `prefix.py` | Prefix Tuning 注入逻辑。 |
| `lora.py` | LoRA 注入逻辑。 |
| `sign_converter.py` | 1-bit sign 压缩/解压工具，更偏底层实验工具。 |
| `GPUtil.py` | 仓库内置的 GPU 利用率查询工具。 |
| `test_sign_converter.py` | `sign_converter.py` 的单元测试。 |
| `Finetuning.ipynb` | 微调结果路径整理和结果汇总 notebook。 |
| `scripts/query_ft.sh` | synthetic data 微调批量脚本。 |

### 3.3 `data/`

| 目录 | 作用 |
| --- | --- |
| `data/imdb/` | 本地 IMDB JSONL 数据。 |
| `data/rtpolarity/` | 本地 RT-Polarity JSONL 数据。 |

## 4. `tasks.py` 中支持的任务

### 4.1 synthetic-data 主流程常用任务

| 任务名 | 数据来源 | 主要用途 |
| --- | --- | --- |
| `SST2` / `SynSST2` | HuggingFace `glue/sst2` / synthetic JSONL | 生成、过滤、微调 |
| `RottenTomatoes` / `SynRottenTomatoes` | HuggingFace `rotten_tomatoes` / synthetic JSONL | 生成、过滤、微调 |
| `TwitterEmotion` / `SynTwitterEmotion` | HuggingFace `dair-ai/emotion` 中 `label in [0,1]` / synthetic JSONL | 生成、过滤、微调 |
| `IMDB` / `SynIMDB` | 本地 `data/imdb/*.jsonl` / synthetic JSONL | 生成、微调 |
| `RTPolarity` / `SynRTPolarity` | 本地 `data/rtpolarity/*.jsonl` / synthetic JSONL | 生成、微调 |

### 4.2 Addax 继承下来的通用任务

`tasks.py` 还保留了 CoLA、COPA、BoolQ、MultiRC、CB、WiC、WSC、ReCoRD、RTE、SQuAD、DROP 等通用任务适配层。  
这些任务不是 README 里的主链路重点，但 `run.py` 仍然支持。

## 5. 第三方依赖及作用

`requirements.txt` 中列的是最小依赖集合，项目实际还会间接使用这些库。

| 第三方包 | 作用 | 在项目中的主要落点 |
| --- | --- | --- |
| `torch` | 张量计算、自动求导、训练与推理 | `gradmm/`, `addax/` 全部主流程 |
| `torchvision` | PyTorch 生态依赖，当前仓库几乎未直接使用 | 环境兼容 |
| `torchaudio` | PyTorch 生态依赖，当前仓库几乎未直接使用 | 环境兼容 |
| `transformers` | 加载 tokenizer、CausalLM、Trainer | `generate.py`, `run.py`, `trainer.py` |
| `accelerate` | 大模型自动分配设备、分布式/FSDP/DeepSpeed 支持 | `addax/trainer.py` |
| `peft` | PEFT 兼容检查 | `addax/trainer.py` |
| `datasets` | 读取 HuggingFace 数据集和本地 JSONL | `tasks.py`, `data_utils.py`, `filtering.py` |
| `scikit-learn` | 分层抽样、线性探测所需逻辑 | `data_utils.py`, `tasks.py`, `trainer.py` |
| `absl-py` | 一些训练生态依赖，当前仓库中未直接大量使用 | 兼容依赖 |
| `nltk` | 生成任务评估生态依赖 | Addax 继承代码 |
| `rouge_score` | 文本生成评估生态依赖 | Addax 继承代码 |
| `wandb` | 实验日志与可视化 | `gradmm/generate.py`, `addax/run.py` |
| `tqdm` | 进度条 | 全局 |
| `jupyterlab` | 运行 notebook | `Filtering.ipynb`, `Finetuning.ipynb` |
| `ipywidgets` | notebook 交互控件支持 | notebook |
| `numpy` | 数值计算 | 全局，通常由上面库自动带上 |
| `regex` | token 文本过滤 | `gradmm/generate.py` |
| `packaging` | 版本比较 | `addax/trainer.py` |
| `pandas` | 结果汇总 | `addax/Finetuning.ipynb`，需要手工补装 |

建议额外执行：

```bash
pip install pandas
```

## 6. 环境安装

### 6.1 使用 `venv`（推荐写进文档的标准方案）

Windows PowerShell:

```powershell
cd D:\学习记录\导师项目\研究\caiqiyue_file\GRADMM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install pandas
```

Linux / macOS / Git Bash:

```bash
cd /path/to/GRADMM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install pandas
```

### 6.2 使用 Conda（仓库 README 的方案）

```bash
conda create -n gradmm python=3.11
conda activate gradmm
pip install -r requirements.txt
pip install pandas
```

### 6.3 GPU 说明

- 生成和微调都默认假设有 GPU。
- 如果你要用 GPU 版 PyTorch，请按你本机 CUDA 版本先安装对应的 PyTorch，再执行 `pip install -r requirements.txt`。
- `scripts/admm.sh` / `admm_dp.sh` / `query_ft.sh` 都依赖 `CUDA_VISIBLE_DEVICES`。

### 6.4 Windows 使用注意事项

- 仓库里的批量脚本是 `.sh`，在 PowerShell 里不能直接当 Bash 脚本运行。
- 如果你在 Windows 下想直接跑这些脚本，建议用 Git Bash 或 WSL。
- 在 PowerShell 下更稳妥的做法是直接执行对应的 `python ...` 命令。
- `query_ft.sh` 里用了 `md5sum`，这也更适合在 Git Bash / WSL 里运行。

### 6.5 W&B 说明

如果不想上传到线上，可以先设为离线模式：

PowerShell:

```powershell
$env:WANDB_MODE = "offline"
```

Bash:

```bash
export WANDB_MODE=offline
```

## 7. 实验怎么启动

### 7.1 Step 1: 生成 synthetic data

最直接的入口是：

```bash
cd gradmm
python generate.py \
  --rng_seed 42 \
  --dataset sst2 \
  --split validation \
  --batch_size 50 \
  --n_steps 30 \
  --n_gen_samples 100 \
  --subset_size 50 \
  --n_gen 10 \
  --gen_bs 10 \
  --use_auto_gen_tokens true \
  --model_name phi \
  --opt_alg admm \
  --admm_rho 0.5 \
  --admm_inner_steps 50 \
  --work_base_dir ./synthetic_data/test \
  --grad_clip 1.0 \
  --topk 200
```

批量跑法：

```bash
cd gradmm
bash ./scripts/admm.sh
```

带差分隐私：

```bash
cd gradmm
bash ./scripts/admm_dp.sh
```

### 7.2 Step 2: 过滤 synthetic data

推荐两种方式：

1. 交互式：`gradmm/Filtering.ipynb`
2. 命令行：`gradmm/filtering.py`

命令行示例：

```bash
cd gradmm
python filtering.py \
  --dataset sst2 \
  --file_dir ./synthetic_data \
  --json_file synthetic_data \
  --filter_method top_score \
  --top_n 50 \
  --coeff_perplexity 0 \
  --clean true
```

### 7.3 Step 3: 用 synthetic data 微调

直接命令：

```bash
cd addax
python run.py \
  --trainer regular \
  --model_name microsoft/phi-1_5 \
  --task_name SynSST2 \
  --syn_data_path ../gradmm/synthetic_data/your_run/synthetic_data.jsonl \
  --output_dir ./synthetic_data_FT/demo/output \
  --num_train 100 \
  --num_eval 1000 \
  --num_eval_to_keep 100 \
  --kept_eval_as_train false \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --max_steps 200 \
  --learning_rate 1e-5 \
  --trainer regular \
  --train_as_classification \
  --eval_strategy steps \
  --save_strategy steps \
  --eval_steps 50 \
  --save_steps 50 \
  --overwrite_output_dir
```

批量脚本：

```bash
cd addax
bash ./scripts/query_ft.sh
```

### 7.4 Step 4: 汇总结果

- `addax/Finetuning.ipynb` 中的 `Print fine-tuning paths`：整理待微调的 synthetic data 路径。
- `addax/Finetuning.ipynb` 中的 `Collect fine-tuning results`：读取多个 `main_results.json` 汇总实验结果。

## 8. 核心输出目录与文件

### 8.1 生成阶段

通常输出在：

```text
gradmm/synthetic_data/<run_tag>/
```

常见文件：

| 文件 | 含义 |
| --- | --- |
| `synthetic_data.jsonl` | 最终 synthetic 文本及其损失指标。 |
| `summary_metrics.json` | 均值指标汇总。 |
| `summary_metrics.pkl` | 汇总指标的中间缓存。 |
| `real_train_data.jsonl` | 本轮用于做梯度匹配的真实样本。 |
| `real_init_data.jsonl` | 每轮被选作初始化的真实样本。 |
| `pos_generations.pkl` / `neg_generations.pkl` | 正负类生成缓存。 |
| `rng_states.pth` | 断点续跑的 RNG 状态。 |
| `non_english_tokens.json` / `english_tokens.json` | token 过滤调试文件。 |

### 8.2 过滤阶段

过滤后的文件通常直接写回同一个 run 目录，文件名会附加：

- `clean/remove/relabel`
- `top{N}_score_alpha{alpha}`
- `per_label`
- `interleave`
- `balance_score`

### 8.3 微调阶段

通常输出在：

```text
addax/synthetic_data_FT/<time>/result/<tag>/output/
```

常见文件：

| 文件 | 含义 |
| --- | --- |
| `args.json` | 当前微调运行的完整参数。 |
| `main_results.json` | 核心评估指标、GPU 统计、训练时长。 |
| `trainer_state.json` | HuggingFace Trainer 状态。 |
| `checkpoint-*` | 中间 checkpoint。 |

## 9. 生成阶段参数说明：`gradmm/generate.py`

下面只解释仓库自定义的参数，即 `gradmm/args_factory.py` 里定义的全部参数。

### 9.1 数据与目标梯度

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--rng_seed` | `42` | 全局随机种子。 |
| `--dataset` | 必填 | 数据集名，可选 `sst2`、`rotten_tomatoes`、`TwitterEmotion`、`imdb`、`rtpolarity`。 |
| `--split` | 必填 | 从哪个 split 抽真实样本。 |
| `--data_loader` | `batch` | 真实数据加载方式，`batch` 或 `cluster`。 |
| `--n_clusters` | `10` | `cluster` 模式下的簇数量。 |
| `-b, --batch_size` | `1` | 每轮用于估计真实平均梯度的真实样本 batch 大小。 |
| `--n_gen_samples` | `1000` | 为生成阶段预抽取的真实样本总数。 |
| `--subset_size` | `100` | 实际参与本次实验的真实样本数上限。 |
| `--n_fewshot` | `0` | few-shot prefix 使用的真实示例数。 |
| `--skip_first_samples` | `0` | 断点恢复时跳过前几轮生成。 |
| `--save_avg_grad` | `false` | 只保存平均梯度后退出。 |

### 9.2 模型与损失

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--device` | `cuda` | 运行设备。 |
| `--model_name` | `phi` | 当前映射到 `microsoft/phi-1_5`。 |
| `--loss` | `cos` | 梯度匹配损失：`cos`、`dlg`、`tag`。 |
| `--embed_loss` | `dlg` | embedding 正则损失：`cos`、`dlg`、`tag`、`cos_mapped_embeds`。 |
| `--coeff_perplexity` | `0.0` | perplexity 项系数。 |
| `--coeff_reg` | `0.0` | embedding / norm 正则项系数。 |
| `--reg_loss_type` | `norm` | 正则项类型，`norm` 或 `embed`。 |
| `--tag_factor` | `None` | `loss=tag` 时的 L1 权重。 |
| `--last_layer_gradient` | `true` | 只对最后一层做真实梯度匹配。 |

### 9.3 初始化

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--init` | `random_normal` | 初始 embedding 策略：`real_first`、`real_closest`、`random_normal`、`random_embed`。 |
| `--init_candidates` | `500` | 随机初始化候选个数，会选 reconstruction loss 最好的那个。 |
| `--init_size` | `1.4` | 初始 embedding 归一化后的目标范数。 |

### 9.4 优化与 ADMM

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--opt_alg` | `admm` | 优化算法，可选 `adam`、`bfgs`、`bert-adam`、`admm`、`admm_sgd`。 |
| `--n_steps` | `30` | 外层优化步数。 |
| `--lr` | `0.008` | 学习率。 |
| `--lr_decay` | `0.9` | 学习率衰减系数。 |
| `--lr_decay_type` | `StepLR` | 学习率策略，`StepLR` 或 `LambdaLR`。 |
| `--lr_max_it` | `None` | `LambdaLR` 时的线性衰减总步数，为空时等于 `n_steps`。 |
| `--admm_rho` | `0.7` | ADMM 惩罚系数。 |
| `--admm_inner_steps` | `10` | 每个外层 ADMM step 中的内层更新步数。 |
| `--grad_clip` | `None` | 对 synthetic embedding 梯度裁剪。 |
| `--gen_grad_clip` | `""` | 对真实梯度估计时的裁剪方式：空字符串、`norm`、`elem`。 |

### 9.5 生成与离散化

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--n_gen` | `10` | 生成轮数；每轮会做正类和负类各一次生成。 |
| `--gen_bs` | `1` | 每次生成的 synthetic 样本数。 |
| `--gen_max_tokens` | `30` | synthetic 文本最大 token 长度。 |
| `--use_auto_gen_tokens` | `false` | 自动把 `gen_max_tokens` 设为当前真实样本平均长度。 |
| `--conversion_method` | `topk` | embedding 到 token 的投影方式：`proj`、`topk`、`concat`。 |
| `--topk` | `50` | `topk` 投影时候选 token 上限。 |
| `--include_prefix` | `false` | 投影时是否把 prefix 拼接到生成序列前面。 |
| `--prefix_option` | `fixed` | prefix 选择策略：`fixed` 或 `random`。 |
| `--n_prefix` | `1` | 使用多少个 prefix。 |
| `--independent_gen` | `true` | 是否让每轮生成彼此独立；`false` 时会累计前轮梯度。 |

### 9.6 token 过滤与输出

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--work_base_dir` | `./synthetic_data/` | 生成结果根目录。 |
| `--save_every` | `1` | 每隔多少轮落盘一次缓存。 |
| `--overwrite` | `false` | 如果输出目录已存在，是否直接删除重跑。 |
| `--print_every` | `10` | 每隔多少步打印一次优化日志。 |
| `--print_full` | `true` | 是否打印当前生成文本和完整损失。 |
| `--alpha` | `0.001` | 历史参数，当前主流程里不是核心控制项。 |
| `--drop_non_english_tokens` | `false` | 过滤掉非英文 token。 |
| `--use_sample_tokens_only` | `false` | 只允许使用真实样本中出现过的 token。 |
| `--use_topk` | `false` | 保留参数，当前主流程主要由 `conversion_method` 控制。 |
| `--drop_change_line_characters` | `true` | 去掉换行和大量特殊字符 token。 |

### 9.7 baseline / DP

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--baseline` | `false` | 切到 baseline 默认配置。 |
| `--use_dp` | `false` | 是否对真实平均梯度加入差分隐私噪声。 |
| `--dp_c` | `1.0` | 单样本梯度裁剪阈值。 |
| `--dp_epsilon` | `0.05` | DP 隐私预算。 |
| `--dp_delta` | `1e-4` | DP `delta`。 |

## 10. 过滤阶段参数说明：`gradmm/filtering.py`

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model_name` | `phi` | 过滤时用的判别模型。 |
| `--dataset` | 必填 | 当前只显式支持 `sst2`、`rotten_tomatoes`、`TwitterEmotion`。 |
| `--split` | `train` | 真实数据 split，主要供梯度辅助方法使用。 |
| `--gen_bs` | `1` | 生成批大小，主要供梯度辅助方法使用。 |
| `--n_gen_samples` | `100` | 真实样本池大小。 |
| `--n_fewshot` | `0` | few-shot 例子数。 |
| `--random_seed` | `42` | 随机种子。 |
| `--pos_label` | `positive` | 正类文本标签。 |
| `--neg_label` | `negative` | 负类文本标签。 |
| `--use_instruction` | `true` | few-shot 分类时是否加 instruction prompt。 |
| `--use_fewshot` | `true` | few-shot 分类时是否加示例。 |
| `--filter_score` | `cls` | 过滤分数方式，目前只支持 `cls`。 |
| `--filter_method` | `remove` | 过滤方式：`remove`、`relabel`、`top_score`、`bottom_score`、`first`、`greedy_selection`。 |
| `--file_dir` | 必填 | generation run 的父目录。 |
| `--json_file` | 必填 | 要读取的 JSONL 文件 stem，通常是 `synthetic_data`。 |
| `--coeff_perplexity` | `1` | 在 `top_score/bottom_score` 中使用的分数系数，公式是 `rec_loss_ids + coeff_perplexity * perplexity`。 |
| `--top_n` | `100` | 每个标签保留多少条样本。 |
| `--clean` | `true` | 输出前是否移除 prompt 残留文本。 |
| `--balance_score` | `false` | 是否进一步按均值分数平衡类别。 |
| `--per_label` | `true` | 是否按标签分别取前 `top_n`。 |
| `--interleave_label` | `false` | 输出时是否按标签交错排列。 |

说明：

- `remove`：标签不匹配就删掉。
- `relabel`：标签不匹配时改成模型预测标签。
- `top_score`：保留分数最低的一批样本。
- `bottom_score`：保留分数最高的一批样本。
- `first`：直接取前 `N` 条。
- `greedy_selection`：代码里仍有未完全打通的参数链路，不建议当主流程使用。

## 11. 微调阶段参数说明：`addax/run.py`

`addax/run.py` 的参数由两部分组成：

1. 仓库自己定义的 `OurArguments`
2. HuggingFace `TrainingArguments` 继承来的通用训练参数

下面先列仓库自定义参数，再列脚本中实际常用的 HF 参数。

### 11.1 `OurArguments`（仓库自定义）

#### 数据与采样

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `task_name` | `SST2` | 任务名；synthetic 任务通常用 `SynSST2`、`SynRottenTomatoes`、`SynTwitterEmotion`、`SynIMDB`、`SynRTPolarity`。 |
| `syn_data_path` | `synthetic_data/SST2` | synthetic JSONL 路径。 |
| `num_train` | `0` | 训练样本数；对 `Syn*` 任务会在 `main()` 里自动改成 synthetic 数据总数。 |
| `num_dev` | `None` | dev 样本数。 |
| `num_eval` | `None` | eval 样本数。 |
| `num_eval_to_keep` | `0` | 从 validation 中额外保留多少样本用于 dev/train。 |
| `kept_eval_as_train` | `false` | 是否把保留下来的 eval 样本直接当训练集。 |
| `mix_train_val` | `false` | 是否把保留的 validation 样本混入 train。 |
| `num_train_sets` | `None` | 采样多少个 train/demo 集合。 |
| `train_set_seed` | `None` | 固定某一个 train/demo 集合随机种子。 |
| `result_file` | `None` | 自定义结果输出文件。 |

#### 模型加载

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `model_name` | `facebook/opt-125m` | 要微调的 HuggingFace 模型名。 |
| `load_float16` | `false` | 是否以 FP16 加载模型。 |
| `load_bfloat16` | `false` | 是否以 BF16 加载模型。 |
| `load_int8` | `false` | 是否以 INT8 加载模型。 |
| `max_length` | `2048` | tokenizer 最大长度。 |
| `no_auto_device` | `false` | 是否禁用 `device_map=auto`。 |

#### 校准与训练模式

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `sfc` | `false` | 是否使用 surface form competition 校准。 |
| `icl_sfc` | `false` | ICL 版本的 SFC 校准。 |
| `trainer` | `none` | 训练模式：`none`、`regular`、`sgd` 等。主流程通常用 `regular`。 |
| `only_train_option` | `true` | 训练时只在答案 option 部分计算损失。 |
| `train_as_classification` | `false` | 是否把 LM 训练改成分类式损失。 |
| `report_train` | `false` | 是否额外汇报 train 集指标。 |
| `verbose` | `false` | 是否打印更详细日志。 |
| `no_eval` | `false` | 是否跳过评估。 |

#### 参数高效微调

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `prefix_tuning` | `false` | 是否开启 Prefix Tuning。 |
| `num_prefix` | `5` | prefix token 数。 |
| `no_reparam` | `true` | 是否关闭 prefix 的 reparameterization。 |
| `prefix_init_by_real_act` | `true` | 是否用真实激活初始化 prefix。 |
| `lora` | `false` | 是否开启 LoRA。 |
| `lora_alpha` | `16` | LoRA alpha。 |
| `lora_r` | `8` | LoRA rank。 |

#### 生成 / 评测相关

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `sampling` | `false` | 生成任务是否采样。 |
| `temperature` | `1.0` | 生成温度。 |
| `num_beams` | `5` | beam search 数。 |
| `top_k` | `None` | top-k 采样。 |
| `top_p` | `0.95` | nucleus sampling。 |
| `max_new_tokens` | `15` | 最多生成多少新 token。 |
| `eos_token` | 换行符 | 生成停止 token。 |

#### 其他高级开关

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `zo_eps` | `1e-3` | MeZO 相关参数。 |
| `gradient_ratio` | `0.3` | SeMO 相关参数。 |
| `select_method` | `random` | SeMO 的 backward sample 选择方式。 |
| `beta` | `0.001` | SeMO 梯度混合系数。 |
| `debug_semo` | `false` | SeMO debug。 |
| `use_cpu` | `false` | 是否强制用 CPU。 |
| `save_model` | `false` | 训练结束后是否显式保存模型。 |
| `tag` | `""` | 输出目录附加 tag。 |
| `linear_probing` | `false` | 是否走线性探测路径。 |
| `lp_early_stopping` | `false` | 线性探测是否提前停止。 |
| `head_tuning` | `false` | 是否只调输出头。 |
| `untie_emb` | `false` | 是否解绑 embedding / lm_head。 |
| `non_diff` | `false` | 是否使用不可导目标。 |
| `save_on_interrupt` | `false` | 收到中断信号时是否自动保存。 |
| `no_save_weights` | `false` | checkpoint 时是否不保存权重。 |

### 11.2 `query_ft.sh` 中实际用到的 HuggingFace `TrainingArguments`

| 参数 | 在脚本中的设置 | 含义 |
| --- | --- | --- |
| `--output_dir` | 动态生成 | 当前微调任务输出目录。 |
| `--logging_steps` | `10` | 每 10 step 打印一次日志。 |
| `--lr_scheduler_type` | `linear` | 学习率调度器类型。 |
| `--load_best_model_at_end` | 打开 | 训练结束后回滚到最佳 checkpoint。 |
| `--eval_strategy` | `steps` | 按 step 做评估。 |
| `--save_strategy` | `steps` | 按 step 保存。 |
| `--eval_steps` | `50` | 每 50 step 评估一次。 |
| `--save_steps` | `50` | 每 50 step 保存一次。 |
| `--overwrite_output_dir` | 打开 | 覆盖已有输出目录。 |
| `--save_only_model` | 打开 | 只保存模型相关内容。 |
| `--per_device_train_batch_size` | `16` | 单卡 batch size。 |
| `--gradient_accumulation_steps` | `1` | 梯度累积步数。 |
| `--max_steps` | `200` | 训练总步数。 |
| `--learning_rate` | 由 `LIST_LR` 控制 | 学习率。 |

## 12. 各脚本中最常改的参数

### 12.1 `gradmm/scripts/admm.sh`

| 变量 | 含义 |
| --- | --- |
| `MODEL` | 生成模型别名。 |
| `dataset` | 数据集名。 |
| `split` | 真实数据 split。 |
| `n_gen_samples` | 真实样本池大小。 |
| `subset_size` | 每轮真正用于估计目标梯度的真实样本量。 |
| `n_gen` | 生成轮数。 |
| `gen_bs` | 每次生成多少条 synthetic 样本。 |
| `use_auto_gen_tokens` | 是否自动按真实样本长度设定生成长度。 |
| `n_steps` | 外层优化步数。 |
| `topk` | 离散化候选 token 上限。 |
| `grad_clip` | synthetic embedding 梯度裁剪阈值。 |
| `opt_alg` | 优化算法。 |
| `admm_inner_steps` | 每次 ADMM 外层更新里的内层步数。 |
| `LIST_RHO` | 要批量尝试的 `rho` 列表。 |
| `LIST_GPU` | 每个任务占用的 GPU 列表。 |
| `base_dir` | 输出根目录。 |

### 12.2 `gradmm/scripts/admm_dp.sh`

除上面参数外，还多三个 DP 参数：

| 变量 | 含义 |
| --- | --- |
| `dp_epsilon` | 隐私预算 `epsilon`。 |
| `dp_delta` | 隐私预算 `delta`。 |
| `dp_c` | 单样本梯度裁剪阈值。 |

### 12.3 `addax/scripts/query_ft.sh`

| 变量 | 含义 |
| --- | --- |
| `task_name` | synthetic 任务名。 |
| `MODEL` | 微调模型名。 |
| `list_syn_data_path` | 要批量微调的 synthetic JSONL 路径列表。 |
| `num_train` | 训练样本数。对 `Syn*` 通常会在 `run.py` 内改成 synthetic 总样本数。 |
| `max_steps` | 总训练步数。 |
| `per_device_train_batch_size` | 单卡 batch size。 |
| `gradient_accumulation_steps` | 梯度累积步数。 |
| `LIST_TRAIN_SET_SEED` | train set 采样随机种子。 |
| `kept_eval_as_train` | 是否把保留的 eval 样本当训练集。 |
| `num_eval_to_keep` | 从 validation 里保留多少样本。 |
| `LIST_LR` | 批量尝试的学习率。 |
| `LIST_GPU` | 并行微调使用的 GPU 列表。 |

## 13. notebook 的推荐使用顺序

### 13.1 `gradmm/Filtering.ipynb`

推荐顺序：

1. `Parameters`
2. `Load model`
3. `Filtering - Clean remove`
4. `(Re)calculate rec_loss_ids per sample`
5. `Extract top score`

关键参数通常是：

- `file_dir`
- `exp_pattern`
- `dataset`
- `filter_method`
- `top_n`
- `coeff_perplexity`

### 13.2 `addax/Finetuning.ipynb`

关键单元：

1. `Print fine-tuning paths`
2. `Collect fine-tuning results`

## 14. 运行时建议与已知注意事项

1. 主流程建议优先使用：`generate.py` -> `Filtering.ipynb`/`filtering.py` -> `run.py`/`query_ft.sh`。
2. `generate.py` 的总输出样本量大致受 `n_gen * gen_bs * 2 * n_prefix` 影响；其中 `2` 来自正类/负类各生成一次。
3. `batch_size` 是“真实样本梯度估计 batch”，不是 synthetic 文本输出条数；后者是 `gen_bs`。
4. `query_ft.sh`、`admm.sh`、`admm_dp.sh` 都会并行起任务，跑前先确认 `LIST_GPU` 和机器 GPU 数量一致。
5. `query_ft.sh` 依赖你先把 synthetic JSONL 路径填进 `list_syn_data_path`。
6. `Filtering.ipynb` 和 `Finetuning.ipynb` 用到 notebook 生态，建议额外安装 `pandas`。
7. `filtering.py` 的 `greedy_selection` 分支当前不是最稳妥的主路径，论文复现实验建议优先使用 `remove/relabel/top_score/bottom_score`。
8. `addax/trainer.py` 里的线性探测路径属于高级实验分支，不是当前 synthetic-data 主流程必需项。

## 15. 一句话记忆版

- 生成看 `gradmm/generate.py`
- 过滤看 `gradmm/filtering.py` / `gradmm/Filtering.ipynb`
- 微调看 `addax/run.py` / `addax/scripts/query_ft.sh`
- 汇总看 `addax/Finetuning.ipynb`
