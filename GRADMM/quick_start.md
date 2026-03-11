# GRADMM 项目速读与实验启动说明

## 1. README 在讲什么

`README.md` 给出的主线是：

1. 安装依赖
2. 在 `gradmm/` 里生成 synthetic data
3. 过滤 synthetic data
4. 在 `addax/` 里做微调并汇总结果

结合代码实际实现后，完整流程更准确地说是：

1. 从真实数据集中抽样
2. 用 `gradmm/generate.py` 做梯度匹配，生成候选合成文本
3. 用 `gradmm/Filtering.ipynb` / `gradmm/filtering.py` 清洗、过滤、重算分数并选样
4. 用 `addax/run.py` 对筛选后的 synthetic data 做微调
5. 用 `addax/Finetuning.ipynb` 汇总微调结果

补充说明：

- README 里写的是 `gradmm/Finetuning.ipynb`，但仓库实际文件在 `addax/Finetuning.ipynb`
- README 没写出 filtering notebook 的具体单元顺序，代码里实际是“先 clean/remove，再重算 `rec_loss_ids`，最后按 score 选 top samples”

## 2. 每个文件夹是做什么的

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `addax/` | 微调与评测框架 | 基于 Addax 改造，负责把 synthetic data 作为训练集喂给语言模型 |
| `data/` | 本地保存的数据集文件 | 当前仓库只放了 `imdb` 和 `rtpolarity` 两个本地 JSONL 数据集 |
| `gradmm/` | 梯度匹配生成与过滤逻辑 | 核心论文实现，负责 synthetic text generation 和 filtering |

根目录文件：

| 文件 | 作用 |
| --- | --- |
| `README.md` | 官方快速说明，给出安装、生成、过滤、微调的大致入口 |
| `requirements.txt` | Python 依赖 |
| `LICENSE` | 许可证 |
| `quick_start.md` | 当前这份整理说明 |

## 3. 每个模块的功能是什么

### 3.1 `gradmm/` 模块

- `gradmm/generate.py`
  生成主入口。负责加载模型、抽取真实样本、计算真实梯度、初始化可优化 embedding、运行 ADMM/梯度匹配优化、把 embedding 投影回 token，并把合成样本写到 `synthetic_data.jsonl`。也支持 DP 版本的梯度裁剪和加噪。
- `gradmm/filtering.py`
  synthetic data 的过滤与选样入口。支持：
  - 用语言模型做 few-shot 分类检查标签是否匹配
  - `remove` / `relabel` 这类清洗策略
  - 按 `rec_loss_ids + alpha * perplexity` 排序选 top/bottom 样本
  - greedy gradient selection
  - 按类别平衡、交错输出、文本清洗等
- `gradmm/data_utils.py`
  数据加载模块。`TextDataset` 会根据数据集名读 HuggingFace 数据集或本地 JSONL；`BatchDatasetLoader` 和 `ClusterDatasetLoader` 负责训练/生成时按 batch 或 cluster 取样。
- `gradmm/args_factory.py`
  `generate.py` 的命令行参数定义。包含数据集、优化算法、ADMM 参数、生成长度、DP 参数、输出目录等。
- `gradmm/utilities.py`
  公共工具函数。包括：
  - 随机种子和 RNG 状态保存/恢复
  - 语言模型梯度计算
  - 梯度距离、余弦相似度
  - embedding 到 token 的最近邻/Top-k 映射
  - prefix 构造
  - perplexity、reconstruction loss、embedding regularization
  - 若干文本后处理工具
- `gradmm/init.py`
  初始化模块。根据 `random_normal`、`random_embed` 或真实样本初始化 embedding，并通过 reconstruction loss 选较好的初始点。
- `gradmm/constants.py`
  定义了 BERT 的特殊 token id。当前这套生成流程里基本不是核心模块，更像保留的历史常量。
- `gradmm/Filtering.ipynb`
  filtering 的 notebook 编排层。实际包含 3 段工作：
  - 参数设置
  - `clean remove`
  - 重算 `rec_loss_ids`
  - 按 score 提取 top 样本
- `gradmm/scripts/admm.sh`
  非 DP 版本的批量生成脚本。默认会遍历多个 `rho`，并把任务分发到多张 GPU。
- `gradmm/scripts/admm_dp.sh`
  DP 版本批量生成脚本。和 `admm.sh` 类似，但会额外打开 `--use_dp true` 并设置 `epsilon/delta/C`。

### 3.2 `addax/` 模块

- `addax/run.py`
  微调主入口。负责：
  - 解析训练参数
  - 读取任务/合成任务
  - 加载模型和 tokenizer
  - 选择训练方式（regular trainer / SGD / ICL 等）
  - 启动训练、验证、测试并落盘结果
- `addax/trainer.py`
  对 HuggingFace `Trainer` 的二次封装。主要增加了：
  - 自定义评测和日志记录
  - 保存 `main_results.json`
  - 训练/验证/测试指标历史
  - 系统资源统计
  这个文件体量最大，本质上是训练基础设施。
- `addax/tasks.py`
  任务与数据集适配层。负责把真实数据集或 synthetic data 包装成统一的 `Sample` / `Dataset` 接口。这里既支持本项目常用的 `SynSST2 / SynRottenTomatoes / SynTwitterEmotion / SynIMDB / SynRTPolarity`，也保留了 Addax 原生的其他任务。
- `addax/templates.py`
  prompt/template 定义。不同任务如何把样本拼成语言模型输入，都在这里定义，例如情感分类统一成 `"... It was great/bad"` 这种模板。
- `addax/utils.py`
  训练和推理辅助函数。包括：
  - prompt 编码
  - collator
  - “只训练 option 部分”的 forward 包装
  - 训练时长/GPU 统计
  - 保存 metrics / predictions
- `addax/metrics.py`
  指标计算，支持 `accuracy`、`per_class_accuracy`、`EM`、`F1`。
- `addax/prefix.py`
  Prefix Tuning 注入逻辑。把 prefix key/value 挂到 attention 层上，并改写 generation 的输入准备逻辑。
- `addax/lora.py`
  LoRA 注入逻辑。把 attention 的部分线性层替换成 LoRA 版本，并冻结非 LoRA 参数。
- `addax/sign_converter.py`
  把张量符号压缩成 1-bit `uint8` 表示的工具，更偏向底层实验/压缩辅助，不是 README 主流程的核心依赖。
- `addax/test_sign_converter.py`
  `sign_converter.py` 的简单单元测试。
- `addax/GPUtil.py`
  一个内置的 GPU 监控工具文件，用于训练期间记录显存占用。
- `addax/Finetuning.ipynb`
  微调辅助 notebook。只有两段关键工作：
  - `Print fine-tuning paths`：枚举筛选后的 synthetic data 路径
  - `Collect fine-tuning results`：汇总多个微调实验的 `main_results.json`
- `addax/scripts/query_ft.sh`
  批量微调脚本。会遍历 synthetic data 路径、学习率、随机种子，并把结果写到 `synthetic_data_FT/`。

### 3.3 `data/` 模块

- `data/imdb/`
  本地 IMDB 数据。文件名里的 `len256` 表示这是预处理后的长度版本。
- `data/rtpolarity/`
  本地 RT-Polarity 数据，JSONL 格式，字段为 `id`、`inputs`、`label`。

## 4. 实验用到的数据集是什么

项目代码实际支持的数据集如下。

| 数据集 | 来源 | 在哪里被使用 | 标签映射 |
| --- | --- | --- | --- |
| `sst2` | HuggingFace `glue/sst2` | 生成、过滤、微调 | `0=bad`, `1=great` |
| `rotten_tomatoes` | HuggingFace `rotten_tomatoes` | 生成、过滤、微调 | `0=bad`, `1=great` |
| `TwitterEmotion` | HuggingFace `dair-ai/emotion`，只保留标签 `0/1` | 生成、过滤、微调 | `0=sadness`, `1=joy` |
| `imdb` | 本地 `data/imdb/*.jsonl` | 生成、微调 | `0=bad`, `1=great` |
| `rtpolarity` | 本地 `data/rtpolarity/*.jsonl` | 生成、微调 | `0=bad`, `1=great` |

补充说明：

- `addax/tasks.py` 里还有 `CoLA`、`Copa`、`BoolQ`、`MultiRC`、`CB`、`WIC`、`WSC`、`ReCoRD`、`RTE`、`SQuAD`、`DROP` 等任务，这些更像是继承自 Addax 的通用能力，不是 README 主流程的 synthetic-data 实验重点。
- `gradmm/filtering.py` 的参数限制里，官方 filtering CLI 只显式支持 `sst2`、`rotten_tomatoes`、`TwitterEmotion` 三个数据集。

## 5. 如何启动实验

### 5.1 环境安装

README 给出的安装方式可以直接用：

```bash
conda create -n gradmm python=3.11
conda activate gradmm
pip install -r requirements.txt
```

建议再补一个：

```bash
pip install pandas
```

原因：`addax/Finetuning.ipynb` 汇总结果时用到了 `pandas`，但 `requirements.txt` 里没有写。

如果你不想把日志上传到 Weights & Biases，可以先设离线模式：

PowerShell:

```powershell
$env:WANDB_MODE = "offline"
```

Bash:

```bash
export WANDB_MODE=offline
```

### 5.2 启动 synthetic data 生成

README 推荐的方式是：

```bash
cd gradmm
bash ./scripts/admm.sh
bash ./scripts/admm_dp.sh
```

这两个脚本的作用分别是：

- `admm.sh`：普通 GRADMM 生成
- `admm_dp.sh`：带差分隐私噪声的生成

运行前建议先改脚本里的这些参数：

- `dataset`
- `split`
- `n_gen_samples`
- `subset_size`
- `n_gen`
- `gen_bs`
- `LIST_RHO`
- `LIST_GPU`

默认脚本会并行占用多张 GPU；如果你只有单卡，建议直接手动跑一条命令，更容易控参。

单次生成的最小可用示例：

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

生成阶段的重要输出通常在：

```text
gradmm/synthetic_data/.../
```

常见文件包括：

- `synthetic_data.jsonl`：生成出的候选合成数据
- `summary_metrics.json`：整体平均指标
- `real_train_data.jsonl`：用于匹配梯度的真实样本
- `real_init_data.jsonl`：被拿来初始化的真实样本
- `pos_generations.pkl` / `neg_generations.pkl`：中间缓存
- `rng_states.pth`：断点续跑用

### 5.3 启动 filtering

官方推荐方式不是直接命令行，而是跑 notebook：

```text
gradmm/Filtering.ipynb
```

这个 notebook 的实际顺序是：

1. `Parameters`
2. `Load model`
3. `Filtering - Clean remove`
4. `(Re)calculate rec_loss_ids per sample`
5. `Extract top score`

你需要重点改的是参数单元里的：

- `file_dir`
- `exp_pattern`
- `--dataset`
- `--filter_method`
- `--top_n`
- `--coeff_perplexity`

根据 notebook 默认代码，筛选阶段会生成类似这样的文件名：

- `synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue.jsonl`
- `synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top50_score_alpha0_per_label_balance_score.jsonl`

含义分别是：

- 第一类文件：先做 clean/remove 之后的版本
- 第二类文件：再按 `rec_loss_ids + alpha * perplexity` 选出 top 样本之后的版本

### 5.4 启动微调

微调阶段一共两步。

第一步：在 notebook 里打印可用 synthetic data 路径

实际文件是：

```text
addax/Finetuning.ipynb
```

在 `# Print fine-tuning paths` 单元里改：

- `file_dir`
- `exp_pattern`
- `json_pattern`

例如 notebook 默认会查找：

```text
synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top*.jsonl
```

第二步：把这些路径贴到批量训练脚本里

编辑：

```text
addax/scripts/query_ft.sh
```

把 `list_syn_data_path=(...)` 填上你在 notebook 里打印出的路径，然后根据数据集设置：

- `SynSST2`
- `SynRottenTomatoes`
- `SynTwitterEmotion`
- `SynIMDB`
- `SynRTPolarity`

然后运行：

```bash
cd addax
bash ./scripts/query_ft.sh
```

这个脚本会遍历：

- 多个 synthetic data 路径
- 多个学习率
- 多个 `train_set_seed`

并把结果写到：

```text
addax/synthetic_data_FT/<timestamp>/result/<experiment_tag>/output/
```

其中最重要的结果文件是：

- `output/main_results.json`
- `output/trainer_state.json`
- `output/args.json`

日志会写到：

```text
addax/logs/admm_syn/
```

### 5.5 汇总微调结果

还是在：

```text
addax/Finetuning.ipynb
```

切到 `# Collect fine-tuning results` 单元，把微调实验目录贴到：

```python
list_exp_path = [
    ...
]
```

运行后 notebook 会读取每个实验目录下的 `output/main_results.json`，汇总成一个 `pandas.DataFrame`。

## 6. 推荐的最小实验路径

如果你只是想把整条链路跑通，建议按这个顺序：

1. 安装依赖，并额外装 `pandas`
2. `cd gradmm`
3. 跑一次单卡 `python generate.py ...`，先用 `sst2`
4. 打开 `gradmm/Filtering.ipynb`，完成 clean/remove 和 top-score 选样
5. 打开 `addax/Finetuning.ipynb`，先打印筛选后 JSONL 路径
6. 把路径粘到 `addax/scripts/query_ft.sh`
7. `cd addax && bash ./scripts/query_ft.sh`
8. 回到 `addax/Finetuning.ipynb` 汇总 `main_results.json`

## 7. 需要提前知道的坑

- README 中的微调 notebook 路径写错了，应该是 `addax/Finetuning.ipynb`，不是 `gradmm/Finetuning.ipynb`
- `requirements.txt` 没有 `pandas`，但结果汇总 notebook 用到了它
- `gradmm/generate.py` 和 `gradmm/filtering.py` 仍内建 `phi -> microsoft/phi-1_5`，但现在也允许直接传 Hugging Face 模型名做 smoke 测试
- 脚本都是 `.sh`，默认按 Bash/WSL/Git Bash 环境写的；如果你在原生 Windows PowerShell 下运行，需要改写脚本或直接运行 Python 命令
- `addax/tasks.py` 里 `RTPolarityDataset` 和 `SynRTPolarityDataset` 使用的是 `../data/rtpolarityy/validation.jsonl`，这里看起来是个路径拼写错误；如果你要跑 `SynRTPolarity` 微调，建议先改成 `../data/rtpolarity/validation.jsonl`
- `gradmm/filtering.py` 现在也支持 `imdb` 和 `rtpolarity`，可以直接走本地 smoke 数据的 filtering 流程

## 8. 一句话总结

这个仓库可以看成两段：

- `gradmm/` 负责“从真实样本梯度反推 synthetic text”
- `addax/` 负责“拿筛选后的 synthetic text 去微调并评估语言模型”

## 9. Tiny 数据与 Python Workflow

为了在本地快速跑通完整链路，仓库现在额外提供了：

- `tools/create_smoke_datasets.py`
  从 `data/` 里抽样生成 `data_smoke/`
- `gradmm/filtering_workflow.py`
  把 `Filtering.ipynb` 的 clean/remove、重算 `rec_loss_ids`、top-score 选样收敛成 CLI
- `addax/finetuning_workflow.py`
  把 `Finetuning.ipynb` 的路径枚举和结果汇总收敛成 CLI

### 9.1 生成 tiny 本地数据

```bash
python tools/create_smoke_datasets.py --sample-size 40
python tools/create_smoke_datasets.py --self-test
```

输出目录：

```text
data_smoke/
├─ imdb/
└─ rtpolarity/
```

### 9.2 推荐 smoke 模型

完整 smoke run 推荐使用：

```text
sshleifer/tiny-gpt2
```

`gradmm` 侧现在支持直接传这个模型名，不必只用 `phi`。

### 9.3 推荐 smoke 命令顺序

1. 生成 tiny 数据：

```bash
python tools/create_smoke_datasets.py --sample-size 40
```

2. 跑一轮 tiny 生成，以 `imdb` 为例：

```bash
cd gradmm
python generate.py \
  --device cpu \
  --model_name sshleifer/tiny-gpt2 \
  --dataset imdb \
  --data_root ../data_smoke \
  --split validation \
  --batch_size 4 \
  --n_steps 2 \
  --n_gen_samples 8 \
  --subset_size 4 \
  --n_gen 2 \
  --gen_bs 2 \
  --gen_max_tokens 8 \
  --opt_alg admm \
  --admm_rho 0.5 \
  --admm_inner_steps 1 \
  --topk 20 \
  --work_base_dir ./synthetic_data/imdb_smoke
```

3. 跑 filtering workflow：

```bash
python filtering_workflow.py \
  all \
  --dataset imdb \
  --model-name sshleifer/tiny-gpt2 \
  --data-root ../data_smoke \
  --file-dir ./synthetic_data \
  --exp-pattern imdb_smoke \
  --gen-bs 2 \
  --top-n 4
```

4. 枚举可用于微调的 JSONL：

```bash
cd ../addax
python finetuning_workflow.py \
  list-paths \
  --file-dir ../gradmm/synthetic_data \
  --exp-pattern imdb_smoke
```

5. 用 tiny 模型做一次微调：

```bash
python run.py \
  --trainer regular \
  --use_cpu \
  --report_to none \
  --model_name sshleifer/tiny-gpt2 \
  --task_name SynIMDB \
  --data_root ../data_smoke \
  --syn_data_path ../gradmm/synthetic_data/imdb_smoke/.../synthetic_data_clean_remove_cls_tiny-gpt2_imdb_positive_negative_instrFalse_fsTrue_top4_score_alpha0.0_per_label_balance_score.jsonl \
  --output_dir ./synthetic_data_FT/imdb_smoke/output \
  --num_train 4 \
  --num_eval 8 \
  --num_eval_to_keep 8 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --max_steps 2 \
  --learning_rate 1e-5 \
  --train_as_classification \
  --eval_strategy steps \
  --save_strategy steps \
  --eval_steps 1 \
  --save_steps 1 \
  --overwrite_output_dir
```

6. 汇总微调结果：

```bash
python finetuning_workflow.py \
  collect-results \
  --exp-path ./synthetic_data_FT/imdb_smoke
```

### 9.4 本地自检

```bash
python tools/create_smoke_datasets.py --self-test
cd gradmm && python filtering_workflow.py --self-test
cd ../addax && python finetuning_workflow.py --self-test
```
