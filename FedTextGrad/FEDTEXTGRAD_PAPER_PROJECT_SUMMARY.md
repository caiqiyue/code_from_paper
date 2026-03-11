# FedTextGrad 论文与当前项目数据集 / 模型 / 超参数汇总

## 说明

本文件基于以下材料整理：

- `FedTextGrad.pdf`
- `FEDTEXTGRAD_PROJECT_GUIDE.md`
- `FEDTEXTGRAD_FEATURE_EXTRACTION_PLAN.md`
- 当前仓库代码，重点参考 `textgrad/tasks/`、`main.py`、`train_*.py`

说明一：论文是 prompt / textual gradient 优化方法，正文里真正稳定出现的“超参数”主要是客户端数、batch size、本地更新步数 / epoch、聚合方式等，而不是传统数值训练里的学习率。  
说明二：论文正文与附录对部分模型名称的粒度不完全一致，例如同时出现 `GPT-4` 与 `GPT-4o`；下面会单独标注。  
说明三：当前仓库“支持的数据集”不等于“仓库里已经内置了数据文件”。大部分数据集是运行时下载，或依赖仓库外部路径。

## 1. 论文中使用了哪些数据集

### 1.1 主实验数据集

论文正文第 3.2 节和表 1 明确使用了 3 个任务：

| 类别 | 数据集 / 任务 | 用途 |
| --- | --- | --- |
| BBH | `Object Counting` | 同构联邦实验 |
| BBH | `Multi-step Arithmetic` | 异构联邦实验中的客户端任务之一 |
| GSM8K | `Math Problems` | 异构联邦实验中的客户端任务之一 |

可直接概括为：论文主实验围绕 `BBH + GSM8K` 展开，其中 BBH 至少用了 `Object Counting` 和 `Multi-step Arithmetic` 两个子任务。

### 1.2 附录扩展实验数据集

论文附录 D 又补充了 LiveBench 上的实验，明确提到：

| 来源 | 具体任务 |
| --- | --- |
| LiveBench Reasoning | `Spatial` |
| LiveBench Reasoning | `Web of Lies V2` |
| LiveBench Reasoning | `Zebra Puzzle` |
| LiveBench Math | `AMPS-Hard` |

### 1.3 Prompt transfer 实验

论文表 4 的 prompt transfer 实验仍然基于：

| 数据集 / 任务 | 用途 |
| --- | --- |
| BBH `Object Counting` | 把大模型上学到的 prompt 迁移到更小模型 |

## 2. 论文中使用了哪些模型

### 2.1 论文主线中明确出现的核心模型

| 模型 | 角色 |
| --- | --- |
| `LLaMA-3.1-8B` | 论文正文图 2 / 第 3.2 节明确给出的当前本地模型，用于主实验展示 |

### 2.2 论文正文与附录中出现的比较模型 / API 模型

论文中可明确识别到的模型名称包括：

| 模型 | 备注 |
| --- | --- |
| `GPT-4` | 正文图 3 明确出现 |
| `GPT-4o` | 附录 B.2 明确出现 |
| `GPT-3.5` | 附录 B.2 明确出现 |
| `DeepSeek-R1-Distill-Llama-70B` | 正文图 3 明确出现 |
| `LLaMA-3.1-405B` | 正文图 3 明确出现 |
| `Llama 3` | 附录 B.2 作为开源 API 模型族出现 |
| `Llama 3.1` | 附录 B.2 作为开源 API 模型族出现 |
| `Qwen 2` | 附录 B.2 作为开源 API 模型族出现 |

### 2.3 Prompt transfer 实验中的模型

论文表 4 明确使用了：

| 模型 | 用途 |
| --- | --- |
| `LLaMA-3.2-11B` | 先在较大模型上优化 prompt |
| `LLaMA-3.2-3B` | 接收迁移后的 prompt，验证能否提升小模型表现 |

### 2.4 关于模型命名的一点说明

- 正文图 3 写的是 `GPT-4`，附录 B.2 写的是 `GPT-4o` / `GPT-3.5`。
- 因此更稳妥的结论是：论文既使用了本地 `LLaMA` 系列模型，也比较了商业 API 模型与开源 API 模型。

## 3. 论文中的超参数 / 实验设置

### 3.1 表 1 明确给出的核心设置

论文表 1 给出的实验设置可以整理为：

| 项目 | 取值 / 设置 |
| --- | --- |
| 同构联邦客户端数 | `3` |
| 异构联邦客户端数 | `3` |
| Batch size `B` | `{1, 3, 10}`，默认主设置为 `3` |
| Local update steps / epochs `E` | `{3, 5, 10}`，默认主设置为 `3` |
| Prompt 聚合方法 | `Naive concat`、`summarization`、`UID-based summarization` |
| 同构数据划分 | 训练集与验证集随机且均匀切给 3 个客户端 |
| 本地 batch 采样 | 随机且可重复抽样 |
| 更新接受规则 | 每轮更新后，用同一个 batch 重新评估；只有 prompt 性能不下降时才保留更新 |

### 3.2 附录中补充出现的设置

| 场景 | 设置 |
| --- | --- |
| 同构 API 模型比较 | `E = 3`、`B = 3`、`3` 个客户端，任务为 BBH `Object Counting` |
| UID 泛化实验 | 训练 epoch 固定为 `3` |
| Prompt transfer | 在 `LLaMA-3.2-11B` 上优化 prompt，再迁移到 `LLaMA-3.2-3B` |

### 3.3 论文没有明确给出的传统数值超参数

论文的重点不是数值参数训练，因此正文中没有像常规深度学习论文那样给出：

- 学习率
- weight decay
- optimizer beta
- warmup ratio

更关键的“超参数”实际上是：

- 客户端数量
- batch size
- 本地更新步数 / epoch
- 聚合方式
- 是否采用“更新后再验证，不提升则回退”的 prompt 接受机制

## 4. 当前项目中有哪些数据集

## 4.1 主训练入口 `load_task(...)` 当前支持的数据集

根据 `FEDTEXTGRAD_PROJECT_GUIDE.md` 和 `textgrad/tasks/__init__.py`，当前主训练入口支持：

| 数据集 / 任务类型 | 说明 |
| --- | --- |
| `BBH_*` | BBH 子任务，代码当前至少覆盖 `object_counting`、`word_sorting`，并可按 `BBH_子任务名` 路由 |
| `GSM8K_DSPy` | GSM8K 的 DSPy 风格切分版本 |
| `prollama` | ProLLaMA 蛋白质超家族分类任务 |
| `livebench_math` | LiveBench Math 总入口 |
| `livebench_math__AMPS_Hard` | LiveBench Math 子任务 |
| `livebench_math__math_comp` | LiveBench Math 子任务 |
| `livebench_math__olympiad` | LiveBench Math 子任务 |
| `livebench_reasoning` | LiveBench Reasoning 总入口 |
| `livebench_reasoning__web_of_lies_v2` | LiveBench Reasoning 子任务 |
| `livebench_reasoning__zebra_puzzle` | LiveBench Reasoning 子任务 |
| `livebench_reasoning__spatial` | LiveBench Reasoning 子任务 |

## 4.2 实例级 / test-time 优化接口支持的数据集

| 数据集 / 任务类型 | 说明 |
| --- | --- |
| `MMLU_*` | 例如 `MMLU_machine_learning`、`MMLU_college_physics` |
| `GPQA_*` | 例如 `GPQA_diamond` |
| `LeetCodeHardEval` | LeetCode Hard 评测集 |

## 4.3 多模态接口支持的数据集

| 数据集 | 说明 |
| --- | --- |
| `mathvista` | 多模态数学推理 |
| `scienceqa` | 多模态科学问答 |

## 4.4 当前仓库内“实际落地数据文件”的情况

当前项目根目录下并没有随仓库一起提交的大型数据集文件；现状是：

| 数据集 | 当前落地方式 |
| --- | --- |
| BBH | 运行时自动从 GitHub 下载并缓存到 `platformdirs.user_cache_dir("textgrad")` |
| GSM8K | 运行时从 Hugging Face 下载 |
| LiveBench Math / Reasoning | 运行时从 Hugging Face 下载 |
| LeetCodeHardEval | 运行时自动下载 |
| ProLLaMA | 依赖仓库外部路径 `../data/ProLLaMA/raw` |
| MathVista / ScienceQA | 运行时从 Hugging Face / 本地缓存读取 |

当前仓库 `resources/` 目录里只有论文框架图 `FedTextGrad_Framework.png`，不包含实际训练数据。

## 4.5 一个需要注意的代码细节

`textgrad/tasks/__init__.py` 里的 `AVAILABLE_DATASETS` 常量目前只显式列出：

- `BBH_object_counting`
- `BBH_word_sorting`
- `GSM8K_DSPy`

但真正的 `load_task(...)` 路由逻辑支持的数据集明显更多，上面 4.1 到 4.3 的列表应以路由逻辑和项目说明文档为准。

## 5. 一页结论

- 论文主实验数据集：`BBH(Object Counting, Multi-step Arithmetic)` + `GSM8K`。
- 论文附录扩展数据集：`LiveBench(Spatial, Web of Lies V2, Zebra Puzzle, AMPS-Hard)`。
- 论文核心模型：主线明确出现 `LLaMA-3.1-8B`，并比较了 `GPT-4 / GPT-4o / GPT-3.5 / DeepSeek-R1-Distill-Llama-70B / LLaMA-3.1-405B / Llama 3 / Llama 3.1 / Qwen 2` 等模型或模型族。
- 论文关键超参数：`3` 个客户端，`B ∈ {1,3,10}`，`E ∈ {3,5,10}`，聚合方法为 `concat / summarization / UID`，并带有“更新后同 batch 回验，不升则回退”的 prompt 接受机制。
- 当前项目代码支持的数据集范围明显比论文主实验更大，除 `BBH` 与 `GSM8K` 外，还支持 `ProLLaMA`、`LiveBench`、`MMLU`、`GPQA`、`LeetCodeHardEval`、`MathVista`、`ScienceQA`；但这些数据多数不是随仓库直接提交，而是运行时下载或依赖外部路径。
