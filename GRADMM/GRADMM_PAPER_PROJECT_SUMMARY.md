# GRADMM 论文与当前项目数据集/模型/超参数汇总

## 1. 结论先看

- GRADMM 论文主实验围绕 5 个情感/情绪分类数据集展开：`SST-2`、`Rotten Tomatoes`、`Twitter Emotion`、`IMDb`、`RT-Polarity`。
- 论文主实验的核心公共生成模型是 `Phi-1.5`；论文主结果里把生成出的 synthetic data 用于训练 `OPT-125M`、`OPT-350M`、`Phi-1.5`，附录还补充了 `OPT-1.3B` 和 `Llama-3.2-1B` 的扩展实验。
- 论文正文里明确写出的关键实验超参数包括：低资源训练样本数 `10` 和 `50`，synthetic/target 样本比例 `{0.5, 1, 2, 4, 8}`，ADMM 更新步数 `30`，学习率 `8e-3`，`rho=0.5`。
- 当前仓库代码主流程支持的核心真实数据集也是这 5 个；但项目根目录下 **本地实际已经存在** 的数据只有 `IMDB` 和 `RT-Polarity` 两套 JSONL 文件，另外 3 套依赖 Hugging Face 在线下载。

## 2. 论文中使用了哪些数据集

论文主实验聚焦于 5 个二分类情感/情绪任务，代码和项目说明文档也与这 5 个任务对齐：

| 数据集 | 任务类型 | 代码中的来源/映射 | 说明 |
| --- | --- | --- | --- |
| `SST-2` | 电影评论情感二分类 | `glue/sst2` | 论文主实验数据集之一 |
| `Rotten Tomatoes` | 电影评论情感二分类 | `rotten_tomatoes` | 论文主实验数据集之一 |
| `Twitter Emotion` | 推文情绪二分类 | `dair-ai/emotion` 中仅保留 `label in [0,1]` | 在当前代码里对应 `sadness` / `joy` |
| `IMDb` | 电影评论情感二分类 | 本地 `data/imdb/*.jsonl` | 仓库自带本地 JSONL |
| `RT-Polarity` | 句子情感二分类 | 本地 `data/rtpolarity/*.jsonl` | 仓库自带本地 JSONL |

补充说明：

- `GRADMM_PROJECT_GUIDE.md` 明确把以上 5 个任务列为 synthetic-data 主流程常用任务。
- `gradmm/args_factory.py` 和 `gradmm/generate.py` 的数据集选项也与上表一致。
- `addax/tasks.py` 为这 5 个真实数据集及其对应 synthetic 版本都提供了任务适配器。

## 3. 论文中使用了哪些模型

### 3.1 生成 synthetic data 时的公共模型

| 模型 | 用途 | 备注 |
| --- | --- | --- |
| `Phi-1.5` | 论文主实验中的公共生成模型 | 论文写明 “most experiments” 使用它；仓库中 `phi -> microsoft/phi-1_5` |
| `OPT-1.3B` | 附录中的扩展公共模型 | 用于补充实验，不是主文主结果的默认模型 |
| `Llama-3.2-1B` | 附录中的扩展公共模型 | 用于补充实验，不是主文主结果的默认模型 |

### 3.2 用 synthetic data 训练/评估的下游模型

| 模型 | 角色 | 备注 |
| --- | --- | --- |
| `OPT-125M` | 下游训练/评估模型 | 论文主结果中明确出现 |
| `OPT-350M` | 下游训练/评估模型 | 论文主结果中明确出现 |
| `Phi-1.5` | 下游训练/评估模型 | 论文主结果中明确出现 |

补充说明：

- 附录中还出现了 `OPT-1.3B` 和 `Llama-3.2-1B` 的扩展实验。
- 但基于当前能直接核对到的论文文字，我更有把握把它们归类为“附录扩展实验中涉及的模型”，不把它们直接并入主结果表中的下游训练模型。

### 3.3 仓库里的模型映射

当前仓库生成阶段只在代码里显式提供了一个模型别名映射：

| 代码别名 | Hugging Face 模型名 | 位置 |
| --- | --- | --- |
| `phi` | `microsoft/phi-1_5` | `gradmm/generate.py`、`gradmm/filtering.py` |

## 4. 论文中使用了哪些关键超参数

下面先列 **论文正文/附录文字中明确写出的超参数**，再列 **仓库脚本里可见的复现实验默认值**。这两类信息要区分开看。

### 4.1 论文正文/附录明确写出的关键超参数

| 超参数 | 取值 | 说明 |
| --- | --- | --- |
| 低资源训练样本数 | `10`、`50` | 论文明确说所有实验在 low-resource setting 下进行 |
| synthetic / target 样本比例 | `{0.5, 1, 2, 4, 8}` | synthetic 数据规模相对于目标训练集大小的比例 |
| 优化算法 | `ADMM` | 论文核心方法 |
| ADMM 更新步数 | `30` | 论文明确写出 |
| 学习率 | `8e-3` | 论文明确写出 |
| `rho` | `0.5` | 论文明确写出 |
| 任务 prompt | 情感任务用 `"It was"` 风格，Twitter Emotion 用 `"Does the tweet express joy or sadness?"` | 论文文字说明与代码模板一致 |

### 4.2 仓库脚本/代码里的相关默认值

这些值对复现实验有帮助，但不应全部直接等同于“论文正文明确报告的超参数”。

#### 生成阶段默认值

来源：`gradmm/args_factory.py`、`gradmm/scripts/admm.sh`、`gradmm/scripts/admm_dp.sh`

| 参数 | 默认值 / 脚本值 | 备注 |
| --- | --- | --- |
| `rng_seed` | `42` | 全局随机种子 |
| `n_steps` | `30` | 与论文主文一致 |
| `lr` | `0.008` | 与论文主文一致 |
| `admm_rho` | `0.7`（代码默认），`0.5`（论文主文），脚本 sweep `{0.01, 0.1, 0.5, 1, 5}` | 代码默认值和论文主文不完全相同 |
| `admm_inner_steps` | `10`（代码默认），`50`（脚本） | 脚本更像实验配置 |
| `n_gen_samples` | `1000`（代码默认），`100`（脚本） | 代码默认与脚本示例不同 |
| `subset_size` | `100`（代码默认），`50`（脚本） | 脚本更贴近 README 示例 |
| `n_gen` | `10` | 每轮正负类各一次生成 |
| `gen_bs` | `1`（代码默认），`10`（脚本） | 脚本做了并行生成 |
| `topk` | `50`（代码默认），`200`（脚本） | 用于 token 投影 |
| `grad_clip` | `None`（代码默认），`1.0`（脚本） | 脚本中显式开启 |
| `use_auto_gen_tokens` | `false`（代码默认），`true`（脚本） | 脚本根据真实样本长度自动设 token 数 |
| `last_layer_gradient` | `true` | 只对最后一层做真实梯度匹配 |

#### DP 生成脚本默认值

来源：`gradmm/scripts/admm_dp.sh`

| 参数 | 值 |
| --- | --- |
| `dp_epsilon` | `0.05` |
| `dp_delta` | `1e-4` |
| `dp_c` | `1.0` |

#### 微调阶段脚本默认值

来源：`addax/scripts/query_ft.sh`

| 参数 | 值 |
| --- | --- |
| `MODEL` | `microsoft/phi-1_5` |
| `num_train` | `100` |
| `max_steps` | `200` |
| `per_device_train_batch_size` | `16` |
| `gradient_accumulation_steps` | `1` |
| `num_eval_to_keep` | `100` |
| 学习率候选 | `7e-6`、`1e-5`、`1.5e-5` |

## 5. 当前项目中有哪些数据集

这里分成三层来看：`主生成流程支持`、`过滤流程支持`、`项目本地实际已有文件`。

### 5.1 `gradmm/generate.py` 主生成流程支持的数据集

| 数据集标识 | 类型 | 来源 |
| --- | --- | --- |
| `sst2` | 真实数据集 | Hugging Face `glue/sst2` |
| `rotten_tomatoes` | 真实数据集 | Hugging Face `rotten_tomatoes` |
| `TwitterEmotion` | 真实数据集 | Hugging Face `dair-ai/emotion` 的二分类子集 |
| `imdb` | 真实数据集 | 本地 `data/imdb/*.jsonl` |
| `rtpolarity` | 真实数据集 | 本地 `data/rtpolarity/*.jsonl` |

### 5.2 `gradmm/filtering.py` 当前过滤流程支持的数据集

| 数据集标识 | 说明 |
| --- | --- |
| `sst2` | 支持 |
| `rotten_tomatoes` | 支持 |
| `TwitterEmotion` | 支持 |

说明：`filtering.py` 当前没有把 `imdb` 和 `rtpolarity` 放进 CLI `choices`，因此过滤阶段的支持范围比生成阶段更小。

### 5.3 `addax/tasks.py` 当前支持的真实任务/数据集

`addax/tasks.py` 中当前可见的真实任务适配器包括：

- `RottenTomatoes`
- `IMDB`
- `RTPolarity`
- `SST2`
- `TwitterEmotion`
- `CoLA`
- `COPA`
- `BoolQ`
- `MultiRC`
- `CB`
- `WiC`
- `WSC`
- `ReCoRD`
- `RTE`
- `SQuAD`
- `DROP`

同时还支持以下 synthetic 任务适配器：

- `SynRottenTomatoes`
- `SynIMDB`
- `SynRTPolarity`
- `SynSST2`
- `SynTwitterEmotion`

### 5.4 项目根目录下当前本地实际存在的数据文件

当前仓库 `data/` 下实际存在的本地 JSONL 数据如下：

| 文件 | 记录数 |
| --- | --- |
| `data/imdb/train_len256.jsonl` | `408` |
| `data/imdb/validation_len256.jsonl` | `462` |
| `data/rtpolarity/train.jsonl` | `1000` |
| `data/rtpolarity/validation.jsonl` | `1000` |

结论：

- **本地实际随仓库提供的数据集**：只有 `IMDB` 和 `RT-Polarity`。
- **代码支持但未直接随仓库落地的数据集**：`SST-2`、`Rotten Tomatoes`、`TwitterEmotion`，这些依赖 Hugging Face `datasets` 在线拉取。

## 6. 论文使用情况与当前项目现状的对应关系

| 维度 | 论文 | 当前项目 |
| --- | --- | --- |
| 主实验数据集 | 5 个情感/情绪二分类任务 | 主生成代码同样支持这 5 个任务 |
| 主生成模型 | 主要是 `Phi-1.5` | 代码中显式内建 `phi -> microsoft/phi-1_5` |
| 扩展公共模型 | `OPT-1.3B`、`Llama-3.2-1B` | 仓库当前没有对应的显式模型映射别名 |
| 过滤流程 | 论文有 filtering 阶段 | 代码里 filtering 目前只覆盖 3 个 HF 数据集 |
| 本地数据文件 | 论文并不要求都本地打包 | 仓库仅打包了 `IMDB`、`RT-Polarity` |

## 7. 建议如何理解这份仓库

- 如果你关注的是“论文主实验到底用了什么”，优先看本文件第 2、3、4 节。
- 如果你关注的是“当前仓库马上能跑什么”，优先看第 5 节。
- 如果你关注的是“论文和代码是否完全一一对应”，答案是：**核心 5 个任务是一致的，但 filtering 支持范围和本地落地数据范围都比论文全景更窄。**

## 8. 本次整理使用的本地依据

- `GRADMM.pdf`
- `GRADMM_PROJECT_GUIDE.md`
- `GRADMM_FUNCTION_EXTRACTION_PLAN.md`
- `README.md`
- `gradmm/args_factory.py`
- `gradmm/generate.py`
- `gradmm/filtering.py`
- `gradmm/scripts/admm.sh`
- `gradmm/scripts/admm_dp.sh`
- `addax/tasks.py`
- `addax/templates.py`
- `addax/scripts/query_ft.sh`
- `data/`
