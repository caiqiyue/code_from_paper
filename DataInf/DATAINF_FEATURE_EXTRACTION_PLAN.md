# DataInf 功能抽取技术方案

## 1. 目标与结论

依据 [docs/THESIS_PLATFORM_PLAN.md](../docs/THESIS_PLATFORM_PLAN.md) 中对 `DataInf` 的定位，`DataInf` 在新平台里只应被封装为一个 `scorer`，职责是“基于效用驱动的坏样本打分器”，而不是整套独立实验流程。

平台文档已经给出 3 个硬约束：

1. `DataInf` 只作为 scorer 接入，不直接照搬其 CLI、实验脚手架和数据集生成流程。
2. scorer 统一接口应为 `score(samples, client_ctx) -> list[ScoredSample]`。
3. `datainf` 方法配置的核心字段为 `score_direction`、`target_module`、`lambda_const_param`。

结合 `DataInf/src` 全部源码模块审阅结果，结论如下：

1. 新项目必须抽取的核心能力只有两类：
   - LoRA 目标参数的逐样本梯度提取。
   - 基于 DataInf 闭式近似的 influence/HVP 计算与坏样本打分。
2. 新项目可选复用的能力有两类：
   - 用于离线准备 LoRA adapter 的 SFT 训练脚本。
   - 用于 smoke test 的合成数据生成脚本。
3. 新项目明确不应抽取的内容包括：
   - `launcher.py`、`simulator.py`、`run_experiment.py`、`configs.py` 这一套原仓库实验驱动逻辑。
   - `train_text_to_image_lora.py` 扩散模型训练脚本。
   - notebooks、README 中的演示流程。

一句话总结：

`DataInf` 在新平台里真正要保留的是 `influence.py + lora_model.py` 的“梯度 -> influence 分数”主链路；其余模块大多只保留参考价值，不进入平台核心。

## 2. 平台侧约束

从 `docs/THESIS_PLATFORM_PLAN.md` 可归纳出 DataInf 适配器的目标边界：

1. DataInf 位于 `adapters/scorers/`，输出统一为 `sample_id -> score`。
2. 平台主流程是按阶段组装的：
   `generator -> scorer -> bad sample selector -> retriever -> critic -> aggregator`
3. 统一样本 schema 已固定，输入样本不再允许依赖 DataInf 自己的私有数据格式。
4. 第一版正式路线以 `openreview`、`bioarxiv` 等文本任务为主，示例配置中 `task_type` 为 `instruction_tuning`。
5. scorer 默认参数中已经预留了：
   - `name: datainf`
   - `score_direction: larger_is_worse`
   - `target_module: lora_or_last_layer`
   - `lambda_const_param: 10`

这意味着新项目需要的是“可插拔的文本 scorer 适配器”，不是“原 DataInf 仓库再跑一遍”。

## 3. DataInf 源码模块审阅结论

### 3.1 模块总表

| 模块 | 原始职责 | 新项目处理方式 | 结论 |
| --- | --- | --- | --- |
| `src/influence.py` | influence/HVP 计算核心 | 直接抽取核心算法，重写 I/O 与接口 | 必须抽取 |
| `src/lora_model.py` | LoRA 模型构建、训练、逐样本梯度提取 | 抽取梯度提取逻辑，训练与路径加载需解耦 | 必须抽取 |
| `src/dataloader.py` | GLUE 数据加载、tokenize、噪声注入 | 只保留“DataLoader 组织模式”，不直接复用 | 部分借鉴 |
| `src/sft_trainer.py` | LLM SFT/LoRA 训练脚本 | 仅作为离线 adapter 准备工具 | 可选复用 |
| `src/generate_sentence-math_datasets.py` | 合成文本/数学数据集 | 仅作为 smoke test 数据工具 | 可选复用 |
| `src/configs.py` | 原仓库 CLI 实验配置工厂 | 不进入新平台，由 YAML 配置替代 | 不抽取 |
| `src/run_experiment.py` | 训练 + influence 计算总控 | 不抽取，平台已有统一 pipeline | 不抽取 |
| `src/launcher.py` | CLI 入口 | 不抽取 | 不抽取 |
| `src/simulator.py` | 随机种子与简单封装 | 不抽取 | 不抽取 |
| `src/train_text_to_image_lora.py` | Stable Diffusion LoRA 训练 | 与当前论文平台文本主线无关 | 不抽取 |

### 3.2 核心模块一：`src/influence.py`

这是必须抽取的算法核心。

#### 可直接复用的能力

1. `IFEngine.compute_val_grad_avg()`：
   - 将验证集梯度做平均，形成分类任务的 validation anchor。
2. `IFEngine.compute_hvp_proposed()`：
   - 实现了 DataInf 论文中的闭式近似 HVP。
   - `lambda_const` 的层级计算方式已经落在代码里，可直接保留为 scorer 的核心实现。
3. `IFEngine.compute_IF()`：
   - 基于训练样本梯度与 HVP 的内积输出 influence score。
4. `IFEngineGeneration`：
   - 面向生成任务，不是对验证集先求平均，而是对每个验证样本单独计算 influence。
   - 这条链路更符合平台主线中 `instruction_tuning` 任务。

#### 新项目中的保留建议

1. 保留 `proposed` 方法作为默认主实现。
2. `identity` 可以保留为低成本基线或调试模式。
3. `LiSSA`、`accurate` 不建议进入第一版平台默认路径：
   - `accurate` 依赖显式矩阵分解，计算和显存代价高。
   - 原仓库里也在 `low_rank > 4` 时主动关闭 `accurate`。

#### 必须改造的点

1. 去除 `save_result()` 这种直接向当前目录写 `results_<run_id>.pkl` 的行为。
2. 不再以原始整数下标作为唯一标识，而要映射回统一 schema 的 `sample_id`。
3. scorer 层要只返回结构化分数，不负责实验文件落盘。
4. 要允许通过配置决定使用：
   - 分类模式 `IFEngine`
   - 生成模式 `IFEngineGeneration`

### 3.3 核心模块二：`src/lora_model.py`

这是必须抽取的“梯度采集层”。

#### `LORAEngine`

原职责是面向 GLUE 分类任务：

1. `build_LORA_model()`：
   - 基于 `AutoModelForSequenceClassification` 构建 LoRA 分类模型。
2. `train_LORA_model()`：
   - 在本地 train/eval dataloader 上完成训练。
3. `compute_gradient()`：
   - 对 train/validation 样本逐条做 forward/backward。
   - 提取 `lora_A`、`lora_B`，同时保留 `modules_to_save.default.out_proj.weight`。

#### `LORAEngineGeneration`

原职责是面向 LLaMA 类生成任务：

1. `load_pretrained_network()`：
   - 加载 base model + LoRA adapter。
2. `load_datasets()`：
   - 从固定目录读取 `datasets/<dataset_name>_train.hf` 与 `*_test.hf`。
3. `create_tokenized_datasets()`：
   - 构造 tokenized dataset 与 `collate_fn`。
4. `compute_gradient()`：
   - 对每个 train/validation 样本做逐条梯度提取。

#### 新项目必须抽取的部分

1. 逐样本梯度计算框架。
2. LoRA 参数过滤逻辑：
   - `lora_A`
   - `lora_B`
   - 可选的最后层参数，如分类路径中的 `out_proj.weight`
3. 训练样本梯度字典和验证样本梯度字典的统一组织格式。

#### 新项目必须改造的点

1. 去掉硬编码路径：
   - 原代码把 adapter 路径写死为 `models/{dataset_name}_13bf`。
   - 新平台必须从 `client_ctx` 或方法配置中读取模型路径。
2. 去掉硬编码数据路径：
   - 原代码默认读取 `project_path/datasets/...`。
   - 新平台必须改为接收上游统一 schema 样本列表或 Dataset 对象。
3. 把“训练”和“梯度提取”分离：
   - scorer 不一定总负责从头训练 LoRA。
   - 更合理的方式是：平台先给 scorer 提供“当前客户端模型快照”，scorer 只负责 target module 梯度提取。
4. 重新定义 `target_module`：
   - 原分类路径既抓 LoRA，也抓输出层。
   - 原生成路径只抓 LoRA。
   - 新平台需要将其变成显式配置项，例如：
     - `lora_only`
     - `lora_or_last_layer`
5. 修正生成路径的 tokenization 假设：
   - 当前 `LORAEngineGeneration.create_tokenized_datasets()` 实际 tokenize 的是 `prompt`，代码里自己也注释了 `text should be more appropriate`。
   - 若平台要评估完整指令样本质量，应该使用统一 schema 中的完整训练文本，而不是只算 prompt 的 loss。

### 3.4 数据接口模块：`src/dataloader.py`

这个文件不应该原样抽取，但值得保留其组织模式。

#### 可以借鉴的点

1. “dataset -> tokenize -> rename label -> collate_fn -> dataloader” 这一条流水线组织方式。
2. 训练集与验证集分别构造 dataloader，再对每条样本做梯度提取。

#### 不应直接复用的点

1. 它只支持 GLUE 分类任务。
2. 它内置了标签翻转噪声注入逻辑，属于原论文 mislabeled detection 设定，不符合新平台主任务。
3. 它把数据集采样上限写死为 `train<=4500`、`validation<=500`，这不适合平台统一配置。

#### 新项目中的处理建议

1. 不保留 `flip_label()` 与 `load_noisy_dataset_by_task()`。
2. 将其抽象为平台自己的 `schema -> task specific batch` 转换器：
   - classification scorer tokenizer
   - instruction tuning scorer tokenizer
3. DataInf scorer 只消费已经分好的本地 train/validation 样本。

### 3.5 实验脚手架模块：`src/configs.py`、`src/run_experiment.py`、`src/launcher.py`、`src/simulator.py`

这 4 个文件共同构成了原始仓库的“单仓库实验驱动层”，不应进入新平台核心。

#### 原始作用

1. `configs.py`
   - 生成 GLUE 任务实验配置。
   - 管理 `noise_ratio`、`low_rank`、`N_repeat` 等原实验参数。
2. `run_experiment.py`
   - 串起 dataloader、LoRA 训练、梯度提取、influence 计算、结果保存。
3. `launcher.py`
   - 提供 `python src/launcher.py run --exp_id=...` 入口。
4. `simulator.py`
   - 只是设置随机种子后调用 `run_experiment_core()`。

#### 在新项目里的处理结论

1. 不抽取代码。
2. 只借鉴其“执行顺序”，即：
   - 准备样本
   - 准备模型
   - 提取梯度
   - 计算 influence
   - 输出排序结果
3. 参数迁移到平台 YAML 中维护，不再使用 `eval(config_xxx())` 这类入口方式。

### 3.6 可选工具模块：`src/sft_trainer.py`

这个文件不属于 scorer 核心，但对于文本生成版 DataInf 很有价值。

#### 适合复用的场景

1. 新平台还没有现成的本地 LoRA 训练器。
2. 需要预先训练一份 LoRA adapter，再用 DataInf 只做梯度和打分。
3. 需要为 smoke test 或离线对照实验快速产出 adapter。

#### 不适合直接塞进 scorer 的原因

1. 它是完整训练脚本，不是一个可复用函数库。
2. 参数解析、模型加载、trainer 构建都直接在文件顶层执行。
3. 它的职责更像“离线模型准备工具”，而不是 runtime scorer 逻辑。

#### 结论

保留为可选辅助工具，不纳入第一版 `datainf_scorer` 核心实现。

### 3.7 可选 smoke test 模块：`src/generate_sentence-math_datasets.py`

这个文件也不属于平台生产主链路，但适合作为本地验证工具。

#### 可复用价值

1. 能快速生成小规模文本生成任务数据。
2. 便于单机验证：
   - LoRA 训练是否可跑通。
   - 逐样本梯度提取是否可跑通。
   - influence 排名输出是否符合预期。

#### 结论

保留为测试辅助脚本，不进入新平台 scorer 主链路。

### 3.8 明确排除模块：`src/train_text_to_image_lora.py`

这是 Stable Diffusion 的 LoRA 训练脚本。

当前 `THESIS_PLATFORM_PLAN.md` 的主路线是文本生成、样本筛选、文本 critique 与聚合，第一版没有图像扩散实验要求，因此该模块不在本次抽取范围。

结论：不抽取。

## 4. 新项目真正需要的 DataInf 功能

### 4.1 P0 必须落地的功能

| 功能 | 作用 | 对应原模块 |
| --- | --- | --- |
| 逐样本训练梯度提取 | 为每个候选样本建立 train gradient | `src/lora_model.py` |
| 逐样本验证梯度提取 | 构造 validation anchor 或 validation-wise influence | `src/lora_model.py` |
| DataInf 闭式近似 HVP | 用 `lambda_const_param` 计算近似逆 Hessian 乘积 | `src/influence.py` |
| influence 分数计算 | 产出 `sample_id -> score` 的坏样本分数 | `src/influence.py` |
| score direction 统一 | 和平台配置对齐，默认 `larger_is_worse` | `src/influence.py` + 平台适配层 |
| target module 过滤 | 按配置提取 `lora_only` 或 `lora_or_last_layer` | `src/lora_model.py` |

### 4.2 P1 建议保留的功能

| 功能 | 作用 | 对应原模块 |
| --- | --- | --- |
| `identity` HVP 基线 | 低成本调试或对照实验 | `src/influence.py` |
| classification 路径 | 供平台 smoke test 或小模型对照 | `src/lora_model.py`, `src/dataloader.py` |
| 离线 LoRA 训练脚本 | 提前准备 adapter | `src/sft_trainer.py` |
| 小型合成数据生成 | 本地联调与单元测试 | `src/generate_sentence-math_datasets.py` |

### 4.3 P2 可以先不做的功能

| 功能 | 原模块 | 暂不纳入原因 |
| --- | --- | --- |
| `accurate` HVP | `src/influence.py` | 显存和时间代价高 |
| `LiSSA` HVP | `src/influence.py` | 不是平台默认主线 |
| 扩散模型 LoRA 训练 | `src/train_text_to_image_lora.py` | 与当前文本平台主路线无关 |
| 原仓库 CLI 和实验配置 | `src/launcher.py`, `src/configs.py` | 平台已有统一 YAML 与 pipeline |

## 5. 功能到模块的抽取映射

### 5.1 推荐抽取映射表

| 新平台能力 | 需要抽取的原模块 | 抽取粒度 | 说明 |
| --- | --- | --- | --- |
| `DataInfScorer.score()` 主入口 | `src/influence.py`, `src/lora_model.py` | 核心重构 | 用统一 schema 包一层 scorer 适配器 |
| `InfluenceCore.compute_scores()` | `src/influence.py` | 算法级抽取 | 保留 `proposed` 为默认 |
| `GradientExtractor.compute_train_val_grads()` | `src/lora_model.py` | 逻辑级抽取 | 统一 train/validation 梯度格式 |
| `TargetParamSelector` | `src/lora_model.py` | 局部抽取 | 用配置控制 LoRA/last-layer 过滤 |
| `BatchBuilder` | `src/dataloader.py` | 思路借鉴 | 不直接抄 GLUE loader，重写为 schema 版 |
| `OfflineLoraTrainer` | `src/sft_trainer.py` | 可选包装 | 仅在平台缺少训练器时引入 |
| `SmokeDataBuilder` | `src/generate_sentence-math_datasets.py` | 可选保留 | 只用于测试 |

### 5.2 推荐的新平台文件拆分

建议不要把 DataInf 原文件原样搬入平台，而是拆成 4 个清晰部件：

1. `thesis_platform/adapters/scorers/datainf_scorer.py`
   - 实现统一接口 `score(samples, client_ctx) -> list[ScoredSample]`
2. `thesis_platform/adapters/scorers/datainf_core.py`
   - 放 `compute_hvp_proposed()`、`compute_if_scores()` 等纯算法逻辑
3. `thesis_platform/adapters/scorers/datainf_gradient.py`
   - 放逐样本梯度提取、target module 过滤、train/val dataloader 构建
4. `thesis_platform/configs/methods/scorer/datainf.yaml`
   - 放方法默认参数，如 `lambda_const_param`、`target_module`、`score_direction`

这样做的原因是：

1. 算法逻辑和模型/数据逻辑分离，后续替换 backbone 更容易。
2. 生成任务与分类任务可以复用同一套 influence 核心。
3. 平台后续对接 `GRADMM`、`IRA` 时可以共用 scorer 外壳。

## 6. 新平台里的 DataInf scorer 设计建议

### 6.1 输入

`score(samples, client_ctx)` 中建议至少包含以下信息：

1. `samples`
   - 当前客户端待打分样本列表。
   - 每条都带统一 schema 中的 `sample_id`、`instruction`、`response`、`text` 等字段。
2. `client_ctx`
   - 当前客户端模型或 adapter 路径。
   - 当前轮次的本地 validation 集。
   - tokenizer/model backbone 信息。
   - `task_type`
   - `target_module`
   - `lambda_const_param`
   - `device`

### 6.2 输出

建议统一输出：

```python
[
  {
    "sample_id": "...",
    "score": 0.123,
    "score_name": "datainf_proposed",
    "score_direction": "larger_is_worse",
    "meta": {
      "target_module": "lora_or_last_layer",
      "lambda_const_param": 10,
      "n_train": 100,
      "n_val": 20
    }
  }
]
```

### 6.3 核心执行流程

建议新平台中的 DataInf scorer 流程固定为：

1. 从统一 schema 样本构造 task-specific batch。
2. 加载客户端当前模型快照或 LoRA adapter。
3. 对待打分样本提取 train gradients。
4. 对本地 validation 样本提取 val gradients。
5. 运行 `compute_hvp_proposed(lambda_const_param)`。
6. 计算 influence 分数。
7. 按 `score_direction` 统一方向。
8. 返回排序后的 `ScoredSample` 列表，供上游 `top_k_bad` 选择器使用。

## 7. 必须处理的实现风险

### 7.1 任务形态风险

原仓库同时支持：

1. 分类任务：`IFEngine` + `LORAEngine`
2. 生成任务：`IFEngineGeneration` + `LORAEngineGeneration`

而平台主线是文本生成/指令微调，所以第一版应优先实现“生成任务 scorer”，分类路径只保留为 smoke test 或对照。

### 7.2 数据格式风险

原仓库数据接口高度耦合：

1. 分类路径绑定 GLUE。
2. 生成路径绑定 `prompt/text/answer/...` 这种特定列名。
3. 生成路径还把 adapter 与 dataset 目录命名写死。

如果不提前做 schema 适配层，后面会反复在 DataInf、GRADMM、IRA 三套 scorer 上重复改数据接口。

### 7.3 分数语义风险

原代码直接输出 `-if_tmp_value`，但平台层已经把“分数方向”单独参数化为 `score_direction`。因此：

1. scorer 内部不要把“正分一定代表坏样本”写死。
2. 必须统一交给配置层决定“越大越差”还是“越小越差”。

### 7.4 计算成本风险

原始 DataInf 除 `proposed` 外还实现了 `LiSSA` 和 `accurate`。如果第一版就把全部模式都接入，会明显增加：

1. 显存占用
2. 客户端打分时间
3. 平台调试复杂度

建议第一版只开放：

1. `proposed`
2. 可选 `identity`

## 8. 最终抽取清单

### 8.1 必须抽取

1. `src/influence.py`
   - 重点抽取 `IFEngineGeneration` 与 `compute_hvp_proposed()/compute_IF()`
   - 分类版 `IFEngine` 可作为兼容路径保留
2. `src/lora_model.py`
   - 重点抽取逐样本梯度提取逻辑与 target parameter 过滤逻辑
   - 需要彻底去掉硬编码模型路径和数据路径

### 8.2 可选抽取

1. `src/sft_trainer.py`
   - 作为离线 LoRA adapter 准备工具
2. `src/generate_sentence-math_datasets.py`
   - 作为 smoke test 数据工具
3. `src/dataloader.py`
   - 只借鉴 batch 构建思路，不直接复用

### 8.3 不抽取

1. `src/configs.py`
2. `src/run_experiment.py`
3. `src/launcher.py`
4. `src/simulator.py`
5. `src/train_text_to_image_lora.py`

## 9. 推荐落地顺序

1. 先做 `instruction_tuning` 版 `datainf_scorer`，只保留 `proposed`。
2. 再补 `target_module` 配置化，支持 `lora_only` 与 `lora_or_last_layer`。
3. 再补 classification smoke test 兼容路径。
4. 最后再决定是否保留 `identity` 基线和离线 `sft_trainer` 包装。

## 10. 最终结论

面向新的创新项目，`DataInf` 需要抽取的不是原仓库的大部分文件，而是一个非常明确的最小核心：

1. `src/influence.py` 中的 influence 计算内核。
2. `src/lora_model.py` 中的逐样本梯度提取逻辑。
3. `src/dataloader.py` 中可借鉴的数据批处理思路。

其中：

1. `influence.py + lora_model.py` 是平台版 `datainf_scorer` 的核心依赖。
2. `sft_trainer.py + generate_sentence-math_datasets.py` 只作为辅助工具。
3. `configs.py + launcher.py + simulator.py + run_experiment.py + train_text_to_image_lora.py` 不进入新平台主链路。

因此，新的平台实现应以“重构式抽取”而不是“目录搬运式复用”为原则。
