# FedTextGrad 功能抽取技术方案

## 1. 目标与总判断

依据 [docs/THESIS_PLATFORM_PLAN.md](../docs/THESIS_PLATFORM_PLAN.md) 中与 `FedTextGrad` 相关的设计，`FedTextGrad` 在新平台中的定位不是一个完整联邦训练框架，而是两类可插拔模块的来源：

1. `critic`
   - 负责生成文本化 critique。
   - 统一接口为 `critique(paired_samples, client_ctx) -> list[Critique]`
2. `aggregator`
   - 负责服务端聚合多客户端 critique，输出新的 Prompt 更新。
   - 统一接口为 `aggregate(client_critiques, server_ctx) -> PromptUpdate`

平台文档还明确给出了当前默认路线：

1. `critic` 使用 `fedtextgrad_qwen`
2. `aggregator` 候选为 `concat`、`summarization`、`uid`、`dbscan_uid`、`dbscan_uid_tsgdm`
3. 推荐主线是“FedTextGrad 风格对比批判 + DBSCAN-UID 聚合 + TSGD-M 启发式动量平滑”

结合对 `FedTextGrad` 目录全部模块的审阅，最终结论是：

1. 新平台必须抽取的核心能力只有两类：
   - FedTextGrad 风格的“反馈而非重写”文本批判生成逻辑。
   - 服务端基于 LLM 的 prompt 聚合逻辑，尤其是 `summarization` 和 `UID`。
2. 新平台不应搬运 `FedTextGrad` 的整套训练框架：
   - `main.py`
   - `train_centralized.py`
   - `train_homo_fed.py`
   - `train_hetero_fed.py`
   - `eval.py`
   - `textgrad/tasks/*`
3. 新平台第一版不需要完整引入整个 `textgrad` 计算图，只需要抽取其 prompt 设计思想、LLM backend 抽象，以及少量可复用运行时。
4. 平台文档中提到的 `dbscan_uid`、`dbscan_uid_tsgdm`、`compress_to_n_rules`、`redact_*`，原仓库都没有完整现成实现，必须由新平台自行补齐。

一句话总结：

`FedTextGrad` 在新平台里真正要保留的是“如何生成 critique”和“如何用 LLM 聚合文本反馈”，而不是“如何在 BBH/GSM8K 上做联邦 prompt 优化实验”。

## 2. 平台侧约束

从 `THESIS_PLATFORM_PLAN.md` 可以归纳出 5 个直接约束：

1. `FedTextGrad/` 被定位为“文本化批判生成与服务端 Prompt 聚合基线”。
2. `critic` 配置字段为：
   - `engine`
   - `model_name`
   - `prompt_template`
   - `max_new_tokens`
   - `compress_to_n_rules`
   - `redact_enable`
   - `redact_mode`
3. `aggregator` 配置字段为：
   - `name`
   - `summarizer_model`
   - `prompt_template`
   - `max_aggregate_tokens`
4. 平台里 `critic` 和 `aggregator` 是独立可替换模块，不再与某个训练循环绑死。
5. 平台里的数据对象不是 FedTextGrad 原仓库的 benchmark QA 数据，而是统一 schema 下的：
   - `x_bad`
   - `x_real`
   - `critique`
   - `PromptUpdate`

这意味着：

1. 新平台要抽取的是 `FedTextGrad` 的方法内核，不是它的 benchmark/task 层。
2. 新平台应优先做轻量适配器，而不是把 `textgrad` 全家桶塞进主流程。

## 3. 仓库模块审阅结论

## 3.1 模块分组总表

| 模块组 | 代表文件 | 原始职责 | 新平台处理方式 | 结论 |
| --- | --- | --- | --- | --- |
| 顶层实验入口 | `main.py`, `train_*.py`, `eval.py` | 联邦/集中式 prompt 优化实验 | 只借鉴执行思路，不直接复用 | 不抽取 |
| 聚合模板工具 | `utils/prompt_template.py` | summarization / UID 模板 | 直接抽取并改写为平台 prompt | 必须抽取 |
| 文本复杂度工具 | `utils/prompt_complexity.py` | 熵、困惑度、UID 相关分析 | 可做研究分析，不进主链路 | 可选复用 |
| TextGrad 核心运行时 | `textgrad/variable.py`, `model.py`, `config.py` | 文本变量、LLM 模块、全局 backward engine | 只在需要保真复现时引入 | 部分抽取 |
| TextGrad 自动求导 | `textgrad/autograd/*` | LLM 调用、backward prompt、文本反馈生成 | 只抽取 critique 风格与最小运行逻辑 | 部分抽取 |
| TextGrad 优化器 | `textgrad/optimizer/*` | 将反馈改写成新 prompt | 第一版平台不必直接复用 | 可选复用 |
| LLM 引擎层 | `textgrad/engine/*` | OpenAI/vLLM/Ollama 等 backend 封装 | 如果平台缺统一推理层，可复用 | 可选复用 |
| 任务和数据集层 | `textgrad/tasks/*` | BBH/GSM8K/MMLU/GPQA 等 benchmark 数据加载 | 与新平台数据无关 | 不抽取 |
| 多模态扩展 | `textgrad/tasks/multimodal/*`, `textgrad/autograd/multimodal_ops.py` | 图像问答、多模态文本梯度 | 当前论文主线不需要 | 不抽取 |
| 启动脚本和资源 | `scripts/*`, `resources/*` | shell 脚本与配图 | 不进入平台核心 | 不抽取 |

## 3.2 顶层实验与脚手架模块

### `main.py`

原职责：

1. 解析实验参数。
2. 初始化 Comet。
3. 动态加载 `train_centralized.py`、`train_homo_fed.py`、`train_hetero_fed.py`。

对新平台的意义：

1. 可借鉴参数组织方式。
2. 不应直接复用，因为平台已经有统一 YAML 和统一 pipeline。

结论：不抽取。

### `train_centralized.py`

原职责：

1. 加载任务数据集。
2. 用 `textgrad` 生成 textual feedback。
3. 用 `TextualGradientDescent` 更新 system prompt。
4. 在 val/test 上做回评。

对新平台的意义：

1. 证明了“文本反馈 -> prompt 更新”这条链路可以跑通。
2. 但它是单机集中式 prompt 优化流程，不是平台需要的 `critic` 适配器。

结论：不抽取代码，只借鉴 workflow。

### `train_homo_fed.py`

原职责：

1. 将同一任务切分给多个客户端。
2. 每个客户端独立做本地 textual gradient 更新。

关键判断：

1. 该文件没有实现平台真正需要的服务端聚合器。
2. 其核心仍然是“本地 prompt 训练”，而不是“生成 critique 并上传”。

结论：不抽取。

### `train_hetero_fed.py`

这是顶层最重要的参考文件，但也不应整体搬运。

它包含 3 个对平台最有价值的内容：

1. 多客户端 prompt 更新后的服务端聚合逻辑。
2. `summarization` 聚合模板的实际使用方式。
3. `sum_uid` 聚合模板的实际使用方式。

它也暴露了 3 个关键边界：

1. 原仓库聚合的是“客户端 prompt”，不是“客户端 critique”。
2. 原仓库聚合方法名叫 `sum_uid`，而平台文档统一命名为 `uid`。
3. 原仓库没有 `dbscan_uid` 和 `dbscan_uid_tsgdm`。

结论：

1. 必须抽取它的“服务端聚合思路”。
2. 不抽取其本地训练循环和 benchmark 数据流程。

### `eval.py`

原职责：

1. 并发跑 benchmark 样本评估。
2. 统计 accuracy。
3. 支持验证集回退。

对新平台的意义很弱，因为平台评估指标和样本对象都不同。

结论：不抽取。

## 3.3 `utils/` 模块

### `utils/prompt_template.py`

这是必须抽取的模块。

它只定义了 3 个对象，但都直接有用：

1. `SUMMARIZATION_TEMPLATE`
   - LLM 汇总基线的 server prompt 模板。
2. `UID_TEMPLATE`
   - UID 风格聚合模板。
3. `FORMATTING_INSTRUCTION`
   - 强制最终回答格式。

新平台中的处理建议：

1. `SUMMARIZATION_TEMPLATE` 直接演化为 `prompts/aggregation/summarize_v1.txt`
2. `UID_TEMPLATE` 直接演化为 `prompts/aggregation/uid_v1.txt`
3. `FORMATTING_INSTRUCTION` 不宜原样保留为数学答案格式，而应改造成“输出新 Prompt / 输出规则摘要”的格式约束

结论：必须抽取，但要改写目标输出格式。

### `utils/prompt_complexity.py`

原职责：

1. 计算 entropy、compression rate、TF-IDF、perplexity。
2. 计算信息密度均匀性。

这个模块当前不在训练主链路里。

它对新平台的价值主要是：

1. 可做 critique 冗余率分析。
2. 可做 UID 类聚合的分析指标。
3. 可作为论文消融或分析工具。

结论：可选复用，不进入第一版平台主链路。

## 3.4 `textgrad` 核心运行时

### `textgrad/variable.py`

这是 TextGrad 的核心数据结构。

职责：

1. `Variable` 表示文本变量。
2. 保存 `value`、`gradients`、`gradients_context`、`predecessors`。
3. `backward()` 负责沿计算图回传 textual gradients。

对新平台的意义：

1. 如果要高度保真复现 FedTextGrad 原始 textual gradient 机制，它是必要基础。
2. 但平台的 `critic` 接口只是“输入 paired samples，输出 critique”，不一定需要显式计算图。

结论：

1. 第一版平台不建议强依赖 `Variable`。
2. 仅当要复现“response -> eval -> backward -> feedback”完整原机制时才抽取。

### `textgrad/config.py`

职责：

1. 维护全局 singleton backward engine。
2. `set_backward_engine()` 和 `validate_engine_or_get_default()` 提供统一 backward LLM。

对新平台的意义：

1. 如果平台会采用原生 textgrad 风格 backward，必须抽取。
2. 如果平台直接用显式 `critic_llm.generate(prompt)`，则不需要。

结论：部分抽取。

### `textgrad/model.py`

职责：

1. `BlackboxLLM` 把一个 LLM 封装成可放入 textgrad 图里的模块。
2. 内部使用 `LLMCall`。

对新平台的意义：

1. 如果平台复用完整 textgrad 运行时，它是必要封装。
2. 如果平台直接调用已有 inference backend，则不需要。

结论：部分抽取。

### `textgrad/loss.py`

职责：

1. 定义 `TextLoss`
2. 定义 `MultiFieldEvaluation`
3. 定义 `MultiFieldTokenParsedEvaluation`
4. 定义 `MultiChoiceTestTime`
5. 定义 `ImageQALoss`

这些模块本质上是 benchmark 评测器，不是平台的 critique 适配器。

结论：不进入第一版平台核心。

## 3.5 `textgrad/autograd` 模块

这是最值得抽取“思想而不是整包代码”的部分。

### `textgrad/autograd/llm_backward_prompts.py`

这是必须重点抽取的模板模块。

它提供了 FedTextGrad 最关键的方法风格：

1. `BACKWARD_SYSTEM_PROMPT`
   - 明确要求模型只给“反馈和批判”，不要直接重写变量。
2. `OBJECTIVE_INSTRUCTION_BASE`
   - 强调反馈只围绕目标提升。
3. `EVALUATE_VARIABLE_INSTRUCTION`
   - 强调对某一段文本变量给出可执行改进建议。

对新平台的价值非常高，因为平台的 `critic` 本质上就需要：

1. 输出规则/批判，而不是直接生成新 prompt。
2. 给出“如何修复 x_bad 相对 x_real 暴露的问题”。

结论：必须抽取，作为 `contrastive_critic_v1` 的核心风格来源。

### `textgrad/autograd/llm_ops.py`

职责：

1. `LLMCall` 前向调用 LLM。
2. 在 backward 阶段，构造反馈 prompt，让 backward engine 生成 textual gradient。
3. 支持普通调用、格式化调用、带 in-context examples 调用。

对新平台的意义：

1. 这里藏着 FedTextGrad 原始“文本反馈生成”的实际实现。
2. 但平台的 `critic` 接口不一定要靠计算图回传来做。

推荐处理：

1. 不直接搬 `LLMCall.backward()` 整套机制。
2. 重点抽取其中的 prompt 构造思路：
   - 给定上下文
   - 给定目标
   - 仅输出反馈
   - 不直接输出重写后的最终文本

结论：部分抽取。

### `textgrad/autograd/string_based_ops.py`

职责：

1. 把字符串规则函数包装成可回传 textual feedback 的节点。
2. 用于那些“前向评估是程序规则，不是 LLM”的场景。

对新平台的价值：

1. 如果后续平台要对 critique 做规则检查或 programmatic evaluator，再生成 textual feedback，这个思路有复用价值。
2. 第一版平台不需要。

结论：可选复用。

### `textgrad/autograd/algebra.py`

职责：

1. `Sum` 做简单拼接。
2. `Aggregate` 做“先拼接、再在 backward 中归并反馈”。
3. `_reduce_gradients_mean()` 用 LLM 做反馈摘要。

关键判断：

1. 文件自己就标了 `Aggregate` 是 WIP。
2. 这个 `Aggregate` 不是平台文档里的最终聚合器实现。
3. 在 `train_hetero_fed.py` 里，真正起作用的是：
   - 先 `aggregate(system_prompt_list)` 得到拼接文本
   - 再用 `SUMMARIZATION_TEMPLATE` / `UID_TEMPLATE` 走一次显式 LLM 汇总

因此，新平台不应把 `Aggregate` 当成成熟服务端聚合器直接搬运。

结论：只借鉴“先拼接再总结”的思路，不直接抽代码。

### `textgrad/autograd/functional.py`

职责：

1. 提供 `sum()`、`aggregate()`、`llm_call()`、`formatted_llm_call()` 函数式接口。

结论：

1. `sum()` / `aggregate()` 逻辑可由平台自己重写为简单拼接。
2. `llm_call()` 与 `formatted_llm_call()` 可以只保留思路。

### `textgrad/autograd/reduce_prompts.py`

职责：

1. 定义反馈 reduce 的系统提示和拼接格式。

价值：

1. 可作为“客户端 critique 摘要”或“多条规则压缩”提示词参考。

结论：可选复用。

### 多模态自动求导模块

包括：

1. `textgrad/autograd/multimodal_ops.py`
2. `textgrad/autograd/multimodal_backward_prompts.py`

这些模块和当前文本主线无关。

结论：不抽取。

## 3.6 `textgrad/optimizer` 模块

### `textgrad/optimizer/optimizer.py`

职责：

1. `TextualGradientDescent`
   - 根据 feedback 生成新的文本变量。
2. `TextualGradientDescentwithMomentum`
   - 加入历史反馈/历史版本。

这个模块很有研究价值，但对当前平台并不是第一版必需品。

原因：

1. 平台里的 `critic` 只负责生成 critique。
2. 平台里的 `aggregator` 直接输出 `PromptUpdate` 即可，不一定要经历“变量 -> optimizer -> 新变量”的完整 TextGrad 机制。
3. 平台文档里 `dbscan_uid_tsgdm` 的 “TSGD-M” 是启发式动量模块，不等于直接复用原优化器。

结论：

1. 第一版不必直接抽取。
2. 若后续要做“基于聚合 critique 的 prompt 重写器”，这个模块可以作为 P1/P2 参考来源。

### `textgrad/optimizer/optimizer_prompts.py`

职责：

1. 定义优化器如何把反馈转成新变量。

对新平台的价值：

1. 可作为未来 `prompt_updater` 模块的参考。
2. 不是当前 `critic` / `aggregator` 的核心依赖。

结论：可选复用。

## 3.7 `textgrad/engine` 模块

这一层本质是 LLM backend 适配层。

### 核心模块

1. `textgrad/engine/__init__.py`
   - `get_engine()` 做 engine name 到 provider class 的路由
2. `textgrad/engine/base.py`
   - 定义 `EngineLM` 与 `CachedEngine`
3. `textgrad/engine/textgrad_openai.py`
   - 支持 OpenAI、Ollama-compatible、vLLM API
4. `textgrad/engine/local_model_openai_api.py`
   - 支持传入外部 OpenAI-compatible client
5. `textgrad/engine/textgrad_vllm.py`
   - 本地 vLLM 推理

### Provider 扩展模块

1. `anthropic.py`
2. `gemini.py`
3. `cohere.py`
4. `together.py`
5. `engine_utils.py`

对新平台的判断：

1. 如果平台已有统一 LLM 推理层，这一组不应再重复接入。
2. 如果平台还没有 inference adapter，这一层是最值得复用的 FedTextGrad 代码之一。
3. 对平台文档给出的 `critic.engine=local_vllm` 场景，`textgrad_vllm.py` 和 `textgrad_openai.py` 的实现尤其有参考价值。

结论：可选复用，取决于平台是否已有统一 inference backend。

## 3.8 `textgrad/tasks` 与 benchmark 数据层

这一层包含：

1. `tasks/__init__.py`
2. `tasks/base.py`
3. `big_bench_hard.py`
4. `gsm8k.py`
5. `mmlu.py`
6. `gpqa.py`
7. `leetcode.py`
8. `prollama.py`
9. `livebench.py`
10. `livebenchmath.py`
11. `livebenchreason.py`
12. `tasks/multimodal/*`

这些模块的职责是：

1. 自动下载或读取 benchmark 数据。
2. 构建 task-specific `train/val/test`
3. 构建原 FedTextGrad 论文中的评测器。

但新平台的数据是：

1. `congressional`
2. `openreview`
3. `bioarxiv`
4. 统一 schema 的合成样本与真实样本

因此这整个任务层都与当前创新项目主线不匹配。

结论：不抽取。

## 4. 新项目真正需要的 FedTextGrad 功能

## 4.1 P0 必须抽取

| 功能 | 作用 | 对应原模块 |
| --- | --- | --- |
| FedTextGrad 风格 feedback-only 批判模板 | 生成 critique，而不是直接改写 prompt | `textgrad/autograd/llm_backward_prompts.py` |
| 服务端 `summarization` 聚合模板 | 作为 LLM 汇总基线 | `utils/prompt_template.py`, `train_hetero_fed.py` |
| 服务端 `uid` 聚合模板 | 作为 FedTextGrad 基线聚合器 | `utils/prompt_template.py`, `train_hetero_fed.py` |
| 简单 `concat` 聚合基线 | 直接拼接 critique | `train_hetero_fed.py`, `textgrad/autograd/functional.py` |
| 最小化 LLM 调用抽象 | 为 critic/aggregator 提供后端推理入口 | `textgrad/engine/__init__.py`, `textgrad/engine/base.py`, `textgrad/engine/textgrad_openai.py`, `textgrad/engine/textgrad_vllm.py` |

## 4.2 P1 可选抽取

| 功能 | 作用 | 对应原模块 |
| --- | --- | --- |
| 保真 TextGrad 运行时 | 如果要复现原始文本梯度反向传播 | `textgrad/variable.py`, `textgrad/config.py`, `textgrad/model.py`, `textgrad/autograd/llm_ops.py` |
| 规则函数反向批判 | 如果要接程序化 evaluator | `textgrad/autograd/string_based_ops.py` |
| 历史反馈/动量型 prompt 改写 | 可为未来 prompt_updater 模块提供参考 | `textgrad/optimizer/optimizer.py`, `textgrad/optimizer/optimizer_prompts.py` |
| 文本复杂度/UID 分析 | 用于 critique 冗余与聚合分析 | `utils/prompt_complexity.py` |
| 外部 provider backend | 当平台需接 Anthropic/Gemini/Together 等 | `textgrad/engine/anthropic.py`, `gemini.py`, `cohere.py`, `together.py` |

## 4.3 P2 明确不抽取

| 功能 | 对应原模块 | 不抽取原因 |
| --- | --- | --- |
| 实验入口与 Comet logging | `main.py`, `train_*.py`, `eval.py` | 平台已有统一 pipeline，不复用论文实验脚手架 |
| Benchmark 任务数据集 | `textgrad/tasks/*` | 与平台数据 schema 不一致 |
| 多模态文本梯度 | `textgrad/autograd/multimodal_ops.py`, `tasks/multimodal/*` | 当前文本论文主线不需要 |
| shell 启动脚本 | `scripts/*` | 只是运行示例 |
| 配图和文档资源 | `resources/*`, `README.md` | 不属于平台实现 |

## 5. 功能到模块的抽取映射

## 5.1 推荐抽取映射表

| 新平台能力 | 需要抽取的原模块 | 抽取粒度 | 说明 |
| --- | --- | --- | --- |
| `fedtextgrad_critic.py` | `textgrad/autograd/llm_backward_prompts.py` | 核心重写 | 继承“只给反馈、不直接重写”的 prompt 风格 |
| `fedtextgrad_critic.py` | `textgrad/autograd/llm_ops.py` | 思路借鉴 | 不建议整套计算图搬运，重点借 prompt 构造逻辑 |
| `summarization.py` | `utils/prompt_template.py`, `train_hetero_fed.py` | 直接抽取 + 改写 | 基于 critique 列表做 server summarize |
| `uid.py` | `utils/prompt_template.py`, `train_hetero_fed.py` | 直接抽取 + 改写 | 原仓库 `sum_uid` 对应平台 `uid` |
| `concat.py` | `train_hetero_fed.py` | 简单重写 | 直接拼接 critique 文本 |
| `llm_backend.py` | `textgrad/engine/*` | 可选抽取 | 若平台尚无统一推理 backend，则强烈建议复用 |
| `prompt_updater.py` | `textgrad/optimizer/*` | 可选借鉴 | 仅在未来需要 TextGrad 风格重写器时引入 |

## 5.2 推荐的新平台文件拆分

建议在平台中拆成如下部件，而不是原样复制 `FedTextGrad`：

1. `thesis_platform/adapters/critics/fedtextgrad_critic.py`
   - 输入 `(x_bad, x_real)` 或其结构化 `PairedSample`
   - 输出 `Critique`
2. `thesis_platform/adapters/aggregators/concat.py`
   - 直接拼接客户端 critique
3. `thesis_platform/adapters/aggregators/summarization.py`
   - LLM 汇总基线
4. `thesis_platform/adapters/aggregators/uid.py`
   - FedTextGrad 风格 UID 聚合器
5. `thesis_platform/prompts/critique/contrastive_critic_v1.txt`
   - 由 FedTextGrad 的 backward prompt 风格演化而来
6. `thesis_platform/prompts/aggregation/summarize_v1.txt`
7. `thesis_platform/prompts/aggregation/uid_v1.txt`
8. `thesis_platform/infra/llm_backends/`
   - 如果平台没有现成统一推理层，可引入 `textgrad/engine/*` 的精简版

## 6. 与原仓库的关键差异

这是本次抽取里最重要的部分。

## 6.1 原仓库聚合的是 prompt，不是 critique

`train_hetero_fed.py` 的聚合输入是 `system_prompt_list`，不是客户端上传的 critique。

而平台设计中的聚合输入是：

1. 每客户端上传的 critique 规则
2. 再由服务端输出新的 Prompt 更新

因此：

1. 可以复用原始 `SUMMARIZATION_TEMPLATE` 和 `UID_TEMPLATE` 的思想。
2. 不能把原模板不改地直接用于平台。

## 6.2 原仓库的 textual gradient 来自“回答-评测”链，而不是“坏样本-真实样本”链

原 FedTextGrad 的典型路径是：

1. 模型回答问题
2. evaluator 判断回答是否正确
3. backward engine 根据评测结果给 response/system prompt 生成 textual feedback

而平台需要的是：

1. 输入 `x_bad`
2. 输入召回的 `x_real`
3. 让 critic 直接产生“对比式 critique”

因此：

1. 原仓库没有现成的 `contrastive_critic_v1`。
2. 需要基于 `llm_backward_prompts.py` 的风格重新写一套对比批判模板。

## 6.3 原仓库的 `UID` 对应实现名是 `sum_uid`

平台文档里命名为 `uid`，而原仓库代码里实际 `aggregate_method` 使用的是 `sum_uid`。

因此新平台应统一对外命名为：

1. `uid`

内部说明中注明：

1. 它来源于 FedTextGrad 原始 `sum_uid` 逻辑。

## 6.4 原仓库没有 `dbscan_uid` 和 `dbscan_uid_tsgdm`

这一点必须明确：

1. `THESIS_PLATFORM_PLAN.md` 已把 `dbscan_uid`、`dbscan_uid_tsgdm` 列为阶段 3 聚合器候选。
2. 但 `FedTextGrad` 原仓库里没有对应 Python 模块实现。

所以：

1. `summarization` 和 `uid` 可从原仓库抽取。
2. `dbscan_uid` 和 `dbscan_uid_tsgdm` 必须在平台中重新实现。

## 6.5 原仓库没有 `compress_to_n_rules` 和 `redact_*`

平台 critic 配置里有：

1. `compress_to_n_rules`
2. `redact_enable`
3. `redact_mode`

这些能力在 FedTextGrad 原仓库中没有现成实现模块。

因此：

1. 规则压缩器需要平台自己实现。
2. 脱敏器需要平台自己实现。

FedTextGrad 最多能提供“如何生成原始 critique”的风格参考。

## 7. 新平台里的实现建议

## 7.1 `fedtextgrad_critic` 的推荐实现

推荐不要直接复用完整 `textgrad.Variable.backward()` 计算图，而是做一个平台原生轻量 critic：

1. 输入
   - `x_bad`
   - `x_real`
   - 当前 round 的上下文
   - 当前 Prompt 或任务描述
2. 构造对比式 prompt
   - 保留 FedTextGrad 的“只给反馈，不直接重写”的约束
   - 强化“指出 x_bad 相比 x_real 暴露的缺陷”
3. 调用指定 backend LLM
4. 输出结构化 `Critique`

这样做的优点：

1. 接口自然符合平台 `critique(paired_samples, client_ctx)` 规范。
2. 不依赖原 benchmark evaluator。
3. 不依赖 `Variable` 计算图。
4. 更容易接入 `compress_to_n_rules` 与 `redact_enable`。

## 7.2 `summarization` 与 `uid` 聚合器的推荐实现

推荐实现非常直接：

1. 收集所有客户端 critique
2. 截断到 `max_aggregate_tokens`
3. 按选择的 prompt template 构造 server prompt
4. 用聚合模型输出 `PromptUpdate`

其中：

1. `concat`
   - 不调用 LLM，直接拼接 critique
2. `summarization`
   - 使用改写后的 `SUMMARIZATION_TEMPLATE`
3. `uid`
   - 使用改写后的 `UID_TEMPLATE`

## 7.3 `dbscan_uid` 与 `dbscan_uid_tsgdm`

这两个模块不应该尝试从原仓库“找现成代码”，因为没有。

推荐路线：

1. `dbscan_uid`
   - 先对 critique embedding
   - 再 `DBSCAN`
   - 每簇选代表 critique
   - 最后走 `uid` prompt 汇总
2. `dbscan_uid_tsgdm`
   - 在 `dbscan_uid` 基础上
   - 给聚合器加入历史轮次动量记忆
   - 可以参考 `TextualGradientDescentwithMomentum` 的思路，但不要直接复制实现

## 8. 最终抽取清单

## 8.1 必须抽取

1. `utils/prompt_template.py`
   - 抽取 `SUMMARIZATION_TEMPLATE`
   - 抽取 `UID_TEMPLATE`
2. `train_hetero_fed.py`
   - 抽取 `summarization`/`sum_uid` 的服务端聚合流程
3. `textgrad/autograd/llm_backward_prompts.py`
   - 抽取 FedTextGrad 风格的 feedback-only prompt 设计原则

## 8.2 可选抽取

1. `textgrad/engine/__init__.py`
2. `textgrad/engine/base.py`
3. `textgrad/engine/textgrad_openai.py`
4. `textgrad/engine/local_model_openai_api.py`
5. `textgrad/engine/textgrad_vllm.py`
6. `textgrad/autograd/llm_ops.py`
7. `textgrad/variable.py`
8. `textgrad/config.py`
9. `textgrad/optimizer/optimizer.py`
10. `utils/prompt_complexity.py`

## 8.3 不抽取

1. `main.py`
2. `train_centralized.py`
3. `train_homo_fed.py`
4. `eval.py`
5. `textgrad/tasks/*`
6. `textgrad/tasks/multimodal/*`
7. `textgrad/autograd/multimodal_ops.py`
8. `scripts/*`
9. `resources/*`

## 9. 推荐实施顺序

1. 先实现 `fedtextgrad_critic.py`
   - 仅保留 feedback-only 风格
   - 不引入完整 textgrad 计算图
2. 再实现 `concat.py`
3. 再实现 `summarization.py`
4. 再实现 `uid.py`
5. 然后补 `compress_to_n_rules` 和 `redact`
6. 最后自行实现 `dbscan_uid` 与 `dbscan_uid_tsgdm`

## 10. 最终结论

面向新的创新项目，`FedTextGrad` 需要抽取的不是原仓库的大多数模块，而是一个很清晰的最小核心：

1. FedTextGrad 风格的 textual feedback 设计原则。
2. `summarization` 与 `UID` 的服务端聚合模板。
3. 可选的 LLM backend 抽象。

其中：

1. `utils/prompt_template.py` 和 `textgrad/autograd/llm_backward_prompts.py` 是最直接的核心来源。
2. `train_hetero_fed.py` 只保留其服务端聚合思路。
3. `textgrad` 计算图、optimizer、benchmark task 层都不应整体搬入平台。
4. `dbscan_uid`、`dbscan_uid_tsgdm`、规则压缩、脱敏，必须作为平台新增实现。

因此，新的平台实现应采用“轻量适配 + prompt 重构 + 缺失模块自研”的方案，而不是“把 FedTextGrad 仓库作为子系统整体接入”。
