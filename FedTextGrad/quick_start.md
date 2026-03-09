# FedTextGrad 快速说明

## 1. 项目整体在做什么

这个项目实现了论文 **Can Textual Gradient Work in Federated Learning?** 对应的实验代码。核心思想不是用数值梯度更新参数，而是让 LLM 先生成回答，再让另一个 LLM 根据回答质量生成“文本梯度 / 文本反馈”，最后再用这些反馈去改写 `system prompt`。  

项目里有三条主实验线：

- `train_centralized.py`：集中式 Prompt 优化。
- `train_homo_fed.py`：同构联邦场景，多个客户端处理同一任务的数据切分。
- `train_hetero_fed.py`：异构联邦场景，不同客户端处理不同任务，再做 Prompt 聚合。

`main.py` 负责统一接收参数并动态调用这三个训练模块。

---

## 2. 根目录和每个文件夹的作用

| 路径 | 作用 |
| --- | --- |
| `README.md` | 项目原始说明，包含论文背景、安装方式、基础运行示例。 |
| `requirements.txt` | Python 依赖列表。核心依赖包括 `openai`、`datasets`、`torch`、`comet_ml`。 |
| `main.py` | 统一实验入口。解析参数、初始化 Comet、设置随机种子、动态加载训练模块。 |
| `eval.py` | 数据集评测工具。并发执行样本评估，统计 accuracy。 |
| `train_centralized.py` | 集中式训练逻辑。 |
| `train_homo_fed.py` | 同构联邦训练逻辑。 |
| `train_hetero_fed.py` | 异构联邦训练逻辑。 |
| `resources/` | README 中用到的图片资源。当前只有框架图。 |
| `scripts/` | Bash 启动脚本，提供服务启动和实验启动示例。 |
| `utils/` | 项目额外工具函数，主要是 Prompt 聚合模板和文本复杂度分析。 |
| `textgrad/` | 本地内置的 TextGrad 框架实现，包含变量系统、自动求“文本梯度”、LLM 引擎封装、优化器、任务数据集。 |

说明：

- README 里提到 `logs/`，仓库当前没有预置这个目录，但运行时会自动生成。
- `scripts/` 下是 `.sh`，更适合 Linux/macOS 或 Git Bash；在 Windows PowerShell 下，直接执行 `python main.py ...` 更稳妥。

---

## 3. 各模块职责

### 3.1 顶层模块

| 模块 | 功能 |
| --- | --- |
| `main.py` | 定义实验参数，如 `task`、`evaluation_engine`、`test_engine`、`batch_size`、`max_epochs`、`max_steps`、`aggregate_method`、`homo_split_num`、`proximal_update` 等；初始化 Comet；按 `--module` 加载训练脚本并调用 `run_training()`。 |
| `eval.py` | `eval_sample()` 对单样本跑推理与评测；`eval_dataset()` 用线程池并发评估全数据集；`run_validation_revert()` 做验证集回退。 |
| `train_centralized.py` | 单任务集中式训练。流程是：加载任务数据集 -> 初始化初始 system prompt -> 计算 batch 上的 textual feedback -> 用 `TextualGradientDescent` 更新 prompt -> 在验证集与测试集上评估 -> 保存 best/last prompt。 |
| `train_homo_fed.py` | 同构联邦训练。把同一任务的数据按客户端数切分，每个客户端各自维护一份 prompt 和 optimizer，然后分别本地更新。当前实现里**没有像异构联邦那样显式做跨客户端 prompt 聚合**，更像“多客户端独立本地更新 + 最后取当前 prompt 评估”。 |
| `train_hetero_fed.py` | 异构联邦训练。每个任务视作一个客户端，本地更新后，按 `aggregate_method` 对多个 prompt 做聚合，再把聚合后的 prompt 同步回所有客户端。 |

### 3.2 `utils/`

| 模块 | 功能 |
| --- | --- |
| `utils/prompt_template.py` | 定义异构联邦聚合时使用的 Prompt 模板，包括普通总结模板、UID 模板和最终格式约束。 |
| `utils/prompt_complexity.py` | 计算文本复杂度指标，如熵、压缩率、TF-IDF、困惑度、token 长度、信息密度均匀性。当前主训练流程未直接调用。 |

### 3.3 `scripts/`

| 模块 | 功能 |
| --- | --- |
| `scripts/run_centralized.sh` | 集中式实验的 Bash 示例命令。 |
| `scripts/run_homo_fed.sh` | 名义上是同构联邦脚本，但当前内容实际上调用了 `train_hetero_fed`，且参数名写成了不存在的 `--homo_split`，不能直接照抄。 |
| `scripts/run_hetero_fed.sh` | 异构联邦 Bash 示例命令。 |
| `scripts/vllm_serve.sh` | 启动本地 vLLM OpenAI 兼容服务。 |
| `scripts/sglang_serve.sh` | 启动本地 SGLang 服务。 |

### 3.4 `resources/`

| 模块 | 功能 |
| --- | --- |
| `resources/FedTextGrad_Framework.png` | 论文框架图，README 展示用。 |

### 3.5 `textgrad/` 根模块

| 模块 | 功能 |
| --- | --- |
| `textgrad/__init__.py` | 暴露对外 API，初始化 JSONL 日志器，并导出 `Variable`、`BlackboxLLM`、`TextualGradientDescent`、`get_engine`、`sum`、`aggregate` 等对象。 |
| `textgrad/config.py` | 维护全局单例 backward engine；`set_backward_engine()` 用于注册负责“反向文本反馈”的模型。 |
| `textgrad/defaults.py` | 默认 role description 常量。 |
| `textgrad/prompts.py` | 梯度文本保存模板。 |
| `textgrad/model.py` | `BlackboxLLM` 封装，把 LLM 变成可参与 TextGrad 图计算的模块。 |
| `textgrad/loss.py` | 各类“文本损失”定义，包括单文本评价、多字段评价、带标签解析的评价、多选测试时评价、图像问答评价。 |
| `textgrad/variable.py` | TextGrad 的核心数据结构 `Variable`。保存值、前驱、梯度、梯度上下文、反向函数，并支持 `backward()` 和计算图可视化。 |
| `textgrad/py.typed` | 类型标记文件。 |

### 3.6 `textgrad/autograd/`

| 模块 | 功能 |
| --- | --- |
| `textgrad/autograd/__init__.py` | 导出 autograd 常用接口。 |
| `textgrad/autograd/function.py` | 定义 `Function`、`BackwardContext`、`Module` 抽象基类。 |
| `textgrad/autograd/functional.py` | 提供函数式接口，如 `sum()`、`aggregate()`、`llm_call()`、`formatted_llm_call()`。 |
| `textgrad/autograd/algebra.py` | 实现 `Sum` 和 `Aggregate` 两种图操作；`Aggregate` 在反向阶段会对多个文本梯度做摘要式归并。 |
| `textgrad/autograd/llm_ops.py` | 单模态 LLM 调用及其反向传播逻辑；支持普通调用、格式化字段调用、带 in-context examples 的调用。 |
| `textgrad/autograd/llm_backward_prompts.py` | 反向阶段生成“文本梯度”时使用的系统提示词和模板。 |
| `textgrad/autograd/string_based_ops.py` | 把普通字符串规则函数包装成可反向传播的 TextGrad `Function`。 |
| `textgrad/autograd/multimodal_ops.py` | 多模态 LLM 调用及其反向传播逻辑。 |
| `textgrad/autograd/multimodal_backward_prompts.py` | 多模态反向传播模板。 |
| `textgrad/autograd/reduce_prompts.py` | 聚合多个梯度时的 reduce/summarize 提示模板。 |

### 3.7 `textgrad/engine/`

| 模块 | 功能 |
| --- | --- |
| `textgrad/engine/__init__.py` | 根据名字构造对应 LLM 引擎；支持 OpenAI、Azure、Ollama/OpenAI-compatible API、vLLM API、Anthropic、Gemini、Together、Cohere、本地 vLLM。 |
| `textgrad/engine/base.py` | 定义 `EngineLM` 抽象基类和 `CachedEngine` 缓存基类。 |
| `textgrad/engine/engine_utils.py` | 图像字节类型识别工具。 |
| `textgrad/engine/textgrad_openai.py` | OpenAI 风格接口引擎，也是 Ollama 和 vLLM OpenAI-compatible 接口的核心实现。支持文本和多模态输入。 |
| `textgrad/engine/local_model_openai_api.py` | 允许传入外部 OpenAI 兼容 client，例如 LM Studio。 |
| `textgrad/engine/textgrad_vllm.py` | 直接用 `vllm.LLM` 本地推理，而不是走 OpenAI-compatible API。 |
| `textgrad/engine/anthropic.py` | Anthropic Claude 接口封装，支持文本和多模态输入。 |
| `textgrad/engine/gemini.py` | Google Gemini 接口封装。 |
| `textgrad/engine/cohere.py` | Cohere 接口封装。 |
| `textgrad/engine/together.py` | Together AI 接口封装。 |

### 3.8 `textgrad/optimizer/`

| 模块 | 功能 |
| --- | --- |
| `textgrad/optimizer/__init__.py` | 导出优化器。 |
| `textgrad/optimizer/optimizer.py` | 核心优化器实现。`TextualGradientDescent` 会读取梯度和上下文，然后让 LLM 生成 `<IMPROVED_VARIABLE>...</IMPROVED_VARIABLE>` 中的新 prompt。 |
| `textgrad/optimizer/optimizer_prompts.py` | 优化器用到的系统提示词和更新模板，支持约束、动量和 in-context examples。 |

### 3.9 `textgrad/tasks/`

| 模块 | 功能 |
| --- | --- |
| `textgrad/tasks/__init__.py` | 任务路由器。`load_task()` 根据任务名返回 `train/val/test/eval_fn`；`load_instance_task()` 返回实例级任务。 |
| `textgrad/tasks/base.py` | 数据集抽象基类和轻量级 `DataLoader`。 |
| `textgrad/tasks/big_bench_hard.py` | BIG-Bench Hard 数据集封装；自动下载 JSON，并切成 `train/val/test`。 |
| `textgrad/tasks/gsm8k.py` | GSM8K 与 `GSM8K_DSPy` 数据集封装。 |
| `textgrad/tasks/mmlu.py` | MMLU 数据集封装及实例级 test-time optimization 接口。 |
| `textgrad/tasks/gpqa.py` | GPQA 数据集封装及实例级 test-time optimization 接口。 |
| `textgrad/tasks/prollama.py` | 氨基酸序列超家族分类数据集封装。 |
| `textgrad/tasks/leetcode.py` | LeetCode Hard 评测集封装，会自动下载 `leetcode-hard.jsonl`。 |
| `textgrad/tasks/livebench.py` | LiveBench 数学与推理数据集的早期封装版本。 |
| `textgrad/tasks/livebenchmath.py` | LiveBench Math 数据集封装和子任务结果解析逻辑。 |
| `textgrad/tasks/livebenchreason.py` | LiveBench Reasoning 数据集封装和子任务结果解析逻辑。 |
| `textgrad/tasks/multimodal/__init__.py` | 多模态任务入口。 |
| `textgrad/tasks/multimodal/scienceqa.py` | ScienceQA 图像问答数据集封装、答案抽取和评测。 |
| `textgrad/tasks/multimodal/mathvista.py` | MathVista 图像数学推理数据集封装、答案抽取和评测。 |

### 3.10 `textgrad/utils/`

| 模块 | 功能 |
| --- | --- |
| `textgrad/utils/image_utils.py` | 图像 URL 检查和缓存下载工具。 |

---

## 4. 当前实验实际会用到哪些数据集

### 4.1 README 和脚本直接覆盖的实验数据集

1. **BIG-Bench Hard (BBH)**
   - 代码入口：`textgrad/tasks/big_bench_hard.py`
   - 任务名示例：
     - `BBH_object_counting`
     - `BBH_word_sorting`
     - `BBH_multistep_arithmetic_two`
   - 数据来源：从 `https://github.com/suzgunmirac/BIG-Bench-Hard` 对应任务 JSON 自动下载。
   - 划分方式：前 50 条做 train，50 到 149 做 val，150 之后做 test。

2. **GSM8K_DSPy**
   - 代码入口：`textgrad/tasks/gsm8k.py`
   - 数据来源：Hugging Face `gsm8k/main`。
   - 划分方式：读取官方 train/test 后重新整理，并做固定随机打乱；当前实现使用：
     - `train = official_train[:50]`
     - `val = official_train[200:300]`
     - `test = official_test[300:400]`

### 4.2 代码里还支持，但当前 README 主路径没有直接用到的数据集

- `prollama`
- `livebench_math`
- `livebench_reasoning`
- `MMLU_*`
- `GPQA_*`
- `LeetCodeHardEval`
- `mathvista`
- `scienceqa`

这些更像是本地 `textgrad` 框架一起带进来的扩展任务接口，不是当前三条训练主脚本的默认实验重点。

---

## 5. 实验如何启动

### 5.1 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果你要跑多模态或扩展任务，可能还需要额外安装：

```powershell
pip install pillow scikit-learn nltk transformers sympy python-Levenshtein
```

### 5.2 配置模型 API

这个项目把模型分成两类：

- `evaluation_engine`：负责给回答打分、生成 textual feedback。
- `test_engine`：负责真正执行回答并被优化。

最常见的两种接法：

1. OpenAI 官方 API

```powershell
$env:OPENAI_API_KEY="你的Key"
```

2. Ollama / OpenRouter / 其他 OpenAI-compatible 接口

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434/v1"
$env:OLLAMA_API_KEY="任意非空字符串或真实Key"
```

说明：

- 代码里 `ollama-xxx` 前缀最终会走 OpenAI-compatible 接口封装。
- 如果你使用第三方兼容接口，也可以沿用 `OLLAMA_BASE_URL` / `OLLAMA_API_KEY` 这组环境变量。

### 5.3 可选：先启动本地服务

如果用 vLLM：

```powershell
$env:TRANSFORMERS_OFFLINE="1"
$env:CUDA_VISIBLE_DEVICES="0"
python -m vllm.entrypoints.openai.api_server --model=meta-llama/Meta-Llama-3.2-11B-Vision-Instruct --port=8003
```

如果用 SGLang：

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.2-11B-Vision-Instruct --port 10003
```

### 5.4 运行集中式实验

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434/v1"
$env:OLLAMA_API_KEY="test"
python main.py `
  --evaluation_engine ollama-meta-llama/llama-3.2-11b-vision-instruct `
  --test_engine ollama-meta-llama/llama-3.2-11b-vision-instruct `
  --task BBH_object_counting `
  --module train_centralized `
  --proximal_update
```

### 5.5 运行同构联邦实验

建议直接用下面这条命令，不要直接照抄 `scripts/run_homo_fed.sh`：

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434/v1"
$env:OLLAMA_API_KEY="test"
python main.py `
  --evaluation_engine ollama-meta-llama/llama-3.2-11b-vision-instruct `
  --test_engine ollama-meta-llama/llama-3.2-11b-vision-instruct `
  --task BBH_object_counting `
  --module train_homo_fed `
  --max_steps 3 `
  --homo_split_num 3 `
  --proximal_update
```

### 5.6 运行异构联邦实验

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434/v1"
$env:OLLAMA_API_KEY="test"
python main.py `
  --evaluation_engine ollama-meta-llama/llama-3.2-11b-vision-instruct `
  --test_engine ollama-meta-llama/llama-3.2-11b-vision-instruct `
  --task BBH_object_counting BBH_multistep_arithmetic_two GSM8K_DSPy `
  --module train_hetero_fed `
  --max_steps 3 `
  --aggregate_method summarization `
  --proximal_update
```

`aggregate_method` 当前实现支持：

- `concat`
- `summarization`
- `sum_uid`

### 5.7 关键参数说明

| 参数 | 作用 |
| --- | --- |
| `--task` | 任务名。集中式/同构联邦通常给一个，异构联邦可给多个。 |
| `--evaluation_engine` | 负责反馈和评测的模型。 |
| `--test_engine` | 被优化的模型。 |
| `--batch_size` | 每步训练使用的样本数。 |
| `--max_epochs` | 训练 epoch 数。 |
| `--max_steps` | 每个 epoch 内的局部更新步数上限。 |
| `--aggregate_method` | 异构联邦中客户端 prompt 聚合方式。 |
| `--homo_split_num` | 同构联邦客户端数。 |
| `--proximal_update` | 若更新后指标没有提升，则回退到旧 prompt。 |
| `--do_not_run_larger_model` | 跳过使用 `evaluation_engine` 作为大模型做 0-shot 参考测试。 |
| `--comet_mode` | `offline` 或 `online`。默认离线记录。 |
| `--comet_log_path` | Comet 离线日志和导出 prompt 的保存目录。 |

---

## 6. 实验输出会保存到哪里

默认输出位置：

- `./logs/`：`textgrad` JSONL 日志。
- `./logs/comet_results/`：Comet 离线结果目录。
- `./logs/comet_results/<task>_last_prompt.txt`：最后一次 prompt。
- `./logs/comet_results/<task>_best_prompt.txt` 或 `*_best_agg_prompt.txt`：最佳 prompt。

训练过程中还会在终端打印：

- train/updated train accuracy
- validation accuracy
- test accuracy
- update success rate

---

## 7. README 与当前代码的几个重要不一致点

1. README 写的是 `sh scripts/vllm_serve.py`，但仓库里实际文件是 `scripts/vllm_serve.sh`。
2. `scripts/run_homo_fed.sh` 当前内容并不是同构联邦正确启动命令：
   - 它调用的是 `train_hetero_fed`
   - 它使用了不存在的参数 `--homo_split`
   - 正确参数名是 `--homo_split_num`
3. README 里“Data Preparation”只列了部分数据集；代码里实际还支持 LiveBench、MMLU、GPQA、ProLLaMA、ScienceQA、MathVista、LeetCode 等扩展任务。
4. BBH 数据自动下载依赖 `wget`。如果是在 Windows PowerShell 原生环境中运行，机器上没有 `wget` 时，BBH 下载阶段可能失败；这时建议：
   - 用 Git Bash 运行
   - 或手动下载对应 JSON 到缓存目录
   - 或自行把下载逻辑改成 `requests` / `Invoke-WebRequest`

---

## 8. 一句话结论

如果只想最快跑通主流程，建议优先用：

- 集中式：`BBH_object_counting + train_centralized`
- 同构联邦：`BBH_object_counting + train_homo_fed`
- 异构联邦：`BBH_object_counting + BBH_multistep_arithmetic_two + GSM8K_DSPy + train_hetero_fed`

并直接执行 `python main.py ...`，不要完全依赖仓库里的 `.sh` 脚本。
