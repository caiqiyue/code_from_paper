# FedTextGrad 项目结构与运行说明

本说明基于当前仓库代码整理，目标是把这个实验平台讲清楚，包括：

- 项目整体结构和调用链
- 每个 Python 包/子包/模块分别负责什么
- 如何创建虚拟环境、安装依赖、补充可选第三方包
- 如何启动集中式、同构联邦、异构联邦实验
- 需要配置哪些 CLI 参数和环境变量
- 当前代码里哪些参数是真正生效的

同时，源码中已经补充了以下可读性增强：

- 为缺失 docstring 的类、函数、方法补齐了说明
- 在 `main.py`、`eval.py`、`train_centralized.py`、`train_homo_fed.py`、`train_hetero_fed.py` 的关键步骤右侧补充了注释
- 修正了 `scripts/run_homo_fed.sh` 与真实 CLI 参数不一致的问题
- 将 `BBH` 数据下载从外部 `wget` 改为 Python 标准库下载，避免 Windows 环境下载失败

## 1. 项目整体定位

FedTextGrad 是一个把 Textual Gradient 引入联邦学习场景的实验仓库。

它的核心思路不是更新数值参数，而是更新“系统提示词”：

1. 用 `textgrad.Variable` 把 prompt、模型输出、评价结果都包装成计算图节点。
2. 用评估模型把“当前回答是否更好”转成文本反馈。
3. 用 `TextualGradientDescent` 根据文本反馈重写 prompt。
4. 在集中式、同构联邦、异构联邦三种训练脚本里，重复执行“生成回答 -> 评分 -> 文本反传 -> prompt 更新 -> 回退/接受”的循环。
5. 在异构联邦脚本中，再把多个客户端 prompt 做拼接或摘要聚合。

## 2. 当前项目结构

```text
FedTextGrad/
├─ README.md
├─ quick_start.md
├─ requirements.txt
├─ main.py
├─ eval.py
├─ train_centralized.py
├─ train_homo_fed.py
├─ train_hetero_fed.py
├─ resources/
│  └─ FedTextGrad_Framework.png
├─ scripts/
│  ├─ run_centralized.sh
│  ├─ run_homo_fed.sh
│  ├─ run_hetero_fed.sh
│  ├─ vllm_serve.sh
│  └─ sglang_serve.sh
├─ textgrad/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ defaults.py
│  ├─ loss.py
│  ├─ model.py
│  ├─ prompts.py
│  ├─ variable.py
│  ├─ autograd/
│  ├─ engine/
│  ├─ optimizer/
│  ├─ tasks/
│  │  └─ multimodal/
│  └─ utils/
└─ utils/
   ├─ prompt_template.py
   └─ prompt_complexity.py
```

## 3. 运行主链路

### 3.1 从入口到训练脚本

`main.py` 是统一入口，主流程如下：

1. 解析 CLI 参数。
2. 初始化 Comet 实验记录。
3. 设置随机种子。
4. 根据 `--module` 动态加载训练脚本。
5. 调用对应脚本里的 `run_training(args, experiment)`。

### 3.2 训练脚本内部的共同模式

三个训练脚本都会做下面几件事：

1. 根据 `--evaluation_engine` 和 `--test_engine` 创建模型接口。
2. 通过 `textgrad.tasks.load_task(...)` 加载数据集与任务评估函数。
3. 把系统 prompt 包装成 `tg.Variable`。
4. 用 `tg.BlackboxLLM` 把“模型 + prompt”封装成一个可调用对象。
5. 对每个 batch：
   - 生成回答
   - 计算任务分数
   - 把分数或 `<ACCURACY>` 标签结果转成可反传的 textual loss
   - `total_loss.backward()`
   - `optimizer.step()`
   - 用同一个 batch 重新评估更新后的 prompt
   - 如果性能下降则回退到旧 prompt
6. 在验证集/测试集上汇总结果并写入 `logs` 与 Comet。

### 3.3 三种训练模式差异

| 模式 | 入口模块 | 数据组织方式 | 聚合方式 |
| --- | --- | --- | --- |
| 集中式 | `train_centralized.py` | 单任务、单 prompt | 不做客户端聚合 |
| 同构联邦 | `train_homo_fed.py` | 单任务，训练/验证数据按客户端随机切分 | 当前实现没有显式使用 `aggregate_method`，更接近多客户端独立更新 |
| 异构联邦 | `train_hetero_fed.py` | 多任务，每个任务视作一个客户端 | 支持 `concat`、`summarization`、`sum_uid` |

## 4. Python 包、子包、模块职责

先说明一件事：

- 严格按 Python 包定义，当前正式 Python 包是 `textgrad` 及其子包。
- 根目录的 `main.py` / `train_*.py` / `eval.py` 是脚本模块。
- 根目录的 `utils/` 是辅助模块目录，不是带 `__init__.py` 的独立安装包。

### 4.1 根目录脚本模块

| 模块 | 作用 |
| --- | --- |
| `main.py` | 统一 CLI 入口，负责解析参数、初始化 Comet、动态加载训练模块 |
| `eval.py` | 样本级/数据集级评估工具，负责并发评估与验证集回退逻辑 |
| `train_centralized.py` | 集中式 prompt 优化训练脚本 |
| `train_homo_fed.py` | 同构联邦训练脚本，把同一任务随机切给多个客户端 |
| `train_hetero_fed.py` | 异构联邦训练脚本，把多个任务视作多个客户端并在服务器端聚合 prompt |

### 4.2 `textgrad` 顶层包

| 模块 | 作用 |
| --- | --- |
| `textgrad/__init__.py` | 初始化 JSON logger，并统一导出 `Variable`、`BlackboxLLM`、`TextualGradientDescent`、`get_engine` 等公共 API |
| `textgrad/config.py` | 维护全局 backward engine 单例，解决“反向文本梯度由哪个模型来生成”的问题 |
| `textgrad/defaults.py` | 定义默认角色描述，如 system prompt / input / output 的默认文字 |
| `textgrad/loss.py` | 定义文本评价损失模块，包括单字段评价、多字段评价、多选题 test-time critic、图像问答评价器 |
| `textgrad/model.py` | 定义 `BlackboxLLM`，把“模型后端 + 系统提示词”包装成可调用模块 |
| `textgrad/prompts.py` | 保存梯度文本模板常量，属于 prompt 片段资源文件 |
| `textgrad/variable.py` | 整个 TextGrad 计算图的核心节点类型，负责保存值、梯度、前驱、grad_fn，并执行拓扑反传 |

### 4.3 `textgrad.autograd` 子包

| 模块 | 作用 |
| --- | --- |
| `textgrad/autograd/__init__.py` | 导出 autograd 常用算子与类 |
| `textgrad/autograd/algebra.py` | 定义 `sum` / `aggregate` 这类变量代数运算，以及梯度聚合/归约逻辑 |
| `textgrad/autograd/function.py` | 定义 `Function`、`BackwardContext`、`Module` 三个核心抽象基类 |
| `textgrad/autograd/functional.py` | 提供函数式接口，比如 `sum(...)`、`aggregate(...)`、`llm_call(...)` |
| `textgrad/autograd/llm_backward_prompts.py` | 保存 LLM backward 所需的模板字符串 |
| `textgrad/autograd/llm_ops.py` | 定义纯文本 LLM 调用算子，包括普通调用、格式化调用、带 in-context examples 的调用 |
| `textgrad/autograd/multimodal_backward_prompts.py` | 保存多模态 backward 提示模板 |
| `textgrad/autograd/multimodal_ops.py` | 定义图文混合输入的多模态 LLM 调用算子 |
| `textgrad/autograd/reduce_prompts.py` | 为梯度归约/摘要生成提示词 |
| `textgrad/autograd/string_based_ops.py` | 把字符串规则函数包装成可接入 TextGrad 反传的算子 |

### 4.4 `textgrad.engine` 子包

| 模块 | 作用 |
| --- | --- |
| `textgrad/engine/__init__.py` | 引擎分发器，根据 `engine_name` 选择具体后端，并校验多模态能力 |
| `textgrad/engine/base.py` | 抽象引擎接口 `EngineLM` 与磁盘缓存 mixin `CachedEngine` |
| `textgrad/engine/textgrad_openai.py` | OpenAI 兼容接口封装，同时兼容 OpenAI、Azure OpenAI、Ollama/OpenAI-compatible API、vLLM API |
| `textgrad/engine/anthropic.py` | Anthropic Claude 系列封装 |
| `textgrad/engine/gemini.py` | Gemini 系列封装 |
| `textgrad/engine/cohere.py` | Cohere 系列封装 |
| `textgrad/engine/together.py` | Together AI 系列封装 |
| `textgrad/engine/local_model_openai_api.py` | 外部 OpenAI-compatible client 适配器，比如 LM Studio |
| `textgrad/engine/textgrad_vllm.py` | 直接在本地进程内加载 vLLM/HF 模型，而不是通过 OpenAI 风格 HTTP API |
| `textgrad/engine/engine_utils.py` | 图像字节类型识别工具 |

### 4.5 `textgrad.optimizer` 子包

| 模块 | 作用 |
| --- | --- |
| `textgrad/optimizer/__init__.py` | 导出优化器别名 |
| `textgrad/optimizer/optimizer.py` | 定义 `Optimizer`、`TextualGradientDescent`、`TextualGradientDescentwithMomentum` |
| `textgrad/optimizer/optimizer_prompts.py` | 构造优化器提示词，决定怎样把“变量值 + 梯度反馈 + 约束 + 历史”组装成 prompt |

### 4.6 `textgrad.tasks` 子包

| 模块 | 作用 |
| --- | --- |
| `textgrad/tasks/__init__.py` | 任务路由器，根据任务名创建数据集对象与评估函数 |
| `textgrad/tasks/base.py` | 定义统一数据集抽象和轻量 DataLoader |
| `textgrad/tasks/big_bench_hard.py` | BBH 数据集加载、自动下载、拆分 train/val/test，并实现整数答案匹配 |
| `textgrad/tasks/gsm8k.py` | GSM8K 标准加载器和 DSPy 风格切分版本 |
| `textgrad/tasks/gpqa.py` | GPQA 数据集，以及面向 instance-level/test-time 优化的数据集封装 |
| `textgrad/tasks/mmlu.py` | MMLU 数据集，以及面向 instance-level/test-time 优化的数据集封装 |
| `textgrad/tasks/leetcode.py` | LeetCode Hard Eval 数据集接口 |
| `textgrad/tasks/livebench.py` | 一组较轻量的 LiveBench 包装与字符串匹配评价函数 |
| `textgrad/tasks/livebenchmath.py` | LiveBench Math 数据集与较复杂的数学答案解析/等价性判断工具 |
| `textgrad/tasks/livebenchreason.py` | LiveBench Reasoning 数据集与任务专用评分逻辑 |
| `textgrad/tasks/prollama.py` | ProLLaMA 蛋白质超家族分类数据集包装 |

### 4.7 `textgrad.tasks.multimodal` 子包

| 模块 | 作用 |
| --- | --- |
| `textgrad/tasks/multimodal/__init__.py` | 多模态任务路由器 |
| `textgrad/tasks/multimodal/mathvista.py` | MathVista 数据集、图像压缩、答案抽取、匹配与评价 |
| `textgrad/tasks/multimodal/scienceqa.py` | ScienceQA 数据集、图像压缩、答案解析和选项匹配 |

### 4.8 `textgrad.utils` 子包

| 模块 | 作用 |
| --- | --- |
| `textgrad/utils/image_utils.py` | URL 检查和图像下载缓存工具 |

### 4.9 根目录 `utils/` 辅助模块目录

| 模块 | 作用 |
| --- | --- |
| `utils/prompt_template.py` | 定义异构联邦聚合时用到的 prompt 合并模板和最终格式约束 |
| `utils/prompt_complexity.py` | 计算 entropy / compression rate / TF-IDF / perplexity / token length / uniformity 等文本复杂度指标；属于分析工具，不是主训练必需模块 |

## 5. 数据与任务支持情况

### 5.1 主训练入口 `load_task(...)` 当前支持的任务类型

| 任务名模式 | 数据来源 | 说明 |
| --- | --- | --- |
| `BBH_*` | BigBenchHard 原始 JSON，首次运行自动下载并缓存 | 例如 `BBH_object_counting` |
| `GSM8K_DSPy` | Hugging Face `gsm8k` | 使用 DSPy 风格切分 |
| `prollama` | 本地 JSON 文件 | 默认期望路径是 `../data/ProLLaMA/raw` |
| `livebench_math` / `livebench_math__子任务` | Hugging Face `livebench/math` | 支持 `AMPS_Hard`、`math_comp`、`olympiad` 等子任务 |
| `livebench_reasoning` / `livebench_reasoning__子任务` | Hugging Face `livebench/reasoning` | 支持 `web_of_lies_v2`、`zebra_puzzle`、`spatial` |

### 5.2 额外存在但不直接走主训练入口的任务接口

| 接口 | 说明 |
| --- | --- |
| `load_instance_task(...)` | 给 instance-level / test-time 优化任务使用，比如 `MMLU_*`、`GPQA_*`、`LeetCodeHardEval` |
| `load_multimodal_instance_task(...)` | 给多模态任务使用，比如 `mathvista`、`scienceqa` |

### 5.3 数据准备注意事项

1. `BBH_*` 任务首次运行会自动下载数据到 `platformdirs.user_cache_dir("textgrad")` 对应缓存目录。
2. `GSM8K_DSPy` 依赖 Hugging Face `datasets` 下载。
3. `prollama` 需要你自己准备数据文件，代码默认读取：
   - `../data/ProLLaMA/raw/train_split.json`
   - `../data/ProLLaMA/raw/test_split.json`
4. LiveBench 数学/推理任务依赖 Hugging Face 下载。
5. 多模态任务依赖图像处理相关依赖和支持多模态输入的引擎。

## 6. 如何创建虚拟环境与安装依赖

以下命令按 Windows PowerShell 编写。

### 6.1 创建虚拟环境

```powershell
cd D:\学习记录\导师项目\研究\caiqiyue_file\FedTextGrad
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果你使用的是 Conda，也可以：

```powershell
conda create -n fedtextgrad python=3.11 -y
conda activate fedtextgrad
python -m pip install --upgrade pip
```

### 6.2 安装核心依赖

```powershell
pip install -r requirements.txt
```

`requirements.txt` 覆盖的是主训练路径所需的核心包，包括：

- `openai`
- `tenacity`
- `python-dotenv`
- `pandas`
- `platformdirs`
- `datasets`
- `diskcache`
- `graphviz`
- `gdown`
- `litellm`
- `pillow`
- `httpx`
- `numpy`
- `tqdm`
- `torch`
- `comet_ml`

### 6.3 按功能安装可选第三方包

不是所有后端都在 `requirements.txt` 中，按你要用的后端补装即可：

| 使用场景 | 需要额外安装 |
| --- | --- |
| Anthropic Claude | `pip install anthropic` |
| Gemini | `pip install google-generativeai` |
| Cohere | `pip install cohere` |
| Together AI | `pip install together` |
| 直接本地 vLLM 后端 (`engine_name` 形如 `vllm-...`) | `pip install vllm transformers` |
| SGLang 服务端 | `pip install "sglang[all]"` |
| FlashInfer 加速 SGLang | `pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/` |
| LiveBench Math 相关解析 | `pip install sympy scikit-learn` |
| `utils/prompt_complexity.py` 分析工具 | `pip install nltk scikit-learn transformers` |

补充说明：

1. 如果你要跑 GPU 版 PyTorch，请安装和你的 CUDA 版本匹配的 PyTorch wheel。
2. `graphviz` Python 包只解决 Python 接口；如果你要真正导出计算图，系统层面通常还需要 Graphviz 可执行程序。

## 7. 运行前要配置的环境变量

### 7.1 实验记录

| 环境变量 | 是否必须 | 作用 |
| --- | --- | --- |
| `COMET_API_KEY` | 仅当 `--comet_mode online` 时必须 | 在线写入 Comet ML |
| `COMET_OFFLINE_DIRECTORY` | 不需要手动设 | `main.py` 会自动设置为 `--comet_log_path` |

### 7.2 模型后端

| 后端 | 典型 `engine_name` | 需要的环境变量 | 说明 |
| --- | --- | --- | --- |
| OpenAI | `gpt-4o`、`gpt-3.5-turbo-0125` | `OPENAI_API_KEY` | 默认直接连 OpenAI |
| Azure OpenAI | `azure-gpt-35-turbo` 等 | `AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_API_BASE`、`AZURE_OPENAI_API_VERSION` | 走 AzureChatOpenAI |
| Ollama / OpenAI-compatible API | `ollama-模型名` | `OLLAMA_BASE_URL`、`OLLAMA_API_KEY` | 代码里把所有 OpenAI-compatible 第三方服务都当成这一路径处理 |
| vLLM OpenAI API | `vllm-api-llama3`、`vllm-api-llama3.1` | `VLLM_BASE_URL`、`VLLM_API_KEY` | 当前代码只内置了两个 alias |
| 直接本地 vLLM | `vllm-meta-llama/Meta-Llama-3-8B-Instruct` | 无固定环境变量 | 直接在本地进程内加载模型 |
| Anthropic | `claude-...` | `ANTHROPIC_API_KEY` | 支持多模态能力校验 |
| Gemini | `gemini-...` | `GOOGLE_API_KEY` | 使用 `google-generativeai` |
| Cohere | `command-r-plus` 等 | `COHERE_API_KEY` | Cohere chat 后端 |
| Together | `together-...` | `TOGETHER_API_KEY` | Together AI chat 后端 |

### 7.3 PowerShell 设置环境变量示例

```powershell
$env:OPENAI_API_KEY = "your-openai-key"
$env:COMET_API_KEY = "your-comet-key"
```

或使用 OpenAI-compatible / Ollama 风格接口：

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434/v1"
$env:OLLAMA_API_KEY = "ollama"
```

## 8. 如何启动实验

### 8.1 重要说明

1. 当前仓库里的 `scripts/*.sh` 是 Bash 示例脚本。
2. 你现在的环境是 Windows PowerShell，最稳妥的方式是直接运行 `python main.py ...`。
3. 如果你使用 Git Bash 或 WSL，才适合直接执行 `.sh`。

### 8.2 集中式训练

```powershell
$env:OPENAI_API_KEY = "your-openai-key"
python .\main.py `
  --module train_centralized `
  --task BBH_object_counting `
  --evaluation_engine gpt-4o `
  --test_engine gpt-3.5-turbo-0125 `
  --batch_size 3 `
  --max_epochs 3 `
  --max_steps 3 `
  --proximal_update
```

### 8.3 同构联邦训练

```powershell
$env:OPENAI_API_KEY = "your-openai-key"
python .\main.py `
  --module train_homo_fed `
  --task BBH_object_counting `
  --evaluation_engine gpt-4o `
  --test_engine gpt-3.5-turbo-0125 `
  --batch_size 3 `
  --max_epochs 3 `
  --max_steps 3 `
  --homo_split_num 3 `
  --proximal_update
```

说明：

- 这个脚本只读取 `args.task[0]`，所以虽然 `--task` 支持多个值，但同构脚本里只有第一个会被使用。
- 当前实现没有真正消费 `--aggregate_method`。

### 8.4 异构联邦训练

```powershell
$env:OPENAI_API_KEY = "your-openai-key"
python .\main.py `
  --module train_hetero_fed `
  --task BBH_object_counting BBH_multistep_arithmetic_two GSM8K_DSPy `
  --evaluation_engine gpt-4o `
  --test_engine gpt-3.5-turbo-0125 `
  --batch_size 3 `
  --max_epochs 3 `
  --max_steps 3 `
  --aggregate_method summarization `
  --proximal_update
```

如果你要显式测试三种聚合方式，当前代码真正支持的是：

- `concat`
- `summarization`
- `sum_uid`

### 8.5 使用 Ollama / OpenAI-compatible 本地或第三方服务

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434/v1"
$env:OLLAMA_API_KEY = "ollama"

python .\main.py `
  --module train_centralized `
  --task BBH_object_counting `
  --evaluation_engine ollama-meta-llama/llama-3.2-11b-vision-instruct `
  --test_engine ollama-meta-llama/llama-3.2-11b-vision-instruct `
  --proximal_update
```

### 8.6 启动本地服务端示例

#### vLLM API 服务

```powershell
$env:TRANSFORMERS_OFFLINE = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.2-11B-Vision-Instruct --port 8003
```

#### SGLang 服务

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.2-11B-Vision-Instruct --port 10003
```

## 9. CLI 参数说明

当前统一在 `main.py` 定义的参数如下。

| 参数 | 类型 | 默认值 | 适用范围 | 说明 |
| --- | --- | --- | --- | --- |
| `--task` | `str ...` | 无 | 全部 | 任务名列表；集中式和同构脚本只使用第一个任务，异构脚本会把多个任务都当客户端 |
| `--evaluation_engine` | `str` | `gpt-4o` | 全部 | 负责评分、反向文本梯度、聚合摘要时调用的模型 |
| `--test_engine` | `str` | `gpt-3.5-turbo-0125` | 全部 | 真正执行任务回答的模型 |
| `--batch_size` | `int` | `3` | 全部 | 每次本地 prompt 更新使用的样本数 |
| `--max_epochs` | `int` | `3` | 全部 | epoch 数 |
| `--max_steps` | `int` | `3` | 全部 | 每个 epoch 或每个客户端的本地更新步数上限；不同脚本的 break 判断略有差异 |
| `--seed` | `int` | `42` | 全部 | 随机种子 |
| `--do_not_run_larger_model` | flag | 关闭 | 全部 | 关闭初始参考模型的 0-shot 评估 |
| `--aggregate_method` | `str` | `summarization` | 主要用于异构联邦 | 聚合方式，当前真实支持 `concat` / `summarization` / `sum_uid` |
| `--homo_split_num` | `int` | `3` | 同构联邦 | 同构联邦下客户端数量 |
| `--comet_mode` | `offline/online` | `offline` | 全部 | Comet 记录模式 |
| `--comet_project_name` | `str` | `fedtextgrad` | 全部 | Comet 项目名 |
| `--comet_log_path` | `str` | `./logs/comet_results/` | 全部 | Comet 离线记录与 prompt 文件输出目录 |
| `--proximal_update` | flag | 关闭 | 全部 | 每次更新后用同一 batch 重评，如果性能下降就回退 prompt |
| `--module` | `str` | 无 | 全部 | 当前真实可运行值为 `train_centralized`、`train_homo_fed`、`train_hetero_fed` |

## 10. 哪些参数在不同模块中真正生效

| 参数 | `train_centralized.py` | `train_homo_fed.py` | `train_hetero_fed.py` |
| --- | --- | --- | --- |
| `task` | 只取第一个 | 只取第一个 | 使用整个列表 |
| `evaluation_engine` | 生效 | 生效 | 生效 |
| `test_engine` | 生效 | 生效 | 生效 |
| `batch_size` | 生效 | 生效 | 生效 |
| `max_epochs` | 生效 | 生效 | 生效 |
| `max_steps` | 生效 | 生效 | 生效 |
| `homo_split_num` | 不使用 | 生效 | 不使用 |
| `aggregate_method` | 不使用 | 当前实现不使用 | 生效 |
| `proximal_update` | 生效 | 生效 | 生效 |

## 11. 输出结果保存到哪里

### 11.1 日志与指标

- 包级 logger 会在 `./logs/` 下写入 JSONL 风格日志文件。
- Comet 离线日志默认写到 `./logs/comet_results/`。

### 11.2 Prompt 快照

训练脚本会把 prompt 保存到 `--comet_log_path` 指向的目录，例如：

- `任务名_last_prompt.txt`
- `任务名_best_prompt.txt`
- `任务名_best_agg_prompt.txt`

### 11.3 结果指标

训练脚本会记录：

- 0-shot test/validation accuracy
- 每步训练前后 batch 分数
- best validation / best aggregation validation
- final test accuracy
- update success rate

## 12. 当前代码层面的关键信息与注意事项

1. `main.py` 现在已经把帮助文本修正为与真实支持的模块和聚合方式一致。
2. `scripts/run_homo_fed.sh` 已修正为 `train_homo_fed` 且使用真实参数 `--homo_split_num`。
3. `BBH` 数据下载已改为 Python 标准库实现，不再要求系统安装 `wget`。
4. `utils/prompt_complexity.py` 是辅助分析工具，不属于主训练必要依赖。
5. Windows PowerShell 下建议直接执行 `python main.py ...`，不要默认执行 `.sh`。
6. `prollama` 任务对本地数据路径有要求，如果你不跑这个任务，可以不用准备这部分数据。
7. `train_homo_fed.py` 当前更偏“同任务多客户端独立更新”，并没有像 `train_hetero_fed.py` 那样实现显式 prompt 聚合分支。

## 13. 建议的最小可跑通路径

如果你只是想先把平台跑起来，推荐按下面顺序：

1. 创建虚拟环境并执行 `pip install -r requirements.txt`
2. 先使用 OpenAI 或现成的 OpenAI-compatible API，避免本地模型服务带来的额外复杂度
3. 先运行集中式：

```powershell
$env:OPENAI_API_KEY = "your-openai-key"
python .\main.py --module train_centralized --task BBH_object_counting --evaluation_engine gpt-4o --test_engine gpt-3.5-turbo-0125 --proximal_update
```

4. 再尝试同构联邦：

```powershell
python .\main.py --module train_homo_fed --task BBH_object_counting --evaluation_engine gpt-4o --test_engine gpt-3.5-turbo-0125 --homo_split_num 3 --proximal_update
```

5. 最后再尝试异构联邦和不同聚合方式：

```powershell
python .\main.py --module train_hetero_fed --task BBH_object_counting BBH_multistep_arithmetic_two GSM8K_DSPy --evaluation_engine gpt-4o --test_engine gpt-3.5-turbo-0125 --aggregate_method sum_uid --proximal_update
```

---

如果你后续还想继续补充，我建议下一步可以做三件事：

1. 给 `requirements.txt` 区分核心依赖和可选依赖。
2. 把 `scripts/*.sh` 再补一套 PowerShell 版本。
3. 在根目录再增加一份“实验配置样例文件”，把常用组合参数固化下来。
