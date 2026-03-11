# `thesis_platform.model_downloaders` 使用说明

## 1. 这个包是干什么的

`thesis_platform/model_downloaders` 是论文复现实验里“模型下载”子系统的实现目录。

它的职责是：

- 定义每个模型对应的下载器模块。
- 把这些下载器注册成统一可调用的名字。
- 提供批量下载、跳过已存在模型、失败后继续后续模型、写下载报告等通用能力。
- 目前所有具体下载器都基于 Hugging Face Hub 的 `snapshot_download(...)` 下载模型快照。

这个包本身不负责模型推理，只负责把模型文件下载到项目约定的位置。

---

## 2. 下载到哪里

所有模型都会下载到：

`D:/学习记录/导师项目/研究/caiqiyue_file/thesis_platform/open_model/`

这是由 `common.py` 里的 `models_root()` 固定决定的，和你当前终端所在目录无关。

每个模型默认放在：

`thesis_platform/open_model/<model_name>/`

例如：

- `thesis_platform/open_model/opt_125m/`
- `thesis_platform/open_model/roberta_large/`
- `thesis_platform/open_model/llama_3_1_8b_instruct/`

批量下载完成后，还会生成总报告：

`thesis_platform/open_model/download_report.json`

每个模型目录下还会生成：

`thesis_platform/open_model/<model_name>/metadata.json`

一个典型目录结构如下：

```text
thesis_platform/
  open_model/
    download_report.json
    opt_125m/
      metadata.json
      config.json
      tokenizer.json
      pytorch_model.bin
      ...
```

---

## 3. 包内每个模块是干什么的

### 3.1 核心模块

| 模块 | 作用 | 关键对象 / 备注 |
| --- | --- | --- |
| `__init__.py` | 包入口。导入所有具体模型模块以触发注册，并向外暴露统一 API。 | 对外常用导出：`download_models`、`list_model_downloaders`、`create_model_downloader` |
| `base.py` | 定义下载器基类和标准结果对象。 | `BaseModelDownloader`、`ModelDownloadResult` |
| `common.py` | 提供下载目录、路径转换、UTC 时间戳、删除目录等工具函数。 | `models_root()` 返回 `thesis_platform/open_model/` |
| `registry.py` | 负责注册下载器类，并按名字实例化下载器。 | `register_model_downloader`、`create_model_downloader` |
| `hf.py` | Hugging Face 通用下载器实现。 | `HuggingFaceModelDownloader`，内部调用 `model_info` 和 `snapshot_download` |
| `controller.py` | 批量列出、解析、下载多个模型，并写总报告。 | `list_model_downloaders()`、`resolve_model_downloaders()`、`download_models()` |

### 3.2 具体模型模块

下面这些模块各自只做一件事：定义一个具体模型的下载器类，并通过装饰器注册到系统里。

| 模块 | 注册名 `name` | 默认 Hugging Face 仓库 `repo_id` | 是否默认下载 | 说明 |
| --- | --- | --- | --- | --- |
| `deepseek_r1_distill_llama_70b.py` | `deepseek_r1_distill_llama_70b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 否，`optional=True` | FedTextGrad 提到的 DeepSeek R1 Distill Llama 70B |
| `llama_2_13b_chat_hf.py` | `llama_2_13b_chat_hf` | `NousResearch/Llama-2-13b-chat-hf` | 否，`optional=True` | DataInf 用的 Llama 2 13B Chat 社区镜像 |
| `llama_3_1_405b_instruct.py` | `llama_3_1_405b_instruct` | `RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic` | 否，`optional=True` | FedTextGrad 提到的可选 405B 模型 |
| `llama_3_1_8b_instruct.py` | `llama_3_1_8b_instruct` | `unsloth/Meta-Llama-3.1-8B-Instruct` | 是 | 默认下载的 Llama 3.1 8B Instruct |
| `llama_3_2_11b_vision_instruct.py` | `llama_3_2_11b_vision_instruct` | `unsloth/Llama-3.2-11B-Vision-Instruct` | 否，`optional=True` | FedTextGrad 文档里使用的 11B Vision Instruct |
| `llama_3_2_3b_instruct.py` | `llama_3_2_3b_instruct` | `unsloth/Llama-3.2-3B-Instruct` | 否，`optional=True` | Prompt-transfer 目标模型 |
| `opt_125m.py` | `opt_125m` | `facebook/opt-125m` | 是 | GRADMM 使用的 OPT-125M |
| `opt_1_3b.py` | `opt_1_3b` | `facebook/opt-1.3b` | 否，`optional=True` | GRADMM 附录实验使用的 OPT-1.3B |
| `opt_350m.py` | `opt_350m` | `facebook/opt-350m` | 是 | GRADMM 使用的 OPT-350M |
| `phi_1_5.py` | `phi_1_5` | `microsoft/phi-1_5` | 是 | GRADMM 使用的 Phi-1.5 |
| `roberta_large.py` | `roberta_large` | `roberta-large` | 是 | DataInf 使用的 RoBERTa-large |
| `stable_diffusion_v1_5.py` | `stable_diffusion_v1_5` | `runwayml/stable-diffusion-v1-5` | 是 | DataInf 使用的 Stable Diffusion v1.5 |

### 3.3 哪些模型默认会下载

如果不传任何参数，系统默认只下载非 `optional` 的模型：

- `llama_3_1_8b_instruct`
- `opt_125m`
- `opt_350m`
- `phi_1_5`
- `roberta_large`
- `stable_diffusion_v1_5`

如果使用 `--include-optional`，则会把可选模型也纳入默认集合。

---

## 4. 下载流程是怎样的

整体流程如下：

1. `__init__.py` 导入所有具体模型模块。
2. 每个具体模型模块通过 `@register_model_downloader` 注册自己的下载器类。
3. `controller.py` 根据模型名创建下载器实例。
4. 每个下载器调用 `BaseModelDownloader.download(...)` 执行统一流程。
5. 具体下载动作由 `HuggingFaceModelDownloader.perform_download(...)` 完成。
6. 下载成功后写 `metadata.json`，批量任务结束后写 `download_report.json`。

默认行为还包括：

- 如果目标目录和 `metadata.json` 已存在，且没有传 `force=True`，则跳过下载。
- 如果传了 `force=True`，会重新下载。
- 如果某个模型下载失败，总控不会中止整个批次，而是继续下载后面的模型，并把错误写入 `download_report.json`。

---

## 5. 怎么用这些模块下载模型

### 5.1 推荐方式：使用命令行入口

最推荐使用项目已经提供好的入口脚本：

`python -m thesis_platform.scripts.download_models`

先安装依赖：

```powershell
cd D:\学习记录\导师项目\研究\caiqiyue_file
pip install -r thesis_platform\requirements.txt
```

常用命令如下。

列出所有已注册模型：

```powershell
python -m thesis_platform.scripts.download_models --list
```

下载默认模型集：

```powershell
python -m thesis_platform.scripts.download_models
```

下载默认模型集并包含可选模型：

```powershell
python -m thesis_platform.scripts.download_models --include-optional
```

只下载指定模型：

```powershell
python -m thesis_platform.scripts.download_models --names opt_125m roberta_large
```

强制重下：

```powershell
python -m thesis_platform.scripts.download_models --names opt_125m --force
```

覆盖某个模型的默认仓库：

```powershell
python -m thesis_platform.scripts.download_models `
  --names llama_3_1_8b_instruct `
  --repo-override llama_3_1_8b_instruct=custom-user/Llama-3.1-8B-Instruct
```

### 5.2 Python API：批量下载

如果你想在代码里调用，推荐使用包导出的总控函数。

```python
from thesis_platform.model_downloaders import download_models

report = download_models(
    names=["opt_125m", "roberta_large"],
    force=False,
    include_optional=False,
    repo_overrides=None,
)

print(report)
```

`report` 是一个字典，包含：

- 本次请求下载了哪些模型
- 下载了多少个、跳过了多少个、失败了多少个
- 每个模型的状态、目标路径、元数据路径、错误信息等

### 5.3 Python API：创建单个下载器

如果你只想下载一个模型，也可以通过注册表按名字创建：

```python
from thesis_platform.model_downloaders import create_model_downloader

downloader = create_model_downloader("opt_125m")
result = downloader.download(force=False)
print(result.to_dict())
```

### 5.4 直接使用某个具体模块

每个具体模块本质上都是一个下载器类，你也可以直接导入并调用：

```python
from thesis_platform.model_downloaders.llama_3_1_8b_instruct import Llama31_8BInstructDownloader

downloader = Llama31_8BInstructDownloader()
result = downloader.download()
print(result.to_dict())
```

这种方式是可行的，但通常不如 `create_model_downloader(...)` 或 `download_models(...)` 统一。

---

## 6. 下载结果里会写什么

### 6.1 单个模型的 `metadata.json`

每个模型下载成功后，都会写一个 `metadata.json`，里面通常会包含：

- `name`
- `description`
- `repo_id`
- `default_repo_id`
- `resolved_repo_id`
- `optional`
- `repo_overridden`
- `source_policy`
- `downloaded_at`
- `target_path`
- `required_paths`
- `source_type`
- `repo_validation`

其中：

- `default_repo_id` 是模块里写死的默认仓库。
- `resolved_repo_id` 是最终实际使用的仓库，可能被 `repo_override` 覆盖。
- `repo_validation` 是调用 Hugging Face `model_info(...)` 校验得到的信息。

### 6.2 批量任务的 `download_report.json`

总报告会记录：

- `requested_names`
- `include_optional`
- `counts.downloaded`
- `counts.skipped`
- `counts.failed`
- `results`

如果某个模型失败，错误不会吞掉，而是会写到该模型对应的 `error` 字段里。

---

## 7. 几个容易混淆的点

### 7.1 `optional=True` 是什么意思

表示这个模型不会出现在“默认下载集合”里，但你仍然可以：

- 用 `--include-optional` 一并下载
- 用 `--names ...` 单独点名下载

### 7.2 `community_mirror_only=True` 是什么意思

这类模型要求最终使用的仓库必须是 Transformers 兼容的 Hugging Face 仓库。

也就是说，如果你给这类模型传了 `repo_override`，指向了一个非 Transformers 仓库，例如只提供 GGUF 文件的仓库，下载前校验就会报错。

当前带这个约束的主要是各类 Llama 模型。

### 7.3 `force=True` 会做什么

对于 Hugging Face 下载器，如果目标目录已经存在，重新下载前会先删除旧目录，再重新拉取快照。

### 7.4 当前工作目录会影响下载位置吗

不会。

下载根目录是按照 `thesis_platform` 包路径固定解析的，不是按命令执行时的 shell 当前目录决定的。

---

## 8. 推荐的实际使用方式

如果你的目标只是把项目所需模型准备好，建议按下面顺序使用：

1. 在仓库根目录安装依赖：`pip install -r thesis_platform/requirements.txt`
2. 先看可用模型：`python -m thesis_platform.scripts.download_models --list`
3. 先下载默认模型：`python -m thesis_platform.scripts.download_models`
4. 如果某些大模型或额外模型需要补齐，再用 `--names` 或 `--include-optional`
5. 如果某个默认仓库失效，再用 `--repo-override`

---

## 9. 一个最小示例

只下载 `opt_125m` 和 `roberta_large`：

```powershell
cd D:\学习记录\导师项目\研究\caiqiyue_file
python -m thesis_platform.scripts.download_models --names opt_125m roberta_large
```

下载完成后可以到这里查看：

- `D:/学习记录/导师项目/研究/caiqiyue_file/thesis_platform/open_model/opt_125m/`
- `D:/学习记录/导师项目/研究/caiqiyue_file/thesis_platform/open_model/roberta_large/`
- `D:/学习记录/导师项目/研究/caiqiyue_file/thesis_platform/open_model/download_report.json`

---

## 10. 结论

这个包的设计思路很简单：

- 每个具体模型模块只描述“我是谁、默认从哪个仓库下、是否可选”。
- 真正的通用逻辑都集中在 `base.py`、`hf.py`、`registry.py`、`controller.py`。
- 日常使用时，优先用 `python -m thesis_platform.scripts.download_models`。
- 如果需要嵌入到你自己的代码里，再使用 `download_models(...)` 或 `create_model_downloader(...)`。

