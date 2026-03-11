# `thesis_platform.model_downloaders` 使用说明

## 1. 这个包是做什么的

`thesis_platform/model_downloaders` 是论文复现实验里的模型下载子系统。

它负责：

- 注册每个模型对应的下载器类。
- 统一解析“默认下载哪些模型”。
- 从 Hugging Face 拉取模型快照。
- 跳过已存在模型，或按 `force=True` 重新下载。
- 某个模型失败时继续处理后续模型。
- 生成单模型 `metadata.json` 和批量 `download_report.json`。

它不负责模型推理，只负责把模型文件下载到项目约定目录。

---

## 2. 模型会下载到哪里

所有模型都会下载到：

`thesis_platform/open_model/`

在你的仓库里，实际路径是：

`D:\学习记录\导师项目\研究\caiqiyue_file\thesis_platform\open_model`

每个模型默认放在：

`thesis_platform/open_model/<model_name>/`

例如：

- `thesis_platform/open_model/opt_125m/`
- `thesis_platform/open_model/roberta_large/`
- `thesis_platform/open_model/llama_3_1_8b_instruct/`

批量下载完成后，还会生成：

- `thesis_platform/open_model/download_report.json`

每个模型目录下还会生成：

- `thesis_platform/open_model/<model_name>/metadata.json`

---

## 3. 包内每个模块是做什么的

### 3.1 核心模块

| 模块 | 作用 | 关键对象 |
| --- | --- | --- |
| `__init__.py` | 包入口，导入所有模型模块并暴露统一 API。 | `download_models`、`list_model_downloaders`、`create_model_downloader` |
| `base.py` | 定义下载器基类和标准结果对象。 | `BaseModelDownloader`、`ModelDownloadResult` |
| `common.py` | 提供路径、时间、目录删除、磁盘占用统计等通用工具。 | `models_root()`、`compute_path_size_bytes()`、`format_bytes()` |
| `registry.py` | 注册下载器并按名字选择下载器。 | `register_model_downloader`、`get_registered_model_names()`、`create_model_downloader()` |
| `hf.py` | Hugging Face 通用下载器实现。 | `HuggingFaceModelDownloader` |
| `controller.py` | 批量列出、筛选、下载模型并生成总报告。 | `list_model_downloaders()`、`resolve_model_downloaders()`、`download_models()` |

### 3.2 具体模型模块

这些模块各自只做一件事：声明一个具体模型的下载器，并注册到系统里。

| 模块 | `name` | 默认 `repo_id` | 参数规模 | `optional` | `large_model` |
| --- | --- | --- | --- | --- | --- |
| `deepseek_r1_distill_llama_70b.py` | `deepseek_r1_distill_llama_70b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | `70B` | 是 | 是 |
| `llama_2_13b_chat_hf.py` | `llama_2_13b_chat_hf` | `NousResearch/Llama-2-13b-chat-hf` | `13B` | 是 | 否 |
| `llama_3_1_405b_instruct.py` | `llama_3_1_405b_instruct` | `RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic` | `405B` | 是 | 是 |
| `llama_3_1_8b_instruct.py` | `llama_3_1_8b_instruct` | `unsloth/Meta-Llama-3.1-8B-Instruct` | `8B` | 否 | 否 |
| `llama_3_2_11b_vision_instruct.py` | `llama_3_2_11b_vision_instruct` | `unsloth/Llama-3.2-11B-Vision-Instruct` | `11B` | 是 | 否 |
| `llama_3_2_3b_instruct.py` | `llama_3_2_3b_instruct` | `unsloth/Llama-3.2-3B-Instruct` | `3B` | 是 | 否 |
| `opt_125m.py` | `opt_125m` | `facebook/opt-125m` | `125M` | 否 | 否 |
| `opt_1_3b.py` | `opt_1_3b` | `facebook/opt-1.3b` | `1.3B` | 是 | 否 |
| `opt_350m.py` | `opt_350m` | `facebook/opt-350m` | `350M` | 否 | 否 |
| `phi_1_5.py` | `phi_1_5` | `microsoft/phi-1_5` | `1.5B` | 否 | 否 |
| `roberta_large.py` | `roberta_large` | `roberta-large` | `355M` | 否 | 否 |
| `stable_diffusion_v1_5.py` | `stable_diffusion_v1_5` | `runwayml/stable-diffusion-v1-5` | `~1.0B` | 否 | 否 |

说明：

- `large_model=True` 的判定阈值是大于 `15B`。
- 目前 `>15B` 的模型默认不会进入批量全量下载。

---

## 4. 默认会下载哪些模型

### 4.1 默认批量下载

不传任何参数时，只会下载：

- 非 `optional`
- 且参数规模不超过 `15B`

当前默认集合是：

- `llama_3_1_8b_instruct`
- `opt_125m`
- `opt_350m`
- `phi_1_5`
- `roberta_large`
- `stable_diffusion_v1_5`

### 4.2 `--include-optional`

`--include-optional` 只会把可选模型里不超过 `15B` 的模型加入默认集合，例如：

- `llama_2_13b_chat_hf`
- `llama_3_2_11b_vision_instruct`
- `llama_3_2_3b_instruct`
- `opt_1_3b`

它不会把 `70B`、`405B` 这种大模型自动加入。

### 4.3 `--include-large`

`--include-large` 会把大于 `15B` 的模型也加入默认集合，例如：

- `deepseek_r1_distill_llama_70b`
- `llama_3_1_405b_instruct`

### 4.4 `--names`

`--names` 是显式点名下载。

只要你明确写了模型名，就可以单独下载该模型，包括大于 `15B` 的模型。

例如：

```powershell
python -m thesis_platform.scripts.download_models --names llama_3_1_405b_instruct
```

---

## 5. 下载流程是什么样的

整体流程如下：

1. `__init__.py` 导入所有具体模型模块。
2. 每个模型模块通过 `@register_model_downloader` 注册自己。
3. `controller.py` 根据 `names`、`include_optional`、`include_large` 解析本次要下载的模型集合。
4. 每个下载器调用 `BaseModelDownloader.download(...)` 执行统一流程。
5. 具体下载动作由 `HuggingFaceModelDownloader.perform_download(...)` 完成。
6. 成功后写入 `metadata.json`，批量任务结束后写入 `download_report.json`。

默认行为还包括：

- 如果模型目录和 `metadata.json` 已存在，且没有 `force=True`，则跳过。
- 如果传了 `force=True`，会删除旧目录后重新下载。
- 如果某个模型下载失败，不会中断整个批次，而是记录失败并继续下一个模型。
- 下载成功或跳过时，会统计该模型目录的磁盘占用。

---

## 6. 怎么用这些模块下载模型

### 6.1 推荐方式：命令行入口

先安装依赖：

```powershell
cd D:\学习记录\导师项目\研究\caiqiyue_file
pip install -r thesis_platform\requirements.txt
```

列出所有已注册模型：

```powershell
python -m thesis_platform.scripts.download_models --list
```

下载默认模型集合：

```powershell
python -m thesis_platform.scripts.download_models
```

下载默认模型集合，并包含可选的非大模型：

```powershell
python -m thesis_platform.scripts.download_models --include-optional
```

下载默认模型集合，并把大于 `15B` 的模型也纳入：

```powershell
python -m thesis_platform.scripts.download_models --include-large
```

同时包含可选模型和大模型：

```powershell
python -m thesis_platform.scripts.download_models --include-optional --include-large
```

只下载指定模型：

```powershell
python -m thesis_platform.scripts.download_models --names opt_125m roberta_large
```

显式下载大模型：

```powershell
python -m thesis_platform.scripts.download_models --names deepseek_r1_distill_llama_70b
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

### 6.2 后台下载脚本（Linux 服务器，自动托管 Clash）

如果你是在 Linux 服务器上跑下载，希望断开 SSH 连接后下载仍然继续，可以用：

`thesis_platform/scripts/download_models_include_optional_bg.sh`

这个脚本会按下面的顺序工作：

- 先检查 `clash_for_linux` 是否已经在本机 `127.0.0.1:7890` 提供代理
- 如果没有，就自动启动 `clash_for_linux/clash`
- 自动读取 `clash_for_linux/run.txt` 里的 `http_proxy` / `https_proxy`
- 再在后台执行模型下载

脚本内部实际下载命令是：

```bash
python -m thesis_platform.scripts.download_models --include-optional
```

并且会通过 `nohup` 把 Clash 和模型下载一起托管到后台运行。

首次使用：

```bash
cd /mnt/public/caiqiyue_file/code_from_paper
chmod +x thesis_platform/scripts/download_models_include_optional_bg.sh
conda activate caiqiyue
./thesis_platform/scripts/download_models_include_optional_bg.sh start
```

如果你不想依赖当前激活环境，也可以显式指定 Python：

```bash
PYTHON_BIN=/home/k8smaster/anaconda3/envs/caiqiyue/bin/python \
./thesis_platform/scripts/download_models_include_optional_bg.sh start
```

常用命令：

```bash
./thesis_platform/scripts/download_models_include_optional_bg.sh status
./thesis_platform/scripts/download_models_include_optional_bg.sh logs
./thesis_platform/scripts/download_models_include_optional_bg.sh clash-logs
./thesis_platform/scripts/download_models_include_optional_bg.sh stop
```

这个脚本会把状态文件和日志写到：

- `thesis_platform/open_model/download_models_include_optional.log`
- `thesis_platform/open_model/download_models_include_optional.pid`
- `thesis_platform/open_model/clash_for_linux.log`
- `thesis_platform/open_model/clash_for_linux.pid`

说明：

- `start`：后台启动下载任务
- `status`：同时查看模型下载和 Clash 是否仍在运行
- `logs`：查看模型下载日志
- `clash-logs`：查看 Clash 日志
- `stop`：停止模型下载，并停止由这个脚本托管的 Clash 进程

适用场景：

- 远程 SSH 登录服务器后启动下载
- 关闭终端或断开连接后，Clash 和下载继续运行
- 之后重新登录服务器再查看下载日志、Clash 日志和状态

可选参数：

- `PYTHON_BIN=/path/to/python`：显式指定用于下载的 Python
- `START_CLASH=0`：如果你已经在系统里手动启动了 Clash，可以跳过脚本内的 Clash 启动逻辑

### 6.3 Python API：批量下载

```python
from thesis_platform.model_downloaders import download_models

report = download_models(
    names=["opt_125m", "roberta_large"],
    force=False,
    include_optional=False,
    include_large=False,
    repo_overrides=None,
)

print(report)
```

### 6.4 Python API：创建单个下载器

```python
from thesis_platform.model_downloaders import create_model_downloader

downloader = create_model_downloader("opt_125m")
result = downloader.download(force=False)
print(result.to_dict())
```

### 6.5 直接使用某个具体模块

```python
from thesis_platform.model_downloaders.llama_3_1_8b_instruct import Llama31_8BInstructDownloader

downloader = Llama31_8BInstructDownloader()
result = downloader.download()
print(result.to_dict())
```

通常还是优先用 `download_models(...)` 或命令行入口，统一性更好。

---

## 7. 下载结果里会写什么

### 7.1 单模型 `metadata.json`

每个模型成功下载后，会写一个 `metadata.json`，里面通常包括：

- `name`
- `description`
- `repo_id`
- `default_repo_id`
- `resolved_repo_id`
- `optional`
- `parameter_count_billions`
- `model_size_label`
- `large_model`
- `repo_overridden`
- `source_policy`
- `downloaded_at`
- `target_path`
- `required_paths`
- `source_type`
- `repo_validation`
- `disk_usage_bytes`
- `disk_usage_human`

### 7.2 批量 `download_report.json`

总报告会记录：

- `requested_names`
- `include_optional`
- `include_large`
- `counts.downloaded`
- `counts.skipped`
- `counts.failed`
- `results`

每个模型的结果项里会包含：

- `name`
- `status`
- `target_path`
- `metadata_path`
- `repo_id`
- `parameter_count_billions`
- `model_size_label`
- `large_model`
- `disk_usage_bytes`
- `disk_usage_human`
- `error`

也就是说，汇总报告现在会同时给出：

- 模型规模标签，例如 `8B`、`70B`
- 实际落盘占用，例如 `14.2 GB`

### 7.3 失败时的行为

如果某个模型下载失败：

- 该模型会在报告里标记为 `failed`
- `error` 字段会记录异常信息
- 控制器会继续下载后续模型，不会整批中断

---

## 8. 容易混淆的点

### 8.1 `optional=True` 不是“永远不能默认下载”

它只是表示：

- 默认批量下载时先不下
- 你可以通过 `--include-optional` 或 `--names` 把它纳入

### 8.2 `large_model=True` 的模型默认不会被“全量下载”

这里的“全量下载”指：

```powershell
python -m thesis_platform.scripts.download_models
```

或者：

```powershell
python -m thesis_platform.scripts.download_models --include-optional
```

这两种情况下，大于 `15B` 的模型仍然不会自动下载。

只有两种方式会下大模型：

- 显式传 `--include-large`
- 用 `--names` 明确点名

### 8.3 `community_mirror_only=True` 的含义

这类模型要求最终仓库是 Transformers 兼容的 Hugging Face 仓库。

如果你给它传了一个只包含 GGUF 等非 Transformers 产物的仓库，下载前校验会失败。

### 8.4 `force=True` 会做什么

对 Hugging Face 下载器来说，目标目录已存在时会先删除旧目录，再重新拉取快照。

### 8.5 当前工作目录不会改变下载位置

不会。

下载根目录是按 `thesis_platform` 包路径固定解析的，不是按你执行命令时所在的 shell 目录决定的。

---

## 9. 推荐的实际用法

如果你的目标只是把论文复现所需模型准备好，建议按这个顺序：

1. 安装依赖：`pip install -r thesis_platform/requirements.txt`
2. 先看可用模型：`python -m thesis_platform.scripts.download_models --list`
3. 先下载默认模型：`python -m thesis_platform.scripts.download_models`
4. 如果需要额外的可选中型模型，再加 `--include-optional`
5. 如果确实需要 `>15B` 大模型，再显式用 `--include-large` 或 `--names`
6. 如果默认仓库不可用，再用 `--repo-override`

---

## 10. 一个最小示例

只下载 `opt_125m` 和 `roberta_large`：

```powershell
cd D:\学习记录\导师项目\研究\caiqiyue_file
python -m thesis_platform.scripts.download_models --names opt_125m roberta_large
```

下载完成后可以查看：

- `D:\学习记录\导师项目\研究\caiqiyue_file\thesis_platform\open_model\opt_125m\`
- `D:\学习记录\导师项目\研究\caiqiyue_file\thesis_platform\open_model\roberta_large\`
- `D:\学习记录\导师项目\研究\caiqiyue_file\thesis_platform\open_model\download_report.json`

---

## 11. 结论

这个包的设计思路很直接：

- 每个具体模型模块只描述“我是谁、默认从哪个仓库下、是不是可选、是不是大模型”。
- 通用逻辑集中在 `base.py`、`hf.py`、`registry.py`、`controller.py`。
- 默认批量下载不会自动拉取 `15B` 以上模型。
- 报告里现在会同时给出模型规模和实际磁盘占用。
- 某个模型失败时会记录错误并继续后续下载。
