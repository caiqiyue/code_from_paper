# `thesis_platform/dataset_downloaders` 使用说明

本文档说明 `thesis_platform/dataset_downloaders` 目录下每个模块的作用、如何使用这些模块下载数据集、数据会被下载到哪里，以及不同数据集的特殊行为。

## 1. 这套下载系统的整体作用

这套模块的目标是：

1. 用统一接口下载或生成论文实验需要的数据集。
2. 把原始数据整理到固定目录。
3. 在需要时把原始数据进一步转换成实验实际使用的格式。
4. 为每个数据集写出 `metadata.json`，并为整批下载写出汇总报告。

数据集默认不会下载到你当前 PowerShell 所在目录，而是固定下载到：

```text
D:\学习记录\导师项目\研究\caiqiyue_file\thesis_platform\datasets
```

这是由 `common.py` 里的 `datasets_root()` 决定的。

## 2. 下载后的目录结构

每个数据集一般会落到：

```text
thesis_platform/datasets/<dataset_name>/
  metadata.json
  raw/
  formatted/
```

含义如下：

- `raw/`：原始下载产物。通常是 Hugging Face `save_to_disk()` 结果，或者 GitHub 下载下来的原始 JSON。
- `formatted/`：实验实际消费的格式化结果，例如 JSONL、CSV、或筛选后的 Hugging Face 数据集目录。
- `metadata.json`：该数据集的下载元数据、来源说明、格式说明、切分规则等。

整批下载结束后，还会额外生成：

```text
thesis_platform/datasets/download_report.json
```

这个文件记录本次下载中哪些数据集成功、跳过或失败。

## 3. 最推荐的使用方式

### 3.1 安装依赖

在仓库根目录执行：

```powershell
cd D:\学习记录\导师项目\研究\caiqiyue_file
python -m pip install -r thesis_platform\requirements.txt
```

如果你使用虚拟环境，先激活虚拟环境再执行。

### 3.2 查看有哪些可下载的数据集

```powershell
python -m thesis_platform.scripts.download_datasets --list
```

### 3.3 下载全部已注册数据集

```powershell
python -m thesis_platform.scripts.download_datasets
```

### 3.4 只下载指定数据集

```powershell
python -m thesis_platform.scripts.download_datasets --names glue_sst2 gsm8k
```

### 3.5 强制重新下载

```powershell
python -m thesis_platform.scripts.download_datasets --force
```

说明：

- 如果某个数据集已经准备好，且你没有传 `--force`，系统会跳过它。
- 跳过逻辑由 `BaseDatasetDownloader.is_ready()` 和 `BaseDatasetDownloader.download()` 决定。
- 整批下载时，即使某个数据集失败，系统也会继续处理后面的数据集，并把失败信息写入 `download_report.json`。

## 4. 也可以在 Python 里直接调用

### 4.1 用统一入口调用

```python
from thesis_platform.dataset_downloaders import (
    create_dataset_downloader,
    download_datasets,
    get_registered_dataset_names,
)

print(get_registered_dataset_names())

result = create_dataset_downloader("gsm8k").download(force=False)
print(result.to_dict())

summary = download_datasets(names=["glue_sst2", "gsm8k"], force=False)
print(summary["counts"])
```

### 4.2 直接实例化具体下载器类

```python
from thesis_platform.dataset_downloaders.gsm8k import GSM8KDownloader

result = GSM8KDownloader().download()
print(result.to_dict())
```

推荐优先使用包级入口：

- `create_dataset_downloader(name)`
- `download_datasets(names=None, force=False)`
- `list_dataset_downloaders()`

因为这样更符合当前项目设计。

## 5. 模块分层

这个目录里的模块大致分四类。

### 5.1 基础框架模块

| 模块 | 作用 |
| --- | --- |
| `__init__.py` | 导入所有下载器模块并暴露统一入口，例如 `create_dataset_downloader`、`download_datasets`。 |
| `base.py` | 定义 `BaseDatasetDownloader` 和 `DatasetDownloadResult`，是所有数据集下载器的公共基类。 |
| `registry.py` | 负责注册下载器类、按名称创建下载器、解析名称列表。 |
| `controller.py` | 负责整批下载、统计结果、写出 `download_report.json`。 |
| `common.py` | 提供公共路径函数、时间戳函数、复制/移动/删除文件等辅助函数。 |

### 5.2 通用下载辅助模块

| 模块 | 作用 |
| --- | --- |
| `hf.py` | 定义 `HuggingFaceDatasetDownloader`，适合从 Hugging Face `datasets.load_dataset()` 下载并保存到 `raw/`。 |
| `bbh_utils.py` | 提供 BBH 任务原始 JSON 的下载逻辑，从 GitHub 拉取 `BIG-Bench-Hard` 对应任务文件。 |
| `datainf_generation.py` | 负责调用本地 `DataInf/src/generate_sentence-math_datasets.py` 脚本，并把产物移入 `thesis_platform/datasets/<name>/formatted/`。 |

### 5.3 具体数据集下载器模块

每个具体数据集对应一个模块，内部通常定义一个下载器类，并通过 `@register_dataset_downloader` 注册。

### 5.4 格式化模块

真正把 `raw/` 变成实验可用格式的逻辑，不在本目录，而在：

```text
thesis_platform/dataset_formatters/
```

下载器会通过 `formatter_name` 找到对应 formatter。

## 6. 每个模块是干什么的

### 6.1 框架与辅助模块

#### `base.py`

用途：

- 定义所有下载器的公共行为。
- 统一处理 `dataset_root()`、`raw_path()`、`formatted_path()`、`metadata_path()`。
- 统一处理“是否已准备好”“是否跳过”“写 metadata”“调用 formatter”等流程。

核心行为：

- `download(force=False)` 是单个下载器最重要的入口。
- 如果数据已存在且完整，就返回 `skipped`。
- 否则先执行 `perform_download_raw()`，再执行 formatter，最后写 `metadata.json`。

#### `registry.py`

用途：

- 注册所有下载器类。
- 根据名称创建下载器实例。
- 检查用户输入的数据集名是否合法。

常用函数：

- `get_registered_dataset_names()`
- `create_dataset_downloader(name)`
- `resolve_dataset_names(names)`

#### `controller.py`

用途：

- 一次性下载多个数据集。
- 出错不中断整批流程。
- 输出下载汇总报告。

常用函数：

- `list_dataset_downloaders()`
- `resolve_dataset_downloaders(names=None)`
- `download_datasets(names=None, force=False)`

#### `common.py`

用途：

- 定义固定下载根目录。
- 提供路径转换、删除目录、复制文件、移动文件等通用函数。

最关键的是：

- `datasets_root()` 返回固定目录 `thesis_platform/datasets`
- 不依赖你当前终端所在目录

#### `hf.py`

用途：

- 给“从 Hugging Face 下载数据”的下载器提供通用实现。

工作方式：

1. 子类实现 `build_raw_dataset()`
2. 返回 `dataset, metadata`
3. 基类自动把数据保存到 `raw/`
4. 后续再交给 formatter 处理

#### `bbh_utils.py`

用途：

- 下载 BBH 任务原始 JSON 文件。

数据来源：

- `https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/<task>.json`

#### `datainf_generation.py`

用途：

- 处理 DataInf 系列数据集。
- 这些数据集不是简单在线下载，而是通过本地脚本生成。

关键行为：

- 调用 `DataInf/src/generate_sentence-math_datasets.py`
- 生成临时输出到 `thesis_platform/datasets/`
- 再移动到各自数据集目录下的 `formatted/train.hf` 和 `formatted/test.hf`

## 7. 每个具体数据集模块说明

下表说明每个下载器模块的用途、数据来源、格式化方式和最终结果。

| 模块 | 下载器名称 | 数据来源 | formatter | 最终结果说明 |
| --- | --- | --- | --- | --- |
| `bbh_multistep_arithmetic_two.py` | `bbh_multistep_arithmetic_two` | GitHub 上的 BBH 原始 JSON | `bbh` | 下载原始 JSON 到 `raw/`，再生成 `formatted/train.csv`、`val.csv`、`test.csv` |
| `bbh_object_counting.py` | `bbh_object_counting` | GitHub 上的 BBH 原始 JSON | `bbh` | 下载原始 JSON 到 `raw/`，再生成 `formatted/train.csv`、`val.csv`、`test.csv` |
| `datainf_grammars.py` | `datainf_grammars` | 本地 `DataInf` 脚本生成 | `datainf` | 生成 `formatted/train.hf`、`formatted/test.hf` |
| `datainf_math_with_reason.py` | `datainf_math_with_reason` | 本地 `DataInf` 脚本生成 | `datainf` | 生成 `formatted/train.hf`、`formatted/test.hf` |
| `datainf_math_without_reason.py` | `datainf_math_without_reason` | 本地 `DataInf` 脚本生成 | `datainf` | 生成 `formatted/train.hf`、`formatted/test.hf` |
| `glue_mrpc.py` | `glue_mrpc` | Hugging Face `glue`, subset `mrpc` 的 `train/validation` | `glue_datainf` | `raw/` 保存官方 `train/validation`；`formatted/` 保存按 DataInf 实验规则裁剪后的 GLUE 子集 |
| `glue_qnli.py` | `glue_qnli` | Hugging Face `glue`, subset `qnli` 的 `train/validation` | `glue_datainf` | `raw/` 保存官方 `train/validation`；`formatted/` 保存按 DataInf 实验规则裁剪后的 GLUE 子集 |
| `glue_qqp.py` | `glue_qqp` | Hugging Face `glue`, subset `qqp` 的 `train/validation` | `glue_datainf` | `raw/` 保存官方 `train/validation`；`formatted/` 保存按 DataInf 实验规则裁剪后的 GLUE 子集 |
| `glue_sst2.py` | `glue_sst2` | Hugging Face `glue`, subset `sst2` 的 `train/validation` | `glue_datainf` | `raw/` 保留 GRADMM 需要的官方 `train/validation`；`formatted/` 额外保存 DataInf 需要的裁剪版 GLUE 子集 |
| `glue_wnli.py` | `glue_wnli` | Hugging Face `glue`, subset `wnli` 的 `train/validation` | `glue_datainf` | `raw/` 保存官方 `train/validation`；`formatted/` 保存按 DataInf 实验规则裁剪后的 GLUE 子集 |
| `gsm8k.py` | `gsm8k` | Hugging Face `gsm8k`, subset `main` | `gsm8k` | 保存原始 HF 数据到 `raw/`，并额外生成 DSPy 风格 `formatted/train.jsonl`、`val.jsonl`、`test.jsonl` |
| `imdb.py` | `imdb` | 本地 `GRADMM/data/imdb/*.jsonl` | `imdb` | 不再下载官方 HF IMDB；直接把论文使用的 vendored `train_len256.jsonl` 和 `validation_len256.jsonl` 复制到 `formatted/` |
| `livebench_math_amps_hard.py` | `livebench_math_amps_hard` | Hugging Face `livebench/math` 的 `test` split 中 `task=AMPS_Hard` 的子集 | `livebench` | `raw/` 已经只保留论文任务子集；`formatted/` 再按确定性 64/16/20 切成 `train.jsonl`、`valid.jsonl`、`test.jsonl` |
| `livebench_reasoning_spatial.py` | `livebench_reasoning_spatial` | Hugging Face `livebench/reasoning` 的 `test` split 中 `task=spatial` 的子集 | `livebench` | `raw/` 已经只保留论文任务子集；`formatted/` 再按确定性 64/16/20 切成 `train.jsonl`、`valid.jsonl`、`test.jsonl` |
| `livebench_reasoning_web_of_lies_v2.py` | `livebench_reasoning_web_of_lies_v2` | Hugging Face `livebench/reasoning` 的 `test` split 中 `task=web_of_lies_v2` 的子集 | `livebench` | `raw/` 已经只保留论文任务子集；`formatted/` 再按确定性 64/16/20 切成 `train.jsonl`、`valid.jsonl`、`test.jsonl` |
| `livebench_reasoning_zebra_puzzle.py` | `livebench_reasoning_zebra_puzzle` | Hugging Face `livebench/reasoning` 的 `test` split 中 `task=zebra_puzzle` 的子集 | `livebench` | `raw/` 已经只保留论文任务子集；`formatted/` 再按确定性 64/16/20 切成 `train.jsonl`、`valid.jsonl`、`test.jsonl` |
| `rotten_tomatoes.py` | `rotten_tomatoes` | Hugging Face `rotten_tomatoes` 的 `train/validation` | `identity` | 只下载 GRADMM 实验会用到的 `train/validation`，`formatted_path()` 仍与 `raw/` 相同 |
| `rt_polarity.py` | `rt_polarity` | 本地 `GRADMM/data/rtpolarity` | `rt_polarity` | 不单独下载 `raw/`，直接复制为 `formatted/train.jsonl` 和 `formatted/validation.jsonl` |
| `three_styles_prompted_250_512x512.py` | `three_styles_prompted_250_512x512` | Hugging Face `kewu93/three_styles_prompted_250_512x512` | `identity` | 原始 Hugging Face 数据直接作为实验可用结果，主要落在 `raw/` |
| `twitter_emotion_binary.py` | `twitter_emotion_binary` | Hugging Face `dair-ai/emotion`, subset `split` 的 `train/validation` | `twitter_emotion_binary` | `raw/` 只保留 GRADMM 会用到的 `train/validation`；`formatted/` 再筛选 `label in [0, 1]` 得到 sadness/joy 二分类子集 |

## 8. 常见数据集的实际落盘位置

### 8.1 `glue_sst2`

下载后会同时有：

```text
thesis_platform/datasets/glue_sst2/raw/
thesis_platform/datasets/glue_sst2/formatted/
thesis_platform/datasets/glue_sst2/metadata.json
```

其中：
- `raw/` 保存官方 `train/validation`，便于对齐 GRADMM。
- `formatted/` 保存按 DataInf 论文流程裁剪后的 GLUE 子集。

### 8.2 `gsm8k`

下载后会有：

```text
thesis_platform/datasets/gsm8k/raw/
thesis_platform/datasets/gsm8k/formatted/train.jsonl
thesis_platform/datasets/gsm8k/formatted/val.jsonl
thesis_platform/datasets/gsm8k/formatted/test.jsonl
thesis_platform/datasets/gsm8k/metadata.json
```

### 8.3 `livebench_reasoning_spatial`

下载后会有：

```text
thesis_platform/datasets/livebench_reasoning_spatial/raw/
thesis_platform/datasets/livebench_reasoning_spatial/formatted/train.jsonl
thesis_platform/datasets/livebench_reasoning_spatial/formatted/valid.jsonl
thesis_platform/datasets/livebench_reasoning_spatial/formatted/test.jsonl
thesis_platform/datasets/livebench_reasoning_spatial/metadata.json
```

### 8.4 `datainf_grammars`

下载后会有：

```text
thesis_platform/datasets/datainf_grammars/formatted/train.hf
thesis_platform/datasets/datainf_grammars/formatted/test.hf
thesis_platform/datasets/datainf_grammars/metadata.json
```

DataInf 系列没有独立 `raw/`。

### 8.5 `imdb`

下载后会有：

```text
thesis_platform/datasets/imdb/formatted/train_len256.jsonl
thesis_platform/datasets/imdb/formatted/validation_len256.jsonl
thesis_platform/datasets/imdb/metadata.json
```

`imdb` 不再生成单独的 `raw/`，因为论文实验直接使用仓库自带的 `GRADMM/data/imdb/*.jsonl`。

## 9. 不同数据集的特殊依赖

### 9.1 纯 Hugging Face 下载型

这类数据集主要依赖：

- `datasets`
- 网络访问 Hugging Face

包含：

- `glue_*`
- `gsm8k`
- `livebench_*`
- `rotten_tomatoes`
- `three_styles_prompted_250_512x512`
- `twitter_emotion_binary`

### 9.2 依赖 GitHub 原始文件

包含：

- `bbh_multistep_arithmetic_two`
- `bbh_object_counting`

它们会从 GitHub 的 BBH 仓库下载对应任务 JSON。

### 9.3 依赖本地 `DataInf` 目录

包含：

- `datainf_grammars`
- `datainf_math_with_reason`
- `datainf_math_without_reason`

要求仓库根目录存在：

```text
DataInf/src/generate_sentence-math_datasets.py
```

当前你的仓库里已经有 `DataInf/` 目录，因此这部分结构是存在的。

### 9.4 依赖本地 `GRADMM` 目录

包含：

- `imdb`
- `rt_polarity`

要求存在：

```text
GRADMM/data/imdb
GRADMM/data/rtpolarity
```

当前你的仓库里已经有 `GRADMM/` 目录。

## 10. 典型使用建议

### 10.1 如果你只是想先验证流程能跑通

建议先下载一个最简单的数据集：

```powershell
python -m thesis_platform.scripts.download_datasets --names glue_sst2
```

原因：

- 来源简单
- 不依赖 `GRADMM`
- 不依赖 `DataInf`
- 不需要额外格式转换

### 10.2 如果你想下载 FedTextGrad 相关数据

可以优先尝试：

```powershell
python -m thesis_platform.scripts.download_datasets --names gsm8k bbh_object_counting livebench_reasoning_spatial
```

### 10.3 如果你想下载 DataInf 相关数据

```powershell
python -m thesis_platform.scripts.download_datasets --names datainf_grammars datainf_math_without_reason datainf_math_with_reason
```

### 10.4 如果你想下载 GRADMM 相关数据

```powershell
python -m thesis_platform.scripts.download_datasets --names imdb rt_polarity twitter_emotion_binary rotten_tomatoes glue_sst2
```

## 11. 常见问题

### 11.1 为什么有些数据集只有 `raw/`，没有单独的 `formatted/`？

因为这些数据集使用 `identity` formatter，表示：

- 原始下载结果已经符合实验需要
- 不再额外复制一份到 `formatted/`

典型例子：

- `rotten_tomatoes`
- `three_styles_prompted_250_512x512`

`glue_*` 现在不再使用纯 `identity` 路径：
- `raw/` 保存官方 `train/validation`
- `formatted/` 保存 DataInf 实验真正使用的裁剪版子集

### 11.2 为什么 DataInf 系列没有 `raw/`？

因为它们不是“先下载原始数据，再格式化”，而是直接调用本地生成脚本，最终产物就是 `formatted/train.hf` 和 `formatted/test.hf`。

### 11.3 为什么整批下载时某个数据集失败后程序没有停？

这是 `controller.py` 的设计行为。它会捕获单个下载器异常，把失败记录进结果列表，然后继续处理后续数据集，最后统一写出 `download_report.json`。

## 12. 一句话总结

如果你只想实际使用这套系统，记住下面三条就够了：

1. 在仓库根目录运行 `python -m thesis_platform.scripts.download_datasets --list` 查看可用数据集。
2. 用 `python -m thesis_platform.scripts.download_datasets --names <dataset_name>` 下载指定数据集。
3. 所有数据都会固定下载到 `D:\学习记录\导师项目\研究\caiqiyue_file\thesis_platform\datasets`。
