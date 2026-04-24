# PrE-Text 快速对比实验结果

更新时间：2026-04-24

## 1. 文档用途

本文档记录 `PrE-Text` 在 `screening-balanced` 快速对比实验配置下的 4 个数据集实验结果。

本轮实验的作用不是替代正式实验，而是作为后续新版创新算法 screening 对比的基线结果。

## 2. 实验配置口径

- 实验类型：`single_node_screening`
- 算法：`PrE-Text`
- 运行环境：旧服务器 `paper-2` 分支，`pretext` 虚拟环境，`A6000`
- Stage 2：`llama2_7b + vllm`
- 下游评估：`gpt2 small eval`
- 统一 screening 参数：
  - `train_limit = 256`
  - `eval_limit = 256`
  - `initialization_limit = 1024`
  - `num_prompts = 100`
  - `epochs = 6`

## 3. 四个数据集结果总表

| 实验 | 数据集 | synthetic_train_count | eval_count | best_top1 |
| --- | --- | ---: | ---: | ---: |
| `SP-S-JOBS` | `jobs` | 94 | 256 | 0.2731984829329962 |
| `SP-S-CONG` | `congressional` | 94 | 256 | 0.2949640287769784 |
| `SP-S-FORUMS` | `forums` | 92 | 256 | 0.25014487154722814 |
| `SP-S-MICRO` | `microblog` | 96 | 256 | 0.2762705387848682 |

## 4. 分实验记录

### 4.1 `SP-S-JOBS`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_jobs_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_jobs_screening`
- 结果摘要：
  - `epochs = 6`
  - `synthetic_train_count = 94`
  - `eval_count = 256`
  - `best_top1 = 0.2731984829329962`

### 4.2 `SP-S-CONG`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_congressional_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_congressional_screening`
- 结果摘要：
  - `epochs = 6`
  - `synthetic_train_count = 94`
  - `eval_count = 256`
  - `best_top1 = 0.2949640287769784`

### 4.3 `SP-S-FORUMS`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_forums_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_forums_screening`
- 结果摘要：
  - `epochs = 6`
  - `synthetic_train_count = 92`
  - `eval_count = 256`
  - `best_top1 = 0.25014487154722814`

### 4.4 `SP-S-MICRO`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_microblog_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_microblog_screening`
- 结果摘要：
  - `epochs = 6`
  - `synthetic_train_count = 96`
  - `eval_count = 256`
  - `best_top1 = 0.2762705387848682`

## 5. 当前用途说明

这 4 组结果应作为后续新版创新算法 screening 的对照基线。

后续判断某个创新点是否值得继续投入时，应采用同一套 screening 配置，直接与本表结果进行比较，而不是与正式实验结果交叉比较。
