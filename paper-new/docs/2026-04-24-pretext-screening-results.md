# `paper-2` 分支创新算法流程与快速对比实验结果

更新时间：2026-04-25

## 1. 文档用途

本文档用于统一记录两部分内容：

1. `paper-2` 分支中当前版本新版创新算法的完整具体流程。
2. `PrE-Text` 与新版创新算法在 4 个数据集上的 `screening-balanced` 快速对比实验结果。

本文档的用途不是替代后续 `formal` 正式实验，而是作为“当前创新点是否值得继续投入”的快速筛选依据。

## 2. 实验运行背景

- 运行分支：`paper-2`
- 运行环境：旧服务器 `pretext`
- 运行显卡：`A6000`
- 实验类型：`single_node_screening`
- Stage 2 生成模型：本地 `llama_2_7b_hf + vllm`
- 下游评估：`gpt2 small eval`
- 统一 screening 参数：
  - `train_limit = 256`
  - `eval_limit = 256`
  - `initialization_limit = 1024`
  - `num_prompts = 100`
  - `epochs = 6`

## 3. `paper-2` 分支中创新算法的整个具体流程

当前版本的新版创新算法是在 `PrE-Text` 两阶段框架上做局部替换与增强，核心创新集中在 `Stage 1 selector`，而不是修改 Stage 2 的 bootstrap 思想。

### 3.1 整体流程概览

1. 从初始化数据集 `D_init` 中采样 exemplar。
2. 使用固定模板提示词，通过本地 `llama_2_7b_hf + vllm` 生成 Stage 1 候选样本。
3. 对私有训练样本建立向量表示。
4. 对每个私有样本计算 `importance prior`。
5. 对每个候选样本计算 `Top-Q private support`。
6. 对每个候选样本计算 `genericity penalty`。
7. 对候选集合进行动态 `redundancy penalty` 计算。
8. 通过贪心选择器输出：
   - `selected seeds`
   - `hard negatives`
   - `boundary_state`
9. 使用 `selected seeds` 进入 Stage 2。
10. Stage 2 继续复用 `PrE-Text` 的 `build_bootstrap_prompts` 逻辑构造 bootstrap prompts。
11. Stage 2 直接复用 Stage 1 已经加载好的同一个 `vllm 7B engine`，不重新初始化新的 `LLM(...)`。
12. 生成 synthetic corpus。
13. 将 synthetic corpus 送入统一的 `gpt2 small eval`。
14. 输出：
   - `synthetic_train_count`
   - `eval_count`
   - `best_top1`
   - `best_top3`
   - `best_top5`
   - `best_top10`

### 3.2 Stage 1 详细流程

#### 3.2.1 候选生成

- 输入：
  - 初始化样本 `D_init`
  - 固定 prompt 模板
  - 本地 `llama_2_7b_hf + vllm`
- 输出：
  - 一批 Stage 1 候选样本

当前 screening 配置下，Stage 1 会在较小规模参数下生成候选，但仍保持完整算法链，不是只做简化版流程验证。

#### 3.2.2 私有样本重要性建模

对私有训练样本建立 embedding，并计算每个私有样本的重要性先验 `importance prior`。  
这一部分用于增强“更值得被候选样本覆盖的私有样本”在后续支持度计算中的权重。

#### 3.2.3 候选支持度计算

对每个候选样本，基于私有样本 embedding 计算 `Top-Q private support`。  
它反映的是候选样本在私有数据分布上的支持程度，而不是简单的最近邻单点匹配。

#### 3.2.4 泛化惩罚

对候选样本计算 `genericity penalty`。  
目标是抑制那些过于模板化、过于泛化、虽然“像样本”但对目标任务区分性不强的候选。

#### 3.2.5 冗余惩罚

在选择 seeds 的过程中动态计算 `redundancy penalty`。  
目标是避免多个 seeds 高度相似，导致最终 synthetic corpus 覆盖面变窄。

#### 3.2.6 贪心选择器输出

基于：

- `importance prior`
- `Top-Q private support`
- `genericity penalty`
- `redundancy penalty`

进行综合评分与贪心选择，最终得到：

- `selected seeds`
- `hard negatives`
- `boundary_state`

其中：

- `selected seeds` 用于进入 Stage 2
- `hard negatives` 与 `boundary_state` 用于刻画被拒绝候选与边界信息

### 3.3 Stage 2 详细流程

#### 3.3.1 Prompt 构造

Stage 2 不重新发明生成逻辑，而是继续复用 `PrE-Text` 的 `build_bootstrap_prompts` 机制，根据 Stage 1 输出的 `selected seeds` 构造 bootstrap prompts。

#### 3.3.2 共享 vLLM 生成

这是当前 `paper-2` 分支中的关键工程改动：

- Stage 1 与 Stage 2 共用同一个 `llama_2_7b_hf + vllm` engine
- Stage 2 不再重新初始化新的 `LLM(...)`

这样做的目的有两个：

1. 避免 Stage 2 二次加载 7B 模型造成额外显存峰值。
2. 避免在 screening 与 formal 运行中频繁出现 Stage 2 初始化 OOM。

#### 3.3.3 Synthetic Corpus 导出

Stage 2 将生成得到的 synthetic texts 导出为：

- `llama7b_text_syn.json`

后续下游评估直接读取该 synthetic corpus。

### 3.4 下游评估流程

下游统一走 `gpt2 small eval`，并且当前代码已经保证 `paper-new -> thesis_platform -> PrE-Text eval` 这条链正确传递：

- `train_limit`
- `eval_limit`
- `initialization_limit`

因此，新版创新算法与 `PrE-Text` 的 screening 现在已经能在同一数据规模口径下进行公平比较。

## 4. `PrE-Text` 60 次快速对比实验结果（15 轮 × 4 数据集）

### 4.1 实验运行记录

- 运行分支：`paper-2-genereic`
- 运行环境：旧服务器 `pretext`
- 运行显卡：`A6000`（`CUDA_VISIBLE_DEVICES=0`）
- 实验类型：`single_node_screening`
- 组织方式：15 轮，每轮严格串行 `jobs → congressional → forums → microblog`
- 实验次数：共 60 个实验（4 数据集 × 15 轮）
- 成功率：60/60（100%）
- 运行时间：2026-05-02 15:22:41 ~ 18:58:11（约 3.6 小时）

### 4.2 原始实验数据

| Round | Dataset | Seed | Status | best_top1 | synthetic_train_count |
| :---: | :--- | :---: | :---: | ---: | ---: |
| 1 | jobs | 1 | success | 0.2759 | 95 |
| 1 | congressional | 1 | success | 0.2911 | 96 |
| 1 | forums | 1 | success | 0.2477 | 94 |
| 1 | microblog | 1 | success | 0.2789 | 94 |
| 2 | jobs | 2 | success | 0.2767 | 95 |
| 2 | congressional | 2 | success | 0.2905 | 95 |
| 2 | forums | 2 | success | 0.2449 | 91 |
| 2 | microblog | 2 | success | 0.2781 | 93 |
| 3 | jobs | 3 | success | 0.2750 | 92 |
| 3 | congressional | 3 | success | 0.2937 | 98 |
| 3 | forums | 3 | success | 0.2501 | 93 |
| 3 | microblog | 3 | success | 0.2761 | 92 |
| 4 | jobs | 4 | success | 0.2750 | 95 |
| 4 | congressional | 4 | success | 0.2970 | 93 |
| 4 | forums | 4 | success | 0.2463 | 90 |
| 4 | microblog | 4 | success | 0.2767 | 91 |
| 5 | jobs | 5 | success | 0.2826 | 93 |
| 5 | congressional | 5 | success | 0.2950 | 92 |
| 5 | forums | 5 | success | 0.2494 | 93 |
| 5 | microblog | 5 | success | 0.2757 | 95 |
| 6 | jobs | 6 | success | 0.2784 | 93 |
| 6 | congressional | 6 | success | 0.2906 | 93 |
| 6 | forums | 6 | success | 0.2466 | 95 |
| 6 | microblog | 6 | success | 0.2758 | 96 |
| 7 | jobs | 7 | success | 0.2780 | 94 |
| 7 | congressional | 7 | success | 0.2911 | 95 |
| 7 | forums | 7 | success | 0.2449 | 96 |
| 7 | microblog | 7 | success | 0.2780 | 96 |
| 8 | jobs | 8 | success | 0.2753 | 95 |
| 8 | congressional | 8 | success | 0.2902 | 96 |
| 8 | forums | 8 | success | 0.2463 | 97 |
| 8 | microblog | 8 | success | 0.2765 | 96 |
| 9 | jobs | 9 | success | 0.2799 | 89 |
| 9 | congressional | 9 | success | 0.2969 | 89 |
| 9 | forums | 9 | success | 0.2478 | 95 |
| 9 | microblog | 9 | success | 0.2782 | 96 |
| 10 | jobs | 10 | success | 0.2800 | 95 |
| 10 | congressional | 10 | success | 0.2924 | 96 |
| 10 | forums | 10 | success | 0.2481 | 93 |
| 10 | microblog | 10 | success | 0.2788 | 93 |
| 11 | jobs | 11 | success | 0.2803 | 91 |
| 11 | congressional | 11 | success | 0.2914 | 96 |
| 11 | forums | 11 | success | 0.2440 | 93 |
| 11 | microblog | 11 | success | 0.2772 | 94 |
| 12 | jobs | 12 | success | 0.2772 | 96 |
| 12 | congressional | 12 | success | 0.2982 | 94 |
| 12 | forums | 12 | success | 0.2469 | 96 |
| 12 | microblog | 12 | success | 0.2763 | 93 |
| 13 | jobs | 13 | success | 0.2753 | 95 |
| 13 | congressional | 13 | success | 0.2888 | 98 |
| 13 | forums | 13 | success | 0.2409 | 97 |
| 13 | microblog | 13 | success | 0.2776 | 95 |
| 14 | jobs | 14 | success | 0.2765 | 94 |
| 14 | congressional | 14 | success | 0.2952 | 97 |
| 14 | forums | 14 | success | 0.2476 | 94 |
| 14 | microblog | 14 | success | 0.2783 | 96 |
| 15 | jobs | 15 | success | 0.2735 | 97 |
| 15 | congressional | 15 | success | 0.2926 | 94 |
| 15 | forums | 15 | success | 0.2461 | 92 |
| 15 | microblog | 15 | success | 0.2764 | 95 |

### 4.3 各数据集统计分析

| Dataset | Mean | Std | Min | Max | Median |
| :--- | ---: | ---: | ---: | ---: | ---: |
| jobs | 0.2768 | 0.0024 | **0.2735** | 0.2826 | 0.2765 |
| congressional | 0.2929 | 0.0026 | **0.2888** | 0.2982 | 0.2914 |
| forums | 0.2465 | 0.0023 | **0.2409** | 0.2501 | 0.2466 |
| microblog | 0.2775 | 0.0011 | **0.2757** | 0.2789 | 0.2769 |

### 4.4 各数据集性能最差实验（标记为 PrE-Text 快速对比标准结果）

> **说明**：以下结果为 PrE-Text 在各数据集上性能最差的实验，代表该算法在快速对比场景下的基线标准结果。

| Dataset | Round | Seed | best_top1 | synthetic_train_count | experiment_dir |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **jobs** | 15 | 15 | **0.2735** | 97 | `outputs/repeat15_rounds/round15_sp_s_jobs_screening_seed15` |
| **congressional** | 13 | 13 | **0.2888** | 98 | `outputs/repeat15_rounds/round13_sp_s_congressional_screening_seed13` |
| **forums** | 13 | 13 | **0.2409** | 97 | `outputs/repeat15_rounds/round13_sp_s_forums_screening_seed13` |
| **microblog** | 5 | 5 | **0.2757** | 95 | `outputs/repeat15_rounds/round05_sp_s_microblog_screening_seed5` |

---

## 5. 四个数据集快速对比实验总表（历史单次结果）

| 数据集 | 算法 | synthetic_train_count | eval_count | best_top1 | best_top3 | best_top5 | best_top10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `jobs` | `PrE-Text` | 94 | 256 | 0.2731984829329962 | 0.42237673830594186 | 0.4890012642225032 | 0.5697850821744627 |
| `jobs` | 新版创新算法 | 88 | 256 | 0.2761061946902655 | 0.4275600505689001 | 0.4928571428571429 | 0.5747155499367889 |
| `congressional` | `PrE-Text` | 94 | 256 | 0.2949640287769784 | 0.4601075896039925 | 0.5315963445459848 | 0.6188346620001296 |
| `congressional` | 新版创新算法 | 96 | 256 | 0.2969732322250308 | 0.4610149718063387 | 0.5337999870373971 | 0.6212975565493551 |
| `forums` | `PrE-Text` | 92 | 256 | 0.25014487154722814 | 0.3876762603824609 | 0.4547678835876634 | 0.5375056338935034 |
| `forums` | 新版创新算法 | 90 | 256 | 0.2470542785396948 | 0.3820101732019831 | 0.44935934582448006 | 0.5317751593587019 |
| `microblog` | `PrE-Text` | 96 | 256 | 0.2762705387848682 | 0.4185454082282512 | 0.4803846643739651 | 0.5627945484651636 |
| `microblog` | 新版创新算法 | 88 | 256 | 0.27493312953763854 | 0.41911858361992105 | 0.4793019997452554 | 0.5648325054133232 |

## 6. `PrE-Text` 四个快速对比实验结果（历史单次）

### 6.1 `SP-S-JOBS`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_jobs_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_jobs_screening`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 94`
  - `eval_count = 256`
  - `best_top1 = 0.2731984829329962`
  - `best_top3 = 0.42237673830594186`
  - `best_top5 = 0.4890012642225032`
  - `best_top10 = 0.5697850821744627`

### 6.2 `SP-S-CONG`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_congressional_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_congressional_screening`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 94`
  - `eval_count = 256`
  - `best_top1 = 0.2949640287769784`
  - `best_top3 = 0.4601075896039925`
  - `best_top5 = 0.5315963445459848`
  - `best_top10 = 0.6188346620001296`

### 6.3 `SP-S-FORUMS`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_forums_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_forums_screening`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 92`
  - `eval_count = 256`
  - `best_top1 = 0.25014487154722814`
  - `best_top3 = 0.3876762603824609`
  - `best_top5 = 0.4547678835876634`
  - `best_top10 = 0.5375056338935034`

### 6.4 `SP-S-MICRO`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_microblog_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_microblog_screening`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 96`
  - `eval_count = 256`
  - `best_top1 = 0.2762705387848682`
  - `best_top3 = 0.4185454082282512`
  - `best_top5 = 0.4803846643739651`
  - `best_top10 = 0.5627945484651636`

## 7. 新版创新算法四个快速对比实验结果

### 7.1 `NS-S-JOBS`

- 配置文件：`paper-new/configs/experiments/single_node_screening/ns_s_jobs_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_jobs_screening`
- Stage 2 语料文件：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_jobs_screening/eval/stage2/llama7b_text_syn.json`
- 下游评估摘要：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_jobs_screening/eval/downstream_eval_summary.json`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 88`
  - `eval_count = 256`
  - `best_top1 = 0.2761061946902655`
  - `best_top3 = 0.4275600505689001`
  - `best_top5 = 0.4928571428571429`
  - `best_top10 = 0.5747155499367889`

### 7.2 `NS-S-CONG`

- 配置文件：`paper-new/configs/experiments/single_node_screening/ns_s_congressional_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_congressional_screening`
- Stage 2 语料文件：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_congressional_screening/eval/stage2/llama7b_text_syn.json`
- 下游评估摘要：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_congressional_screening/eval/downstream_eval_summary.json`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 96`
  - `eval_count = 256`
  - `best_top1 = 0.2969732322250308`
  - `best_top3 = 0.4610149718063387`
  - `best_top5 = 0.5337999870373971`
  - `best_top10 = 0.6212975565493551`

### 7.3 `NS-S-FORUMS`

- 配置文件：`paper-new/configs/experiments/single_node_screening/ns_s_forums_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_forums_screening`
- Stage 2 语料文件：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_forums_screening/eval/stage2/llama7b_text_syn.json`
- 下游评估摘要：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_forums_screening/eval/downstream_eval_summary.json`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 90`
  - `eval_count = 256`
  - `best_top1 = 0.2470542785396948`
  - `best_top3 = 0.3820101732019831`
  - `best_top5 = 0.44935934582448006`
  - `best_top10 = 0.5317751593587019`

### 7.4 `NS-S-MICRO`

- 配置文件：`paper-new/configs/experiments/single_node_screening/ns_s_microblog_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_microblog_screening`
- Stage 2 语料文件：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_microblog_screening/eval/stage2/llama7b_text_syn.json`
- 下游评估摘要：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_microblog_screening/eval/downstream_eval_summary.json`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 88`
  - `eval_count = 256`
  - `best_top1 = 0.27493312953763854`
  - `best_top3 = 0.41911858361992105`
  - `best_top5 = 0.4793019997452554`
  - `best_top10 = 0.5648325054133232`

## 8. 当前快速对比结论

从这轮 `screening-balanced` 快速对比结果看：

1. `jobs`
   - 新版创新算法优于 `PrE-Text`
2. `congressional`
   - 新版创新算法优于 `PrE-Text`
3. `forums`
   - `PrE-Text` 优于新版创新算法
4. `microblog`
   - `PrE-Text` 优于新版创新算法

当前判断：

- 这版创新算法不是“完全无效”，因为它已经在 `jobs` 与 `congressional` 上表现出正向趋势。
- 但它也没有达到“全面超过 `PrE-Text`”的程度，因为在 `forums` 与 `microblog` 上仍然落后。
- 因此，这条创新线当前更适合继续做局部修改，而不是直接进入正式实验定稿。

下一步更合理的方向是：

- 围绕 `forums / microblog` 上的劣势继续分析；
- 优先找出当前 selector 设计在这两类数据集上的失效原因；
- 修改后继续进入下一轮 screening，而不是立即扩大为 formal。
