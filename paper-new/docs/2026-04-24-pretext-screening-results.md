# `PrE-Text` 与新版创新算法快速对比实验结果
更新时间：2026-04-25

## 1. 文档用途
本文档统一记录当前 `screening-balanced` 快速对比实验中，两条算法线的结果：

- 对照算法：`PrE-Text`
- 创新算法：新版 selector 创新算法

这份文档的用途不是替代 formal 正式实验，而是作为“创新点是否值得继续投入”的 screening 判断依据。

## 2. 统一实验口径

- 实验类型：`single_node_screening`
- 运行分支：`paper-2`
- 运行环境：旧服务器 `pretext`
- 运行 GPU：`A6000`
- Stage 2：统一使用本地 `llama_2_7b_hf + vllm`
- 下游评估：统一使用 `gpt2 small eval`
- screening 统一参数：
  - `train_limit = 256`
  - `eval_limit = 256`
  - `initialization_limit = 1024`
  - `num_prompts = 100`
  - `epochs = 6`

需要说明：

- `eval_limit = 256` 是 screening 配置的统一缩参设计
- 实际 `eval_count` 由下游评估器真实读取的数据决定，因此不同数据集最终显示的 `eval_count` 可能不同

## 3. 当前版本新版创新算法具体流程

当前版本新版创新算法是在 `PrE-Text` 基础上做的局部修改，核心不在 Stage 2，而在 Stage 1 的 selector 设计与两阶段生成链的工程打通。

### 3.1 算法主流程

1. 使用固定初始提示词和公共初始化样本，通过本地 `llama_2_7b_hf + vllm` 生成 Stage 1 候选样本。
2. 对私有训练样本建立嵌入表示，计算每个私有样本的 `importance prior`。
3. 对候选样本计算 `Top-Q private support`。
4. 对候选样本计算 `genericity penalty`。
5. 对候选样本计算动态 `redundancy penalty`。
6. 通过贪心选择器输出：
   - `selected seeds`
   - `hard negatives`
   - `boundary_state`
7. Stage 1 结束后不释放已加载的 `vllm 7B engine`。
8. Stage 2 继续复用 `PrE-Text` 的 `build_bootstrap_prompts` 逻辑构造 bootstrap prompts。
9. Stage 2 不再重新初始化新的 `LLM(...)`，而是直接复用 Stage 1 已加载好的同一个 `vllm engine` 批量生成 synthetic corpus。
10. 生成结果进入统一的 `gpt2 small eval`，输出 `best_top1 / top3 / top5 / top10`。

### 3.2 本轮工程关键点

当前这版创新算法有一个重要工程变化：

- `Stage 1` 与 `Stage 2` 共享同一个 `llama_2_7b_hf + vllm` engine
- 避免了 `Stage 2` 再次初始化 7B 模型带来的显存峰值和 OOM

## 4. 四个数据集总对比表

| 数据集 | 算法 | synthetic_train_count | eval_count | best_top1 | best_top3 | best_top5 | best_top10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `jobs` | `PrE-Text` | 94 | 256 | 0.2731984829329962 | - | - | - |
| `jobs` | 新版创新算法 | 88 | 1000 | 0.2786768628845812 | 0.42998722778199927 | 0.4955458910642976 | 0.5781126218615104 |
| `congressional` | `PrE-Text` | 94 | 256 | 0.2949640287769784 | - | - | - |
| `congressional` | 新版创新算法 | 95 | 28632 | 0.29074027492465426 | 0.4589837290440988 | 0.5311561089450663 | 0.6199033897436614 |
| `forums` | `PrE-Text` | 92 | 256 | 0.25014487154722814 | - | - | - |
| `forums` | 新版创新算法 | 92 | 1000 | 0.24976645852795124 | 0.3872527328449448 | 0.4511529573725355 | 0.5367192749561598 |
| `microblog` | `PrE-Text` | 96 | 256 | 0.2762705387848682 | - | - | - |
| `microblog` | 新版创新算法 | 91 | 1000 | 0.2768144075953084 | 0.41841079264612324 | 0.48367889593970737 | 0.5665160437839514 |

说明：

- `PrE-Text` 这轮 screening 记录里目前已确认的主指标是 `best_top1`
- 新版创新算法的 screening 结果文件中保留了 `best_top1 / top3 / top5 / top10`
- 因此总表中，`PrE-Text` 暂只放入当前已整理出来的主指标

## 5. `PrE-Text` 四个 screening 实验结果

### 5.1 `SP-S-JOBS`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_jobs_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_jobs_screening`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 94`
  - `eval_count = 256`
  - `best_top1 = 0.2731984829329962`

### 5.2 `SP-S-CONG`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_congressional_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_congressional_screening`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 94`
  - `eval_count = 256`
  - `best_top1 = 0.2949640287769784`

### 5.3 `SP-S-FORUMS`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_forums_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_forums_screening`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 92`
  - `eval_count = 256`
  - `best_top1 = 0.25014487154722814`

### 5.4 `SP-S-MICRO`

- 配置文件：`PrE-Text/configs/experiments/single_node_screening/sp_s_microblog_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/PrE-Text/outputs/pretext_platform/sp_s_microblog_screening`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 96`
  - `eval_count = 256`
  - `best_top1 = 0.2762705387848682`

## 6. 新版创新算法四个 screening 实验结果

### 6.1 `NS-S-JOBS`

- 配置文件：`paper-new/configs/experiments/single_node_screening/ns_s_jobs_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_jobs_screening`
- Stage 2 语料文件：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_jobs_screening/eval/stage2/llama7b_text_syn.json`
- 下游评估摘要：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_jobs_screening/eval/downstream_eval_summary.json`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 88`
  - `eval_count = 1000`
  - `best_top1 = 0.2786768628845812`
  - `best_top3 = 0.42998722778199927`
  - `best_top5 = 0.4955458910642976`
  - `best_top10 = 0.5781126218615104`

### 6.2 `NS-S-CONG`

- 配置文件：`paper-new/configs/experiments/single_node_screening/ns_s_congressional_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_congressional_screening`
- Stage 2 语料文件：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_congressional_screening/eval/stage2/llama7b_text_syn.json`
- 下游评估摘要：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_congressional_screening/eval/downstream_eval_summary.json`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 95`
  - `eval_count = 28632`
  - `best_top1 = 0.29074027492465426`
  - `best_top3 = 0.4589837290440988`
  - `best_top5 = 0.5311561089450663`
  - `best_top10 = 0.6199033897436614`

### 6.3 `NS-S-FORUMS`

- 配置文件：`paper-new/configs/experiments/single_node_screening/ns_s_forums_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_forums_screening`
- Stage 2 语料文件：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_forums_screening/eval/stage2/llama7b_text_syn.json`
- 下游评估摘要：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_forums_screening/eval/downstream_eval_summary.json`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 92`
  - `eval_count = 1000`
  - `best_top1 = 0.24976645852795124`
  - `best_top3 = 0.3872527328449448`
  - `best_top5 = 0.4511529573725355`
  - `best_top10 = 0.5367192749561598`

### 6.4 `NS-S-MICRO`

- 配置文件：`paper-new/configs/experiments/single_node_screening/ns_s_microblog_screening.yaml`
- 结果目录：`/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_microblog_screening`
- Stage 2 语料文件：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_microblog_screening/eval/stage2/llama7b_text_syn.json`
- 下游评估摘要：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_s_microblog_screening/eval/downstream_eval_summary.json`
- 结果：
  - `epochs = 6`
  - `synthetic_train_count = 91`
  - `eval_count = 1000`
  - `best_top1 = 0.2768144075953084`
  - `best_top3 = 0.41841079264612324`
  - `best_top5 = 0.48367889593970737`
  - `best_top10 = 0.5665160437839514`

## 7. 当前快速对比结论

从当前这轮 screening 结果看：

1. `jobs`
   - 新版创新算法 `0.2786768628845812`
   - `PrE-Text` `0.2731984829329962`
   - 新版创新算法略高

2. `congressional`
   - 新版创新算法 `0.29074027492465426`
   - `PrE-Text` `0.2949640287769784`
   - `PrE-Text` 略高

3. `forums`
   - 新版创新算法 `0.24976645852795124`
   - `PrE-Text` `0.25014487154722814`
   - 两者接近，`PrE-Text` 略高

4. `microblog`
   - 新版创新算法 `0.2768144075953084`
   - `PrE-Text` `0.2762705387848682`
   - 新版创新算法极小幅领先

当前判断：

- 这版创新算法在 `jobs` 和 `microblog` 上表现为正向
- 在 `congressional` 和 `forums` 上没有超出 `PrE-Text`
- 因此它还不能直接判定为“screening 明确通过”
- 但这版实现至少证明了：
  - 创新算法并非全面失效
  - 共享 `vLLM` 改造后，4 个 screening 全部稳定跑通
  - 这条线仍值得继续做局部修改和下一轮 screening
