# `PrE-Text` 与 `paper-new-2` Stage2 Selector 快速对比实验说明

更新时间：2026-04-25

## 1. 文档用途

本文档用于固定 `paper-new-2` 这版创新算法的快速对比实验口径。

对比双方：

- 对照算法：`PrE-Text`
- 创新算法：`paper-new-2` 的 `Seed-Aware Stage2 Selector`

这轮实验的目标不是给出正式论文结论，而是回答一个更现实的问题：

> 只在 `Stage 2` bootstrap 产物筛选处做单点修改，是否已经出现值得继续投入 formal experiment 的正向趋势。

## 2. 可比性约束

除创新点本身外，下列维度必须保持和 `PrE-Text` / `paper-new` 快速对比实验一致：

- 相同数据集
- 相同 `train_limit = 256`
- 相同 `eval_limit = 256`
- 相同 `initialization_limit = 1024`
- 相同 `num_prompts = 100`
- 相同下游评估链：`gpt2 small eval`
- 相同主判定指标：`best_top1`
- 同时记录 `best_top3 / best_top5 / best_top10`
- Stage 2 生成继续使用本地 `llama_2_7b_hf + vllm`
- 不改 `PrE-Text` 的 Stage 1 DP seed 生成主干

这版创新算法唯一允许的算法差异是：

- `PrE-Text` 原始 bootstrap outputs 进入下游训练前，新增 `seed-aware synthetic corpus selector`

不允许发生的变化：

- 改 Stage 1 选择逻辑
- 改 bootstrap prompt 模板
- 改 bootstrap backend
- 改下游评估模型或指标
- 通过多生成/少生成改变比较口径

## 3. 本版创新算法的实验定义

### 3.1 研究对象

`paper-new-2` 的研究对象不是重新发明整套两阶段框架，而是在 `PrE-Text` 的 Stage 2 之后追加一个局部选择器：

1. `PrE-Text Stage 1` 正常生成 surviving seeds
2. `PrE-Text bootstrap` 正常用 `vllm + 本地 llama_2_7b_hf` 生成 synthetic texts
3. `paper-new-2` 只在 synthetic texts 进入 downstream eval 之前做筛选

### 3.2 创新点

对每个 bootstrap output，基于其对应 prompt 的 seed 元数据计算：

- `Consistency`
- `TemplatePenalty`
- `DuplicatePenalty`

然后做：

- 硬过滤：去掉 baseline 清洗为空、低一致性、明显模板化样本
- 排序选择：在与本次运行经过统一 eval 清洗后可训练语料规模同量级的前提下保留更高质量样本

### 3.3 目标

本轮 screening 只回答：

> 在不改变 `PrE-Text` Stage 1 与 Stage 2 生成主干的前提下，只通过 seed-aware 的后验语料筛选，能否优于原始 `PrE-Text`。

## 4. 配置文件

### 4.1 `PrE-Text` 对照组

- `PrE-Text/configs/experiments/single_node_screening/sp_s_jobs_screening.yaml`
- `PrE-Text/configs/experiments/single_node_screening/sp_s_congressional_screening.yaml`
- `PrE-Text/configs/experiments/single_node_screening/sp_s_forums_screening.yaml`
- `PrE-Text/configs/experiments/single_node_screening/sp_s_microblog_screening.yaml`

### 4.2 `paper-new-2` 创新组

- `paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml`
- `paper-new-2/configs/experiments/single_node_screening/sas_s_congressional_screening.yaml`
- `paper-new-2/configs/experiments/single_node_screening/sas_s_forums_screening.yaml`
- `paper-new-2/configs/experiments/single_node_screening/sas_s_microblog_screening.yaml`

### 4.3 Formal 模板

- `paper-new-2/configs/experiments/single_node_formal/sas_c1_jobs_base.yaml`

该 formal 模板当前只作为后续放大量级的入口，不参与这轮 screening 快速对比。

## 5. 服务器侧执行命令

以下命令默认在 Linux 服务器、`pretext` 环境、项目根目录 `caiqiyue_file` 下执行。

### 5.1 先跑 `PrE-Text` 基线

```bash
python -m pretext_platform.scripts.run_stage1 --config PrE-Text/configs/experiments/single_node_screening/sp_s_jobs_screening.yaml
python -m pretext_platform.scripts.run_bootstrap --config PrE-Text/configs/experiments/single_node_screening/sp_s_jobs_screening.yaml
python -m pretext_platform.scripts.run_eval_small --config PrE-Text/configs/experiments/single_node_screening/sp_s_jobs_screening.yaml
```

如果服务器已经有完整基线脚本，也可以直接沿用你之前跑 `sp_s_*` 的命令体系，不需要强行改流程。

### 5.2 再跑 `paper-new-2` 创新组

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export VLLM_HOST_IP=127.0.0.1
export HOST_IP=127.0.0.1
export PYTHONPATH=paper-new-2

python -m paper_new_stage2_selector.run_stage2_seed_aware_single_node \
  --config paper-new-2/configs/experiments/single_node_screening/sas_s_jobs_screening.yaml
```

其余三个数据集把配置替换成：

- `sas_s_congressional_screening.yaml`
- `sas_s_forums_screening.yaml`
- `sas_s_microblog_screening.yaml`

## 6. 结果检查位置

### 6.1 `PrE-Text`

检查各自输出目录中的：

- `stage2/llama7b_text_syn.json`
- `eval_small_summary.json` 或已有的小模型评估结果文件

### 6.2 `paper-new-2`

检查：

- `paper-new-2/outputs/<experiment_id>/stage2_selected/llama7b_text_syn.json`
- `paper-new-2/outputs/<experiment_id>/stage2_selected/selection_metadata.json`
- `paper-new-2/outputs/<experiment_id>/eval/downstream_eval_summary.json`

其中：

- `selection_metadata.json` 用于分析这版 selector 到底筛掉了哪些样本
- `downstream_eval_summary.json` 用于和 `PrE-Text` 对比主指标

## 7. 结果记录模板

| 数据集 | 算法 | synthetic_train_count | eval_count | best_top1 | best_top3 | best_top5 | best_top10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `jobs` | `PrE-Text` |  |  |  |  |  |  |
| `jobs` | `paper-new-2` Stage2 Selector |  |  |  |  |  |  |
| `congressional` | `PrE-Text` |  |  |  |  |  |  |
| `congressional` | `paper-new-2` Stage2 Selector |  |  |  |  |  |  |
| `forums` | `PrE-Text` |  |  |  |  |  |  |
| `forums` | `paper-new-2` Stage2 Selector |  |  |  |  |  |  |
| `microblog` | `PrE-Text` |  |  |  |  |  |  |
| `microblog` | `paper-new-2` Stage2 Selector |  |  |  |  |  |  |

## 8. 当前建议

这轮 screening 先不要叠加第二个创新点。

也就是说：

- 先只验证 `Stage 2` 后验筛选是否有效
- 暂时不要把 `generation budget control` 一起加进来

只有当这轮结果已经出现明确正向趋势后，才值得进入下一轮增强版 screening。
