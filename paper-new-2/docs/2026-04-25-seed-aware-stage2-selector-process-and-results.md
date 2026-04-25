# `paper-new-2` Seed-Aware Stage2 Selector 流程与快速对比实验结果

更新时间：2026-04-25

## 1. 文档用途

本文档用于固定 `paper-new-2` 这一版创新算法的：

- 具体算法流程
- 与 `PrE-Text` 的差异边界
- 快速对比实验口径
- 四个 screening 数据集的最终结果

本文档对应的是：

- 对照算法：`PrE-Text`
- 创新算法：`paper-new-2` 的 `Seed-Aware Stage2 Selector`

---

## 2. 创新算法的定位

这版创新算法不是重写整套 `PrE-Text` 两阶段框架，而是在 **不改 `Stage 1` 和 `Stage 2` 生成主干** 的前提下，只在 `bootstrap outputs -> downstream eval` 之间插入一个局部选择器。

也就是说：

1. `PrE-Text Stage 1` 仍然负责生成 surviving seeds
2. `PrE-Text Stage 2 bootstrap` 仍然负责用 `vllm + 本地 llama_2_7b_hf` 生成 synthetic texts
3. `paper-new-2` 只负责决定：
   哪些 synthetic texts 更值得保留进入下游训练

---

## 3. 与 `PrE-Text` 的差异边界

除创新点本身外，本版算法尽量保持和 `PrE-Text` / `paper-new` 快速对比实验一致。

保持一致的部分：

- 相同数据集
- 相同 `train_limit = 256`
- 相同 `eval_limit = 256`
- 相同 `initialization_limit = 1024`
- 相同 `num_prompts = 100`
- 相同 `gpt2 small eval`
- 相同主判定指标 `best_top1`
- 相同 Stage 2 生成后端：`vllm + llama_2_7b_hf`
- 不改 `PrE-Text` 的 Stage 1 DP seed 生成逻辑

唯一允许的算法差异：

- 在 `PrE-Text` 原始 bootstrap outputs 进入 downstream eval 之前，增加一个 `seed-aware synthetic corpus selector`

不允许发生的变化：

- 改 Stage 1 selector
- 改 bootstrap prompt 模板
- 改 bootstrap backend
- 改下游评估模型
- 通过扩大或缩小生成预算改变比较口径

---

## 4. 具体算法流程

### 4.1 输入

输入对象有三类：

- `seed_texts`
  `PrE-Text Stage 1` 输出的 surviving seeds
- `prompt_records`
  由 `paper-new-2` 镜像 `PrE-Text` bootstrap prompt builder 生成，并显式保留：
  - `prompt_index`
  - `prompt_text`
  - `seed_texts`
- `raw_outputs`
  `PrE-Text` bootstrap 基于上述 prompts 生成的 synthetic texts

### 4.2 记录映射

`paper-new-2` 会显式保留这条映射链：

- `prompt_text -> seed_texts -> generated_text`

这一步是本版创新里非常关键的一点，因为如果没有这条映射，就无法证明选择器真的是 `seed-aware`，而不是普通的后处理清洗脚本。

### 4.3 样本清洗

每条生成文本会先经过与 `PrE-Text` 下游评测兼容的 baseline cleaning，得到：

- `raw_text`
- `baseline_text`

这样可以避免“选择器看的文本”和“eval 看的文本”不一致。

### 4.4 三个评分项

对每条 bootstrap output，选择器计算三类分数：

1. `Consistency`
   表示生成文本与其来源 seed 集之间的语义一致性。
   当前实现是把生成文本 embedding 与该 prompt 对应的 seed embeddings 做相似度比较，取最接近的匹配强度。

2. `TemplatePenalty`
   惩罚明显模板化、prompt echo、异常短文本、低词汇多样性文本。

3. `DuplicatePenalty`
   惩罚完全重复和近重复文本，避免最终语料被少数模板淹没。

### 4.5 选择规则

选择规则分两层：

1. 硬过滤
   去掉：
   - baseline cleaning 后为空的文本
   - 一致性低于阈值的文本
   - 明显模板化文本

2. 排序选择
   对剩余样本计算总分：

   `final_score = w_consistency * Consistency - w_template * TemplatePenalty - w_duplicate * DuplicatePenalty`

   按 `final_score` 从高到低排序，保留高分样本。

### 4.6 目标数量控制

本轮 screening 不是回答“生成多少更好”，而是回答“哪些样本更该保留”。

因此选择器采用 fixed-target 方式：

- 先统计本轮原始 bootstrap outputs 经过统一 cleaning 后的有效数量
- 再在这个口径下做筛选

这样可以避免把“预算变化”伪装成“质量提升”。

---

## 5. 代码流程落点

本版算法主流程入口：

- `paper-new-2/paper_new_stage2_selector/run_stage2_seed_aware_single_node.py`

核心流程文件：

- `paper-new-2/paper_new_stage2_selector/pipeline.py`

关键模块：

- `bootstrap_bridge.py`
  负责镜像 bootstrap prompt 构造，并保留 `prompt -> seed -> output` 映射
- `corpus_loader.py`
  负责 baseline cleaning
- `consistency.py`
  负责一致性打分
- `template_penalty.py`
  负责模板化惩罚
- `dedup.py`
  负责重复惩罚
- `selector.py`
  负责硬过滤、打分、排序、截断
- `eval_bridge.py`
  负责把选中语料写回 `PrE-Text` 兼容格式并调用 downstream eval

---

## 6. 快速对比实验口径

本轮 screening 使用四个数据集：

- `jobs`
- `congressional`
- `forums`
- `microblog`

对照组配置：

- `PrE-Text/configs/experiments/single_node_screening/sp_s_*.yaml`

创新组配置：

- `paper-new-2/configs/experiments/single_node_screening/sas_s_*.yaml`

判定指标：

- 主指标：`best_top1`
- 同时记录：`best_top3 / best_top5 / best_top10`

---

## 7. 快速对比实验结果

### 7.1 `PrE-Text` 基线 `best_top1`

| 数据集 | `PrE-Text best_top1` |
| --- | ---: |
| `jobs` | `0.2732` |
| `congressional` | `0.2950` |
| `forums` | `0.2501` |
| `microblog` | `0.2763` |

### 7.2 `paper-new-2` 创新组结果

| 数据集 | synthetic_train_count | eval_count | best_top1 | best_top3 | best_top5 | best_top10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `jobs` | `9` | `256` | `0.2729` | `0.4209` | `0.4881` | `0.5721` |
| `congressional` | `17` | `256` | `0.2952` | `0.4569` | `0.5339` | `0.6171` |
| `forums` | `7` | `256` | `0.2409` | `0.3850` | `0.4494` | `0.5316` |
| `microblog` | `11` | `256` | `0.2747` | `0.4157` | `0.4826` | `0.5608` |

### 7.3 `best_top1` 对比结论

| 数据集 | `PrE-Text` | `paper-new-2` | 差值（创新 - 基线） | 结论 |
| --- | ---: | ---: | ---: | --- |
| `jobs` | `0.2732` | `0.2729` | `-0.0003` | 略低 |
| `congressional` | `0.2950` | `0.2952` | `+0.0003` | 略高 |
| `forums` | `0.2501` | `0.2409` | `-0.0092` | 明显低 |
| `microblog` | `0.2763` | `0.2747` | `-0.0015` | 略低 |

---

## 8. 结果解读

这轮 screening 的总体结论是：

- 这版 `Seed-Aware Stage2 Selector` **没有跑出稳定的正结果**
- 4 个数据集里只有 `congressional` 微弱超过 `PrE-Text`
- 其余 `jobs / forums / microblog` 都没有超过基线

从结果形态看，当前版本存在一个比较明显的现象：

- 最终进入小模型训练的 synthetic corpus 很小
- 四个数据集分别只保留了 `9 / 17 / 7 / 11` 条文本

这说明当前 selector 的筛选强度偏大，已经不仅仅是在“去坏样本”，而是在很大程度上压缩了训练语料规模。  
在这种情况下，即使留下来的文本平均质量更高，也可能因为覆盖度不足，最终无法转化成稳定的下游提升。

---

## 9. 当前阶段结论

就 screening 目标而言，这版创新算法当前不能直接进入 formal experiment。

更准确地说：

- 它证明了 `Stage 2` 后验选择这条创新位置是可以实现、可以独立跑通、可以公平对比的
- 但当前这组 selector 规则和阈值，还没有证明自己优于 `PrE-Text`

因此，这轮实验的结论不是“Stage 2 创新位置错误”，而是：

- **当前这版 selector 设计还不够好**
- 尤其在 `selected_count` 过低这一点上，需要优先继续分析和修正

---

## 10. 产物位置

创新组结果文件位于：

- `paper-new-2/outputs/sas_s_jobs_screening/eval/downstream_eval_summary.json`
- `paper-new-2/outputs/sas_s_congressional_screening/eval/downstream_eval_summary.json`
- `paper-new-2/outputs/sas_s_forums_screening/eval/downstream_eval_summary.json`
- `paper-new-2/outputs/sas_s_microblog_screening/eval/downstream_eval_summary.json`

创新组中间选择产物位于：

- `paper-new-2/outputs/<experiment_id>/stage2_selected/llama7b_text_syn.json`
- `paper-new-2/outputs/<experiment_id>/stage2_selected/selection_metadata.json`

其中 `selection_metadata.json` 适合继续分析：

- 每轮到底保留了哪些文本
- 哪些文本被 `Consistency`、`TemplatePenalty`、`DuplicatePenalty` 筛掉
- 为什么 `selected_count` 会压得这么低

