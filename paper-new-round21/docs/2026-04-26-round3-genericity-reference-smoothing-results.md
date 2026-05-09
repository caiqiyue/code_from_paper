# 2026-04-26 Round3 Genericity Reference Smoothing Results

## 1. Round3 结构改动背景

这轮实验对应 `paper-2-genereic` 分支上的新 `Stage 1 selector` 结构微调，核心目标是：

- 不再把 `genericity reference` 建模为“`top-k` reference 邻居的简单均值”
- 改成“`top-k` reference 邻居的排名衰减加权均值”
- 继续验证这类更宽、更平滑的 `genericity` 参考方式，是否能在 `forums` / `microblog` 上减少误罚，同时尽量不破坏 `jobs` / `congressional`

Round3 的共享 base 配置在 [single_node_tuning_round3/_base_selector_tuning_round3.yaml](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/configs/experiments/single_node_tuning_round3/_base_selector_tuning_round3.yaml)。

相对前一版静态参数调优实验，这轮 base 的关键变化是：

- `reference_top_k: 4 -> 6`
- 新增 `reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]`

也就是说，`f1` 组本身就已经不是“旧算法原样重跑”，而是：

- 把 `genericity` 的参考邻域从 `top-4` 扩到 `top-6`
- 并把 `top-6` 的 simple mean 改成 rank-weighted mean

## 2. 各实验组主要调整内容

### 2.1 F1: Weighted Reference K=6

配置文件：

- [single_node_tuning_round3/_f1_weighted_ref_k6.yaml](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/configs/experiments/single_node_tuning_round3/_f1_weighted_ref_k6.yaml)

主要改动：

- `reference_top_k: 4 -> 6`
- `genericity reference aggregation: simple mean -> weighted mean`
- `reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]`

含义：

- 在更宽的 reference 邻域里，前几个近邻仍然主导
- 后几个近邻只提供平滑作用，不再一刀切等权平均

### 2.2 F2: Weighted Reference K=6, Steeper Tail

配置文件：

- [single_node_tuning_round3/_f2_weighted_ref_k6_steeper.yaml](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/configs/experiments/single_node_tuning_round3/_f2_weighted_ref_k6_steeper.yaml)

主要改动：

- 保持 `reference_top_k = 6`
- 保持 weighted mean 结构
- `reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1] -> [1.0, 0.7, 0.45, 0.25, 0.12, 0.05]`

含义：

- 仍然允许 `top-6` 全部参与参考
- 但更强调前部近邻，进一步减弱尾部 reference 邻居的影响

### 2.3 F3: Weighted Reference K=8 with Tail

配置文件：

- [single_node_tuning_round3/_f3_weighted_ref_k8_tail.yaml](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/configs/experiments/single_node_tuning_round3/_f3_weighted_ref_k8_tail.yaml)

主要改动：

- `reference_top_k: 6 -> 8`
- `reference_rank_weights: [1.0, 0.8, 0.6, 0.4, 0.25, 0.1] -> [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.12, 0.05]`

含义：

- 把 `genericity` 的参考邻域进一步拉宽到 `top-8`
- 同时保留明显衰减的尾部，避免简单加宽后把局部判别力完全冲掉

### 2.4 F4: Weighted Reference K=6 + A2/E5 稳健组合

配置文件：

- [single_node_tuning_round3/_f4_weighted_ref_k6_plus_a2e5.yaml](D:/学习记录/导师项目/研究/caiqiyue_file/paper-new/configs/experiments/single_node_tuning_round3/_f4_weighted_ref_k6_plus_a2e5.yaml)

主要改动：

- 保持 `f1` 的 weighted `genericity` 结构
- 叠加前面较稳健的参数方向：
- `density_lambda: 0.50 -> 0.45`
- `novelty_lambda: 0.30 -> 0.35`
- `length_lambda: 0.20 -> 0.10`

含义：

- 一边保留 round3 的 `genericity reference smoothing`
- 一边叠加前两轮里相对更稳健的 `A2/E5` 参数方向

## 3. 实验结果

下面汇总的是服务器最终产出的 `downstream_eval_summary.json` 中的 `metrics` 字段。所有实验均已 `status = completed`。

### 3.1 F1 结果

| experiment | synthetic_train_count | eval_count | best_top1 | best_top3 | best_top5 | best_top10 |
|---|---:|---:|---:|---:|---:|---:|
| `ns_tune3_f1_jobs` | 93 | 256 | 0.27920353982300883 | 0.42996207332490516 | 0.4964601769911504 | 0.5754740834386852 |
| `ns_tune3_f1_congressional` | 87 | 256 | 0.292954825328926 | 0.46030202864735237 | 0.535290686369823 | 0.6251215244020999 |
| `ns_tune3_f1_forums` | 95 | 256 | 0.24724744060266562 | 0.384843216792222 | 0.45232116412336615 | 0.5393084798145644 |
| `ns_tune3_f1_microblog` | 92 | 256 | 0.27862692650617754 | 0.4207744236403006 | 0.48337791364157434 | 0.5664246592790727 |

### 3.2 F2 结果

| experiment | synthetic_train_count | eval_count | best_top1 | best_top3 | best_top5 | best_top10 |
|---|---:|---:|---:|---:|---:|---:|
| `ns_tune3_f2_jobs` | 88 | 256 | 0.2761061946902655 | 0.4275600505689001 | 0.4928571428571429 | 0.5747155499367889 |
| `ns_tune3_f2_congressional` | 95 | 256 | 0.2919826301121265 | 0.4566724998379675 | 0.5311426534448117 | 0.6201309222891956 |
| `ns_tune3_f2_forums` | 90 | 256 | 0.24834202562616703 | 0.38619535123301785 | 0.45109780439121755 | 0.5345438155946172 |
| `ns_tune3_f2_microblog` | 93 | 256 | 0.27900904343395744 | 0.4175901159088014 | 0.48337791364157434 | 0.5637498407846134 |

### 3.3 F3 结果

| experiment | synthetic_train_count | eval_count | best_top1 | best_top3 | best_top5 | best_top10 |
|---|---:|---:|---:|---:|---:|---:|
| `ns_tune3_f3_jobs` | 93 | 256 | 0.27869785082174464 | 0.42812895069532236 | 0.4952591656131479 | 0.5797092288242731 |
| `ns_tune3_f3_congressional` | 96 | 256 | 0.2969732322250308 | 0.4610149718063387 | 0.5337999870373971 | 0.6212975565493551 |
| `ns_tune3_f3_forums` | 94 | 256 | 0.24679672912240036 | 0.38683922477625393 | 0.4547034962333398 | 0.5390509303972699 |
| `ns_tune3_f3_microblog` | 94 | 256 | 0.27748057572283785 | 0.4179085466819513 | 0.4810215259202649 | 0.5633677238568335 |

### 3.4 F4 结果

| experiment | synthetic_train_count | eval_count | best_top1 | best_top3 | best_top5 | best_top10 |
|---|---:|---:|---:|---:|---:|---:|
| `ns_tune3_f4_jobs` | 92 | 256 | 0.27686472819216185 | 0.4251580278128951 | 0.4924778761061947 | 0.5743994943109987 |
| `ns_tune3_f4_congressional` | 89 | 256 | 0.292954825328926 | 0.45926502041609957 | 0.5327629788061443 | 0.6232419469829542 |
| `ns_tune3_f4_forums` | 90 | 256 | 0.2470542785396948 | 0.3820101732019831 | 0.44935934582448006 | 0.5317751593587019 |
| `ns_tune3_f4_microblog` | 94 | 256 | 0.2744873264552286 | 0.41943701439307096 | 0.48019360591007515 | 0.5627945484651636 |

## 4. 按数据集查看 Round3 最优组

这里只按 `best_top1` 做一层快速筛选。

### 4.1 Jobs

- `f1`: `0.27920353982300883`
- `f2`: `0.2761061946902655`
- `f3`: `0.27869785082174464`
- `f4`: `0.27686472819216185`

Round3 内部最优：

- `jobs -> f1`

### 4.2 Congressional

- `f1`: `0.292954825328926`
- `f2`: `0.2919826301121265`
- `f3`: `0.2969732322250308`
- `f4`: `0.292954825328926`

Round3 内部最优：

- `congressional -> f3`

### 4.3 Forums

- `f1`: `0.24724744060266562`
- `f2`: `0.24834202562616703`
- `f3`: `0.24679672912240036`
- `f4`: `0.2470542785396948`

Round3 内部最优：

- `forums -> f2`

### 4.4 Microblog

- `f1`: `0.27862692650617754`
- `f2`: `0.27900904343395744`
- `f3`: `0.27748057572283785`
- `f4`: `0.2744873264552286`

Round3 内部最优：

- `microblog -> f2`

## 5. 一句话记录

Round3 的结论可以先压成一句话：

- `f1` 是“weighted genericity smoothing` 的本体版本
- `f2` 是“更强调前部近邻、弱化尾部 reference 邻居”的版本
- `f3` 是“更宽 reference 邻域”的版本
- `f4` 是“weighted genericity smoothing + A2/E5 稳健参数组合”的版本

从 Round3 内部结果看：

- `jobs` 最好的是 `f1`
- `congressional` 最好的是 `f3`
- `forums` 和 `microblog` 最好的是 `f2`

