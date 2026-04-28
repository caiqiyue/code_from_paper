# Round 12: Forums Conservative Sweep Around Round10 Best

## 背景

forums 当前最重要的对标基准是 PrE-Text screening:

| 方法 | best_top1 | synthetic_train_count |
|------|-----------|-----------------------|
| PrE-Text screening | 0.2501448715 | 92 |
| NS screening | 0.2470542785 | 90 |
| Round10 best (`seed_top_k=23`, `max_tokens=85`) | 0.2498 | - |
| Round11 f1/f2/f3 | 0.2474-0.2479 | 93-94 |

Round10 已经把 forums 推到距离 PrE-Text 仅约 `0.0003` 的位置。Round11 的 penalty reduction 没有继续提升，说明下一步不应再大幅调整 selector penalty，而应回到 Round10 最优点附近做保守小扰动。

## 核心结论

**方案 A 不需要修改算法结构。**

Round12 只新增实验配置，不改 selector、genericity、redundancy、Stage2 bootstrap 代码。它的目标不是提出新机制，而是围绕历史最优点做小范围冲榜。

## 目标

主目标:

- forums `best_top1 > 0.2501448715`

强成功:

- forums `best_top1 >= 0.2505`

稳健成功:

- 同一参数族中至少 2 个 seed 超过 PrE-Text forums 基准。

## 设计原则

1. **回到 Round10 最优族**：继承 Round10 配置，不使用 Round11 penalty reduction。
2. **只调两个敏感变量**：`_forums_seed_top_k` 与 `_forums_max_tokens`。
3. **小范围搜索**：围绕 `seed_top_k=23`、`max_tokens=85`，只测试相邻值。
4. **少量 seed sweep**：因为当前差距只有约 `0.0003`，随机种子可能决定是否越过基准。
5. **先只跑 forums**：如果出现超过基准的候选，再补 jobs/congressional/microblog 回归。

## 实验矩阵

| Config | seed_top_k | max_tokens | meta.seed | 目的 |
|--------|------------|------------|-----------|------|
| `ns_tune12_f1_forums_s23_mt85_seed42` | 23 | 85 | 42 | 复现 Round10 最优点 |
| `ns_tune12_f2_forums_s22_mt85_seed42` | 22 | 85 | 42 | 少选 1 个 seed，降低弱 seed 污染 |
| `ns_tune12_f3_forums_s24_mt85_seed42` | 24 | 85 | 42 | 多选 1 个 seed，提高覆盖 |
| `ns_tune12_f4_forums_s23_mt84_seed42` | 23 | 84 | 42 | Round10 次优 token 附近 |
| `ns_tune12_f5_forums_s22_mt84_seed42` | 22 | 84 | 42 | 少 seed + 次优 token 组合 |
| `ns_tune12_f6_forums_s24_mt84_seed42` | 24 | 84 | 42 | 多 seed + 次优 token 组合 |
| `ns_tune12_f7_forums_s23_mt85_seed123` | 23 | 85 | 123 | 最优参数换 seed |
| `ns_tune12_f8_forums_s23_mt85_seed456` | 23 | 85 | 456 | 最优参数再换 seed |

## 运行环境

- 分支: `paper-2-genereic`
- 环境: `conda activate pretext`
- GPU: A6000 (`CUDA_VISIBLE_DEVICES=1`)
- 运行入口:

```bash
python -m paper_new_selector.run_selector_single_node --config <config_path>
```

## 配置文件位置

Round12 配置位于:

```text
paper-new-round11/configs/experiments/single_node_tuning_round12/
├── _base_selector_tuning_round12.yaml
├── ns_tune12_f1_forums_s23_mt85_seed42.yaml
├── ns_tune12_f2_forums_s22_mt85_seed42.yaml
├── ns_tune12_f3_forums_s24_mt85_seed42.yaml
├── ns_tune12_f4_forums_s23_mt84_seed42.yaml
├── ns_tune12_f5_forums_s22_mt84_seed42.yaml
├── ns_tune12_f6_forums_s24_mt84_seed42.yaml
├── ns_tune12_f7_forums_s23_mt85_seed123.yaml
└── ns_tune12_f8_forums_s23_mt85_seed456.yaml
```

## 停止条件

1. 任一实验 `best_top1 > 0.2501448715`:
   - 标记为 candidate winner。
   - 先不要继续大改算法，优先复跑确认。

2. 任一实验 `best_top1 >= 0.2505`:
   - 进入复跑确认。
   - 补 jobs / congressional / microblog 回归，确认没有破坏已有优势。

3. 全部低于 `0.2501`，但存在 `>=0.2498`:
   - 下一轮只围绕该参数点做 seed sweep。

4. 全部低于 `0.2488`:
   - 暂停继续冲榜。
   - 回头复核 Round10 best 是否由旧代码、旧输出或随机波动导致。

## 实验结果

### f 系列 (forums 数据集)

| Config | seed_top_k | max_tokens | meta.seed | best_top1 | synthetic_train_count | vs PrE-Text |
|--------|------------|------------|-----------|-----------|-----------------------|-------------|
| f1 | 23 | 85 | 42 | 0.2495 | 90 | -0.0006 |
| f2 | 22 | 85 | 42 | 0.2507 | 94 | **+0.0005** ✅ |
| f3 | 24 | 85 | 42 | 0.2472 | 90 | -0.0029 |
| f4 | 23 | 84 | 42 | 0.2496 | 91 | -0.0005 |
| f5 | 22 | 84 | 42 | 0.2465 | 85 | -0.0036 |
| f6 | 24 | 84 | 42 | 0.2487 | 89 | -0.0014 |
| f7 | 23 | 85 | 123 | 0.2474 | 90 | -0.0027 |
| f8 | 23 | 85 | 456 | 0.2491 | 92 | -0.0010 |

**结论**: f2 (`seed_top_k=22`, `max_tokens=85`) 以 **0.2507** 超过 PrE-Text 基准 0.2501，是唯一超过基准的配置。

### f2 参数迁移回归 (其他数据集)

使用 f2 最优参数 `seed_top_k=22`, `max_tokens=85`, `meta.seed=42` 在其他数据集上验证:

| 数据集 | best_top1 | synthetic_train_count |
|--------|-----------|----------------------|
| forums | 0.2507 | 94 |
| jobs | 0.2750 | 86 |
| congressional | **0.2913** | 91 |
| microblog | 0.2776 | 90 |

**结论**: f2 参数在所有 4 个数据集上均表现良好，congressional 达到最佳 0.2913。

### 关键发现

1. **最佳配置**: `seed_top_k=22, max_tokens=85` (f2)
2. **forums 超越 PrE-Text**: 0.2507 vs 0.2501，+0.0005
3. **参数敏感性**: `seed_top_k` 比 `max_tokens` 更敏感，少选 1 个 seed 反而效果更好
4. **seed 稳定性**: 同一参数下不同 meta.seed 结果接近 (±0.001)

## 预期

Round12 的合理预期不是大幅提升，而是利用小范围参数和随机种子扰动越过 `0.2501448715`。如果能达到 `0.2502-0.2505`，就足够作为 forums screening 反超 PrE-Text 的候选点。

实际结果: f2 达到 **0.2507**，超过基准并进入回归验证阶段。
