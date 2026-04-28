# Round 13: Unified Seed Top-K Sweep

## 背景

当前 screening 结果已经证明新版算法可以通过不同配置分别超过 PrE-Text:

| 数据集 | PrE-Text best_top1 | 当前可用最好 NS | 状态 |
|--------|--------------------|-----------------|------|
| jobs | 0.2731984829 | 0.2761061947 | 超过 |
| congressional | 0.2949640288 | 0.2969732322 | 超过 |
| forums | 0.2501448715 | 0.2506599704 | 超过 |
| microblog | 0.2762705388 | 0.2776079480 | 超过 |

但这些结果不是来自同一个统一配置。Round12 f2 参数 (`seed_top_k=22`, `max_tokens=85`) 让 forums 和 microblog 超过 PrE-Text，也让 jobs 保持超过，但 congressional 明显下降到 `0.2913344999`，低于 PrE-Text。

因此 Round13 的目标是寻找一个统一配置，让四个数据集都超过 PrE-Text，即使每个数据集的提升幅度不大。

## 核心假设

`max_tokens=85` 已经在 forums 上验证为关键高点，并且在 microblog 上也表现良好。当前最主要的冲突变量是 `seed_top_k`:

- `seed_top_k=22` 对 forums/microblog 有利；
- 但 congressional 在 `seed_top_k=22` 下明显退化；
- congressional 是短文本、结构化、领域单一数据集，可能需要更少、更精的 seeds。

Round13 因此固定 `max_tokens=85`，扫描较低的统一 `seed_top_k` 区间，寻找兼顾四个数据集的临界点。

## 目标

主目标:

- 找到至少一个统一配置，使四个数据集的 `best_top1` 全部超过对应 PrE-Text screening 基准。

PrE-Text 基准:

| 数据集 | PrE-Text best_top1 |
|--------|--------------------|
| jobs | 0.2731984829329962 |
| congressional | 0.2949640287769784 |
| forums | 0.25014487154722814 |
| microblog | 0.2762705387848682 |

## 设计原则

1. **统一配置**: 同一个 `seed_top_k` 和 `max_tokens` 应用于 jobs、congressional、forums、microblog。
2. **不做 dataset-aware override**: Round13 不使用 `_forums_seed_top_k` 或 `_forums_max_tokens`。
3. **不改算法结构**: 只新增实验配置，不修改 selector、genericity、redundancy 或 Stage2 代码。
4. **优先保护 congressional**: 因为当前统一 f2 参数唯一失败点是 congressional。
5. **保留 forums/microblog 的成功条件**: 固定 `max_tokens=85`，避免偏离 Round12 f2 的关键设置。

## 实验矩阵

统一参数:

- `bootstrap.max_tokens: 85`
- `meta.seed: 42`

扫描参数:

| 组别 | seed_top_k | max_tokens |
|------|------------|------------|
| u18 | 18 | 85 |
| u19 | 19 | 85 |
| u20 | 20 | 85 |
| u21 | 21 | 85 |
| u22 | 22 | 85 |

每个组别跑四个数据集，因此总共 20 个实验。

## 配置文件位置

配置位于:

```text
paper-new-round11/configs/experiments/single_node_tuning_round13/
```

命名规则:

```text
ns_tune13_u<seed_top_k>_<dataset>_s<seed_top_k>_mt85_seed42.yaml
```

示例:

```text
ns_tune13_u20_congressional_s20_mt85_seed42.yaml
```

## 实验结果

### 完整结果矩阵

| seed_top_k | jobs | congressional | forums | microblog | all_win |
|------------|------|---------------|--------|-----------|---------|
| 18 | 0.2751 | 0.2912 | 0.2477 | 0.2777 | ❌ |
| 19 | 0.2751 | 0.2966 | 0.2499 | 0.2771 | ❌ |
| 20 | 0.2794 | 0.2912 | 0.2484 | 0.2742 | ❌ |
| 21 | 0.2786 | 0.2928 | 0.2481 | 0.2751 | ❌ |
| 22 | 0.2791 | 0.2939 | **0.2507** | 0.2770 | ❌ |

### vs PrE-Text 对比

| seed_top_k | jobs vs 0.2732 | congressional vs 0.2950 | forums vs 0.2501 | microblog vs 0.2763 |
|------------|----------------|-------------------------|------------------|-------------------|
| 18 | ✅ +0.0019 | ❌ -0.0038 | ❌ -0.0024 | ✅ +0.0014 |
| 19 | ✅ +0.0019 | ✅ +0.0017 | ❌ -0.0002 | ✅ +0.0008 |
| 20 | ✅ +0.0062 | ❌ -0.0038 | ❌ -0.0017 | ❌ -0.0020 |
| 21 | ✅ +0.0054 | ❌ -0.0021 | ❌ -0.0020 | ❌ -0.0012 |
| 22 | ✅ +0.0059 | ❌ -0.0011 | ✅ +0.0005 | ✅ +0.0007 |

### 关键发现

1. **没有找到统一 winner**: 所有 seed_top_k 配置都无法让四个数据集同时超过 PrE-Text
2. **forums 最佳**: seed_top_k=22 时 forums 达到 0.2507，超过 PrE-Text
3. **congressional 最佳**: seed_top_k=19 时 congressional 达到 0.2966，但 forums 只有 0.2499
4. **jobs/microblog 较鲁棒**: 多个 seed_top_k 值都能超过 PrE-Text
5. **冲突本质**: forums 需要较大的 seed_top_k (22)，而 congressional 需要较小的 (19)

### 各数据集最佳配置

| 数据集 | 最佳 seed_top_k | 最佳结果 | vs PrE-Text |
|--------|-----------------|----------|-------------|
| jobs | 20 | 0.2794 | +0.0062 |
| congressional | 19 | 0.2966 | +0.0017 |
| forums | 22 | 0.2507 | +0.0005 |
| microblog | 18 | 0.2777 | +0.0014 |

### 结论

统一静态 `seed_top_k` 无法同时让四个数据集超过 PrE-Text。forums 和 congressional 对 seed_top_k 的敏感方向相反（前者需要大值，后者需要小值）。

下一步可能方向:
1. 接受极小的 dataset-family rule，针对 congressional 使用不同的 seed_top_k
2. 或重新审视 selector penalty/redundancy 参数，可能能解耦这个冲突
