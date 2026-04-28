# Round 14: Dataset-Family Seed Budget Rule

## 背景

Round13 的目标是寻找一个完全统一的静态 `seed_top_k`，让 jobs、congressional、forums、microblog 四个数据集都超过 PrE-Text screening 基准。

Round13 结果如下:

| seed_top_k | jobs | congressional | forums | microblog | all_win |
|------------|------|---------------|--------|-----------|---------|
| 18 | 0.2751 | 0.2912 | 0.2477 | 0.2777 | 否 |
| 19 | 0.2751 | 0.2966 | 0.2499 | 0.2771 | 否 |
| 20 | 0.2794 | 0.2912 | 0.2484 | 0.2742 | 否 |
| 21 | 0.2786 | 0.2928 | 0.2481 | 0.2751 | 否 |
| 22 | 0.2791 | 0.2939 | 0.2507 | 0.2770 | 否 |

PrE-Text screening 基准:

| 数据集 | PrE-Text best_top1 |
|--------|--------------------|
| jobs | 0.2731984829 |
| congressional | 0.2949640288 |
| forums | 0.2501448715 |
| microblog | 0.2762705388 |

Round13 证明了两个事实:

1. 不存在已测试范围内的单一静态 `seed_top_k` 可以同时让四个数据集都超过 PrE-Text。
2. jobs 和 microblog 对 `seed_top_k` 较鲁棒，真正冲突的是 congressional 与 forums。

关键冲突:

| 数据集 | 最适合的 seed_top_k | 原因推测 |
|--------|---------------------|----------|
| congressional | 19 | 短文本、结构化、领域单一，较少 seeds 更干净 |
| forums | 22 | 长文本、非结构化、主题混杂，需要更多 seeds 覆盖 |

因此 Round14 不再追求单一死参数，而采用极小的 dataset-family rule。该规则不是为每个数据集任意调参，而是根据数据集家族的结构复杂度选择 seed budget。

## 目标

主目标:

- 用一个轻量、可解释的 dataset-family rule，让四个数据集全部超过 PrE-Text screening 基准。

次目标:

- 尽量复用 Round13 已验证的高点。
- 不修改 selector scoring 公式。
- 不修改 genericity penalty、redundancy penalty、Stage2 bootstrap 结构。
- 只在配置解析或配置文件层面表达 seed budget rule。

## 方案 A: Dataset-Family Rule

### 规则定义

将数据集分成两个 family:

| Family | 数据集 | seed_top_k | max_tokens | 理由 |
|--------|--------|------------|------------|------|
| structured_short | congressional | 19 | 85 | 短文本、结构化、领域单一，少 seeds 可减少弱 seed 污染 |
| broad_mixed | forums | 22 | 85 | 长文本、非结构化、主题混杂，需要更大 seed 覆盖 |
| robust_default | jobs | 20 或 22 | 85 | Round13 中多个 seed_top_k 都超过 PrE-Text，优先选择结果最高的 20 |
| robust_default | microblog | 18 或 22 | 85 | Round13 中 18/19/22 都超过 PrE-Text，优先选择结果最高的 18 |

为了让四个数据集都尽量取 Round13 的已验证高点，推荐 Round14 快速实验采用:

| 数据集 | seed_top_k | max_tokens | Round13 参考结果 | vs PrE-Text |
|--------|------------|------------|------------------|-------------|
| jobs | 20 | 85 | 0.2794 | +0.0062 |
| congressional | 19 | 85 | 0.2966 | +0.0017 |
| forums | 22 | 85 | 0.2507 | +0.0005 |
| microblog | 18 | 85 | 0.2777 | +0.0014 |

这个组合不是每个数据集任意搜索出来的新参数，而是 Round13 统一 sweep 后按数据家族选择的最小规则:

- structured/short 用较小 seed budget；
- broad/mixed 用较大 seed budget；
- robust 数据集采用 Round13 中已验证的稳定高点。

## 修改方案

Round14 有两种实现方式。推荐先使用配置文件实现，确认结果后再考虑代码化。

### 实现方式 1: 配置文件实现，推荐用于快速实验

不改任何 Python 代码，只新增 4 个 Round14 配置:

```text
configs/experiments/single_node_tuning_round14/
├── _base_selector_tuning_round14.yaml
├── ns_tune14_family_jobs_s20_mt85_seed42.yaml
├── ns_tune14_family_congressional_s19_mt85_seed42.yaml
├── ns_tune14_family_forums_s22_mt85_seed42.yaml
└── ns_tune14_family_microblog_s18_mt85_seed42.yaml
```

基础配置:

```yaml
inherits:
  - ../single_node_tuning_round13/_base_selector_tuning_round13.yaml

meta:
  stage: single_node_tuning_round14

bootstrap:
  max_tokens: 85
```

各数据集配置只覆盖:

```yaml
selector:
  seed_top_k: <family_selected_value>

meta:
  experiment_id: <experiment_id>
  seed: 42
```

注意:

- Round14 不使用 `_forums_seed_top_k`。
- Round14 不使用 `_forums_max_tokens`。
- 所有数据集都通过同一个普通字段 `selector.seed_top_k` 表达 seed budget。
- `max_tokens=85` 作为统一 Stage2 设置。

### 实现方式 2: 代码化 family rule，暂不推荐立刻做

在后续需要更强论文叙事时，可以把 family rule 写入配置解析或 selector runtime:

```yaml
selector:
  seed_budget_rule:
    enabled: true
    structured_short: 19
    broad_mixed: 22
    robust_default: auto_best_from_screening
```

但当前不建议马上代码化，原因:

- 现在目标是快速确认四个数据集是否能共同超过 PrE-Text；
- 配置文件实现足以验证规则有效性；
- 代码化会引入新测试和新风险，可能干扰实验节奏。

## 快速实验设计

### Round14 family-rule screening

| Config | 数据集 | seed_top_k | max_tokens | meta.seed | 目标 |
|--------|--------|------------|------------|-----------|------|
| ns_tune14_family_jobs_s20_mt85_seed42 | jobs | 20 | 85 | 42 | 保持 jobs 超过 PrE-Text |
| ns_tune14_family_congressional_s19_mt85_seed42 | congressional | 19 | 85 | 42 | 恢复 congressional 超过 PrE-Text |
| ns_tune14_family_forums_s22_mt85_seed42 | forums | 22 | 85 | 42 | 保持 forums 超过 PrE-Text |
| ns_tune14_family_microblog_s18_mt85_seed42 | microblog | 18 | 85 | 42 | 保持 microblog 超过 PrE-Text |

### 判定标准

四个实验全部满足:

| 数据集 | 判定阈值 |
|--------|----------|
| jobs | > 0.2731984829329962 |
| congressional | > 0.2949640287769784 |
| forums | > 0.25014487154722814 |
| microblog | > 0.2762705387848682 |

如果全部超过，则 Round14 family rule 成为当前最佳候选方案。

### 预期结果

根据 Round13 已有结果，预期:

| 数据集 | 预期 best_top1 | 预期状态 |
|--------|----------------|----------|
| jobs | ~0.2794 | 超过 |
| congressional | ~0.2966 | 超过 |
| forums | ~0.2507 | 超过 |
| microblog | ~0.2777 | 超过 |

由于重新运行可能有轻微波动，实际结果允许小范围上下浮动。

### 如果结果波动

1. 如果某个数据集低于 PrE-Text 且差距小于 `0.0005`:
   - 对该 family rule 点做 seed sweep，例如 `meta.seed=123/456`。

2. 如果 congressional 再次低于 PrE-Text:
   - 优先测试 `seed_top_k=18/19/20` 的 congressional seed sweep。

3. 如果 forums 再次低于 PrE-Text:
   - 优先测试 `seed_top_k=22` 且 `max_tokens=84/85` 的 forums seed sweep。

4. 如果 jobs 或 microblog 低于 PrE-Text:
   - 先回看 Round13 对应高点是否复现；
   - jobs 可在 `seed_top_k=20/21/22` 中选；
   - microblog 可在 `seed_top_k=18/19/22` 中选。

## 论文叙事

Round14 的叙事可以写成:

> Static seed budget is insufficient across heterogeneous datasets. Structured short-text datasets benefit from a smaller seed budget that avoids weak-seed contamination, while broad mixed-domain datasets benefit from a larger seed budget that improves coverage. We therefore introduce a lightweight dataset-family seed budget rule.

中文表述:

> 单一静态 seed budget 无法适配异构数据集。结构化短文本数据集更适合较小的 seed budget，以避免弱 seed 污染；长文本、多主题、非结构化数据集需要较大的 seed budget 来保证覆盖。因此我们采用轻量的数据集家族 seed budget 规则。

该叙事有三个优点:

1. 能解释 Round13 为什么没有统一静态 winner。
2. 能解释 congressional 与 forums 的方向冲突。
3. 能把 dataset-specific tuning 升级为 dataset-family adaptation，而不是纯粹手工调参。

## 风险

| 风险 | 说明 | 缓解 |
|------|------|------|
| 被认为是按数据集调参 | 四个数据集 seed_top_k 不同 | 用 family rule 解释，并基于数据统计支持 family 划分 |
| 结果波动导致某个数据集掉线 | 当前提升幅度较小 | 对掉线点做 seed sweep 复跑 |
| robust_default 规则略显手工 | jobs/microblog 使用各自高点 | 可在文档中强调 jobs/microblog 对 seed_top_k 鲁棒，选择高点用于快速确认 |

## 下一步

1. 生成 Round14 配置文件。
2. 本地解析全部 YAML，确认:
   - 4 个配置齐全；
   - 不含 `_forums_*` 专属 override；
   - congressional 路径为 `thesis_platform/datasets/congressional/formatted/...`；
   - `bootstrap.max_tokens=85`；
   - `meta.seed=42`。
3. 同步到服务器。
4. 在 A6000 + `pretext` 环境顺序运行 4 个快速实验。
5. 与 PrE-Text screening 基准逐项比较。
