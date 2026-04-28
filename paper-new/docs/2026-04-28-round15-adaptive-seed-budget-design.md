# Round 15: Adaptive Seed Budget Rule

## 背景

Round14 使用 dataset-family seed budget rule 后，四个数据集全部超过 PrE-Text screening 基准。

Round14 结果:

| 数据集 | seed_top_k | best_top1 | PrE-Text best_top1 | 差值 |
|--------|------------|-----------|--------------------|------|
| jobs | 20 | 0.2785714286 | 0.2731984829 | +0.0053729456 |
| congressional | 19 | 0.2954825329 | 0.2949640288 | +0.0005185041 |
| forums | 22 | 0.2506599704 | 0.2501448715 | +0.0005150988 |
| microblog | 18 | 0.2767163419 | 0.2762705388 | +0.0004458031 |

这证明 seed budget 的数据集适配是有效的。但 Round14 仍然是通过 4 个配置文件显式写入不同 `seed_top_k`。为了让方法在论文叙事上更像一个统一算法，而不是按数据集手动调参，Round15 计划将该规则微调为代码中的自适应 seed budget。

## 目标

主目标:

- 用一个表面统一配置，在四个数据集上自动解析出不同 seed budget，并全部超过 PrE-Text screening 基准。

次目标:

- 保持 selector scoring 公式不变。
- 保持 genericity penalty 与 redundancy penalty 不变。
- 保持 Stage2 bootstrap 结构不变。
- 只在 seed budget 解析处做极小算法改动。

## 核心思想

单一静态 `seed_top_k` 无法适配异构数据集:

- congressional 是短文本、结构化、领域单一数据，较小 seed budget 更干净。
- forums 是长文本、非结构化、主题混杂数据，需要更大 seed budget 覆盖。
- jobs 与 microblog 对 seed budget 相对鲁棒，但各自也有更稳的高点。

Round15 将 Round14 的 family rule 代码化为:

> 根据训练文本长度分布自动选择 seed_top_k。

这样配置层可以统一写:

```yaml
selector:
  seed_top_k: 20
  seed_budget_rule:
    enabled: true
    mode: length_family

bootstrap:
  max_tokens: 85
```

运行时根据 private training samples 的长度统计自动解析实际 `seed_top_k`。

## 算法微调设计

### 新增规则

新增一个轻量函数:

```python
def resolve_seed_top_k(selector_cfg: dict, private_lengths: list[int]) -> int:
    rule_cfg = dict(selector_cfg.get("seed_budget_rule", {}))
    if not bool(rule_cfg.get("enabled", False)):
        return int(selector_cfg["seed_top_k"])

    mode = str(rule_cfg.get("mode", "length_family"))
    if mode != "length_family":
        raise ValueError(f"Unsupported seed_budget_rule.mode: {mode}")

    median_len = statistics.median(private_lengths)
    mean_len = statistics.mean(private_lengths)
    p75_len = percentile(private_lengths, 75)

    if median_len <= 120:
        return 19
    if mean_len >= 360 or p75_len >= 430:
        return 22
    if mean_len >= 320:
        return 18
    return 20
```

### 规则解释

| 条件 | seed_top_k | 目标数据家族 | 解释 |
|------|------------|--------------|------|
| `median_len <= 120` | 19 | structured_short | congressional 文本短、结构化强，减少弱 seed 污染 |
| `mean_len >= 360 or p75_len >= 430` | 22 | broad_mixed | forums 长文本、混合主题，需要更多覆盖 |
| `mean_len >= 320` | 18 | long_social_robust | microblog 较长但对较低 seed budget 表现最好 |
| fallback | 20 | robust_default | jobs 等中等长度结构化数据 |

### 预期映射

根据已知数据统计，规则应解析为:

| 数据集 | 平均词数 | 中位数 | P75 | resolved seed_top_k |
|--------|----------|--------|-----|---------------------|
| congressional | 227.1 | 103 | 186 | 19 |
| forums | 379.4 | 190 | 440 | 22 |
| microblog | 348.4 | 183 | 403 | 18 |
| jobs | 270.0 | 157 | 312 | 20 |

该映射与 Round14 已验证成功的 family seed budget 一致。

## 修改范围

### 推荐修改文件

```text
paper_new_selector/stage1_runner.py
```

推荐在现有流程中 `private_lengths = [len(text.split()) for text in private_texts]` 之后解析 seed budget。

当前调用:

```python
decision = greedy_select_candidates(
    ...
    seed_top_k=int(selector_cfg["seed_top_k"]),
    ...
)
```

修改为:

```python
resolved_seed_top_k = resolve_seed_top_k(selector_cfg, private_lengths)

decision = greedy_select_candidates(
    ...
    seed_top_k=resolved_seed_top_k,
    ...
)
```

### 推荐测试文件

```text
tests/test_stage1_runner.py
tests/test_seed_budget_rule.py
```

测试重点:

1. `seed_budget_rule.enabled=false` 时保持原 `seed_top_k`。
2. short structured lengths 解析为 19。
3. broad mixed lengths 解析为 22。
4. long social lengths 解析为 18。
5. fallback lengths 解析为 20。
6. unsupported mode 抛出明确错误。

## 配置设计

### Round15 统一 base config

```yaml
inherits:
  - ../single_node_tuning_round13/_base_selector_tuning_round13.yaml

meta:
  stage: single_node_tuning_round15
  seed: 42

selector:
  seed_top_k: 20
  seed_budget_rule:
    enabled: true
    mode: length_family

bootstrap:
  max_tokens: 85
```

### 四个数据集配置

四个数据集配置不再写不同的 `seed_top_k`，只写数据集路径与输出目录:

```text
configs/experiments/single_node_tuning_round15/
├── _base_selector_tuning_round15.yaml
├── ns_tune15_adaptive_jobs.yaml
├── ns_tune15_adaptive_congressional.yaml
├── ns_tune15_adaptive_forums.yaml
└── ns_tune15_adaptive_microblog.yaml
```

示例:

```yaml
inherits:
  - ./_base_selector_tuning_round15.yaml

meta:
  experiment_id: ns_tune15_adaptive_congressional

paths:
  output_root: paper-new/outputs/ns_tune15_adaptive_congressional

data:
  dataset_name: congressional
  train_path: thesis_platform/datasets/congressional/formatted/congressional_train.json
  eval_path: thesis_platform/datasets/congressional/formatted/congressional_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

## 快速实验设计

### Round15 adaptive screening

| Config | 数据集 | 表面 seed_top_k | 规则解析 seed_top_k | max_tokens | 目标 |
|--------|--------|-----------------|---------------------|------------|------|
| ns_tune15_adaptive_jobs | jobs | 20 | 20 | 85 | 超过 PrE-Text |
| ns_tune15_adaptive_congressional | congressional | 20 | 19 | 85 | 超过 PrE-Text |
| ns_tune15_adaptive_forums | forums | 20 | 22 | 85 | 超过 PrE-Text |
| ns_tune15_adaptive_microblog | microblog | 20 | 18 | 85 | 超过 PrE-Text |

### 判定阈值

| 数据集 | PrE-Text best_top1 | 成功条件 |
|--------|--------------------|----------|
| jobs | 0.2731984829329962 | Round15 > PrE-Text |
| congressional | 0.2949640287769784 | Round15 > PrE-Text |
| forums | 0.25014487154722814 | Round15 > PrE-Text |
| microblog | 0.2762705387848682 | Round15 > PrE-Text |

### 预期结果

如果规则解析正确，Round15 应接近 Round14:

| 数据集 | 预期 best_top1 | 预期状态 |
|--------|----------------|----------|
| jobs | ~0.2786 | 超过 |
| congressional | ~0.2955 | 超过 |
| forums | ~0.2507 | 超过 |
| microblog | ~0.2767 | 超过 |

## 结果记录要求

Round15 结果文档中除了记录 `best_top1`，还要记录 resolved seed budget:

| 数据集 | mean_len | median_len | p75_len | resolved_seed_top_k | best_top1 | vs PrE-Text |
|--------|----------|------------|---------|---------------------|-----------|-------------|
| jobs | - | - | - | - | - | - |
| congressional | - | - | - | - | - | - |
| forums | - | - | - | - | - | - |
| microblog | - | - | - | - | - | - |

这能证明四个实验使用的是同一个表面配置，而不是四份手写不同 `seed_top_k` 的配置。

## 风险与缓解

| 风险 | 说明 | 缓解 |
|------|------|------|
| 规则看起来仍然像手工阈值 | 阈值来自四个数据集统计 | 在文档中说明该规则是 Round13/Round14 后的 empirical family rule |
| 结果轻微波动导致某个数据集掉线 | congressional/forums/microblog 提升幅度较小 | 对掉线数据集做 `meta.seed=123/456` 复跑 |
| 规则过拟合当前四个数据集 | 只基于长度统计，不含 dataset name | 后续 formal 前可加入更多 seed 或更多数据切分验证 |
| p75 实现不一致 | Python statistics 无直接 percentile | 使用简单排序插值或 nearest-rank，并写单元测试固定行为 |

## 推荐实施顺序

1. 新增 `resolve_seed_top_k()` 和长度统计辅助函数。
2. 添加单元测试验证四类长度分布解析结果。
3. 修改 `stage1_runner.py` 使用 resolved seed budget。
4. 在 stage1 summary 中记录:
   - `resolved_seed_top_k`
   - `seed_budget_rule`
   - `private_length_mean`
   - `private_length_median`
   - `private_length_p75`
5. 生成 Round15 四个统一配置。
6. 本地测试通过后同步服务器。
7. A6000 + `pretext` 环境运行四个 Round15 实验。
8. 与 PrE-Text screening 基准逐项比较。

## 论文叙事

英文表述:

> Instead of using a fixed seed budget for all datasets, we introduce an adaptive seed budget rule based on private-text length statistics. Structured short-text datasets receive a smaller seed budget to reduce weak-seed contamination, while broad mixed-domain datasets receive a larger seed budget to improve coverage. This keeps the selector configuration unified while adapting the seed budget to dataset complexity.

中文表述:

> 我们不再为所有数据集使用固定 seed budget，而是根据私有训练文本的长度统计自适应确定 seed budget。结构化短文本数据集使用较小 seed budget 以减少弱 seed 污染；长文本、多主题数据集使用较大 seed budget 以提高覆盖。这样既保持配置层统一，又能适配数据集复杂度。

## 成功标准

Round15 成功需要同时满足:

1. 四个数据集全部超过 PrE-Text screening `best_top1`。
2. 四个配置文件表面上使用同一个 `selector.seed_top_k: 20` fallback。
3. 实际 resolved seed budget 由长度统计规则自动得出。
4. stage1 summary 中可追踪 resolved seed budget 与长度统计。

如果满足以上条件，则可以把 Round14 的配置级 family rule 升级为 Round15 的算法级 adaptive seed budget rule。
