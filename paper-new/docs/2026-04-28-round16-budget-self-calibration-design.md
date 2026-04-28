# Round 16: Budget Self-Calibration Design

## 背景

Round14 通过 dataset-family seed budget rule 实现了四个数据集全部超过 PrE-Text screening 基准：

| 数据集 | seed_top_k | best_top1 | PrE-Text | vs PrE-Text |
|--------|------------|-----------|----------|-------------|
| jobs | 20 | 0.2786 | 0.2732 | +0.0054 |
| congressional | 19 | 0.2955 | 0.2950 | +0.0005 |
| forums | 22 | 0.2507 | 0.2501 | +0.0005 |
| microblog | 18 | 0.2767 | 0.2763 | +0.0004 |

Round15 进一步把配置级 family rule 升级为算法级 adaptive seed budget rule。修复后，Round15 也实现了 4/4 超过 PrE-Text：

| 数据集 | resolved_seed_top_k | best_top1 | vs PrE-Text | 状态 |
|--------|---------------------|-----------|-------------|------|
| jobs | 20 | 0.2737 | +0.0005 | ✅ |
| congressional | 19 | 0.2970 | +0.0020 | ✅ |
| forums | 22 | 0.2507 | +0.0005 | ✅ |
| microblog | 18 | 0.2754 | +0.0004 | ✅ |

Round15 的问题不在结果，而在方法叙事：

1. 当前规则仍然依赖手工阈值。
2. `19/20/22/18` 的分配虽然有效，但更像 empirical rule，而不是 selector 自身的自然自适应行为。
3. 如果后续换数据切片、换训练上限、换数据集，阈值规则可能再次需要人工修补。

因此 Round16 的目标不是再去“修阈值”，而是把 seed budget 的决策逻辑进一步算法化。

## Round16 目标

主目标：

- 用更自然的 adaptive algorithm 替代 Round15 的硬阈值 budget rule。
- 保持四个数据集稳定超过 PrE-Text screening 基准。
- 在论文叙事上，把 budget 选择解释为 selector 内部自校准，而不是人工 family mapping。

次目标：

- 不修改 Stage2 bootstrap 主结构。
- 不修改 candidate generation 主流程。
- 尽量复用 Stage1 已有信号：private support、genericity penalty、redundancy、embedding coverage。
- 让调试信息可追踪，方便解释为什么某个数据集选择了某个 budget。

## 核心思路

Round16 不再依据 private text 的长度统计直接决定 `seed_top_k`，而是让 selector 在同一批候选样本上，对多个候选 budget 进行轻量自校准：

1. Stage1 正常生成 candidate texts。
2. 固定同一批 `candidate_vectors/private_support/genericity_penalty`。
3. 对多个候选 `seed_top_k` 分别运行一次 greedy selector。
4. 对每个 budget 对应的 selected set 计算 internal utility。
5. 自动选择 utility 最优的 budget。

新的预算选择逻辑可以写成：

> The selector chooses the seed budget that maximizes internal selection quality under a support-coverage-compactness tradeoff.

这比 Round15 的“长度统计阈值 -> budget”更像自然算法，因为 budget 是由 selector 自己在当前数据上的内部质量信号决定的。

## 推荐方案

推荐采用：

**Budget Self-Calibration + 轻量 tie-breaker**

不推荐一开始就做 bootstrap-heavy 的 full stability calibration。原因：

- Round16 需要先验证“无阈值化”是否成立；
- 先做 compact 版本更容易解释和调试；
- 如果 compact 版本已经实现 4/4 超过 PrE-Text，再决定要不要上更重的稳定性机制。

## 算法设计

### 候选 budget 集合

第一版固定：

```python
candidate_seed_top_k = [18, 19, 20, 21, 22]
```

理由：

1. 完全继承 Round13/14/15 已验证过的有效区间。
2. 不引入新的大搜索空间。
3. 计算成本可控。

后续如果确有必要，再扩展到 `[17, 18, 19, 20, 21, 22, 23]`。

### 自校准流程

在同一轮 Stage1 中：

```python
for k in candidate_seed_top_k:
    decision_k = greedy_select_candidates(..., seed_top_k=k, ...)
    metrics_k = evaluate_budget_selection(decision_k, ...)
    utility_k = combine_budget_metrics(metrics_k, k, calibration_cfg)

resolved_seed_top_k = argmax_k utility_k
final_decision = decision_resolved_seed_top_k
```

这里 `decision_k` 是同一批 candidates 上不同 budget 的轻量 selector 结果，因此不会额外增加 LLM 生成成本，增加的主要是向量和分数层面的计算。

### utility 结构

第一版推荐：

```python
utility(k) = (
    w_support * support_score
    - w_genericity * genericity_score
    - w_redundancy * redundancy_score
    + w_coverage * coverage_score
    - w_budget * budget_cost
)
```

推荐初始权重：

```python
w_support = 1.0
w_genericity = 0.5
w_redundancy = 0.3
w_coverage = 0.4
w_budget = 0.1
```

这些不是最终定值，而是 Round16 第一轮实验的初始默认值。

### 各项定义

#### 1. support_score

定义：

```python
support_score = mean(private_support[idx] for idx in selected_indices)
```

意义：

- 保证 selected seeds 仍以贴近 private 分布为主目标。
- 这是 utility 中最核心、权重最大的项。

#### 2. genericity_score

定义：

```python
genericity_score = mean(genericity_penalty[idx] for idx in selected_indices)
```

意义：

- 防止更大 budget 通过吸收更“泛”的样本来虚假提高 coverage。
- 保持 selector 不偏向过于安全但无信息量的样本。

#### 3. redundancy_score

定义：

- 取 selected set 内部两两相似度的平均值，或复用现有 redundancy 计算逻辑中的聚合统计。

推荐：

```python
redundancy_score = mean_pairwise_cosine(selected_vectors)
```

意义：

- 防止更大 budget 只是多选重复样本。
- 与 coverage 配合，形成“覆盖增加”和“集合塌缩”之间的平衡。

#### 4. coverage_score

推荐定义：

对每个 private sample，计算它与 selected set 中最近 selected sample 的最大相似度：

```python
cover_i = max(cosine(private_i, selected_j) for selected_j in selected_set)
coverage_score = mean(cover_i)
```

辅助记录但第一版不直接进 utility 的统计：

```python
coverage_p25 = percentile(cover_i, 25)
coverage_min = min(cover_i)
coverage_gain = coverage_score(k) - coverage_score(k-1)
```

意义：

- forums 需要更大 budget 的本质不是“文本更长”，而是 private manifold 更散、更混杂；
- coverage score 能更自然地区分“需要更多 seeds 才能覆盖”的数据；
- microblog 虽然也长，但如果在较低 budget 下 coverage 已经饱和，那么继续加 budget 只会带来 genericity/redundancy 成本，最终更小的 budget 会胜出。
- 因此 Round16 不只记录 coverage 的绝对值，也记录相邻 budget 的边际增益，用来判断 coverage 是否已经接近饱和。

#### 5. budget_cost

定义：

```python
budget_cost = (k - min(candidate_seed_top_k)) / (
    max(candidate_seed_top_k) - min(candidate_seed_top_k)
)
```

意义：

- 不是惩罚大 budget 本身；
- 而是在 utility 接近时，偏向更紧凑、更干净的 selected set；
- 避免算法天然滑向更大 budget。

### tie-breaker 设计

如果 top-2 utility 很接近，直接选最大值容易被噪声扰动。

推荐规则：

```python
if utility_gap <= epsilon:
    prefer smaller k
    unless coverage_gain(larger_k, smaller_k) >= coverage_gain_min
```

推荐初始值：

```python
epsilon = 0.01
coverage_gain_min = 0.005
```

解释：

- 当两个 budget 总体质量非常接近时，默认选更小、更紧凑的 budget；
- 只有更大的 budget 在 coverage 上提供了明确收益，才允许它胜出。

这使算法既不过分保守，也不会无脑偏向更大 budget。

### metric normalization

Round16 第一版必须先对各项 metric 做归一化，再进入 utility 组合。否则：

- `support/genericity/redundancy/coverage` 量纲未必一致；
- `epsilon=0.01` 这样的 tie-break 阈值会失去稳定语义；
- 权重 sweep 会混入“量纲差异”而不是真正的 trade-off。

推荐：

```python
normalized_metric = (metric - min_metric_across_candidate_k) / (
    max_metric_across_candidate_k - min_metric_across_candidate_k + 1e-8
)
```

即：

- 对同一数据集、同一轮 calibration 中的候选 budget 集合做组内归一化；
- utility 权重只作用在归一化后的 metric 上；
- `epsilon` 也定义在归一化 utility 空间里。

### private-signal 边界

Round16 的 `coverage_score` 会使用 `private_vectors` 做内部校准，因此必须明确 private-signal 边界：

1. `coverage_score` 只用于 Stage1 运行时的内部 budget calibration。
2. 不保存任何 per-private-sample 的覆盖轨迹、最近邻 id、逐样本相似度。
3. 落盘日志只允许保存聚合统计，例如：
   - `coverage_mean`
   - `coverage_p25`
   - `coverage_min`
   - `resolved_seed_top_k`
   - `per_budget_metrics`
4. Round16 不额外扩大 private artifact 的持久化范围。
5. 如果后续 formal 版本需要更强 privacy 叙事，再单独设计 noisy coverage 或 cluster-level coverage 变体；Round16 先不把它和当前实验混在一起。

这意味着：

- Round16 可以继续沿用当前 selector 实验范式；
- 但不应把 `coverage_score` 描述成“无代价新增的 privacy-preserving signal”。

## 与 Round15 的区别

Round15:

```text
private length stats -> threshold rule -> resolved seed_top_k
```

Round16:

```text
candidate set + selector internal metrics -> utility comparison -> resolved seed_top_k
```

Round16 的 budget 决策不再显式依赖：

- dataset family
- length threshold
- hand-coded mapping

因此论文叙事上更自然。

## 代码开发路径

### 开发目录

Round16 的新算法开发目录固定为：

[`/Users/apple/Desktop/code_from_paper/paper-new-round-16`](/Users/apple/Desktop/code_from_paper/paper-new-round-16)

推荐开发起点：

1. 先把 [`/Users/apple/Desktop/code_from_paper/paper-new-round11`](/Users/apple/Desktop/code_from_paper/paper-new-round11) 复制到 `paper-new-round-16`
2. 所有 Round16 代码修改只在 `paper-new-round-16` 内进行
3. `paper-new-round11` 保持为 Round15 成功基线，不再混改

### 推荐修改模块

核心新增文件：

```text
paper_new_selector/budget_calibration.py
```

职责：

- 计算 selected set 的 calibration metrics
- 计算 utility
- 在多个 budget 间选最优值

推荐保留 `stage1_runner.py` 为 orchestrator，不在其中堆叠过多校准细节。

#### 新增函数建议

```python
compute_selected_support_score(...)
compute_selected_genericity_score(...)
compute_selected_redundancy_score(...)
compute_selected_coverage_score(...)
compute_budget_cost(...)
combine_budget_metrics(...)
resolve_seed_top_k_by_self_calibration(...)
```

### Stage1 集成方式

当前 Round15 是：

```python
resolved_seed_top_k = resolve_seed_top_k(selector_cfg, private_lengths)
decision = greedy_select_candidates(..., seed_top_k=resolved_seed_top_k, ...)
```

Round16 推荐改成：

```python
calibration_result = resolve_seed_top_k_by_self_calibration(
    selector_cfg=selector_cfg,
    candidate_vectors=candidate_vectors,
    candidate_texts=candidate_texts,
    private_vectors=private_vectors,
    private_support=private_support,
    genericity_penalty=genericity_penalty,
)

resolved_seed_top_k = calibration_result["resolved_seed_top_k"]
decision = calibration_result["decision"]
```

这样可以避免为最终 budget 再重复运行一次 selector。

### 配置设计

Round16 推荐新增配置段：

```yaml
selector:
  seed_budget_rule:
    enabled: true
    mode: self_calibrated
    candidate_seed_top_k: [18, 19, 20, 21, 22]
    utility:
      support_weight: 1.0
      genericity_weight: 0.5
      redundancy_weight: 0.3
      coverage_weight: 0.4
      budget_weight: 0.1
    tiebreak:
      epsilon: 0.01
      coverage_gain_min: 0.005
      prefer_smaller_budget: true
```

注意：

- Round16 不再使用 length-family 阈值规则；
- `mode=self_calibrated` 与 Round15 的 `mode=length_family` 明确区分；
- 后续如果需要，也能保留两种 mode 共存，便于对照实验。

### 输出记录

Round16 必须在 Stage1 summary 中新增完整校准轨迹：

```json
"seed_budget": {
  "mode": "self_calibrated",
  "candidate_seed_top_k": [18, 19, 20, 21, 22],
  "resolved_seed_top_k": 22,
  "per_budget_metrics": {
    "18": {...},
    "19": {...},
    "20": {...},
    "21": {...},
    "22": {...}
  },
  "selected_utility": 0.713,
  "runner_up_seed_top_k": 21,
  "runner_up_utility": 0.706,
  "utility_gap": 0.007,
  "tiebreak_applied": true
}
```

这对于后续实验分析和论文图表都非常重要。

另外，Round16 建议显式把 Stage1 summary 写入独立文件，例如：

```text
paper-new/outputs/<experiment_id>/stage1_summary.json
paper-new/outputs/<experiment_id>/stage1_budget_calibration.json
```

而不是只依赖 stdout 中的大 JSON 输出。这样后续服务器实验在抽取 `resolved_seed_top_k`、`per_budget_metrics`、`utility_gap` 时会更稳定，也更方便自动汇总。

## 开发计划

### Phase 0: 建立 Round16 开发副本

目标：

- 复制 `paper-new-round11` 到 `paper-new-round-16`
- 确认 Round15 成功代码在 Round16 目录内能正常加载

执行项：

1. 复制工作目录
2. 运行基础单测
3. 确认 Round15 配置仍可解析
4. 确认 Round16 输出目录写文件逻辑可正常工作

### Phase 1: 新增校准模块与单元测试

目标：

- 先独立实现 `budget_calibration.py`
- 在不接入 `stage1_runner.py` 的情况下先把 utility 和 metric 逻辑跑通

执行项：

1. 实现 coverage / redundancy / utility 计算函数
2. 为每个函数编写单元测试
3. 设计构造样例，验证 larger budget 不会无脑获胜

建议测试文件：

```text
tests/test_budget_calibration.py
tests/test_seed_budget_rule.py  # 可保留旧模式兼容性测试
```

### Phase 2: 接入 Stage1

目标：

- 把 self-calibration 接入现有 Stage1
- 保证 `mode=self_calibrated` 时走新逻辑

执行项：

1. 更新 `stage1_runner.py`
2. 保留 `length_family` mode 兼容能力
3. Stage1 summary 记录 per-budget metrics
4. 为 `run_stage1()` 增加 integration test

### Phase 3: 配置与日志

目标：

- 生成 Round16 基础配置和实验配置
- 让日志足够解释预算选择

执行项：

1. 新增 `single_node_tuning_round16/` 配置目录
2. 生成 base config
3. 生成各实验 YAML
4. 验证 `load_yaml_config()` 后字段正确

### Phase 4: 快速实验与参数筛选

目标：

- 先确认 self-calibration 行为方向是对的
- 再扩大实验覆盖

执行项：

1. 先跑 forums + microblog 两个关键数据集
2. 看 `resolved_seed_top_k` 是否接近 22 / 18
3. 根据 per-budget utility breakdown 微调权重
4. 产出一套主配置和一套备选配置
5. 从 Phase 4 结束开始冻结权重；后续验证阶段不再回改权重

### Phase 5: 四数据集完整验证

目标：

- 找到一个 Round16 自校准方案，实现四个数据集全部超过 PrE-Text

执行项：

1. 用已冻结的主配置与备选配置运行四数据集 full screening
2. 对优胜配置做 seed robustness 验证
3. 与 Round15 做逐项比较

## 实验设计

Round16 的实验需要比 Round15 更广一些，因为我们不只是验证一个 rule，而是在验证一个新算法。

### 实验组 A: 单元行为验证

目的：

- 验证 self-calibration 的算法行为是否符合预期

内容：

1. synthetic toy cases
2. support 高但 redundancy 高的 case
3. coverage 高但 genericity 高的 case
4. top-2 utility 很接近时的 tie-break case

成功标准：

- utility 方向符合设计直觉
- tie-break 触发行为可复现

### 实验组 B: 关键数据集快速验证

目的：

- 用最少实验判断新算法有没有成为“自然替代”的潜力

配置：

| Config | 数据集 | 目标 |
|--------|--------|------|
| r16_probe_forums | forums | 应倾向选 22 左右 |
| r16_probe_microblog | microblog | 应倾向选 18 左右 |

成功标准：

- forums 的 resolved seed budget 应呈现偏大倾向
- microblog 的 resolved seed budget 应呈现偏小倾向
- 两者 best_top1 均不低于 PrE-Text

说明：

- `resolved_seed_top_k` 在 probe 阶段是诊断信号，不是硬成功标准；
- 如果某个数据集选出的 budget 与 Round15 不同，但 downstream 更好或不退化，应视为算法成功证据而不是失败。

### 实验组 C: 权重小扫

目的：

- 校准 utility 权重，而不是只跑一个默认权重

推荐实验矩阵：

| 组别 | support | genericity | redundancy | coverage | budget |
|------|---------|------------|------------|----------|--------|
| c1 | 1.0 | 0.5 | 0.3 | 0.4 | 0.1 |
| c2 | 1.0 | 0.5 | 0.3 | 0.5 | 0.1 |
| c3 | 1.0 | 0.5 | 0.4 | 0.4 | 0.1 |
| c4 | 1.0 | 0.6 | 0.3 | 0.4 | 0.1 |
| c5 | 1.0 | 0.5 | 0.3 | 0.4 | 0.15 |

建议先在：

- forums
- congressional

这两个冲突最强的数据集上跑。

成功标准：

- 找到至少一组权重，使 forums 和 congressional 同时过线
- 实验组 C 结束时锁定一套主配置和一套备选配置
- 后续实验组 D/E/F/G 只做验证，不再修改权重
- 同时记录各 metric 的原始分布与归一化后分布，确认 sweep 在比较真实 trade-off，而不是量纲偏置

### 实验组 D: 四数据集完整验证

目的：

- 检查一套统一自校准算法是否 4/4 全部超过 PrE-Text

运行：

| Config family | 数据集 |
|---------------|--------|
| Round16 best utility weights | jobs |
| Round16 best utility weights | congressional |
| Round16 best utility weights | forums |
| Round16 best utility weights | microblog |

成功标准：

- jobs > 0.2731984829329962
- congressional > 0.2949640287769784
- forums > 0.25014487154722814
- microblog > 0.2762705387848682

实验协议：

1. 先在实验组 C 锁定主配置和备选配置。
2. 实验组 D 只运行这两套锁定配置。
3. 不允许根据实验组 D 的结果再回头修改权重，并把修改后的配置重新包装成同一轮最终结果。
4. 如果主配置失败而备选配置成功，记录为“备选配置胜出”；如果两者都失败，进入新的设计轮次。

### 实验组 E: seed robustness

目的：

- 防止 Round16 只是 seed=42 的偶然高点

推荐：

| 数据集 | seeds |
|--------|-------|
| forums | 42, 123, 456 |
| congressional | 42, 123, 456 |
| jobs | 42, 123 |
| microblog | 42, 123 |

成功标准：

- 至少多数 seed 保持超过 PrE-Text
- forums / congressional 不出现大面积掉线

### 实验组 E2: Round15 正式对照

目的：

- 证明 Round16 不只是“也能过线”，而是可以作为 Round15 的自然替代

对照方式：

| Baseline | 数据集 |
|----------|--------|
| Round15 length_family | jobs |
| Round15 length_family | congressional |
| Round15 length_family | forums |
| Round15 length_family | microblog |

记录项：

- resolved_seed_top_k
- best_top1
- synthetic_train_count
- per_budget_metrics 摘要
- utility_gap

成功标准：

- Round16 4/4 不低于 PrE-Text
- 与 Round15 相比，不出现系统性退化
- 如果个别数据集略低于 Round15，但整体更自然且更稳，需要在结论中明确权衡，而不是自动宣称替代成功

### 实验组 F: 候选 budget 集合敏感性

目的：

- 看 `candidate_seed_top_k` 的边界是否影响结果

推荐对照：

1. `[18, 19, 20, 21, 22]`
2. `[17, 18, 19, 20, 21, 22, 23]`
3. `[18, 20, 22]`

成功标准：

- 主结论在合理 budget 候选集合下保持稳定

### 实验组 G: ablation

目的：

- 证明 Round16 的关键增益到底来自哪一项

推荐消融：

| Ablation | 说明 |
|----------|------|
| no_coverage | utility 去掉 coverage |
| no_budget_cost | utility 去掉 budget penalty |
| no_tiebreak | utility 最大值直接选 |
| no_redundancy | utility 去掉 redundancy |

重点观察：

- forums 是否在 no_coverage 下掉回较小 budget
- 所有数据集是否在 no_budget_cost 下偏向更大 budget

## 配置规划

Round16 推荐目录：

```text
paper-new-round-16/configs/experiments/single_node_tuning_round16/
├── _base_selector_tuning_round16.yaml
├── probes/
├── weight_sweep/
├── full_run/
├── seed_robustness/
└── ablations/
```

这样实验组织会比 Round13/14/15 更清晰，也便于后续服务器批量调度。

## 风险

### 风险 1: utility 偏向更大 budget

表现：

- 所有数据集都往 22 靠

缓解：

- 加入 budget cost
- 启用 close-call tie-break
- 跑 no_budget_cost ablation 对照

### 风险 2: coverage 定义太强，反而牺牲 precision

表现：

- forums 变好，但 congressional/jobs 退化

缓解：

- coverage 权重不要过大
- genericity/redundancy 保持为明确约束项

### 风险 3: internal utility 和 downstream best_top1 不完全一致

表现：

- utility 选出的 budget 在 selector 内部看起来合理，但评估不一定最好

缓解：

- Round16 第一轮必须保留权重 sweep
- 用四数据集对 utility 进行 empirical calibration

### 风险 4: 实现复杂度过高

表现：

- `stage1_runner.py` 膨胀，调试困难

缓解：

- 新建 `budget_calibration.py`
- 把 metric 计算与 orchestration 分离

## 预期收益

如果 Round16 成功，将比 Round15 更强：

1. 结果层面：仍保持 4/4 超过 PrE-Text。
2. 方法层面：不再依赖长度阈值。
3. 叙事层面：seed budget 由 selector 内部自校准自动选择。
4. 泛化层面：对新的数据切片或新的数据集更有希望直接适配。

## 论文叙事建议

英文表述：

> We replace the hand-designed adaptive seed-budget rule with a self-calibrated budget selection mechanism. Given a shared candidate pool, the selector evaluates multiple seed-budget candidates and chooses the one that maximizes an internal utility defined over private support, genericity control, redundancy suppression, and coverage of the private manifold. This makes the seed budget an emergent property of the selector rather than a manually specified dataset rule.

中文表述：

> 我们进一步用自校准预算选择机制替代手工设计的 adaptive seed-budget rule。在共享候选池上，selector 对多个候选 seed budget 分别执行轻量选择，并根据 private support、genericity 抑制、redundancy 控制以及 private manifold 覆盖构成的内部效用函数自动选择最优 budget。这样 seed budget 不再是人工指定的数据集规则，而成为 selector 内部自适应决策的自然结果。

## 成功标准

Round16 成功至少需要满足：

1. 不使用长度阈值或 dataset-family 显式映射。
2. `mode=self_calibrated` 下 selector 自动选择 budget。
3. Stage1 summary 中可完整追踪每个 budget 的 utility breakdown。
4. 至少存在一套统一自校准配置，使四个数据集全部超过 PrE-Text。
5. forums 与 congressional 的冲突能够通过 internal utility 自然解开，而不是靠手工 if/else。

更理想的成功标准：

6. 相比 Round15，utility 选择逻辑具有更强的解释性。
7. 在 `meta.seed` 变化下结果具有基本鲁棒性。
8. ablation 能说明 coverage / budget_cost / tie-break 的必要性。

## 推荐实施顺序

1. 在 `paper-new-round-16` 建立 Round16 开发副本。
2. 新增 `budget_calibration.py` 和单元测试。
3. 接入 `stage1_runner.py`，保留 `length_family` 向后兼容。
4. 生成 Round16 probe / sweep / ablation 配置。
5. 先跑 forums + microblog probe。
6. 再跑 forums + congressional 的权重小扫。
7. 选出最优权重后跑四数据集 full screening。
8. 对优胜配置补 seed robustness 和 ablation。

如果这一路跑通，Round16 就可以作为“从 empirical rule 走向自然 adaptive algorithm”的关键版本。
