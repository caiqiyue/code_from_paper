# Round 11: Forums Penalty Strength Reduction (Option A)

## Background

### Problem Statement

forums 数据集始终无法超越 PrE-Text 基线 (0.2501)，最佳成绩为 0.2498，差距仅 0.0003。

通过 Round 6-10 的参数调优实验，发现：
- max_tokens 细粒度搜索已验证 85 为最优值
- seed_top_k 细粒度搜索已验证 23 为最优值
- 无论怎么调参，forums 都卡在 ~0.2498

这说明问题不在参数调优，而是**算法框架对 forums 特殊数据类型的适配性**。

### Root Cause Analysis

forums 数据集的特殊性：
| 特性 | forums | 其他数据集 |
|------|--------|------------|
| 平均长度 | 379 词 | 227-348 词 |
| 结构化程度 | 非结构化（论坛帖子） | 较结构化 |
| 领域混杂度 | 高（各种话题混合） | 较单一 |
| 超越基线 | ❌ | ✅ |

**核心问题**：
- `genericity_penalty` 基于与初始化语料的 cosine similarity
- forums 的词汇本就多样、非结构化，容易被误判为"generic"
- `redundancy_penalty` 基于与已选中的 cosine similarity
- forums 的长文本 + 领域混杂 = 容易被过度惩罚

## Solution Design

### Option A: 降低 Forums 的 Penalty 强度

利用现有的 forums 参数覆盖机制，在 `stage1_runner.py` 中为 forums 设置独立的参数值。

#### 修改文件

**1. `paper_new_selector/stage1_runner.py`**

**关键修复**：将 forums override 移到 genericity_penalty 计算**之前**，确保 gate_high 和 low_scale 正确应用。

```python
_dataset_name = str(config.get("data", {}).get("dataset_name", ""))
if _dataset_name == "forums":
    _overrides = [
        ("_forums_lambda_generic", "lambda_generic"),
        ("_forums_lambda_redundancy", "lambda_redundancy"),
        ("_forums_seed_top_k", "seed_top_k"),
        ("_forums_gate_low", "genericity_gate_low"),
        ("_forums_gate_high", "genericity_gate_high"),
        ("_forums_low_scale", "genericity_gate_low_scale"),
        ("_forums_mid_scale", "genericity_gate_mid_scale"),
    ]
    for _src_key, _tgt_key in _overrides:
        if _src_key in selector_cfg:
            selector_cfg[_tgt_key] = float(selector_cfg[_src_key])
# 然后计算 genericity_penalty 使用已覆盖的参数
```

### 参数调整理由

| 参数 | 通用值 | Forums 值 | 理由 |
|------|--------|-----------|------|
| lambda_generic | 0.35 | 0.15 | forums 词汇多样，降低 genericity 惩罚 |
| lambda_redundancy | 0.25 | 0.10 | forums 长文本，降低冗余惩罚 |
| genericity_gate_low | 0.78 | 0.85 | 提高阈值，减少被判定为"太通用"的候选 |
| genericity_gate_high | 0.90 | 0.95 | 提高阈值，允许更多样化 |
| genericity_gate_low_scale | 0.10 | 0.20 | 放宽低 genericity 的惩罚 |
| genericity_gate_mid_scale | 0.45 | 0.60 | 放宽中等 genericity 的惩罚 |

## Experiment Design

### Round 11 实验配置

**目标**: 验证降低 penalty 强度是否能帮助 forums 超越 0.2501 基线

| 实验 | 数据集 | lambda_generic | lambda_redundancy | gate_low | gate_high | mid_scale |
|------|--------|----------------|-------------------|----------|-----------|-----------|
| ns_tune11_f1 | forums | 0.15 | 0.10 | 0.85 | 0.95 | 0.60 |
| ns_tune11_f2 | forums | 0.20 | 0.15 | 0.82 | 0.92 | 0.50 |
| ns_tune11_f3 | forums | 0.10 | 0.08 | 0.88 | 0.96 | 0.70 |

### 回归测试

确保其他数据集不受影响：
| 实验 | 数据集 | 预期结果 |
|------|--------|----------|
| ns_tune11_jobs | jobs | ~0.2798 (保持) |
| ns_tune11_congressional | congressional | ~0.2970 (保持) |
| ns_tune11_microblog | microblog | ~0.2820 (保持) |

## Implementation Status

✅ **已完成**:
1. `paper_new_selector/stage1_runner.py` - 修改 override 逻辑顺序，增加 gate_high 和 low_scale
2. `configs/experiments/single_node_tuning_round11/_base_selector_tuning_round11.yaml` - 基础配置
3. 6 个实验配置 - 3 个 forums + 3 个回归测试

## Files Created/Modified

```
paper-new-round11/
├── paper_new_selector/
│   └── stage1_runner.py          # ✅ 已修改: override 移到 genericity_penalty 之前
├── configs/experiments/
│   └── single_node_tuning_round11/
│       ├── _base_selector_tuning_round11.yaml  ✅
│       ├── ns_tune11_f1_forums.yaml            ✅
│       ├── ns_tune11_f2_forums.yaml            ✅
│       ├── ns_tune11_f3_forums.yaml            ✅
│       ├── ns_tune11_jobs.yaml                 ✅
│       ├── ns_tune11_congressional.yaml        ✅
│       └── ns_tune11_microblog.yaml            ✅
```

## Expected Outcome

- forums: 0.2498 → >0.2501 (超越基线)
- 其他数据集: 保持现有水平

## Risk Mitigation

- **风险**: 过度降低 penalty 导致选择质量下降
- **缓解**: 回归测试确保其他数据集不受影响
- **备用**: 如果 forums 性能下降，恢复到 Round 8 配置

## Next Steps

1. 同步 round11 代码到服务器
2. 执行 6 个实验
3. 分析结果