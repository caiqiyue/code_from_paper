# Round 8: Forums Bootstrap max_tokens 调优实验

**日期**: 2026-04-28  
**状态**: ✅ 已完成（实验失败，发现新 insights）

---

## 1. 问题分析

### 1.1 当前最佳实验结果

| 数据集 | 当前最佳 | seed_top_k | PrE-Text | 差异 | 状态 |
|--------|---------|-----------|----------|------|------|
| jobs | 0.2798 | 23 | 0.2718 | +0.0080 | ✅ 超越 |
| congressional | 0.2970 | 6 | 0.2906 | +0.0064 | ✅ 超越 |
| microblog | 0.2820 | 23 | 0.2683 | +0.0140 | ✅ 超越 |
| forums | 0.2498 | 23 | 0.2501 | -0.0003 | ❌ 未超越 |

### 1.2 根本原因分析

**合成语料长度压缩严重：**

| 数据集 | 原始平均词数 | 合成平均词数 | 压缩比 |
|--------|------------|------------|--------|
| forums | 379.4 | 42.6 | **11%** |
| jobs | 270.0 | 47.8 | 18% |
| congressional | 227.1 | 43.2 | 19% |
| microblog | 348.4 | 48.5 | 14% |

**关键发现：**
- forums 的信息压缩最严重（仅保留 11%）
- 其他数据集保留 14-19%
- max_tokens=85 限制了生成长文本的能力

### 1.3 假设

增加 max_tokens 可以让 forums 的合成语料更完整，从而提升下游任务性能。

---

## 2. 解决方案

### 2.1 代码修改

**修改文件**: `paper-new-round5/paper_new_selector/pretext_bridge.py`

**修改位置**: `prepare_bootstrap_runtime` 函数

**新增代码**:
```python
_dataset_name = str(config.get("data", {}).get("dataset_name", ""))
if _dataset_name == "forums":
    _bootstrap_overrides = [
        ("_forums_max_tokens", "max_tokens"),
    ]
    for _src_key, _tgt_key in _bootstrap_overrides:
        if _src_key in config.get("selector", {}):
            if "bootstrap" not in config:
                config["bootstrap"] = {}
            config["bootstrap"][_tgt_key] = float(config["selector"][_src_key])
```

---

## 3. 实验设计

| 实验 ID | 数据集 | seed_top_k | max_tokens | 描述 |
|---------|--------|-----------|-----------|------|
| ns_tune8_f1_forums | forums | 23 | 150 | max_tokens 增大测试 |

---

## 4. 实验结果

### 4.1 实际结果

| 配置 | max_tokens | 合成平均词数 | best_top1 | vs PrE-Text |
|------|-----------|-------------|-----------|-------------|
| 对照组 (s09) | 85 | 42.6 | 0.2498 | -0.0003 ❌ |
| 实验组 (f1) | 150 | 64.0 | **0.2465** | **-0.0036** ❌ |

### 4.2 关键发现

**❌ max_tokens 增大反而使性能下降！**

- 合成语料确实变长了（42.6 → 64.0 词，+50%）
- 但下游 best_top1 从 0.2498 下降到 0.2465（-0.0033）
- 距离 PrE-Text 基线从 -0.0003 扩大到 -0.0036

---

## 5. 新insights

### 5.1 假设验证结果

**原假设被否定**：增加 max_tokens 不能提升 forums 性能，反而使其下降。

### 5.2 新发现

对于 forums 数据集：
- **较短的合成文本表现更好**（0.2498 vs 0.2465）
- 长文本可能引入更多噪声
- 短文本质量更高、更集中

### 5.3 可能的解释

1. **模型能力限制**：LLaMA-2-7B 可能更适合生成短文本
2. **任务特性**：forums 下游任务可能更关注文本的核心信息，而非完整长度
3. **噪声累积**：生成长文本时，模型可能在后期产生更多无关内容

---

## 6. 结论

| 假设 | 结果 | 说明 |
|------|------|------|
| 增大 max_tokens 能提升 forums 性能 | ❌ **失败** | 性能反而下降 0.0033 |

**需要探索其他方向来提升 forums 性能。**

---

## 7. 下一步方向（待讨论）

1. **减少 max_tokens**：尝试 50 或 60
2. **调整 seed_top_k**：回退到 seed_top_k=6 或其他值
3. **调整 genericity gate 参数**：尝试不同的 gate_low/high
4. **改变 candidate initialization**：使用更接近 forums 领域的数据
5. **接受现状**：forums 差距极小（0.0003），可能属于实验误差范围

---

## 8. 附录

### 8.1 代码修改位置

```python
# pretext_bridge.py, prepare_bootstrap_runtime 函数

def prepare_bootstrap_runtime(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)

    _dataset_name = str(config.get("data", {}).get("dataset_name", ""))
    if _dataset_name == "forums":
        _bootstrap_overrides = [
            ("_forums_max_tokens", "max_tokens"),
        ]
        for _src_key, _tgt_key in _bootstrap_overrides:
            if _src_key in config.get("selector", {}):
                if "bootstrap" not in config:
                    config["bootstrap"] = {}
                config["bootstrap"][_tgt_key] = float(config["selector"][_src_key])

    repo_root = resolve_repo_root()
    ...
```

### 8.2 配置文件

**_base_selector_tuning_round8.yaml**:
```yaml
inherits:
  - ../single_node_tuning_round4/_base_selector_tuning_round4.yaml

meta:
  stage: single_node_tuning_round8
```

**ns_tune8_f1_forums.yaml**:
```yaml
inherits:
  - ./_base_selector_tuning_round8.yaml

selector:
  _forums_seed_top_k: 23
  _forums_max_tokens: 150

meta:
  experiment_id: ns_tune8_f1_forums

paths:
  output_root: paper-new/outputs/ns_tune8_f1_forums

data:
  dataset_name: forums
  train_path: thesis_platform/datasets/pretext_forums/formatted/forums_train.json
  eval_path: thesis_platform/datasets/pretext_forums/formatted/forums_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```