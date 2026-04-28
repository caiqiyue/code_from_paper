# Round 8: Forums Bootstrap max_tokens 调优实验设计

**日期**: 2026-04-28  
**状态**: 待实施  
**目的**: 解决 forums 数据集未能超越 PrE-Text 基线的问题

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

### 2.1 代码修改说明

**重要更正**：`max_tokens` 是在 `bootstrap` 配置中的，不是在 `selector` 中。

- `selector` override 在 `stage1_runner.py` 中处理
- `bootstrap` 配置在 `pretext_bridge.py` 的 `prepare_bootstrap_runtime` 函数中处理

**修改文件**: `paper-new-round5/paper_new_selector/pretext_bridge.py`

**修改位置**: `prepare_bootstrap_runtime` 函数（约 line 100-110）

**修改内容**:
在 `bootstrap_cfg` 字典构建之前，添加 override 逻辑：

```python
# 在 line 100 之前添加
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

bootstrap_cfg = {
    ...
}
```

### 2.2 修改理由

1. **最小改动**：只增加几行代码
2. **向后兼容**：不影响已有配置的行为
3. **可逆性**：如实验失败，可快速回滚
4. **隔离性**：只在 forums 数据集时生效

---

## 3. 实验设计

### 3.1 设计原则

根据 screening 方法论，保持三个已成功数据集（jobs/congressional/microblog）的现有最佳参数不变，只针对 forums 问题进行定向调整。

### 3.2 现有最佳参数（保持不变）

| 数据集 | seed_top_k | max_tokens | 状态 |
|--------|-----------|-----------|------|
| jobs | 23 | 85 | ✅ 已超越 |
| congressional | 6 | 85 | ✅ 已超越 |
| microblog | 23 | 85 | ✅ 已超越 |

### 3.3 实验配置

**实验组**: seed_top_k=23, max_tokens=150 for forums

| 实验 ID | 数据集 | seed_top_k | max_tokens | 描述 |
|---------|--------|-----------|-----------|------|
| ns_tune8_f1_forums | forums | 23 | 150 | max_tokens 增大测试 |

**对照组**: seed_top_k=23, max_tokens=85 for forums（已有）

| 实验 ID | 数据集 | seed_top_k | max_tokens | best_top1 | 状态 |
|---------|--------|-----------|-----------|-----------|------|
| ns_tune7_s09_forums | forums | 23 | 85 | 0.2498 | ❌ 未超越 |

### 3.4 预期结果

| max_tokens | 预期合成词数 | 预期 best_top1 | 预期 vs PrE-Text |
|-----------|------------|----------------|------------------|
| 85 (当前) | ~42 词 | 0.2498 | -0.0003 ❌ |
| 150 (实验) | ~70-80 词 | ? | ? |

---

## 4. 实施计划

### 4.1 步骤 1: 修改代码

1. 修改 `pretext_bridge.py`，在 `prepare_bootstrap_runtime` 函数中增加 `_forums_max_tokens` override
2. 验证修改语法正确
3. 本地测试 import 正常

### 4.2 步骤 2: 创建实验配置

创建配置目录和文件：
```
paper-new-round5/configs/experiments/single_node_tuning_round8/
├── _base_selector_tuning_round8.yaml
└── ns_tune8_f1_forums.yaml
```

### 4.3 步骤 3: 子智能体审核

1. 检查代码修改是否正确
2. 检查配置文件继承链
3. 检查路径、名称是否正确

### 4.4 步骤 4: 同步到服务器并执行

1. 手动同步本地代码到服务器
2. 在 A6000 GPU 上执行 `ns_tune8_f1_forums` 实验
3. 记录结果

---

## 5. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| max_tokens 增大导致显存不足 | 中 | 使用 A6000（48GB），风险低 |
| 生成时间显著增加 | 低 | 增加约 2x，仍可接受 |
| 破坏其他数据集的现有优势 | 极低 | 只影响 forums，不改变其他数据集配置 |

---

## 6. 成功标准

**主目标**：forums 的 best_top1 > PrE-Text 0.2501

**次目标**：forums 的 best_top1 有显著提升（至少 +0.001）

---

## 7. 附录

### 7.1 代码修改位置

```python
# pretext_bridge.py, prepare_bootstrap_runtime 函数

def prepare_bootstrap_runtime(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    
    # ========== 新增 override 逻辑 (约 line 85-93) ==========
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
    # ========== 新增 override 逻辑结束 ==========
    
    repo_root = resolve_repo_root()
    ...
```

### 7.2 配置文件

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

### 7.3 配置路径汇总

```
paper-new-round5/
├── paper_new_selector/
│   └── pretext_bridge.py           # 待修改：增加 _forums_max_tokens override
└── configs/experiments/
    └── single_node_tuning_round8/   # 新建目录
        ├── _base_selector_tuning_round8.yaml
        └── ns_tune8_f1_forums.yaml