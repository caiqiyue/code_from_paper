# Round 6 全面参数调优实验实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不损害 jobs/microblog/congressional 已超越基线的前提下，通过多维度参数探索（λ_generic、λ_redundancy、seed_top_k、gate_low、mid_scale）突破 forums 的 PrE-Text 基线 (0.2501)。

**Architecture:** 在 `stage1_runner.py` 中通过 `_override_*` 配置键支持 per-dataset 参数覆盖，实现路径 1（硬编码数据集名）。

**Tech Stack:** Python 3.10, YAML config, paper-new-round5 pipeline

---

## 实验设计：10 配置 × 4 数据集 = 40 实验

| 配置 | λ_generic | λ_redundancy | seed_top_k | gate_low | mid_scale | 描述 |
|------|-----------|--------------|------------|----------|-----------|------|
| c01 | 0.35 | 0.25 | 10 | 0.78 | 0.45 | **默认基准** |
| c02 | 0.30 | 0.25 | 10 | 0.78 | 0.45 | 降低 generic 权重 |
| c03 | 0.25 | 0.25 | 10 | 0.78 | 0.45 | 进一步降低 |
| c04 | 0.20 | 0.25 | 10 | 0.78 | 0.45 | |
| c05 | 0.15 | 0.25 | 10 | 0.78 | 0.45 | |
| c06 | 0.10 | 0.25 | 10 | 0.78 | 0.45 | |
| c07 | 0.05 | 0.25 | 10 | 0.78 | 0.45 | |
| c08 | 0.01 | 0.25 | 10 | 0.78 | 0.45 | **极低 generic** |
| c09 | 0.35 | 0.20 | 10 | 0.78 | 0.45 | 降低 redundancy |
| c10 | 0.35 | 0.15 | 10 | 0.78 | 0.45 | 进一步降低 redundancy |

每个配置在 4 个数据集（jobs, forums, microblog, congressional）上各跑 1 次，共 40 个实验。

预期执行时间：~40 × 5-6min ≈ 3-4 小时（可过夜）

---

## stage1_runner.py override 机制

**修改位置:** `paper-new-round5/paper_new_selector/stage1_runner.py` line 195-198

**Override 参数键（按数据集名自动匹配）:**
| 配置键 | 触发数据集 | 效果 |
|--------|-----------|------|
| `_forums_lambda_generic` | forums | 覆盖 λ_generic |
| `_forums_lambda_redundancy` | forums | 覆盖 λ_redundancy |
| `_forums_seed_top_k` | forums | 覆盖 seed_top_k |
| `_forums_gate_low` | forums | 覆盖 gate_low |
| `_forums_mid_scale` | forums | 覆盖 mid_scale |

**代码逻辑:**
```python
_dataset_name = str(config.get("data", {}).get("dataset_name", ""))
if _dataset_name == "forums":
    _override_key = "_forums_lambda_generic"
    if _override_key in selector_cfg:
        selector_cfg["lambda_generic"] = float(selector_cfg[_override_key])
    _override_key = "_forums_lambda_redundancy"
    if _override_key in selector_cfg:
        selector_cfg["lambda_redundancy"] = float(selector_cfg[_override_key])
    # ... seed_top_k, gate_low, mid_scale 同理
```

---

## 文件结构

```
paper-new-round5/
├── paper_new_selector/
│   └── stage1_runner.py                              # 修改: 扩展 override 支持
└── configs/experiments/
    └── single_node_tuning_round6/                    # 40 个实验配置
        ├── _base_selector_tuning_round6.yaml         # 基础配置 (g1 gate)
        ├── _c01.yaml ~ _c10.yaml                     # 10 个 group configs
        ├── ns_tune6_c01_{jobs,forums,microblog,congressional}.yaml  # 4×10=40 leaf

old_automation/
└── run_round6_queue.py                               # 40 实验队列脚本
```

---

## 实现步骤

### Task 1: 修改 stage1_runner.py — 扩展 override 支持

**Files:**
- Modify: `paper-new-round5/paper_new_selector/stage1_runner.py:195-198`

```python
# Dataset-specific parameter override for forums tuning
_dataset_name = str(config.get("data", {}).get("dataset_name", ""))
if _dataset_name == "forums":
    _overrides = [
        ("_forums_lambda_generic", "lambda_generic"),
        ("_forums_lambda_redundancy", "lambda_redundancy"),
        ("_forums_seed_top_k", "seed_top_k"),
        ("_forums_gate_low", "genericity_gate_low"),
        ("_forums_mid_scale", "genericity_gate_mid_scale"),
    ]
    for _src_key, _tgt_key in _overrides:
        if _src_key in selector_cfg:
            selector_cfg[_tgt_key] = float(selector_cfg[_src_key])
```

- [ ] **Step 1: 修改 stage1_runner.py**
- [ ] **Step 2: 验证语法** `python -m py_compile stage1_runner.py`

---

### Task 2: 创建基础配置 + 10 个 group configs

**Files:**
- Create: `_base_selector_tuning_round6.yaml`
- Create: `_c01.yaml` ~ `_c10.yaml`

- [ ] **Step 1: 写入 _base_selector_tuning_round6.yaml**（g1 gate, length_modulation=false）
- [ ] **Step 2: 写入 _c01.yaml ~ _c10.yaml**（每个对应一组参数覆盖）

---

### Task 3: 创建 40 个 leaf 实验配置

**Files:**
- Create: `ns_tune6_c{01~10}_{jobs,forums,microblog,congressional}.yaml` (40 个)

每个 leaf 继承对应的 group config。例如 `ns_tune6_c05_forums.yaml` 继承 `_c05.yaml`，并设置 `dataset_name: forums`。

- [ ] **Step 1: 创建 c01~c05 组各 4 个 leaf**
- [ ] **Step 2: 创建 c06~c10 组各 4 个 leaf**

---

### Task 4: 创建 40 实验队列运行脚本

**Files:**
- Create: `old_automation/run_round6_queue.py`

实验 ID 格式: `ns_tune6_c{dataset_config}_{dataset_name}`

- [ ] **Step 1: 创建 run_round6_queue.py**
- [ ] **Step 2: 验证语法**

---

## 自审检查清单

- [ ] 10 个 group configs 参数值与设计表一致
- [ ] 40 个 leaf configs 数据集名称正确（jobs/forums/microblog/congressional）
- [ ] 40 个 leaf configs 路径正确（train_path/eval_path/initialization_path）
- [ ] 队列脚本 40 个实验 ID 完整
- [ ] py_compile 验证通过
