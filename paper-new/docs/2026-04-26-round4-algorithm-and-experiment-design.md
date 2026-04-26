# Round 4：三段式条件泛化惩罚算法与实验设计

**日期：** 2026-04-26  
**分支：** paper-2-genereic  
**阶段：** Round 4 调参实验（16个实验）

---

## 一、背景与问题

### 1.1 已有机制回顾

Stage 1 候选句筛选器的核心评分公式为：

```
score(c) = support(c) - λ_generic × genericity(c) - λ_redundancy × redundancy(c)
```

其中 `genericity(c)` 衡量候选句 `c` 与公共初始化语料（C4-en）的相似程度——相似度越高，说明该候选句越"通用"，不具备对当前数据集的特征性表达能力，应被惩罚压低得分。

Round 3 引入了**加权参考平滑**（Weighted Reference Smoothing）：将 genericity 参考向量从简单均值改为对 top-k 邻居的秩加权均值，使得 genericity 估计对单一异常向量更鲁棒。

### 1.2 现存问题

Round 3 结果分析显示：

- **jobs / congressional**：较 PrE-Text baseline 有显著改进，泛化惩罚有效
- **forums / microblog**：改进幅度小于预期，部分实验甚至不如 baseline

根因分析：

> `forums` 和 `microblog` 数据集的文本风格是**对话式、口语化**的，其候选句本身就与 C4-en 存在一定距离（genericity 原始分数集中在 0.75–0.88 中间段），**不是真正的模板化通用文本**。对这类中间段候选句施加与高泛化候选句相同强度的惩罚，会错误地压低有价值候选句的得分，导致筛选器偏向选择冗余但"看起来不通用"的候选句。

### 1.3 根本解决方案

引入**三段式条件泛化惩罚门控（Three-Band Conditional Genericity Gate）**：

- **低分段**（score ≤ gate_low）：候选句与初始化语料差异大，根本不需要惩罚 → 将惩罚乘以极小系数（low_scale ≈ 0.10）
- **中分段**（gate_low < score ≤ gate_high）：候选句有轻微泛化倾向，施加缓和惩罚 → 乘以 mid_scale（0.30–0.45）
- **高分段**（score > gate_high）：候选句高度通用，施加完整惩罚 → scale = 1.0（不改变）

---

## 二、创新算法：三段式条件泛化惩罚

### 2.1 算法整体流程

```
输入：
  候选句向量集合 {c_1, ..., c_n}（Stage 1 生成的候选摘要句）
  参考语料向量集合 {r_1, ..., r_m}（C4-en 初始化集合的嵌入）
  超参数：reference_top_k, reference_rank_weights
  门控参数：gate_low, gate_high, gate_low_scale, gate_mid_scale
  惩罚权重：lambda_generic

流程：
  For each candidate c_i:
    1. 计算 raw genericity score（见 2.2）
    2. 根据 raw score 确定门控系数 g（见 2.3）
    3. gated_penalty = raw_score × g
    4. stage1_score(c_i) = support(c_i) - lambda_generic × gated_penalty - lambda_redundancy × redundancy(c_i)

输出：
  每个候选句的 stage1_score → 用于贪心筛选
```

### 2.2 Raw Genericity Score 计算

**文件：** `paper-new/paper_new_selector/genericity.py`，函数 `compute_genericity_penalty`（第 55–93 行）

```
步骤：
1. 遍历所有参考向量，计算 candidate_vector 与每个 reference_vector 的余弦相似度
2. 取 top-k 相似度（reference_top_k = 6）
3. 对 top-k 分数用秩加权均值（reference_rank_weights = [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]）：

   raw_score = Σ(sim_k × w_k) / Σ(w_k)

4. 将 raw_score 裁剪到 [0, 1]
```

秩加权均值（Round 3 引入）的作用：越排名靠前（越相似）的参考向量权重越大，这比简单平均更能反映候选句与初始化分布"最近邻"的接近程度，同时对单一异常参考向量更鲁棒。

### 2.3 门控函数（三段式）

**文件：** `paper-new/paper_new_selector/genericity.py`，函数 `apply_genericity_gate`（第 40–52 行）

```python
def apply_genericity_gate(
    *, score, gate_low, gate_high, low_scale, mid_scale
) -> float:
    if score <= gate_low:
        return low_scale      # 低分段：极小惩罚
    if score <= gate_high:
        return mid_scale      # 中分段：缓和惩罚
    return 1.0                # 高分段：完整惩罚
```

**门控后的 penaly 计算：**

```
gated_penalty = raw_score × apply_genericity_gate(raw_score, ...)
             = raw_score × gate_scale
```

注意：`gated_penalty` 保持了与 `raw_score` 的单调关系，并非简单地将低分段置零——低分段依然有极小但非零的惩罚（`low_scale=0.10`），保证评分的连续性。

### 2.4 三段式惩罚的直觉图示

```
gated_penalty
     |
1.0  |                              ╔══════════╗  (高分段，full scale)
     |                             ╔╝
0.45 |              ╔══════════════╝           (中分段，mid_scale=0.45)
     |             ╔╝
0.10 |╔════════════╝                           (低分段，low_scale=0.10)
     |
     +-----|--------|-----|-------------------→ raw_score
          0        0.78  0.90   1.0
                gate_low gate_high
```

（纵轴为 `gate_scale`；实际 `gated_penalty = raw_score × gate_scale`，斜率保持单调）

### 2.5 Stage 1 Runner 中的集成

**文件：** `paper-new/paper_new_selector/stage1_runner.py`，第 168–178 行

```python
genericity_penalty = compute_genericity_penalties(
    candidate_vectors=candidate_vectors,
    reference_vectors=reference_vectors,
    reference_top_k=int(selector_cfg["reference_top_k"]),
    reference_rank_weights=list(selector_cfg.get("reference_rank_weights", [])),
    apply_gate=True,
    gate_low=float(selector_cfg.get("genericity_gate_low", 0.0)),
    gate_high=float(selector_cfg.get("genericity_gate_high", 1.0)),
    low_scale=float(selector_cfg.get("genericity_gate_low_scale", 1.0)),
    mid_scale=float(selector_cfg.get("genericity_gate_mid_scale", 1.0)),
)
```

`apply_gate=True` 始终开启，门控参数从实验配置 YAML 中读取，全部走 `selector_cfg` 命名空间。

---

## 三、配置体系

### 3.1 配置继承树

```
single_node_formal/_base_selector_formal.yaml
└── single_node_tuning_round4/_base_selector_tuning_round4.yaml  ← 公共基础
    ├── _g1_conditional_genericity_default.yaml         (g1：默认门控)
    │   ├── ns_tune4_g1_jobs.yaml
    │   ├── ns_tune4_g1_congressional.yaml
    │   ├── ns_tune4_g1_forums.yaml
    │   └── ns_tune4_g1_microblog.yaml
    ├── _g2_conditional_genericity_soft_mid.yaml        (g2：更软中段)
    │   ├── ns_tune4_g2_jobs.yaml
    │   ├── ns_tune4_g2_congressional.yaml
    │   ├── ns_tune4_g2_forums.yaml
    │   └── ns_tune4_g2_microblog.yaml
    ├── _g3_conditional_genericity_early_high.yaml      (g3：更早触发高段)
    │   ├── ns_tune4_g3_jobs.yaml
    │   ├── ns_tune4_g3_congressional.yaml
    │   ├── ns_tune4_g3_forums.yaml
    │   └── ns_tune4_g3_microblog.yaml
    └── _g4_conditional_genericity_plus_a2.yaml         (g4：g1 + length_lambda)
        ├── ns_tune4_g4_jobs.yaml
        ├── ns_tune4_g4_congressional.yaml
        ├── ns_tune4_g4_forums.yaml
        └── ns_tune4_g4_microblog.yaml
```

### 3.2 公共基础参数（`_base_selector_tuning_round4.yaml`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `train_limit` | 256 | 训练样本数 |
| `eval_limit` | 256 | 评测样本数 |
| `initialization_limit` | 1024 | 初始化参考样本数 |
| `candidate_count` | 24 | 每轮生成候选句总数 |
| `generated_per_round` | 8 | 每轮生成数 |
| `max_rounds` | 4 | 最大迭代轮数 |
| `seed_top_k` | 6 | 种子候选保留数 |
| `hard_negative_top_k` | 6 | 难负例保留数 |
| `reference_top_k` | 6 | genericity top-k |
| `reference_rank_weights` | [1.0, 0.8, 0.6, 0.4, 0.25, 0.1] | 秩加权权重 |
| `genericity_gate_low` | **0.78** | 低/中分界阈值 |
| `genericity_gate_high` | **0.90** | 中/高分界阈值 |
| `genericity_gate_low_scale` | **0.10** | 低分段惩罚系数 |
| `genericity_gate_mid_scale` | **0.45** | 中分段惩罚系数 |
| `eval.mode` | pretext_small | 用 GPT-2 小模型评测 |

---

## 四、实验组设计

### 4.1 实验组一览

| 组别 | 组文件 | 相比基础配置的改动 | 设计意图 |
|------|--------|------------------|----------|
| **g1**（默认） | `_g1_conditional_genericity_default.yaml` | 无覆盖，使用基础参数 | 三段式门控的基准行为；验证默认参数（0.78/0.90/0.10/0.45）是否已足够改善 forums/microblog |
| **g2**（更软中段） | `_g2_conditional_genericity_soft_mid.yaml` | `genericity_gate_mid_scale: 0.30` | 对中分段候选句施加更轻的惩罚（0.30 vs 0.45），进一步保护口语化文本中得分在 0.78–0.90 区间的候选句 |
| **g3**（更早高段） | `_g3_conditional_genericity_early_high.yaml` | `gate_low: 0.75`, `gate_high: 0.86` | 将高分段触发点前移，对得分 >0.86 的候选句施加完整惩罚（比 g1 的 0.90 更严格），同时低分保护区间扩展到 0.75 以下 |
| **g4**（协同 A2） | `_g4_conditional_genericity_plus_a2.yaml` | `length_lambda: 0.10` | 在 g1 门控基础上叠加 Round 1 发现的最优 length 参数（A2 组），测试两者是否有正向协同效应 |

### 4.2 各组参数对比

| 参数 | g1（默认） | g2（软中段） | g3（早高段） | g4（协同） |
|------|-----------|-------------|-------------|-----------|
| `gate_low` | 0.78 | 0.78 | **0.75** | 0.78 |
| `gate_high` | 0.90 | 0.90 | **0.86** | 0.90 |
| `gate_low_scale` | 0.10 | 0.10 | 0.10 | 0.10 |
| `gate_mid_scale` | 0.45 | **0.30** | 0.45 | 0.45 |
| `length_lambda` | — | — | — | **0.10** |

### 4.3 四个数据集

| 数据集 | 简称 | 文本类型 | 主要挑战 |
|--------|------|---------|---------|
| `jobs` | JOBS | 招聘JD，格式化 | 高泛化候选句需强力压制 |
| `congressional` | CONG | 国会听证，正式 | 相似，需抑制模板化措辞 |
| `forums` | FORUMS | 论坛讨论，口语 | 中间段候选句易被过度惩罚 |
| `microblog` | MICRO | 微博/推文，口语 | 同 forums，过度惩罚问题更突出 |

### 4.4 全部 16 个实验

| 实验 ID | 实验标签 | 数据集 | 组别 | 配置路径 |
|---------|---------|--------|------|---------|
| `ns_tune4_g1_jobs` | NS-T4-G1-JOBS | jobs | g1 | `single_node_tuning_round4/ns_tune4_g1_jobs.yaml` |
| `ns_tune4_g1_congressional` | NS-T4-G1-CONG | congressional | g1 | `single_node_tuning_round4/ns_tune4_g1_congressional.yaml` |
| `ns_tune4_g1_forums` | NS-T4-G1-FORUMS | forums | g1 | `single_node_tuning_round4/ns_tune4_g1_forums.yaml` |
| `ns_tune4_g1_microblog` | NS-T4-G1-MICRO | microblog | g1 | `single_node_tuning_round4/ns_tune4_g1_microblog.yaml` |
| `ns_tune4_g2_jobs` | NS-T4-G2-JOBS | jobs | g2 | `single_node_tuning_round4/ns_tune4_g2_jobs.yaml` |
| `ns_tune4_g2_congressional` | NS-T4-G2-CONG | congressional | g2 | `single_node_tuning_round4/ns_tune4_g2_congressional.yaml` |
| `ns_tune4_g2_forums` | NS-T4-G2-FORUMS | forums | g2 | `single_node_tuning_round4/ns_tune4_g2_forums.yaml` |
| `ns_tune4_g2_microblog` | NS-T4-G2-MICRO | microblog | g2 | `single_node_tuning_round4/ns_tune4_g2_microblog.yaml` |
| `ns_tune4_g3_jobs` | NS-T4-G3-JOBS | jobs | g3 | `single_node_tuning_round4/ns_tune4_g3_jobs.yaml` |
| `ns_tune4_g3_congressional` | NS-T4-G3-CONG | congressional | g3 | `single_node_tuning_round4/ns_tune4_g3_congressional.yaml` |
| `ns_tune4_g3_forums` | NS-T4-G3-FORUMS | forums | g3 | `single_node_tuning_round4/ns_tune4_g3_forums.yaml` |
| `ns_tune4_g3_microblog` | NS-T4-G3-MICRO | microblog | g3 | `single_node_tuning_round4/ns_tune4_g3_microblog.yaml` |
| `ns_tune4_g4_jobs` | NS-T4-G4-JOBS | jobs | g4 | `single_node_tuning_round4/ns_tune4_g4_jobs.yaml` |
| `ns_tune4_g4_congressional` | NS-T4-G4-CONG | congressional | g4 | `single_node_tuning_round4/ns_tune4_g4_congressional.yaml` |
| `ns_tune4_g4_forums` | NS-T4-G4-FORUMS | forums | g4 | `single_node_tuning_round4/ns_tune4_g4_forums.yaml` |
| `ns_tune4_g4_microblog` | NS-T4-G4-MICRO | microblog | g4 | `single_node_tuning_round4/ns_tune4_g4_microblog.yaml` |

---

## 五、预期结果与分析框架

### 5.1 期望的改进方向

**对 forums / microblog：**
- g1/g2 相比 Round 3 应显著改善：低中段口语候选句不再被过度压制，筛选器能选出更具领域特征的句子
- g2（mid_scale=0.30）对 forums/microblog 的改善应 ≥ g1，因为口语文本 genericity 分布集中在中分段

**对 jobs / congressional：**
- g1 应与 Round 3 持平或略有波动，高分段惩罚强度未变
- g3（gate_high=0.86）对 jobs/congressional 可能更好：更早触发完整惩罚，对正式文本中的模板措辞打压更及时

### 5.2 关键分析指标

主指标：`best_top1`（一轮最优 top-1 ROUGE/BERTScore）  
辅助指标：`best_top3`, `best_top5`, `best_top10`

### 5.3 组间比较矩阵

| 比较 | 意义 |
|------|------|
| g1 vs Round 3 | 三段式门控本身对各数据集的基础改善 |
| g2 vs g1 | 更软中段对口语数据集的额外收益 |
| g3 vs g1 | 更早高段惩罚对正式文本的额外收益 |
| g4 vs g1 | length_lambda 协同效应是否存在 |
| g2 vs g3（forums/micro） | 软中段 vs 早高段对口语数据集的最优选择 |

### 5.4 成功标准

- **forums/microblog**：≥1 个组别超越 PrE-Text baseline（`best_top1` 绝对值提升 > 0）
- **jobs/congressional**：所有组别不低于 Round 3 最优值（确保无回退）
- **跨组一致性**：最优组的优势在 4 个数据集上方向一致（无明显跨数据集反转）

---

## 六、实验执行

### 6.1 环境配置

| 项目 | 值 |
|------|----|
| 服务器 | node03（k8smaster） |
| GPU | NVIDIA RTX A6000（index=1，49GB VRAM） |
| CUDA_DEVICE_ORDER | PCI_BUS_ID |
| CUDA_VISIBLE_DEVICES | 1 |
| conda 环境 | pretext |
| 代码分支 | paper-2-genereic |
| 工作目录 | `/mnt/public/caiqiyue_file/code_from_paper/paper-new` |
| 执行方式 | 串行队列，16 个实验逐一执行 |

### 6.2 执行命令模板

```bash
# 激活环境
source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext

# 环境变量
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export VLLM_HOST_IP=127.0.0.1
export HOST_IP=127.0.0.1

# 运行单个实验（以 g1_jobs 为例）
cd /mnt/public/caiqiyue_file/code_from_paper/paper-new
python -m paper_new_selector.run_selector_single_node \
    --config configs/experiments/single_node_tuning_round4/ns_tune4_g1_jobs.yaml
```

### 6.3 队列自动化

队列管理文件：`old_automation/old_experiment_queue.py`  
队列状态文件：`old_automation/old_experiment_queue_state.json`  
串行执行脚本：`old_automation/run_round4_queue.py`（Python 实现，避免 bash 数组兼容性问题）

---

## 七、代码实现清单

| 文件 | 修改内容 |
|------|---------|
| `paper_new_selector/genericity.py` | 新增 `apply_genericity_gate`（第 40–52 行）；`compute_genericity_penalty` 增加 `apply_gate/gate_*` 参数（第 55–93 行）；`compute_genericity_penalties` 透传参数（第 96–121 行） |
| `paper_new_selector/stage1_runner.py` | 调用 `compute_genericity_penalties` 时传入 `apply_gate=True` 及四个门控参数（第 168–178 行） |
| `configs/experiments/single_node_formal/_base_selector_formal.yaml` | 新增四个门控默认值（`gate_low/high/low_scale/mid_scale`） |
| `configs/single_node_jobs_selector.yaml` | 同上 |
| `configs/experiments/single_node_tuning_round4/` | 新增 21 个配置文件（1 base + 4 group + 16 leaf） |
| `tests/test_support.py` | 新增 `test_genericity_gate_uses_low_mid_high_scales`（第 51 行）和 `test_genericity_penalty_applies_gate_to_raw_score`（第 83 行） |
| `tests/test_stage1_runner.py` | 新增 `test_stage1_runner_passes_genericity_gate_config_to_genericity`（第 322 行） |
| `old_automation/old_experiment_queue.py` | 将 Round 3 队列（ROUND3_GROUPS/f1-f4）替换为 Round 4 队列（ROUND4_GROUPS/g1-g4） |
| `old_automation/old_experiment_queue_state.json` | 重置为 16 个 Round 4 实验，状态全部 pending |
