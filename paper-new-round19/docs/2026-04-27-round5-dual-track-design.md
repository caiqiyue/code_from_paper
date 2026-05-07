# Round 5 双轨并行设计：gate 网格扩展 + 长度自适应惩罚

**日期**：2026-04-27
**状态**：设计稿（待用户复审 → 进入 writing-plans）
**前置文档**：
- `paper-new/docs/2026-04-26-round4-algorithm-and-experiment-design.md`（含第十节 forums 多指标硬约束分析）
- `paper-new/docs/2026-04-24-pretext-screening-results.md`（PrE-Text 基线）
- `paper-new/docs/2026-04-26-round3-genericity-reference-smoothing-results.md`（Round 3 加权平滑）

---

## 0. 目标与约束

### 0.1 终极目标

让创新算法在 **jobs / congressional / forums / microblog** 四个数据集上的下游评估 best_top1 **同时超过 PrE-Text 基线**。优先寻找单一全局参数实现这一目标；若无法实现，回退到 per-dataset 超参组合。

### 0.2 当前状态（Round 4 收尾）

| 数据集 | PrE-Text | Round 4 最佳 | 来自配置 | Δ |
|---|---|---|---|---|
| jobs | 0.2732 | 0.2784 | g2 | **+0.0052** ✅ |
| congressional | 0.2950 | 0.2979 | g3 | **+0.0029** ✅ |
| forums | 0.2501 | 0.2500 | g1 | **−0.0001** ❌（顽疾）|
| microblog | 0.2763 | 0.2790 | g3 | **+0.0027** ✅ |

### 0.3 forums 多指标硬约束（已写入 Round 4 文档第十节）

forums 在 R4/g1 上 4 个指标对 PrE-Text 的差距：

| 指标 | Δ vs PrE-Text |
|---|---|
| best_top1 | −0.0001（噪声内）|
| best_top3 | **−0.0021（落后）** |
| best_top5 | **−0.0060（明显落后）** |
| best_top10 | +0.0001（噪声内）|

**结论**：forums 是真·全指标落后，不是 top1 单点噪声。**genericity-gate 这一维已榨干，必须引入新算法维度。**

### 0.4 g 配置间的内在冲突

- jobs 想要 g2（mid_scale=0.30，软中段）
- congressional/microblog 想要 g3（gate_low=0.75, gate_high=0.86，高段提前）
- forums 想要 g1（默认 0.78/0.90/0.45），且 **g3 让 forums 退步**

→ 在 gate 单一维度上，"一套全局参数同赢"在物理上不成立。

---

## 1. 整体架构

### 1.1 双轨并行策略

| 方向 | 工作目录 | 改动类型 | 假设要验证 |
|---|---|---|---|
| **方向 1**（gate 网格扩展）| `paper-new/`（现有）| 仅 config 扩展，**算法代码不动** | 在 g1-g4 现有点位之间是否存在"折中带"，能同时拿下 forums 和 congressional/microblog |
| **方向 2a**（长度自适应惩罚）| `paper-new-round5/`（从 paper-new copy 新建）| 改 `genericity.py` 和 `stage1_runner.py`，新增长度因子 | 长度作为 mid-band 候选词的二级判别器是否有效，能否把 forums 拉过 PrE-Text |

### 1.2 执行顺序

- 两条路在同一张 A6000 上**串行**跑（`CUDA_VISIBLE_DEVICES=1`）
- 顺序：先方向 1（12 个正式实验，~50 分钟）→ 再方向 2a 的 sanity check（1 个 forums 实验，~5 分钟）→ 方向 2a 正式实验（16 个，~80 分钟）
- 总实验数：**29 次运行 = 12（方向 1）+ 1（sanity）+ 16（方向 2a 正式）**，总时长 ~135 分钟

### 1.3 一致性约束

- 两条路的非算法控制参数**全部继承自 Round 4 base**：
  - train_limit=256, eval_limit=256, initialization_limit=1024
  - bootstrap.num_prompts=100, max_tokens=85
  - eval small_epochs=6, gpt2 small
- 种子和数据集划分与 Round 4 保持一致，确保结果可直接横向对比
- **方向 2a 的 α=0 必须严格等价 Round 4 g1**（写入回归测试）

---

## 2. 方向 1：gate 网格扩展（paper-new/）

### 2.1 目录与文件

```
paper-new/configs/experiments/single_node_tuning_round4_ext/
├── _base_selector_tuning_round4_ext.yaml      ← inherits round4 base
├── _g5_compromise_low_high_mid.yaml
├── _g6_low_only_early.yaml
├── _g7_mid_softer_lite.yaml
├── ns_tune4_g5_{jobs,congressional,forums,microblog}.yaml   (×4)
├── ns_tune4_g6_{jobs,congressional,forums,microblog}.yaml   (×4)
└── ns_tune4_g7_{jobs,congressional,forums,microblog}.yaml   (×4)
```

合计：1 base + 3 group + 12 leaf = 16 个新 yaml 文件。

### 2.2 三组配置参数

| 配置 | gate_low | gate_high | low_scale | mid_scale | 假设 |
|---|---|---|---|---|---|
| **g5（折中带）**| 0.76 | 0.88 | 0.10 | 0.35 | g3 的 early-low + g2 的 soft-mid 是否能同时见效 |
| **g6（仅 low 提前）**| 0.75 | **0.90** | 0.10 | 0.45 | g3 的提升只来自 gate_low；gate_high 提前才是 forums 退步元凶 |
| **g7（mid 略软）**| 0.78 | 0.90 | 0.10 | **0.40** | g2 的 0.30 太软伤 forums，0.40 是 g1 与 g2 的安全中间值 |

### 2.3 自动化执行

- 新建 `old_automation/run_round4_ext_queue.py`，复用 Round 4 串行 Python runner 模式
- 12 个实验顺序：(g5, g6, g7) × (jobs, congressional, forums, microblog)
- 在 tmux session `round4_ext` 内执行

### 2.4 成功判据

- **强成功**：g5/g6/g7 中存在某一组，4 个数据集 best_top1 **全部超过** PrE-Text（jobs ≥ 0.2732, congressional ≥ 0.2950, forums ≥ 0.2501, microblog ≥ 0.2763）
- **弱成功**：没有"全胜"组，但 forums 的最佳值（不限组）≥ 0.2501，且 jobs/congressional/microblog 上至少各有一组维持原 Round 4 的胜利
- **失败**：forums 仍 < 0.2501 → 必须依赖方向 2a 的结果

### 2.5 文档输出

实验结果追加到 `paper-new/docs/2026-04-26-round4-algorithm-and-experiment-design.md` 的"第十一节：Round 4 扩展（gate 网格 g5-g7）"。

---

## 3. 方向 2a：长度自适应惩罚（paper-new-round5/）

### 3.1 数学定义

**长度度量定义**：candidate_texts 在 stage1_runner 中是 `list[str]`（每个元素是完整文档字符串，而非 token 列表）。其长度定义为 **word 数**：
```
L_c = len(c.split())
```
这与项目已有的 `private_lengths` 度量保持一致（参见 `paper-new/paper_new_selector/stage1_runner.py:138`）。**不要使用 `len(c)`**（那是字符数）；也不要引入额外 tokenizer。

`L_ref` 为批内全部候选词长度的**中位数**。

**Round 4 输出**：`gated_penalty(c) = raw_score(c) × gate_scale(raw_score(c))`

**Round 5 在末尾叠加长度因子**：

```
length_factor(c) = clip( (L_ref / max(L_c, 1)) ^ alpha, factor_min, factor_max )
final_penalty(c) = gated_penalty(c) × length_factor(c)
```

- α > 0 → 长候选词受**更少**惩罚（"长 = 受保护"）
- α < 0 → 短候选词受**更少**惩罚（"短 = 受保护"）
- α = 0 → 退化为 Round 4 g1，length_factor 恒为 1

钳制范围（默认值）：`factor_min=0.2, factor_max=5.0`。但**该默认值需要在实施阶段由离线分布检查确认**，见 §3.4.1。

### 3.2 代码改动

#### 3.2.1 新建分支

`paper-new-round5/` 从 `paper-new/` 整体 copy 而来（`cp -r paper-new paper-new-round5`），保留所有原代码、tests、configs，再在其上做以下改动。

#### 3.2.2 `paper_new_selector/genericity.py`

**新增函数**：
```python
def compute_length_factors(
    lengths: List[int],
    alpha: float,
    l_ref_strategy: str = "batch_median",
    factor_min: float = 0.2,
    factor_max: float = 5.0,
) -> List[float]:
    """根据候选词长度计算调制因子。alpha=0 时全部返回 1.0。"""
```

**修改 `compute_genericity_penalties` 签名**（注意：现有代码使用 `*` 强制 keyword-only 参数，**必须保持这个风格**；参数名沿用 `candidate_vectors` / `reference_top_k` / `reference_rank_weights`，不是 `candidates` / `rank_weights`）：

```python
def compute_genericity_penalties(
    *,
    candidate_vectors: list[list[float]],
    reference_vectors: list[list[float]],
    reference_top_k: int,
    reference_rank_weights: list[float] | None = None,
    apply_gate: bool = False,         # 与现有默认值一致
    gate_low: float = 0.0,
    gate_high: float = 1.0,
    low_scale: float = 1.0,
    mid_scale: float = 1.0,
    # ↓↓↓ Round 5 新增 ↓↓↓
    candidate_lengths: list[int] | None = None,
    length_modulation_enabled: bool = False,
    length_alpha: float = 0.0,
    length_factor_min: float = 0.2,
    length_factor_max: float = 5.0,
) -> list[float]:
    ...
```

**同时必须修改单条版本 `compute_genericity_penalty`**（因为 `compute_genericity_penalties` 是它的 list comprehension 包装），让单条函数也接受单候选的 `candidate_length` 参数；外层批处理函数计算 `L_ref` 后逐个透传。

**调用结构**：
```python
def compute_genericity_penalties(*, candidate_vectors, ..., candidate_lengths=None, length_modulation_enabled=False, ...):
    l_ref = _median(candidate_lengths) if (length_modulation_enabled and candidate_lengths) else None
    return [
        compute_genericity_penalty(
            candidate_vector=v,
            ...,
            candidate_length=l,
            l_ref=l_ref,
            length_modulation_enabled=length_modulation_enabled,
            length_alpha=length_alpha,
            length_factor_min=length_factor_min,
            length_factor_max=length_factor_max,
        )
        for v, l in zip(candidate_vectors, candidate_lengths or [None] * len(candidate_vectors))
    ]
```

#### 3.2.3 `paper_new_selector/stage1_runner.py`

- 在调用 `compute_genericity_penalties` 之前，先用 `[len(text.split()) for text in candidate_texts]` 计算 `candidate_lengths`（与 `private_lengths` 在第 138 行的算法完全一致）
- 从 `selector_cfg` 读出 `length_modulation_enabled / length_alpha / length_factor_min / length_factor_max`，全部带默认值（`.get(..., 默认值)`）以保证旧 config 不传也能跑
- 通过 keyword arg 透传给 `compute_genericity_penalties`

#### 3.2.4 base config

`paper-new-round5/configs/experiments/single_node_tuning_round5/_base_selector_tuning_round5.yaml`：
```yaml
inherits:
  - ../single_node_tuning_round4/_base_selector_tuning_round4.yaml

selector:
  length_modulation_enabled: false   # 默认 disabled
  length_alpha: 0.0
  length_factor_min: 0.2
  length_factor_max: 5.0
```

### 3.3 r1/r2/r3/r4 配置

| 组 | α | 含义 | 文件名 |
|---|---|---|---|
| **r1** | +0.3 | 长候选受较少惩罚（适度）| `_r1_protect_long_moderate.yaml` |
| **r2** | −0.3 | 短候选受较少惩罚（适度）| `_r2_protect_short_moderate.yaml` |
| **r3** | +0.6 | 长候选受较少惩罚（强）| `_r3_protect_long_strong.yaml` |
| **r4** | −0.6 | 短候选受较少惩罚（强）| `_r4_protect_short_strong.yaml` |

每组 config 仅覆盖：
```yaml
selector:
  length_modulation_enabled: true
  length_alpha: <±0.3 or ±0.6>
```

全部叠加在 **g1 默认 gate** 之上（gate_low=0.78, gate_high=0.90, low_scale=0.10, mid_scale=0.45）——其余继承自 Round 4 base。

叶子 config：`ns_tune5_r{1,2,3,4}_<dataset>.yaml`，共 16 个。

### 3.4 边界与异常处理

| 场景 | 处理 |
|---|---|
| `L_c = 0`（不应发生）| `max(L_c, 1)` 防止除零 |
| 批内只有 1 个候选 | 中位数 = 唯一值 → factor = 1，自然退化 |
| 极短候选 + α=+0.6 | factor 被 `factor_max=5.0` 钳住 |
| 极长候选 + α=+0.6 | factor 被 `factor_min=0.2` 钳住 |
| `length_modulation_enabled=false` | 跳过 factor 计算，零开销，**严格等价 Round 4 g1** |

#### 3.4.1 钳制范围的离线分布检查（实施前必做）

由于不同数据集候选词的 word 数分布差异很大（forums 短帖 ~10-30 words，congressional ~40-200 words），`(L_ref / L_c) ^ 0.6` 在极端样本上很容易撞到 `factor_min=0.2` 或 `factor_max=5.0` 的边界，导致 α 被"饱和"，r3/r4 实际退化为 r1/r2 的稍强版本。

**实施时必须执行的离线检查**：

1. 在 `paper-new-round5/` 改完代码后，在 r1-r4 实验跑之前：
   - 选取 **forums** 数据集（候选词长度方差最大）
   - 跑一次 stage1 candidate generation（不需要进 stage2 / eval）
   - dump 一批的 `candidate_lengths`，离线计算 `L_ref / L_c` 的 5%/95% 分位
2. 检查在 α=+0.6 和 α=−0.6 下，`factor` 落在 `[factor_min, factor_max]` 之内的比例
3. **可接受阈值**：α=±0.6 时，**至少 80% 的候选词 factor 不触边界**。否则需要：
   - 把 `factor_min/max` 调宽（例如 0.1 / 10.0），或
   - 缩小 r3/r4 的 α 绝对值（例如改为 ±0.5）
4. 检查通过后，将"实际使用的 factor_min/max"写入 base config 并固定

**这一步的 log 与决策记录写入** `paper-new-round5/docs/2026-04-27-round5-length-adaptive-design.md` 的"附录 A：钳制范围离线检查"。

### 3.5 测试设计（必须全部 pass 才能跑实验）

| # | 测试名 | 验证内容 |
|---|---|---|
| 1 | `test_length_factor_neutral_when_alpha_zero` | α=0 时所有 factor=1.0 |
| 2 | `test_length_factor_protects_longer_when_alpha_positive` | α>0 时长候选 factor < 短候选 factor |
| 3 | `test_length_factor_protects_shorter_when_alpha_negative` | α<0 时方向相反 |
| 4 | `test_length_factor_clipped_to_min_max` | 极端长度被钳到 [factor_min, factor_max] |
| 5 | `test_genericity_with_length_disabled_matches_round4` | **关键回归**：disabled 时输出与 Round 4 完全一致 |
| 6 | `test_stage1_runner_passes_length_config_to_genericity` | config → runner → genericity 链路通畅 |

测试 1-5 写入 `tests/test_support.py`（或新建 `tests/test_length_modulation.py`），测试 6 写入 `tests/test_stage1_runner.py`。

### 3.6 sanity check 流程

正式跑 r1-r4 之前，必须做 **α=0 复现实验**：

1. 新建 `ns_tune5_r0_forums.yaml`：`length_modulation_enabled=true, length_alpha=0.0`
2. 在 `paper-new-round5/` 上跑该单组 forums 实验
3. **多层一致性比对（必须全部通过）**：
   - **(a) Stage 1 中间产物文本一致性**：r0 和 Round 4 g1 在 forums 上产生的 `selected_texts` 文本（stage1 输出）**逐字符一致**——这是最强的判据，因为只要 candidate scoring 有任何数值偏差，selected_texts 都会不同
   - **(b) hard_negative_texts 一致性**：同样要求逐字符一致
   - **(c) 下游评估 best_top1**：差异 < 0.0001 视为通过
   - 三项必须**全部通过**才算 sanity 通过；其中 (a) 和 (b) 是数值正确性的精确判据，(c) 是端到端的 sanity
4. 若任一项不通过，停下来排查代码副作用，修复后重跑
5. 通过后再跑 r1-r4 的 16 个正式实验

sanity check 的文本一致性比对在 stage1 完成后即可做，秒级；不需要等整个 eval 跑完。整体 sanity check 不计入 16 个正式实验，单独走（约 5-10 分钟）。

**比对实施提示**：Round 4 g1 forums 的 stage1 artifacts 在 `paper-new/outputs/ns_tune4_g1_forums/` 下，需要确保仍然存在；若已被清理，重跑一次 Round 4 g1 forums 作为对照基准（约 5 分钟）。

### 3.7 自动化执行

- 新建 `old_automation/run_round5_queue.py`（基于 Round 4 runner，CWD 改为 `paper-new-round5`）
- 16 个实验顺序：(r1, r2, r3, r4) × (jobs, congressional, forums, microblog)
- 在 tmux session `round5` 内执行

### 3.8 成功判据

- **强成功**：r1-r4 中存在某一组，4 个数据集 best_top1 **全部超过** PrE-Text
- **方向性成功**：r1/r3（α>0）显著优于 r2/r4（α<0）或反之，明确"长 vs 短保护"哪边对路 → 决定 Round 6 重点
- **失败但有用**：r1-r4 全部都不超过 g1 baseline → 长度调制无效，作为 ablation 写入论文

### 3.9 文档输出

- `paper-new-round5/docs/2026-04-27-round5-length-adaptive-design.md` ← 本设计的 paper-new-round5 内副本
- `paper-new-round5/docs/2026-04-27-round5-length-adaptive-results.md` ← 实验结果与分析

---

## 4. 集成成功判据（Round 5 整体）

按优先级从高到低：

1. **理想结局**：方向 1 的 g5/g6/g7 之一或方向 2a 的 r1-r4 之一，**单一配置**在 4 个数据集 best_top1 全部超过 PrE-Text → 论文主打这个全局配置
2. **次优结局**：单一配置不存在，但合并所有 Round 4 + Round 5 配置后，存在 per-dataset 选择使每个数据集都超过 PrE-Text（特别是 forums）→ 论文用 per-dataset best 表格
3. **算法 finding**：方向 2a 揭示长度调制方向（长保护 vs 短保护），即使没赢 PrE-Text，也是有故事的 ablation
4. **底线**：jobs/congressional/microblog 在 Round 5 任一新配置上不退步至 < PrE-Text 基线

---

## 5. 风险与回滚

| 风险 | 概率 | 缓解 |
|---|---|---|
| 方向 2a 代码改动引入副作用，破坏 Round 4 已有结果 | 低 | sanity check（α=0）+ 测试 5（disabled 时输出严格一致）双重防护；`paper-new-round5/` 与 `paper-new/` 物理隔离 |
| 长度调制在所有 r 组都无效 | 中 | 这本身是论文 ablation 的有效结论；不影响 Round 4 已有胜利 |
| 方向 1 的 g5/g6/g7 都无法改善 forums | 中（forums 顽疾）| 方向 2a 的 16 实验是冗余保险 |
| GPU 占用冲突 | 低 | 单卡（`CUDA_VISIBLE_DEVICES=1`）串行执行 |
| 28 个实验中途失败 | 低 | runner 记录每个实验 stdout/stderr 到独立 log，失败不中断队列 |

---

## 6. 实施步骤大纲（详细 task 在 writing-plans 阶段产出）

1. **方向 1 准备**（在 `paper-new/` 内）
   - 新建 `configs/experiments/single_node_tuning_round4_ext/` 及 16 个 yaml
   - 新建 `old_automation/run_round4_ext_queue.py`
2. **方向 1 执行**：跑 12 个 g5/g6/g7 实验
3. **方向 1 结果整理**：写入 Round 4 文档第十一节
4. **方向 2a 准备**：
   - `cp -r paper-new paper-new-round5`
   - 改 `genericity.py` / `stage1_runner.py`
   - 写 6 个新测试
   - 新建 base + 4 个 r 组 + 16 个叶子 config
   - 新建 `run_round5_queue.py`
5. **方向 2a sanity check**：α=0 forums 单组实验，对比 Round 4 g1
6. **方向 2a 执行**：跑 r1-r4 共 16 个实验
7. **方向 2a 结果整理**：写入 `paper-new-round5/docs/`
8. **集成分析**：合并 Round 4 + Round 5 全部结果，确定最终交付配置（单一全局 vs per-dataset）

---

## 7. 待 writing-plans 阶段澄清的子问题

- 方向 1 的 12 个叶子 yaml 是否完全照搬 Round 4 的 yaml 模板（仅改 `inherits` 路径与组名）
- run_round4_ext_queue.py / run_round5_queue.py 是否完全沿用 run_round4_queue.py 的结构
- §3.4.1 离线分布检查的具体实施方式（哪个脚本、输出格式）

这些不影响整体设计，写实施计划时由 writing-plans 处理即可。

---

## 8. 修订记录

- **2026-04-27 初版** — brainstorming 后写入
- **2026-04-27 v2** — 通过 feasibility-reviewer 子智能体审核后修订：
  - 修正 §3.1：候选词长度定义为 `len(c.split())`（word 数），与项目已有的 `private_lengths` 对齐；明确禁止使用 `len(c)`（字符数）
  - 修正 §3.2.2：`compute_genericity_penalties` 函数签名遵循 keyword-only（`*`）风格，参数名沿用 `candidate_vectors` / `reference_top_k` / `reference_rank_weights`，`apply_gate=False` 默认值；明确同时改单条版本 `compute_genericity_penalty`
  - 修正 §3.2.3：明确使用 `[len(text.split()) for text in candidate_texts]`，与第 138 行 `private_lengths` 一致
  - 新增 §3.4.1：r1-r4 实验前必须执行钳制范围离线分布检查，避免 α 被饱和
  - 加强 §3.6：sanity check 增加 `selected_texts` / `hard_negative_texts` 文本逐字符一致性比对，不再只看 best_top1
