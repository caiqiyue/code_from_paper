# E1 新增基线设计 · C4-only / Aug-PE / DP-Prompt

**日期**：2026-05-30  
**背景**：导师要求在 E1 主对比实验中补充更多基线，扩充"不微调大模型的合成样本生成"比较维度。

---

## 1. 目标

将 E1 对比表从 5 个方法扩展为 8 个方法，新增 3 个"不微调 LLM"基线：

| 类别 | 方法 | 角色 |
|------|------|------|
| 原有 | PrE-Text, WASP, DPGA, round19, round23 | 不变 |
| 新增 | **C4-only** | 无算法下限，ε=∞（无隐私保护），说明算法本身价值 |
| 新增 | **Aug-PE** | PE 家族最优化版（ICML 2024），同类最强竞争对手 ⚠️集中式假设 |
| 新增 | **DP-Prompt** | 完全不同范式（本地改写私有文档），跨范式厚度 |

> ⚠️ **Aug-PE 集中式说明**：Aug-PE 假设服务器可直接访问全部私有数据（集中式 DP），其他方法均为联邦场景。这一差异须在论文写作时明确说明，不影响合成样本质量的对比（所有方法最终均以 best_top1 评估）。

---

## 2. 实验设置

**完全对齐 E1**，不做任何更改：

- 数据集：jobs / congressional / forums / microblog（seen 4 个）
- Seeds：与 E1 一致（repeat10 → repeat30）
- 评估指标：best_top1（主），best_top3/5/10（附录）
- 评估流水线：与现有实验保持一致的输入输出格式

---

## 3. 技术架构

**三个算法各自独立构建**，参考 pretext / round23 的实验代码组织方式（配置管理、manifest 格式、sidecar JSON、runner 接口），但不直接调用它们的内部模块：

```
数据加载 → [内部合成算法（独立实现）] → 合成样本 → 下游任务评估 → best_top1
                     ↑
         参考 pretext/round23 的构建模式，各自独立实现
```

**复用范围（模式/格式层面）**：
- Runner 框架（`round23_dynamic_experiment_runner.py`）：注册新 method，调度逻辑不变
- Manifest TSV 格式、Config YAML 命名规范、Sidecar JSON 格式：完全对齐
- 运行脚本（`.sh`）：格式仿照 `run_e7_all_modes.sh`

**不复用**：三个算法各自的数据加载、合成生成、评估调用均独立实现，不 import pretext_platform 或 dp-prompt 内部模块。

---

## 4. 各方法技术设计

### 4.1 C4-only

**核心思想**：随机选种子，无 DP 算法，ε=∞（无隐私保护）

**算法**：
1. 从本地预处理文件（`pretext_initialization_c4_en/formatted/initialization.json`）随机采样 nsyn 条文本（`random.choices(init_pool, k=nsyn)`，nsyn 按现有框架约定计算）
2. 用 LLaMA few-shot 扩充（独立实现，参考 PrE-Text Stage2 逻辑）
3. 输出合成样本集，评估 best_top1

**实现参考**：PrE-Text 代码中的 `c4-only` 模式，去掉 Stage1 的 DP histogram 循环，其余逻辑自行实现。

---

### 4.2 Aug-PE

**论文**：Differentially Private Synthetic Data via Foundation Model APIs 2: Text（ICML 2024 Spotlight，arxiv 2403.01749）

**核心思想**：PE 框架的集中式优化版，黑盒 LLM API，不微调模型

**算法**：
1. 从 GitHub 获取 Aug-PE 代码（AI-secure/aug-pe，需验证 repo 名）
2. 用 Aug-PE 的 Private Evolution 核心逻辑选出高质量 seeds（集中式 DP histogram，改进版 PE）
3. 用 LLaMA few-shot 扩充（独立实现）
4. 输出合成样本集，评估 best_top1

**环境策略**：以服务器上的 `pretext` conda 环境为主。若 Aug-PE 存在依赖冲突（如 `sentence-transformers` 版本不兼容），**修改 Aug-PE 源码**以兼容 pretext 环境中的等效 API，不使用 subprocess 隔离。核心算法逻辑保持不变，只替换冲突的依赖调用。

---

### 4.3 DP-Prompt

**论文**：Locally Differentially Private Document Generation Using Zero Shot Prompting（EMNLP 2023 Findings）

**核心思想**：对每条私有文档做零样本 LLM 改写，不微调模型

**算法**：
1. 对每条私有文档构造 prompt（完整模板：`"Review: {text}\nParaphrase of the review:"`，源自 `dp-prompt/dp_prompt/prompting/templates.py`）
2. 用预训练 LLM 采样生成改写文本（温度控制，可选 logit clipping）
3. 输出合成改写样本集，评估 best_top1

**隐私机制说明**：文档级改写通过温度采样增加随机性，是一种经验性隐私保护手段；代码中真正形式化的 LDP 机制（Laplace、Mahalanobis）仅用于 word-level 模块。论文写作时应准确描述为"温度采样的文档级随机改写"，避免直接声称满足形式化 LDP 定义。

**与其他方法的关键区别**：DP-Prompt 直接处理私有文档，其他方法通过公开数据间接生成合成样本。这一差异在论文对比表中须明确注明，不影响统一用 best_top1 进行评估。

**现状**：本机已有修改过的代码（`/Users/apple/Desktop/code_from_paper/dp-prompt`），需继续完善并以独立 pipeline 方式接入现有实验框架。

---

## 5. 各方法开发工作量

| 方法 | 工作量 | 关键工作 |
|------|--------|---------|
| C4-only | 低 | 参考 PrE-Text 模式，独立实现随机采样 + 扩充 + 评估 pipeline |
| Aug-PE | 中 | clone 代码 → 必要时修改源码兼容 pretext 环境 → 独立实现整体 pipeline |
| DP-Prompt | 中 | 基于现有 dp-prompt 代码完善 → 独立实现评估对齐 → 接入 runner 框架 |

---

## 6. 参考资料

- Aug-PE 论文：arxiv.org/abs/2403.01749
- DP-Prompt 论文：EMNLP 2023 Findings（本机代码：`dp-prompt/`）
- PrE-Text 代码参考：`/Users/apple/Desktop/code_from_paper/PrE-Text/`
- Round23 框架参考：`paper-new-round23/scripts/`
