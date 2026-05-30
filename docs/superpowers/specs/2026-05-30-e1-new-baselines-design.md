# E1 新增基线设计 · C4-only / Aug-PE / DP-Prompt

**日期**：2026-05-30  
**背景**：导师要求在 E1 主对比实验中补充更多基线，扩充"不微调大模型的合成样本生成"比较维度。

---

## 1. 目标

将 E1 对比表从 5 个方法扩展为 8 个方法，新增 3 个"不微调 LLM"基线：

| 类别 | 方法 | 角色 |
|------|------|------|
| 原有 | PrE-Text, WASP, DPGA, round19, round23 | 不变 |
| 新增 | **C4-only** | 无算法下限，ε=0，说明算法本身价值 |
| 新增 | **Aug-PE** | PE 家族最优化版（ICML 2024），最强同类竞争对手 |
| 新增 | **DP-Prompt** | 完全不同范式（本地改写），跨范式厚度 |

---

## 2. 实验设置

**完全对齐 E1**，不做任何更改：

- 数据集：jobs / congressional / forums / microblog（seen 4 个）
- Seeds：与 E1 一致（repeat10 → repeat30）
- 评估指标：best_top1（主），best_top3/5/10（附录）
- 评估流水线：与现有 pretext/round23 完全相同

---

## 3. 技术架构

所有新方法复用现有外层框架，只替换内部合成算法：

```
数据加载 → [内部合成算法] → 合成样本 → 下游任务评估 → best_top1
                  ↑
            三个方法只换这里
```

外层复用来源：pretext / round23 的数据加载、配置管理、评估流水线。

---

## 4. 各方法技术设计

### 4.1 C4-only

**核心思想**：随机选种子，无 DP 算法，ε=0

**算法**：
1. 从公开 C4 数据集中随机采样 k 条文本（`random.sample(c4_pool, k)`，k=20 与 reference budget 对齐）
2. 用 LLaMA few-shot 扩充（复用 PrE-Text / round23 的 Stage2 扩充逻辑）
3. 输出合成样本集

**实现来源**：直接参考 PrE-Text 代码中的 `c4-only` 逻辑，Stage2 完全复用现有代码。

**与现有方法的区别**：去掉 PrE-Text Stage1 的 DP histogram 选种子，改为 random.sample。

---

### 4.2 Aug-PE

**论文**：Differentially Private Synthetic Data via Foundation Model APIs 2: Text（ICML 2024 Spotlight，arxiv 2403.01749）

**核心思想**：PE 框架的集中式优化版，黑盒 LLM API，不微调模型

**算法**：
1. 从 GitHub 拉取 Aug-PE 代码（AI-secure/aug-pe）
2. 用 Aug-PE 的 Private Evolution 核心逻辑选出高质量 seeds（集中式 DP histogram，改进版 PE）
3. 用 LLaMA few-shot 扩充（复用 Stage2）
4. 输出合成样本集

**与 PrE-Text 的区别**：
- Aug-PE 假设集中式（server 可访问私有数据），PrE-Text 是联邦场景
- Aug-PE 的 PE 算法做了多项工程优化（更好的 embedding、更稳定的迭代）
- Stage2 扩充逻辑在集成时统一使用现有框架的 LLaMA 实现

**实现步骤**：
1. `git clone https://github.com/AI-secure/aug-pe`（或对应 repo）
2. 提取 PE 核心模块，接入现有框架的输入/输出接口
3. 保持 ε 与其他方法一致

---

### 4.3 DP-Prompt

**论文**：Locally Differentially Private Document Generation Using Zero Shot Prompting（EMNLP 2023 Findings）

**核心思想**：对每条私有文档做零样本 LLM 改写，温度控制 Local DP，不微调模型

**算法**：
1. 对每条私有文档构造 prompt（`Review: {原文}\nParaphrase: `）
2. 用预训练 LLM 采样生成改写文本（温度 + 可选 logit clipping）
3. 输出合成改写样本集

**现状**：本机已有修改过的代码（`/Users/apple/Desktop/code_from_paper/dp-prompt`），但尚未达到可直接跑实验的状态，需要继续完善以对齐现有评估流水线。

**与其他方法的关键区别**：DP-Prompt 直接处理私有文档，其他方法通过公开数据间接生成。这个差异在论文写作时需明确说明，不影响放在同一对比表中（最终都是合成样本→下游评估）。

---

## 5. 各方法开发工作量

| 方法 | 工作量 | 关键工作 |
|------|--------|---------|
| C4-only | 低 | 参考 PrE-Text c4-only 逻辑，接入现有框架 |
| Aug-PE | 中 | clone 代码 → 提取 PE 核心 → 接口对齐 |
| DP-Prompt | 中 | 继续完善已有代码 → 对齐评估流水线 |

---

## 6. 参考资料

- Aug-PE 论文：arxiv.org/abs/2403.01749
- DP-Prompt 论文：EMNLP 2023 Findings（本机代码：`dp-prompt/`）
- PrE-Text 代码：`/Users/apple/Desktop/code_from_paper/PrE-Text/`
- E1 实验代码参考：`paper-new-round23/scripts/`
