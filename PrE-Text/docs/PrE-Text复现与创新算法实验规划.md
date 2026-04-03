# PrE-Text 复现实验 & 创新算法实验规划

## 一、PrE-Text 论文实验清单（需要复现的实验）

### 1.1 核心实验矩阵

| 实验 | 数据集 | Epsilon | 模型 | 轮数 | 合成数据量 |
|------|--------|---------|------|------|------------|
| E1 | Jobs | 1.29 | DistilGPT2 | 11 | 2,000,000 |
| E2 | Jobs | 7.58 | DistilGPT2 | 11 | 2,000,000 |
| E3 | Jobs | 1.29 | LLaMA-2-7B | 11 | 50,000 |
| E4 | Jobs | 7.58 | LLaMA-2-7B | 11 | 50,000 |
| E5 | Forums | 1.29 | DistilGPT2 | 11 | 2,000,000 |
| E6 | Forums | 7.58 | DistilGPT2 | 11 | 2,000,000 |
| E7 | Forums | 1.29 | LLaMA-2-7B | 11 | 50,000 |
| E8 | Forums | 7.58 | LLaMA-2-7B | 11 | 50,000 |
| E9 | Microblog | 1.29 | DistilGPT2 | 11 | 2,000,000 |
| E10 | Microblog | 7.58 | DistilGPT2 | 11 | 2,000,000 |
| E11 | Microblog | 1.29 | LLaMA-2-7B | 11 | 50,000 |
| E12 | Microblog | 7.58 | LLaMA-2-7B | 11 | 50,000 |
| E13 | Code | 1.29 | DistilGPT2 | 11 | 2,000,000 |
| E14 | Code | 7.58 | DistilGPT2 | 11 | 2,000,000 |
| E15 | Code | 1.29 | LLaMA-2-7B | 11 | 50,000 |
| E16 | Code | 7.58 | LLaMA-2-7B | 11 | 50,000 |

**关键差异**：当前 PrE-Text 平台使用 `congressional`/`bioarxiv` 数据集，**不是论文原始的 Jobs/Forums/Microblog/Code**。

### 1.2 对比基线实验

| 基线 | 说明 | 需要额外实现 |
|------|------|-------------|
| DP-FedAvg | 传统联邦平均+差分隐私 | 是 |
| DP-FTRL | 差分隐私 FTRL | 是 |
| DP-Prompt | flan-t5-3b 文本到文本 | 是 |
| c4-only | 仅用公共数据微调 | 否 |
| Expand-only | 仅用合成数据微调 | 否 |

### 1.3 消融实验

| 消融项 | 变体 | 目的 |
|--------|------|------|
| Stage 1 影响 | 仅 Stage 1 vs 完整流程 | 验证 bootstrap 贡献 |
| Privacy budget | epsilon ∈ {0.1, 0.5, 1.29, 7.58} | 隐私-效用权衡曲线 |
| Noise scale | sigma ∈ {2.31, 11.3} | 对应两个 epsilon |
| Mask ratio | 10%, 30%, 50% | 变异强度影响 |
| Lookahead steps | 0, 2, 4, 8 | lookahead 窗口影响 |
| 合成数据规模 | 10k, 50k, 500k, 1M, 2M | scaling curve |

### 1.4 当前 PrE-Text 平台状态

| 组件 | 状态 | 说明 |
|------|------|------|
| Stage 1 Private Evolution | ✅ 已实现 | 核心算法完成 |
| Stage 2 Bootstrap (LLaMA-2-7B) | ✅ 已实现 | |
| DistilGPT2 eval | ⚠️ 部分完成 | 缺 c4_checkpoint.pth |
| LLaMA-2-7B eval | ✅ 已实现 | |
| 原始论文数据集 | ❌ 未准备 | Jobs/Forums/Microblog/Code |
| DP-FedAvg 基线 | ❌ 未实现 | |
| DP-FTRL 基线 | ❌ 未实现 | |
| DP-Prompt 基线 | ❌ 未实现 | |

---

## 二、创新算法（FedText-Proto）实验清单

基于[创新算法第三版平台完整流程与差距分析报告](./创新算法第三版平台完整流程与差距分析.md)，当前实现完成度约 35-45%。

### 2.1 P0 实验（核心功能验证）

| 实验 | 配置 | 目标 |
|------|------|------|
| V3-E1 | jobs_real_datainf_v3 (epsilon=1.29) | 验证完整流程能跑通 |
| V3-E2 | jobs_real_datainf_v3 (epsilon=7.58) | 高隐私预算下的效用 |
| V3-E3 | 原型聚类可视化 | 验证 DBSCAN 聚类有效性 |
| V3-E4 | KKT 冲突消解验证 | 验证冲突检测和投影逻辑 |

### 2.2 P1 实验（算法完整性）

| 实验 | 配置 | 目标 |
|------|------|------|
| V3-E5 | 实现 utility-weighted attention | α_k = exp(R_k)/Σexp(R_j) |
| V3-E6 | 动态混合比例 α 搜索 | 0.3, 0.5, 0.7, 0.9 |
| V3-E7 | DataInf 真实梯度计算 | 替代 Roberta 相似度 |
| V3-E8 | 差分隐私真实执行 | gradient clipping |

### 2.3 P2 实验（扩展验证）

| 实验 | 数据集 | 目标 |
|------|--------|------|
| V3-E9 | Forums | 跨数据集泛化 |
| V3-E10 | Microblog | 跨数据集泛化 |
| V3-E11 | Code | 跨数据集泛化 |
| V3-E12 | congressional | 当前平台数据 |

### 2.4 对比实验（创新算法 vs PrE-Text）

| 对比项 | PrE-Text | FedText-Proto | 预期差异 |
|--------|----------|---------------|----------|
| Non-IID 处理 | 无 | 原型聚类+双层 prompt | 客户端分布差异大时优势明显 |
| 隐私保护 | DP 直方图 | DP + 原型聚合 | 相当或更好 |
| 通信效率 | 全量合成数据 | 原型+规则压缩 | FedText-Proto 更高效 |

---

## 三、实验优先级建议

### 第一阶段：PrE-Text 复现（1-2月）

1. 准备 Jobs/Forums/Microblog/Code 数据集（按照论文从 c4-en 构造）
2. 运行 E1-E4（Jobs 数据集 + 两个 epsilon + 两个模型）
3. 运行消融实验子集

### 第二阶段：创新算法验证（2-3月）

1. V3-E1 到 V3-E4 核心流程验证
2. V3-E5 到 V3-E8 算法完整性
3. V3-E9 到 V3-E12 跨数据集验证

### 第三阶段：对比实验（3-4月）

1. 在相同数据集上运行 PrE-Text 和 FedText-Proto
2. 绘制隐私-效用权衡曲线
3. 分析各组件消融贡献

---

## 四、关键文件位置

### PrE-Text 相关

| 文件 | 路径 |
|------|------|
| 论文 PDF | `D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\PrE-Text.pdf` |
| 论文摘要 | `D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\PRETEXT_PAPER_SUMMARY.md` |
| 平台 README | `D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\README.md` |
| 平台核心代码 | `D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\pretext_platform\` |
| 实验配置 | `D:\学习记录\导师项目\研究\caiqiyue_file\PrE-Text\configs\experiments\` |

### 创新算法相关

| 文件 | 路径 |
|------|------|
| 差距分析报告 | `D:\学习记录\导师项目\研究\caiqiyue_file\docs\创新算法第三版平台完整流程与差距分析.md` |
| 平台核心代码 | `D:\学习记录\导师项目\研究\caiqiyue_file\thesis_platform\` |
| V3 实验配置 | `D:\学习记录\导师项目\研究\caiqiyue_file\thesis_platform\configs\experiments\v3\jobs_real_datainf_v3.yaml` |

---

## 五、验证方法

1. **PrE-Text 复现**: 对比论文中的 accuracy/perplexity 曲线
2. **创新算法**: 对比 V3 实现与 PrE-Text 在相同设置下的表现
3. **消融实验**: 通过移除/修改特定组件验证贡献度

---

## 六、PrE-Text 复现所需数据集构造

根据论文，数据集构造方法如下：

### Jobs / Forums / Microblog

- 各自从 `c4-en` 中取对应站点来源的前 11,000 条样本
- 10,000 条作为 private train，1,000 条作为 eval
- 训练集均匀随机拆成 1250 个客户端，每个客户端 8 条样本
- sensitivity = 8

### Code

- 面向 coding / technical topics 的问答数据集
- 构造 1250 个用户客户端
- 每个用户训练时最多保留 128 条 comments
- eval 集来自后续 100 个用户中的前 2000 条样本
- sensitivity = 16（噪声和阈值 H 相对翻倍）
