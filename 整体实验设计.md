# 创新算法 vs PrE-Text 实验设计报告

> 生成时间：2026-04-03
> 适用环境：Linux 服务器（AutoDL RTX 2080 Ti）
> 状态：本文档为**小数据全流程测试实验**设计，用于验证两个平台的主链路完整性与正确性，非论文正式结果实验

---

## 一、Linux 服务器资源总览

### 1.1 硬件配置

| 配置项 | 规格 |
|--------|------|
| **CPU** | 9 核心 |
| **内存** | 48 GB |
| **GPU** | NVIDIA GeForce RTX 2080 Ti, 22 GB VRAM |
| **CUDA 版本** | 12.4 |
| **系统盘** | 30 GB（`/`） |
| **数据盘** | /root/autodl-tmp，250 GB |

> **目录说明**：
> - `/`（系统盘）：实例关机数据不丢失，可存放代码
> - `/root/autodl-tmp`（数据盘）：读写速度更快，**不会随保存镜像一起保存**

### 1.2 项目位置

```
/root/autodl-tmp/caiqiyue/code_from_paper/
├── thesis_platform/          # 创新算法主平台
├── PrE-Text/                 # PrE-Text 对照算法
├── outputs/                  # 实验输出目录
├── GRADMM/
├── FedTextGrad/
├── DataInf/
└── ... (其他算法目录)
```

### 1.3 虚拟环境

| 环境名称 | 路径 | Python 版本 | 状态 | 已安装核心包 |
|----------|------|-------------|------|-------------|
| `caiqiyue` | `/root/miniconda3/envs/caiqiyue` | 3.11 | ✅ 正常（CUDA 12.4 匹配） | torch 2.6.0+cu124, transformers 5.4.0, peft 0.18.1, accelerate 1.13.0, bitsandbytes 0.49.2, datasets 4.8.4, faiss-cpu 1.13.2, sentence-transformers 5.3.0 |
| `pretext` | `/root/miniconda3/envs/pretext` | 3.10 | ⚠️ 已安装但有 CUDA 不匹配 | torch 2.1.2+cu118（CUDA 11.8，与服务器 CUDA 12.4 不匹配），transformers 4.38.2，peft 0.10.0，bitsandbytes 0.43.1（损坏） |

**推荐**：两个项目**统一使用 `caiqiyue` 环境**，因为：
- `caiqiyue` 的 torch 2.6.0+cu124 与服务器 CUDA 12.4 完全匹配
- `pretext` 环境的 torch 2.1.2+cu118 和 bitsandbytes 0.43.1 存在 CUDA 版本冲突

**激活命令**：
```bash
conda activate caiqiyue
```
```

### 1.4 可用数据集（已存在于服务器）

| 数据集 | 路径 | 说明 | 适用于 |
|--------|------|------|--------|
| `pretext_jobs` | `thesis_platform/datasets/pretext_jobs/` | PrE-Text 论文主数据集（Jobs） | 创新算法 + PrE-Text |
| `pretext_forums` | `thesis_platform/datasets/pretext_forums/` | 论坛数据集 | 创新算法 + PrE-Text |
| `pretext_microblog` | `thesis_platform/datasets/pretext_microblog/` | 微博数据集 | 创新算法 + PrE-Text |
| `pretext_congressional` | `thesis_platform/datasets/pretext_congressional/` | 国会演讲数据集 | 创新算法 + PrE-Text |
| `pretext_initialization_c4_en` | `thesis_platform/datasets/pretext_initialization_c4_en/` | C4 英文初始化语料 | PrE-Text Stage 1 |
| `glue_sst2` | `thesis_platform/datasets/glue_sst2/` | 情感分类 | 下游评估 |
| `glue_qnli` | `thesis_platform/datasets/glue_qnli/` | 问答蕴含 | 下游评估 |
| `glue_qqp` | `thesis_platform/datasets/glue_qqp/` | 问题对相似度 | 下游评估 |
| `imdb` | `thesis_platform/datasets/imdb/` | 影评情感 | 下游评估 |
| `rotten_tomatoes` | `thesis_platform/datasets/rotten_tomatoes/` | 影评情感 | 下游评估 |

### 1.5 可用模型（已存在于服务器）

| 模型 | 路径 | 用途 | 显存需求 | 状态 |
|------|------|------|----------|------|
| `all_minilm_l6_v2` | `open_model/all_minilm_l6_v2/` | Embedding / Retriever / Prototype | ~1 GB | ✅ 完好 |
| `roberta_large` | `open_model/roberta_large/` | DataInf Real Scorer 特征编码 | ~1.5 GB | ✅ 完好 |
| `qwen_2_0_5b_instruct` | `open_model/qwen_2_0_5b_instruct/` | 创新算法 Server/Client LLM（推荐） | ~5 GB | ✅ 完好 |
| `llama_3_1_8b_instruct` | `open_model/llama_3_1_8b_instruct/` | 备选 Server LLM（4bit量化） | ~8 GB | ✅ 完好 |
| `llama_2_7b_hf` | `open_model/llama_2_7b_hf/` | PrE-Text Bootstrap / peft_lora 评测 | ~14 GB | ✅ 完好 |
| `distilgpt2` | `open_model/distilgpt2/` | 下游评测（小模型，快速测试） | ~250 MB | ✅ 完好 |

**RTX 2080 Ti 22GB 显存约束下的可用模型组合**：

| 场景 | Server LLM | Client LLM (Critic) | Scorer | 预估显存 |
|------|-----------|---------------------|--------|----------|
| **创新算法（推荐）** | qwen_2_0_5b_instruct | qwen_2_0_5b_instruct | roberta_large | ~10 GB |
| **创新算法（tiny test）** | distilgpt2 | distilgpt2 | all_minilm_l6_v2 | ~1 GB |
| **创新算法（8B量化）** | llama_3_1_8b_instruct (INT4) | qwen_2_0_5b_instruct | roberta_large | ~10 GB |
| **PrE-Text Stage 1** | — | roberta_large (MLM) | — | ~1.5 GB |
| **PrE-Text Bootstrap** | llama_2_7b_hf | — | — | ~14 GB |
| **下游评测（distilgpt2）** | distilgpt2 | — | — | ~500 MB |

---

## 二、两算法完整流程说明

### 2.1 创新算法（thesis_platform）完整流程

创新算法的核心创新在于：**联邦框架下的差分隐私保护 + Prompt 优化**，通过 Critique 机制持续改进 Server 端的生成能力。

#### 流程步骤详解

| 步骤 | 名称 | 模块位置 | 做什么 | 关键输入/输出 |
|:----:|------|---------|-------|--------------|
| **1** | **Generation（生成）** | `adapters/generators/pretext_prompt_generator.py` | Server LLM 根据当前 prompt + 公开种子样例生成合成样本 | 输入：server_prompt, public_seeds；输出：generated_samples.jsonl |
| **2** | **Client Sample Assignment（分配）** | `core/round_runner.py` | 将 global pool 中的合成样本分配给各个 client（可按 personalized_mix_ratio 混合 global/cluster pool） | 输入：generated_samples；输出：client_assigned_samples.jsonl |
| **3** | **Scoring（打分）** | `adapters/scorers/datainf_real_scorer.py` | 用 roberta_large 特征编码，计算每个合成样本的 DataInf 风格 influence score + domain gap（分数越大表示样本越差） | 输入：合成样本 + client 私有数据；输出：scored_samples.jsonl（含 score） |
| **4** | **Bad Sample Selection（坏样本选择）** | `core/selector.py` | 根据 scorer 的分数，选择 top-k（默认2）最差的合成样本 | 输入：scored_samples；输出：selected_bad_samples.jsonl |
| **5** | **Anchor Retrieval（锚点召回）** | `adapters/retrievers/knn_retriever.py` | 用 all_minilm_l6_v2 做 KNN 搜索，召回与 bad sample 最相似的 top-k 真实私有样本作为锚点 | 输入：bad_samples + 私有数据；输出：retrieved_pairs.jsonl |
| **6** | **Critique Generation（批评生成）** | `adapters/critics/fedtextgrad_qwen_critic.py` | LLM 对比 bad sample 与真实锚点样本，生成改进规则（必须以动词开头，如 Add/Use/Remove/Focus） | 输入：retrieved_pairs；输出：client_critiques.jsonl（含多条规则） |
| **7** | **Prototype Extraction（原型提取）** | `algorithms/prototypes/minilm_mean.py` | 用 all_minilm_l6_v2 提取各 client 的原型向量（真实锚点样本的均值 + 归一化），可选 DP clip+noise 隐私保护 | 输入：retrieved_pairs；输出：client_prototypes.jsonl |
| **8** | **Clustering & Aggregation（聚类聚合）** | `algorithms/aggregators/dbscan_core.py` | **① 构建 R_k utility weight → ② DBSCAN 聚类 critique 规则 → ③ 注意力加权排名 → ④ 跨轮记忆合并 → ⑤ 客户端原型聚类 → ⑥ 生成 Cluster-Specific Rules → ⑦ 构建 PromptUpdate** | 输入：critiques + prototypes；输出：prompt_update.json（含 global_rules + cluster_rules + client_cluster_map + memory_rules） |
| **9** | **Prompt Update（提示更新）** | `core/prompt_updater.py` | 提取 global_rules 和 memory_rules，构建新的 Server prompt（含 Base Instruction + Round Guidance + Cluster Guidance + Memory Summary） | 输出：server_prompt.txt |

**每轮输出循环**：更新后的 server_prompt.txt → 回到步骤1进入下一轮

**算法特色**：
- **联邦架构**：多 client 协作，各自在私有数据上评估，共享聚合后的规则
- **Critique 机制**：不直接修改生成数据，而是通过"批评-规则-聚合"间接优化 prompt
- **隐私保护**：通过 DP 对 prototype 向量和权重进行 clip + noise，保护 client 私有数据

---

### 2.2 PrE-Text 算法完整流程

PrE-Text 的核心创新在于：**差分隐私下的私有演化（Private Evolution）**，通过带隐私保护的演化机制生成高质量合成文本。

#### 流程步骤详解

| 步骤 | 名称 | 模块位置 | 做什么 | 关键输入/输出 |
|:----:|------|---------|-------|--------------|
| **1** | **初始化种群** | `stage1.py` | 从 C4 公共语料采样 nsyn 条作为初始父本 | 输入：initialization.json（C4语料）；输出：初始 parent pool |
| **2** | **私有文本 Embedding** | `similarity.py` | 用 all-MiniLM-L6-v2 将所有私有训练文本编码为向量，缓存到 private_embeds.npy | 输入：私有训练集；输出：private_embeds.npy（仅首次计算） |
| **3** | **Lookahead Embedding（前瞻）** | `similarity.py` | 对每个候选文本执行 lookahead=4 次 mask-fill 变异，求每次变异的 embedding 平均 | 输入：candidate pool；输出：lookahead_embeddings |
| **4** | **FAISS 最近邻直方图** | `histogram.py` | 用 FAISS IndexFlatL2 构建索引，统计每个候选被命中的次数构建直方图 | 输入：lookahead_embeds + private_embeds；输出：histogram（命中次数） |
| **5** | **DP 噪声注入** | `histogram.py` | 对直方图加 Gaussian 噪声 N(0, σ) 并阈值裁剪，实现 (ε, δ)-DP 保证 | 输入：histogram + sensitivity + sigma_ratio；输出：noised_histogram |
| **6** | **Survivor 重采样** | `stage1.py` | 按 DP 直方图归一化后的概率分布，有放回采样 nsyn 个 survivor | 输入：noised_histogram；输出：surviving pool |
| **7** | **Mask-Fill 变异** | `variation.py` | 对 survivor 群体执行 t_steps=2 轮 30% mask 的 RoBERTa-large MLM 填充变异 | 输入：surviving pool；输出：new_candidate_pool |
| **8** | **迭代** | — | 重复步骤 3-7，共 rounds=11 轮 | 输出：surviving_text_it{round}.json |
| **9** | **Stage 2 Bootstrap** | `bootstrap.py` | 用 LLaMA-2-7B + 3-shot prompt 对所有 surviving seeds 扩增生成合成语料 | 输入：surviving_texts；输出：llama7b_text_syn.json（大规模合成语料） |
| **10** | **下游评测** | `distilgpt2_eval.py` / `llama2_eval.py` | 在合成语料上微调 DistilGPT2 或 LLaMA-2-7B+LoRA，评测 cross-entropy loss 和 top-k accuracy | 输出：eval_metrics.json |

**每轮输出循环**：surviving pool → variation → new candidate → 回到步骤3进入下一轮

**算法特色**：
- **差分隐私保证**：通过 Gaussian 机制和阈值裁剪，提供严格的 (ε, δ)-DP 隐私保证
- **Lookahead 策略**：用"未来可能的发展方向"评估当前候选，而非仅评估当前状态
- **双阶段生成**：Stage 1 演化 + Stage 2 Bootstrap 扩增，兼顾质量与数量

---

## 三、测试实验设计方案

### 3.1 实验目的

本文档基于当前两个仓库的实际源码，挑选若干组可以用较小数据集跑通的测试性实验，用于验证：

- `thesis_platform` 创新算法主链路是否能从配置解析一路执行到主实验结果与下游评测结果
- `PrE-Text` 对照算法是否能从 `stage1`、`stage2` 一路执行到 `eval_small` 与 `eval_large`
- 合成语料是否能继续进入 GLUE 类下游任务，证明整个实验闭环是完整的

> 这些实验的目标是"证明平台完整性与正确性"，不是论文正式结果实验，因此故意使用了更小的 `train_limit`、`eval_limit`、`rounds`、`num_prompts` 等配置。

### 3.2 执行前提

- 统一使用 `caiqiyue` 虚拟环境：`conda activate caiqiyue`
- 命令以 Bash 为例，假设已通过 SSH 连接到 Linux 服务器
- `thesis_platform` 的命令统一从工作区根目录执行：`/root/autodl-tmp/caiqiyue/code_from_paper`
- `PrE-Text` 的命令统一从其仓库根目录执行：`/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text`

### 3.3 输出位置规则

#### 3.3.1 thesis_platform

实验输出根目录：

```
/root/autodl-tmp/caiqiyue/code_from_paper/outputs/thesis_platform
```

输出规则（两层）：

- 最新运行指针文件：`outputs/thesis_platform/<experiment_id>_latest.json`
- 实际运行目录：`latest.json` 中 `experiment_dir` 字段指向的时间戳目录

> **定位方式**：先看 latest 指针文件，再看真实目录

#### 3.3.2 PrE-Text

实验输出根目录：

```
/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/outputs/pretext_platform
```

直接按实验 ID 建目录：`outputs/pretext_platform/<experiment_id>`

### 3.4 已验证状态

| 实验 | 状态 | 说明 |
|------|------|------|
| **TP-1** (qwen_full_eval) | ✅ 已完成 | `qwen_2_0_5b_instruct` 全流程 + 下游评测，状态 completed |
| **TP-2** (jobs_v3_tiny_linux) | ❌ 已失败 | `llama_3_2_3b_instruct` 模型损坏，无法加载 |
| **PT-1** (validate_tiny_complete_test) | ⏸ 未运行 | 依赖损坏的 `llama_3_2_3b_instruct` |
| **PT-2** (validate_jobs_tiny_thesis_platform) | ⏸ 未运行 | Bootstrap 使用 `llama_2_7b_hf`（完好），可尝试 |

### 3.5 测试实验分组（不需要损坏模型的实验）

| 实验组 | 算法 | 实验目的 | 使用模型 | 推荐顺序 |
|--------|------|----------|----------|----------|
| **TP-1** | 创新算法 | 已完成验证（qwen + 全流程 + 下游评测） | qwen_2_0_5b_instruct | — |
| **TP-1B** | 创新算法 | Tiny 测试（仅 distilgpt2，无需大模型） | distilgpt2 | 1 |
| **TP-2A** | 创新算法 | 创建 qwen 版 2 轮实验（替代损坏的 3B 模型） | qwen_2_0_5b_instruct | 2 |
| **PT-2** | PrE-Text | Stage1 + Bootstrap(7B) + 下游评测 | llama_2_7b_hf + distilgpt2 | 3 |

---

## 四、测试性实验详细说明

### 4.1 TP-1：创新算法 Qwen 版完整性实验（已验证）

> 此实验已完成，状态：✅ completed

#### 实验命令

```bash
cd /root/autodl-tmp/caiqiyue/code_from_paper
conda activate caiqiyue
python -m thesis_platform.scripts.run_experiment \
  --config thesis_platform/configs/experiments/validation/integrity_jobs_real_datainf_qwen_full_eval.yaml
```

#### 命令含义

- 使用 `jobs` 数据集的极小样本配置（`train_limit=6`，`eval_limit=4`）
- 只跑 `1` 轮，客户端和服务端都使用 `qwen_2_0_5b_instruct`
- 配置中已启用 `downstream_eval`，一条命令完成：
  **主实验 → 合成语料导出 → 下游大模型评测（peft_lora 模式）**

#### 过程存储位置

| 内容 | 路径 |
|------|------|
| latest 指针 | `outputs/thesis_platform/integrity_jobs_real_datainf_qwen_full_eval_latest.json` |
| 主实验轮次目录 | `<TP-1_RUN>/round_000/` |
| 下游评测目录 | `<TP-1_RUN>/downstream_eval/` |
| 合成语料目录 | `<TP-1_RUN>/downstream_eval/stage2/` |

> `<TP-1_RUN>` = latest 指针中 `experiment_dir` 指向的真实运行目录

#### 结果存储位置

| 产物 | 路径 |
|------|------|
| 实验汇总 | `<TP-1_RUN>/metrics_summary.json` |
| 解析后完整配置 | `<TP-1_RUN>/resolved_config.json` |
| 下游评测汇总 | `<TP-1_RUN>/downstream_eval/downstream_eval_summary.json` |
| 大模型评测汇总 | `<TP-1_RUN>/downstream_eval/pretext_large_eval_summary.json` |
| 导出的合成语料 | `<TP-1_RUN>/downstream_eval/stage2/llama7b_text_syn.json` |

#### 验收标准

- [x] `metrics_summary.json` 中 `status: "completed"`
- [x] `downstream_eval_summary.json` 中 `status: "completed"`
- [x] `llama7b_text_syn.json` 存在且非空

---

### 4.2 TP-1B：创新算法 Tiny 测试（纯 distilgpt2，无需大模型）

#### 实验命令

```bash
cd /root/autodl-tmp/caiqiyue/code_from_paper
conda activate caiqiyue
python -m thesis_platform.scripts.run_experiment \
  --config thesis_platform/configs/experiments/validation/integrity_jobs_tiny_transformers.yaml
```

#### 命令含义

- 使用 `distilgpt2`（82M 参数）作为 Client 和 Server LLM
- **极小配置**：`train_limit=4`，`eval_limit=4`，`rounds=1`
- 无下游评测（不继承 downstream_eval 配置）
- 全部模型均为 `distilgpt2` 和 `all_minilm_l6_v2`，无需大模型

#### 实验目的

验证创新算法核心流程在极低资源下可运行，用于快速冒烟测试，不依赖任何大模型。

#### 过程存储位置

| 内容 | 路径 |
|------|------|
| latest 指针 | `outputs/thesis_platform/integrity_jobs_tiny_transformers_latest.json` |
| 主实验轮次目录 | `<TP-1B_RUN>/round_000/` |

#### 结果存储位置

| 产物 | 路径 |
|------|------|
| 实验汇总 | `<TP-1B_RUN>/metrics_summary.json` |
| 解析后完整配置 | `<TP-1B_RUN>/resolved_config.json` |

#### 验收标准

- [ ] `metrics_summary.json` 中 `status: "completed"`
- [ ] `round_000/` 目录存在且包含 `generated_samples.jsonl`

---

### 4.3 TP-2A：创新算法 Qwen 版 2 轮完整实验（推荐）

#### 实验命令

**第1步**：在服务器上创建临时配置文件（将 `llama_3_2_3b_instruct` 替换为 `qwen_2_0_5b_instruct`）：

```bash
cat > /root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/configs/experiments/linux/jobs_v3_tiny_qwen_linux.yaml << 'EOF'
# Linux RTX 2080 Ti (22GB) - 用 qwen 替代损坏的 llama_3_2_3b_instruct
inherits:
  - ../../base/paths.yaml
  - ../../base/runtime.yaml
  - ../../methods/generators/pretext_prompt_llm.yaml
  - ../../methods/scorers/datainf_real.yaml
  - ../../methods/retrievers/knn.yaml
  - ../../methods/critics/fedtextgrad_qwen.yaml
  - ../../methods/aggregators/dbscan_attn_tsgdm.yaml
  - ../../methods/prototypes/minilm_mean.yaml
  - ../../methods/routing/personalized_v3.yaml
  - ../../methods/privacy/jobs_eps129.yaml
  - ../../methods/downstream_eval/pretext_large_off.yaml

meta:
  experiment_id: jobs_v3_tiny_qwen_linux
  stage: validation_linux_qwen
  seed: 7

data:
  dataset_name: jobs
  task_type: instruction_tuning
  sample_format: raw_text
  partition_strategy: preserve_buckets
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  public_seed_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
  max_public_seed_samples: 8
  num_clients: 2
  max_samples_per_client: 2
  validation_ratio: 0.5
  train_limit: 8
  eval_limit: 8
  initialization_min_words: 20

llm:
  client:
    engine: transformers
    model_name_or_path: thesis_platform/open_model/qwen_2_0_5b_instruct
    device: auto
    dtype: auto
    temperature: 0.0
    max_new_tokens: 96
    use_chat_template: true
  server:
    engine: transformers
    model_name_or_path: thesis_platform/open_model/qwen_2_0_5b_instruct
    device: auto
    dtype: auto
    temperature: 0.2
    max_new_tokens: 96
    use_chat_template: true

generator:
  generated_per_round: 4
  exemplars_per_prompt: 1
  max_new_tokens: 96
  temperature: 0.0
  initial_prompt: "Generate concise job-related text."

scorer:
  allow_hashing_fallback: true

retriever:
  embedding_model: thesis_platform/open_model/all_minilm_l6_v2
  allow_hashing_fallback: true
  top_k: 1

aggregator:
  allow_hashing_fallback: true
  cluster_min_samples: 1

federation:
  rounds: 2
  top_k_bad: 1
EOF
```

**第2步**：运行主实验：

```bash
cd /root/autodl-tmp/caiqiyue/code_from_paper
conda activate caiqiyue
python -m thesis_platform.scripts.run_experiment \
  --config thesis_platform/configs/experiments/linux/jobs_v3_tiny_qwen_linux.yaml
```

**第3步**：独立下游评测（主实验完成后执行）：

```bash
cd /root/autodl-tmp/caiqiyue/code_from_paper
conda activate caiqiyue
python -m thesis_platform.scripts.run_downstream_eval \
  --experiment-id jobs_v3_tiny_qwen_linux \
  --large-eval-mode gpt2_xl
```

> `--large-eval-mode gpt2_xl` 会自动降级到 `distilgpt2`（`gpt2_xl` 模型文件不存在于服务器）。

#### 命令含义

- 替代原本使用 `llama_3_2_3b_instruct`（已损坏）的 `jobs_v3_tiny_linux.yaml`
- 使用 `qwen_2_0_5b_instruct` 作为 Client 和 Server LLM
- `rounds=2`，`train_limit=8`，`eval_limit=8`，轻量配置
- 第3步下游评测使用 `distilgpt2`，无需大模型

#### 实验目的

验证创新算法在 2 轮迭代下完整运行，且主实验与下游评测解耦可独立运行。

#### 过程存储位置

| 内容 | 路径 |
|------|------|
| latest 指针 | `outputs/thesis_platform/jobs_v3_tiny_qwen_linux_latest.json` |
| 主实验 round_000 | `<TP-2A_RUN>/round_000/` |
| 主实验 round_001 | `<TP-2A_RUN>/round_001/` |
| 下游评测目录 | `<TP-2A_RUN>/downstream_eval/` |

#### 结果存储位置

| 产物 | 路径 |
|------|------|
| 主实验汇总 | `<TP-2A_RUN>/metrics_summary.json` |
| 解析后完整配置 | `<TP-2A_RUN>/resolved_config.json` |
| 下游评测汇总 | `<TP-2A_RUN>/downstream_eval/downstream_eval_summary.json` |
| 导出的合成语料 | `<TP-2A_RUN>/downstream_eval/stage2/llama7b_text_syn.json` |

#### 验收标准

- [ ] `metrics_summary.json` 中 `completed_rounds: 2`
- [ ] `round_000/` 和 `round_001/` 目录均存在
- [ ] `llama7b_text_syn.json` 存在且非空

---

### 4.4 TP-3：创新算法合成语料的 GLUE 衍生验证

#### 实验命令

```bash
cd /root/autodl-tmp/caiqiyue/code_from_paper
conda activate caiqiyue
python -m thesis_platform.scripts.run_glue_eval \
  --experiment-id jobs_v3_tiny_qwen_linux \
  --tasks sst2
```

#### 命令含义

- 读取 TP-2A 生成的合成语料：`<TP-2A_RUN>/downstream_eval/stage2/llama7b_text_syn.json`
- 对 `sst2` 任务执行 GLUE 风格分类评测

#### 实验目的

证明创新算法生成的合成语料能够继续进入衍生下游任务（GLUE 分类），而不只是停留在主实验内部。

#### 过程存储位置

| 内容 | 路径 |
|------|------|
| GLUE 评测目录 | `<TP-2A_RUN>/glue_eval/glue_sst2_eval/` |

#### 结果存储位置

| 产物 | 路径 |
|------|------|
| 单任务汇总 | `<TP-2A_RUN>/glue_eval/glue_sst2_summary.json` |
| GLUE 总汇总 | `<TP-2A_RUN>/glue_eval/glue_summary.json` |

#### 验收标准

- [ ] `glue_summary.json` 存在
- [ ] `glue_sst2_summary.json` 中 `status: "completed"`

---

### 4.5 PT-2：PrE-Text 论文路径微型实验

#### 实验命令

```bash
cd /root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text
conda activate caiqiyue
python -m pretext_platform.scripts.run_pipeline \
  --config configs/experiments/validate_jobs_tiny_thesis_platform.yaml
```

#### 命令含义

- 使用 `jobs` 数据集，继承 `thesis_platform` 数据目录与模型目录
- 执行 **stage1 + bootstrap(stage2) + eval_large**
- Bootstrap 默认使用 `llama_2_7b_hf`（服务器完好）
- 下游评测使用 `peft_lora` 模式（LLaMA-2-7B + LoRA）

#### 实验目的

验证 PrE-Text 完整主流程（Stage1 演化 + Stage2 Bootstrap + 下游评测）在小数据上可闭环运行。

#### 过程存储位置

| 内容 | 路径 |
|------|------|
| 实验目录 | `outputs/pretext_platform/validate_jobs_tiny_thesis_platform_eps758/` |
| Stage1 目录 | `.../validate_jobs_tiny_thesis_platform_eps758/stage1/` |
| Stage2（Bootstrap）目录 | `.../validate_jobs_tiny_thesis_platform_eps758/stage2/` |
| 下游评测目录 | `.../validate_jobs_tiny_thesis_platform_eps758/eval_large/` |

#### 结果存储位置

| 产物 | 路径 |
|------|------|
| 实验汇总 | `.../validate_jobs_tiny_thesis_platform_eps758/metrics_summary.json` |
| Stage1 汇总 | `.../validate_jobs_tiny_thesis_platform_eps758/stage1_summary.json` |
| Stage2 汇总 | `.../validate_jobs_tiny_thesis_platform_eps758/stage2_summary.json` |
| 下游评测汇总 | `.../validate_jobs_tiny_thesis_platform_eps758/eval_large_summary.json` |
| 导出的合成语料 | `.../validate_jobs_tiny_thesis_platform_eps758/stage2/llama7b_text_syn.json` |

#### 验收标准

- [ ] `stage1_summary.json` 中 `status: "completed"`
- [ ] `stage2_summary.json` 中 `status: "completed"`
- [ ] `llama7b_text_syn.json` 存在且非空

---

## 五、推荐执行顺序

```
TP-1B（创新算法 Tiny 冒烟测试，纯 distilgpt2，无需大模型）
   ↓
TP-2A（创新算法 Qwen 版 2 轮实验）
   ↓
TP-3（创新算法 GLUE 衍生验证）
   ↓
PT-2（PrE-Text 论文路径微型实验）
```

**最低验收组合**：TP-1B + PT-2 — 若这两组跑通，说明两个平台各自最小的完整闭环均已成立。

---

## 六、代码修改记录

以下代码修复已同步至服务器，与本地保持一致：

| 修复文件 | 问题 | 修复内容 |
|----------|------|----------|
| `PrE-Text/pretext_platform/core/config.py` | 缺少 PyYAML 时，`_parse_scalar` 不处理行内注释 | 增加引号内 `#` 注释的检测与去除 |
| `PrE-Text/pretext_platform/evaluation/llama2_eval.py` | peft_lora 模式下 LoRA 参数 dtype 不一致，导致 `float != c10::Half` matmul 错误 | `get_peft_model` 后强制将 float32 的 LoRA 参数转为 base model dtype |
| `thesis_platform/scripts/run_downstream_eval.py` | 无 `downstream_eval/stage2/` 时直接报错，不支持从 round artifact 回退 | 增加从 `round_XXX/client_assigned_samples.jsonl` 回退读取的逻辑 |
| `thesis_platform/scripts/run_glue_eval.py` | 同上，run_glue_eval 硬性要求 stage2 文件存在 | 同上修复 + 回退时写入 stage2 文件供下游消费 |

---

## 七、服务器模型文件状态

> 检查时间：2026-04-03

| 模型 | 服务器状态 | 备注 |
|------|------------|------|
| `qwen_2_0_5b_instruct` | ✅ 完好（954MB） | TP-1/TP-2A 主实验使用 |
| `llama_2_7b_hf` | ✅ 完好（38GB） | PT-2 Bootstrap / peft_lora 评测使用 |
| `llama_3_1_8b_instruct` | ✅ 完好（15GB, 4bit 量化） | 备选大模型 |
| `all_minilm_l6_v2` | ✅ 完好 | Embedding/Retriever/Prototype |
| `roberta_large` | ✅ 完好 | Scorer 特征编码 |
| `distilgpt2` | ✅ 完好（3GB） | 下游评测 |
| `llama_3_2_3b_instruct` | ❌ 损坏（仅 1.1GB，应为 ~6GB） | TP-2 / PT-1 不可用，需修复 |
| `Meta-Llama-2-7b-chat-hf` | ❌ 损坏（仅 390MB，应为 ~12GB） | 当前实验未使用 |
| `Meta-Llama-3-8B` | ✅ 完好（15GB） | 当前实验未使用 |
| `flan_t5_3b` | ❌ 不存在 | 当前实验未使用 |

> ⚠️ `llama_3_2_3b_instruct` 模型文件不完整（1.1GB vs 正常 ~6GB），会导致所有依赖它的实验失败。修复方法：
> ```bash
> # 方法1：从本地上传（本地文件已验证完好）
> rsync -avP /本地路径/llama_3_2_3b_instruct/ root@服务器:/root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_3_2_3b_instruct/
>
> # 方法2：在服务器重新下载
> conda activate caiqiyue
> huggingface-cli download --local-dir /root/autodl-tmp/caiqiyue/code_from_paper/thesis_platform/open_model/llama_3_2_3b_instruct meta-llama/Llama-3.2-3B-Instruct
> ```

---

## 八、环境说明补充

### 8.1 服务器上的 conda 环境

执行 `conda env list` 的输出：

```
base                     /root/miniconda3
caiqiyue                 /root/miniconda3/envs/caiqiyue
drugban                  /root/miniconda3/envs/drugban
pretext                  /root/miniconda3/envs/pretext
```

其中 `pretext` 和 `drugban` 实际指向同一路径 `/root/miniconda3/envs/pretext`。

### 8.2 两个环境的差异

| 对比项 | `caiqiyue` 环境 | `pretext` 环境 |
|--------|----------------|----------------|
| Python 版本 | 3.11 | 3.10 |
| torch 版本 | 2.6.0+**cu124**（与服务器 CUDA 12.4 匹配）✅ | 2.1.2+**cu118**（CUDA 11.8，与服务器 CUDA 12.4 不匹配）⚠️ |
| bitsandbytes | 0.49.2 ✅ | 0.43.1（CUDA 库冲突）❌ |
| transformers | 5.4.0 ✅ | 4.38.2 |
| peft | 0.18.1 ✅ | 0.10.0 |
| 是否推荐使用 | ✅ 推荐所有实验使用 | ❌ 不推荐 |

**结论**：所有实验（包括 PrE-Text）统一使用 `conda activate caiqiyue`。

### 8.3 为什么 TP-2（jobs_v3_tiny_linux）失败了？

`jobs_v3_tiny_linux.yaml` 继承自 `configs/base/llm_3b_linux.yaml`，该配置将 `llama_3_2_3b_instruct` 设为 Client 和 Server LLM。由于该模型文件损坏（1.1GB/6GB），LLM 加载失败，导致 `completed_rounds: 0` 直接退出。TP-2A 通过将 LLM 替换为完好的 `qwen_2_0_5b_instruct` 解决了此问题。
