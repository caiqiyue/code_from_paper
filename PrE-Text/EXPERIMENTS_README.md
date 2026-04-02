# PrE-Text 实验运行指南

> 生成时间：2026-04-02
> 适用环境：Linux 服务器（AutoDL RTX 2080 Ti）
> 实验设计：20260402_创新算法实验设计_增强版.md

---

## 一、目录结构

```
PrE-Text/
├── configs/
│   ├── base/                        # 基础配置
│   │   ├── paths.yaml               #   路径配置
│   │   ├── models.yaml               #   模型配置
│   │   └── runtime.yaml              #   运行时配置
│   ├── templates/                   # 配置模板
│   │   ├── noise_eps05.yaml          #   ε=0.5 严格隐私
│   │   ├── noise_eps129.yaml         #   ε=1.29 平衡隐私
│   │   ├── noise_eps758.yaml         #   ε=7.58 宽松隐私
│   │   ├── glue_eval.yaml            #   GLUE评估模板
│   │   ├── glue_eval_jobs.yaml       #   Jobs GLUE评估
│   │   ├── glue_eval_forums.yaml     #   Forums GLUE评估
│   │   ├── glue_eval_microblog.yaml  #   Microblog GLUE评估
│   │   └── glue_eval_congressional.yaml  # Congressional GLUE评估
│   └── experiments/                 # 实验配置（46组实验）
│       ├── jobs_real_eps05.yaml      #   GP1
│       ├── jobs_real_eps129.yaml     #   GP2
│       ├── jobs_real_eps758.yaml     #   GP3
│       ├── jobs_real_no_privacy.yaml #   GP4
│       ├── forums_real_eps05.yaml    #   Forums ε=0.5
│       ├── forums_real_eps129.yaml   #   GC8
│       ├── forums_real_eps758.yaml   #   Forums ε=7.58
│       ├── forums_real_no_privacy.yaml
│       ├── microblog_real_eps05.yaml #  Microblog ε=0.5
│       ├── microblog_real_eps129.yaml # GC9
│       ├── microblog_real_eps758.yaml
│       ├── microblog_real_no_privacy.yaml
│       ├── congressional_real_eps05.yaml # GC4
│       ├── congressional_real_no_privacy.yaml
│       └── ... (其他验证配置)
│
├── pretext_platform/
│   ├── algorithms/                   # 核心算法
│   │   ├── stage1.py               #   Stage 1: 私有演化
│   │   ├── bootstrap.py            #   Stage 2: Bootstrap生成
│   │   ├── histogram.py            #   DP NN直方图
│   │   ├── variation.py            #   Mask-Fill变异
│   │   └── similarity.py           #   Embedding计算
│   ├── evaluation/                  # 下游评估
│   │   ├── glue_classification_eval.py  # GLUE分类评估
│   │   ├── gpt2_eval.py            #   GPT-2小模型评估
│   │   ├── llama2_eval.py          #   LLaMA-2 PEFT评估
│   │   └── llama32_eval.py         #   LLaMA-3.2评估
│   ├── scripts/                     # 运行脚本
│   │   ├── run_pipeline.py         #   完整Pipeline
│   │   ├── run_glue.py             #   独立GLUE评估
│   │   ├── run_experiments.py      #   综合实验运行器
│   │   ├── run_cross_domain.py     #   跨域迁移实验
│   │   └── run_multi_seed.py       #   多种子实验
│   └── core/                        # 核心模块
│       ├── pipeline.py             #   Pipeline编排
│       ├── config.py              #   配置加载
│       └── ...
│
├── outputs/                         # ⭐ 实验输出目录（gitignore）
│   └── pretext_platform/
│       ├── jobs_real_eps05/        #   实验1输出
│       │   ├── stage1/             #     Stage1产物
│       │   │   ├── surviving_text_it*.json
│       │   │   └── private_embeds.npy
│       │   ├── stage2/             #     Stage2产物
│       │   │   └── llama7b_text_syn.json
│       │   ├── eval_large/         #     大模型评估
│       │   │   └── llama2_models_and_accuracies/
│       │   ├── eval_glue/          #     GLUE评估
│       │   │   ├── glue_sst2_eval/
│       │   │   └── glue_summary.json
│       │   └── metrics_summary.json
│       ├── jobs_real_eps129/       #   实验2输出
│       └── ...
│
├── logs/                            # ⭐ 日志目录（gitignore）
│   ├── jobs_real_eps05/            #   实验日志
│   │   ├── main.log               #     主日志
│   │   ├── error.log              #     错误日志
│   │   └── stages.log             #     分阶段日志
│   └── run_YYYYMMDD_HHMMSS.log    #   批量运行日志
│
├── results/                         # ⭐ 结果目录（gitignore）
│   ├── experiment_list.json        #   实验列表
│   ├── summary.json                #   最终汇总
│   ├── progress.json               #   中间进度
│   ├── jobs_real_eps05.json       #   实验1结果
│   ├── jobs_real_eps129.json       #   实验2结果
│   └── ...
│
├── run_all_experiments.sh          # ⭐ 批量运行脚本
└── EXPERIMENTS_README.md          # 本文档
```

---

## 二、实验配置清单

### 2.1 PrE-Text 实验（共16组）

| 实验ID | 配置文件 | 数据集 | ε值 | 对应设计组 |
|--------|---------|--------|-----|-----------|
| GC2_jobs | `jobs_real_eps129.yaml` | Jobs | 1.29 | GC2 |
| GC4_cong | `congressional_real_eps05.yaml` | Congressional | 0.5 | GC4 |
| GC8_forums | `forums_real_eps129.yaml` | Forums | 1.29 | GC8 |
| GC9_micro | `microblog_real_eps129.yaml` | Microblog | 1.29 | GC9 |
| GP1_jobs | `jobs_real_eps05.yaml` | Jobs | 0.5 | GP1 |
| GP2_jobs | `jobs_real_eps129.yaml` | Jobs | 1.29 | GP2 |
| GP3_jobs | `jobs_real_eps758.yaml` | Jobs | 7.58 | GP3 |
| GP4_jobs | `jobs_real_no_privacy.yaml` | Jobs | ∞ | GP4 |
| forums_eps05 | `forums_real_eps05.yaml` | Forums | 0.5 | - |
| forums_eps758 | `forums_real_eps758.yaml` | Forums | 7.58 | - |
| forums_nopriv | `forums_real_no_privacy.yaml` | Forums | ∞ | - |
| micro_eps05 | `microblog_real_eps05.yaml` | Microblog | 0.5 | - |
| micro_eps758 | `microblog_real_eps758.yaml` | Microblog | 7.58 | - |
| micro_nopriv | `microblog_real_no_privacy.yaml` | Microblog | ∞ | - |
| cong_nopriv | `congressional_real_no_privacy.yaml` | Congressional | ∞ | - |

---

## 三、启动指令

### 3.1 方式一：使用综合脚本（推荐）

```bash
# 进入PrE-Text目录
cd /root/autodl-tmp/PrE-Text

# 1. 烟雾测试（验证环境）
bash run_all_experiments.sh --smoke

# 2. 运行Jobs系列实验
bash run_all_experiments.sh --jobs

# 3. 运行Forums系列实验
bash run_all_experiments.sh --forums

# 4. 运行Microblog系列实验
bash run_all_experiments.sh --microblog

# 5. 运行Congressional系列实验
bash run_all_experiments.sh --congressional

# 6. 运行所有实验
bash run_all_experiments.sh --all

# 7. 带GLUE下游评估运行
bash run_all_experiments.sh --jobs --glue

# 8. 并行运行（最多N个）
bash run_all_experiments.sh --all --parallel 2

# 9. 预览命令（不执行）
bash run_all_experiments.sh --all --dry-run
```

### 3.2 方式二：使用Python模块

```bash
# 进入PrE-Text目录
cd /root/autodl-tmp/PrE-Text

# 列出所有可用实验
python -m pretext_platform.scripts.run_experiments --list

# 运行所有实验（顺序执行）
python -m pretext_platform.scripts.run_experiments --all

# 运行指定实验
python -m pretext_platform.scripts.run_experiments --experiment jobs_real_eps129

# 运行指定实验 + GLUE评估
python -m pretext_platform.scripts.run_experiments --experiment jobs_real_eps129 --with-glue

# 自定义输出目录
python -m pretext_platform.scripts.run_experiments --all \
    --output-base /root/autodl-tmp/outputs \
    --log-dir /root/autodl-tmp/logs \
    --result-dir /root/autodl-tmp/results

# ============================================================
# 新增功能: 并行执行 + 断点续跑
# ============================================================

# 并行运行（最多2个实验同时跑）
python -m pretext_platform.scripts.run_experiments --all --parallel

# 并行运行 + 指定并行数
python -m pretext_platform.scripts.run_experiments --all --parallel --max-parallel 4

# 断点续跑（跳过已完成的实验）
python -m pretext_platform.scripts.run_experiments --all --resume

# 并行 + 续跑（推荐用于长时间实验）
python -m pretext_platform.scripts.run_experiments --all --parallel --resume

# 仅续跑模式运行
python -m pretext_platform.scripts.run_experiments --all --resume --parallel
```

### 3.3 方式三：单独运行每个实验

```bash
# ============================================================
# GC2: PrE-Text Jobs ε=1.29
# ============================================================
python -m pretext_platform.scripts.run_pipeline \
    --config configs/experiments/jobs_real_eps129.yaml

# ============================================================
# GC4: PrE-Text Congressional ε=0.5
# ============================================================
python -m pretext_platform.scripts.run_pipeline \
    --config configs/experiments/congressional_real_eps05.yaml

# ============================================================
# GC8: PrE-Text Forums ε=1.29
# ============================================================
python -m pretext_platform.scripts.run_pipeline \
    --config configs/experiments/forums_real_eps129.yaml

# ============================================================
# GC9: PrE-Text Microblog ε=1.29
# ============================================================
python -m pretext_platform.scripts.run_pipeline \
    --config configs/experiments/microblog_real_eps129.yaml

# ============================================================
# GP1: PrE-Text Jobs ε=0.5 (严格隐私)
# ============================================================
python -m pretext_platform.scripts.run_pipeline \
    --config configs/experiments/jobs_real_eps05.yaml

# ============================================================
# GP2: PrE-Text Jobs ε=1.29 (平衡隐私)
# ============================================================
python -m pretext_platform.scripts.run_pipeline \
    --config configs/experiments/jobs_real_eps129.yaml

# ============================================================
# GP3: PrE-Text Jobs ε=7.58 (宽松隐私)
# ============================================================
python -m pretext_platform.scripts.run_pipeline \
    --config configs/experiments/jobs_real_eps758.yaml

# ============================================================
# GP4: PrE-Text Jobs 无隐私 (ε=∞)
# ============================================================
python -m pretext_platform.scripts.run_pipeline \
    --config configs/experiments/jobs_real_no_privacy.yaml
```

### 3.4 GLUE 下游评估指令

```bash
# ============================================================
# 预先验证本地GLUE数据集是否可用
# ============================================================
# 验证默认路径的GLUE数据集
python -m pretext_platform.scripts.run_glue --validate

# 验证自定义路径的GLUE数据集
python -m pretext_platform.scripts.run_glue --validate --dataset-root /path/to/datasets

# ============================================================
# 运行GLUE评估
# ============================================================

# 对已完成实验运行GLUE评估
python -m pretext_platform.scripts.run_glue \
    --config configs/experiments/jobs_real_eps129.yaml \
    --task sst2

# 运行多个GLUE任务
python -m pretext_platform.scripts.run_glue \
    --config configs/experiments/jobs_real_eps129.yaml \
    --tasks sst2 qqp qnli

# 运行所有GLUE任务
python -m pretext_platform.scripts.run_glue \
    --config configs/experiments/jobs_real_eps129.yaml \
    --tasks all
```

### 3.5 跨域迁移实验

```bash
# Jobs -> Forums 迁移
python -m pretext_platform.scripts.run_cross_domain \
    --source-config configs/experiments/jobs_real_eps129.yaml \
    --target-dataset forums \
    --target-eval-config configs/experiments/forums_real_eps129.yaml

# Jobs -> Microblog 迁移
python -m pretext_platform.scripts.run_cross_domain \
    --source-config configs/experiments/jobs_real_eps129.yaml \
    --target-dataset microblog
```

### 3.6 多种子统计显著性实验

```bash
# 运行多个种子
python -m pretext_platform.scripts.run_multi_seed \
    --config configs/experiments/jobs_real_eps129.yaml \
    --seeds 42 123 456
```

---

## 四、实验中间过程记录

### 4.1 日志文件结构

每个实验运行后，在 `logs/{experiment_id}/` 目录下生成：

```
logs/
└── jobs_real_eps129/
    ├── main.log       # 完整运行日志（包含所有阶段）
    ├── error.log      # 仅错误和警告
    └── stages.log    # 分阶段日志
```

### 4.2 日志内容示例

**main.log**:
```
[2026-04-02 10:00:00] ====================================================
[2026-04-02 10:00:00] EXPERIMENT: jobs_real_eps129
[2026-04-02 10:00:00] START TIME: 2026-04-02T10:00:00
[2026-04-02 10:00:00] LOG DIR: logs/jobs_real_eps129
[2026-04-02 10:00:00] ====================================================
[2026-04-02 10:00:00] Description: PrE-Text Jobs ε=1.29 (11 rounds)
[2026-04-02 10:00:00] Config: configs/experiments/jobs_real_eps129.yaml
[2026-04-02 10:00:05] [STAGE1] Starting Stage 1: Private Evolution
[2026-04-02 10:00:10] [STAGE1] Time for synthetic embeddings: 5.23s
[2026-04-02 10:00:15] [STAGE1] Current step 0
...
```

**error.log**:
```
[2026-04-02 10:05:00] ERROR: Stage 1 failed: OutOfMemoryError: CUDA out of memory
  Exception: OutOfMemoryError: CUDA out of memory
  Traceback:
  ...
```

### 4.3 进度跟踪

运行期间，进度信息保存在 `results/progress.json`：

```json
{
  "total": 16,
  "succeeded": ["jobs_real_eps05", "jobs_real_eps129"],
  "failed": [],
  "in_progress": ["jobs_real_eps758", "jobs_real_no_privacy", ...],
  "results": {...}
}
```

每次实验完成后，进度自动更新。

### 4.4 失败记录

实验失败时：

1. **错误日志**写入 `logs/{experiment_id}/error.log`
2. **结果文件** `results/{experiment_id}.json` 中包含：
   ```json
   {
     "experiment_id": "jobs_real_eps758",
     "status": "FAILED",
     "error": {
       "type": "RuntimeError",
       "message": "CUDA out of memory"
     },
     "stages": {
       "stage1": {"status": "COMPLETED", ...},
       "eval_large": {"status": "FAILED", ...}
     }
   }
   ```
3. **失败汇总**保存在 `results/summary_YYYYMMDD_HHMMSS.txt`

---

## 五、实验结果存放位置

### 5.1 完整输出目录结构

```
outputs/pretext_platform/
└── {experiment_id}/
    ├── resolved_config.json        # 解析后的完整配置
    ├── stage1_summary.json         # Stage1摘要
    ├── stage2_summary.json         # Stage2摘要
    ├── eval_large_summary.json     # 大模型评估摘要
    ├── glue_{task}_summary.json    # GLUE评估摘要（每个task一个）
    ├── metrics_summary.json        # 完整指标汇总
    │
    ├── stage1/                     # Stage1产物
    │   ├── private_embeds.npy     #   私有文本embedding
    │   ├── surviving_text_it0.json #  第0轮存活文本
    │   ├── surviving_text_it1.json #  第1轮存活文本
    │   └── ... (it2-it10)
    │
    ├── stage2/                     # Stage2产物
    │   └── llama7b_text_syn.json  #   50k条Bootstrap合成语料
    │
    ├── eval_large/                 # 大模型评估（eval_large）
    │   └── llama2_models_and_accuracies/
    │       ├── checkpoint0.pth    #   第0轮检查点
    │       ├── checkpoint1.pth    #   第1轮检查点
    │       ├── best_model.pth     #   最佳模型
    │       ├── best_stats.json    #   最佳指标
    │       └── epoch*_stats.json  #   每轮统计
    │
    └── eval_glue/                  # GLUE下游评估
        ├── glue_sst2_eval/         #   SST-2任务
        │   ├── best_model.pth
        │   ├── best_stats.json
        │   └── epoch*_stats.json
        ├── glue_qqp_eval/          #   QQP任务
        ├── glue_qnli_eval/         #   QNLI任务
        └── glue_summary.json       #   所有GLUE任务汇总
```

### 5.2 结果JSON文件

每个实验运行后，在 `results/` 目录生成：

**`results/{experiment_id}.json`** - 单个实验完整结果：
```json
{
  "experiment_id": "jobs_real_eps129",
  "description": "PrE-Text Jobs ε=1.29 (11 rounds)",
  "status": "SUCCESS",
  "start_time": "2026-04-02T10:00:00",
  "end_time": "2026-04-02T18:30:00",
  "total_elapsed_seconds": 30600,
  "output_dir": "outputs/pretext_platform/jobs_real_eps129",
  "log_dir": "logs/jobs_real_eps129",
  "stages": {
    "stage1": {
      "status": "COMPLETED",
      "epsilon": 1.29,
      "delta": 3e-6,
      "nsyn": 1024,
      "rounds": 11,
      "elapsed_hours": 6.5
    },
    "eval_large": {
      "status": "COMPLETED",
      "best_top1": 0.342,
      "epochs": 1
    }
  },
  "glue_results": {
    "sst2": {"best_accuracy": 0.85, "correct": 1700, "total": 2000},
    "qqp": {"best_accuracy": 0.78, "correct": 3120, "total": 4000},
    "qnli": {"best_accuracy": 0.72, "correct": 2880, "total": 4000}
  }
}
```

**`results/summary.json`** - 所有实验汇总：
```json
{
  "timestamp": "2026-04-03T12:00:00",
  "total": 16,
  "succeeded": 14,
  "failed": 2,
  "succeeded_experiments": [...],
  "failed_experiments": [...],
  "parallel": false,
  "with_glue": true
}
```

### 5.3 日志文件

**`logs/run_YYYYMMDD_HHMMSS.log`** - 批量运行完整日志：
```
============================================
EXPERIMENT: jobs_real_eps05
START: 2026-04-02T10:00:00+08:00
CONFIG: configs/experiments/jobs_real_eps05.yaml
============================================
...
END: 2026-04-02T16:30:00+08:00
ELAPSED: 23400s
EXIT CODE: 0
============================================

============================================
EXPERIMENT: jobs_real_eps129
START: 2026-04-02T16:30:00+08:00
CONFIG: configs/experiments/jobs_real_eps129.yaml
============================================
...
```

---

## 六、结果解读

### 6.1 Stage1 隐私参数

从 `stage1_summary.json` 查看：

```json
{
  "stage_name": "stage1",
  "metrics": {
    "epsilon": 1.29,
    "delta": 3e-6,
    "sigma_ratio": 11.3,
    "sensitivity": 8,
    "private_train_count": 1000,
    "nsyn": 1024,
    "rounds": 11
  }
}
```

### 6.2 大模型评估结果

从 `eval_large_summary.json` 查看：

```json
{
  "stage_name": "eval_large",
  "metrics": {
    "epochs": 1,
    "synthetic_train_count": 50000,
    "eval_count": 1000,
    "best_top1": 0.342
  }
}
```

### 6.3 GLUE下游评估结果

从 `eval_glue/glue_summary.json` 查看：

```json
{
  "sst2": {
    "best_accuracy": 0.852,
    "correct": 1704,
    "total": 2000
  },
  "qqp": {
    "best_accuracy": 0.782,
    "correct": 3128,
    "total": 4000
  },
  "qnli": {
    "best_accuracy": 0.721,
    "correct": 2884,
    "total": 4000
  }
}
```

---

## 七、故障排查

### 7.1 CUDA OOM（显存不足）

```
ERROR: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案**：
1. 减小 `batch_size`：
   ```yaml
   eval_large:
     batch_size: 4  # 从8减到4
     eval_batch_size: 1
   ```
2. 减少 `num_prompts`：
   ```yaml
   bootstrap:
     num_prompts: 20000  # 从50000减到20000
   ```

### 7.2 数据集文件未找到

```
FileNotFoundError: [Errno 2] No such file or directory: '../thesis_platform/datasets/jobs_train.json'
```

**解决方案**：检查 `paths.dataset_root` 配置，确保路径正确：
```bash
# 验证数据集存在
ls -la ../thesis_platform/datasets/pretext_jobs/formatted/
```

### 7.3 模型文件未找到

```
FileNotFoundError: Model not found at ../thesis_platform/open_model/llama_2_7b_hf
```

**解决方案**：
```bash
# 验证模型存在
ls -la ../thesis_platform/open_model/

# 确认模型目录名称
ls ../thesis_platform/open_model/ | grep llama
```

### 7.4 vLLM导入失败

```
ImportError: cannot import name 'LLM' from 'vllm'
```

**解决方案**：使用HuggingFace后端（自动回退）：
```yaml
bootstrap:
  generator_backend: huggingface
```

### 7.5 GLUE数据集本地文件缺失

```
Loaded sst2 from huggingface, 1821 samples
```
或
```
FileNotFoundError: No Arrow files in /path/to/glue_sst2/formatted/validation
```

**问题说明**：本地GLUE数据集文件缺失，代码会fallback到HuggingFace在线下载（需要网络）。

**解决方案**：
1. 预先验证本地GLUE数据集：
   ```bash
   python -m pretext_platform.scripts.run_glue --validate
   ```
2. 确保数据集已下载到正确位置：
   ```bash
   ls -la ../thesis_platform/datasets/glue_sst2/formatted/validation/
   ```

---

## 八、实验日程建议

基于实验设计文档（增强版）：

| 日期 | 实验内容 | 预计时长 |
|------|---------|---------|
| Day 1 | GC2 (Jobs) + GC4 (Congressional) | 各8h，并行16h |
| Day 2 | GC8 (Forums) + GC9 (Microblog) | 各8h，并行16h |
| Day 3 | GP1-GP4 (Jobs 隐私曲线) | 各8h，串行32h |
| Day 4-7 | Forums/Microblog 隐私实验 | 各8h |
| Day 8 | 跨域迁移实验 | 8h |
| Day 9-14 | GLUE下游评估 | 各0.5h |

**总实验时间**：约2周（考虑并行）

---

## 九、快速命令参考

```bash
# === 环境验证 ===
# 烟雾测试
bash run_all_experiments.sh --smoke

# 验证GLUE数据集本地文件
python -m pretext_platform.scripts.run_glue --validate

# === 并行执行 ===
# 并行运行所有实验（最多2个）
python -m pretext_platform.scripts.run_experiments --all --parallel

# 并行运行（最多4个）
python -m pretext_platform.scripts.run_experiments --all --parallel --max-parallel 4

# === 断点续跑 ===
# 跳过已完成的实验
python -m pretext_platform.scripts.run_experiments --all --resume

# 并行 + 续跑（推荐）
python -m pretext_platform.scripts.run_experiments --all --parallel --resume

# === 查看进度 ===
# 查看当前进度
cat results/progress.json

# 查看已完成的实验
cat results/summary.json

# === 查看结果 ===
# 查看实验输出
ls -la outputs/pretext_platform/jobs_real_eps129/

# 查看GLUE结果
cat outputs/pretext_platform/jobs_real_eps129/eval_glue/glue_summary.json

# === 查看日志 ===
# 查看错误日志
cat logs/jobs_real_eps129/error.log

# === 单独实验 ===
# 运行单个实验
python -m pretext_platform.scripts.run_pipeline \
    --config configs/experiments/jobs_real_eps129.yaml

# === GLUE评估 ===
# 运行所有GLUE任务
python -m pretext_platform.scripts.run_glue \
    --config configs/experiments/jobs_real_eps129.yaml \
    --tasks all

# === 跨域实验 ===
python -m pretext_platform.scripts.run_cross_domain \
    --source-config configs/experiments/jobs_real_eps129.yaml \
    --target-dataset forums

# === 多种子 ===
python -m pretext_platform.scripts.run_multi_seed \
    --config configs/experiments/jobs_real_eps129.yaml \
    --seeds 42 123 456
```
