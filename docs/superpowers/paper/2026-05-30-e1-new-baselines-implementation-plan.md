# E1 新增基线实现计划 · C4-only / Aug-PE / DP-Prompt

**日期**：2026-05-30  
**基于设计文档**：`docs/superpowers/specs/2026-05-30-e1-new-baselines-design.md`

---

## 总览表

| 方法 | 估计工作量 | 关键依赖 | 输出目录约定 |
|------|-----------|---------|------------|
| **C4-only** | 低（1-2天） | 独立构建完整 pipeline，参考 PrE-Text 模式实现随机采样 + LLaMA 扩充 + 评估；接入 round23 runner 框架 | `paper-new-round23/outputs/e1_c4only_seen_repeat30/` |
| **Aug-PE** | 中（3-5天） | 从 GitHub clone Aug-PE，必要时修改其源码兼容 pretext 环境；独立构建整体 pipeline，参考 PrE-Text Stage1+Stage2 模式 | `paper-new-round23/outputs/e1_augpe_seen_repeat30/` |
| **DP-Prompt** | 中（2-3天） | 基于现有 dp-prompt 代码独立构建 pipeline，参考 round23 runner 接口规范对齐评估链 | `paper-new-round23/outputs/e1_dpprompt_seen_repeat30/` |

**开发顺序**：先做 DP-Prompt（已有代码最多，收尾性工作），再做 C4-only（最简单），最后做 Aug-PE（需要外部依赖）。

**共用基础设施**（三个方法都要用，优先完成）：
1. `round23_dynamic_experiment_runner.py` 中注册三个新 method 名
2. `merge_thesis_e1_main_results.py` 中更新 `METHOD_ORDER`
3. 为每个新方法新增一个 `run_<method>.py` 入口脚本（对齐 `run_round23_keep_k0_baseline.py` 的结构）

---

## 阶段 0：共用基础设施

> 三个方法共享的 runner 注册和结果汇总扩展，在任何单方法开发开始之前完成。

### 0.1 扩展 `round23_dynamic_experiment_runner.py` 支持新 method 名

- [ ] 打开 `/Users/apple/Desktop/code_from_paper/paper-new-round23/scripts/round23_dynamic_experiment_runner.py`
- [ ] 在 `RUN_SCRIPTS` 字典（第 26 行附近）中新增三条：
  ```python
  "e1_c4only": ROUND23_ROOT / "scripts" / "run_e1_c4only_baseline.py",
  "e1_augpe":  ROUND23_ROOT / "scripts" / "run_e1_augpe_baseline.py",
  "e1_dpprompt": ROUND23_ROOT / "scripts" / "run_e1_dpprompt_baseline.py",
  ```
- [ ] 在 `sidecar_suffix_for_method()` 函数中新增三个 case：
  ```python
  if method == "e1_c4only":
      return "_c4only_runtime.json"
  if method == "e1_augpe":
      return "_augpe_runtime.json"
  if method == "e1_dpprompt":
      return "_dpprompt_runtime.json"
  ```
- [ ] 在 `MODE_PATHS` 字典中新增六个 mode（每个方法各两个：smoke + repeat30）：
  ```python
  "e1_c4only_seen_smoke": {
      "manifest_relpath": "e1_c4only_seen_smoke/round23_e1_c4only_seen_smoke_manifest.tsv",
      "log_stem": "round23_e1_c4only_seen_smoke",
      "dataset_split": "seen",
  },
  "e1_c4only_seen_repeat30": {
      "manifest_relpath": "e1_c4only_seen_repeat30/round23_e1_c4only_seen_repeat30_manifest.tsv",
      "log_stem": "round23_e1_c4only_seen_repeat30",
      "dataset_split": "seen",
  },
  "e1_augpe_seen_smoke": {
      "manifest_relpath": "e1_augpe_seen_smoke/round23_e1_augpe_seen_smoke_manifest.tsv",
      "log_stem": "round23_e1_augpe_seen_smoke",
      "dataset_split": "seen",
  },
  "e1_augpe_seen_repeat30": {
      "manifest_relpath": "e1_augpe_seen_repeat30/round23_e1_augpe_seen_repeat30_manifest.tsv",
      "log_stem": "round23_e1_augpe_seen_repeat30",
      "dataset_split": "seen",
  },
  "e1_dpprompt_seen_smoke": {
      "manifest_relpath": "e1_dpprompt_seen_smoke/round23_e1_dpprompt_seen_smoke_manifest.tsv",
      "log_stem": "round23_e1_dpprompt_seen_smoke",
      "dataset_split": "seen",
  },
  "e1_dpprompt_seen_repeat30": {
      "manifest_relpath": "e1_dpprompt_seen_repeat30/round23_e1_dpprompt_seen_repeat30_manifest.tsv",
      "log_stem": "round23_e1_dpprompt_seen_repeat30",
      "dataset_split": "seen",
  },
  ```
- [ ] 修改 `resolve_model_dir_for_spec()` 中的条件判断，对新 method 返回 `None`（这三个方法不使用 round23 controller）：
  ```python
  if spec.method in ("round23_keepk0", "e1_c4only", "e1_augpe", "e1_dpprompt"):
      return None
  ```

### 0.2 决策：结果汇总使用新脚本还是修改现有脚本

**不修改 `merge_thesis_e1_main_results.py`**。原因：

1. 该脚本的 `--mode` 参数只接受 `thesis_main_seen_pilot/repeat10/repeat30` 三个值（`MODE_TO_ROUND19_STEM` 硬编码），无法处理新方法产生的 `e1_c4only_seen_repeat30` 等 mode 名称
2. 新方法的 summary TSV 来自不同 runner mode，不共享同一个 round19 summary 输入

**实际操作**：
- [ ] 跳过此步骤，将新方法结果汇总工作完全推迟到阶段 5 的 `merge_thesis_e1_extended_results.py` 实现
- [ ] `merge_thesis_e1_main_results.py` 保持不变，仅用于现有方法（PrE-Text/round19/WASP/DPGA-TextSyn/round23）的历史结果查询

---

## 阶段 1：DP-Prompt（优先完成）

> 独立构建完整 pipeline：基于现有 dp-prompt 代码的核心改写逻辑，实现完整的"私有文档 LLM 改写 + 评估"流程。不直接调用 dp-prompt 内部 pipeline 函数，而是参考其实现，自行构建符合 round23 runner 接口规范的独立脚本。

### 1.1 审核现有 `dp-prompt` 代码，理解核心逻辑

- [ ] 阅读 `/Users/apple/Desktop/code_from_paper/dp-prompt/dp_prompt/runners/pretext_pipeline.py`：理解核心改写流程（prompt 构造、LLM 调用、结果收集方式）
- [ ] 阅读 `dp-prompt/dp_prompt/prompting/templates.py`：确认完整 prompt 模板（`"Review: {text}\nParaphrase of the review:"`）
- [ ] 阅读 dp-prompt 的 YAML config 结构：确认 `experiment.id`、`data.train_path`、`data.eval_path`、`generation`、`privacy` 等字段含义
- [ ] 阅读现有 dp-prompt eval 脚本，了解下游任务评估的调用方式
- [ ] 运行现有 smoke 测试（`python -m dp_prompt.cli --config configs/experiments/p1_jobs_pretext_style.yaml`），确认基础链路可跑

### 1.2 新建 `run_e1_dpprompt_baseline.py` 入口脚本

- [ ] 新建文件：`/Users/apple/Desktop/code_from_paper/paper-new-round23/scripts/run_e1_dpprompt_baseline.py`
- [ ] CLI 接口（必须与 `run_single_experiment()` 传入的参数对齐）：
  ```
  --config <path>
  --output-root <path>          # runner 传入绝对路径
  --timeout-seconds <int>
  --reference-budget <int>      # runner 始终追加此参数，脚本必须接受但可忽略
  ```
- [ ] 核心逻辑（独立实现，参考 dp-prompt 代码逻辑但不 import 其内部 pipeline 函数）：
  1. 自行实现 YAML 加载，解析 `experiment.id`、`data.dataset_name`、`data.train_path` 等字段（注意字段路径：`experiment.id`，不是 `meta.experiment_id`）
  2. 用 `json.load()` 加载私有训练文档
  3. 对每条私有文档构造 prompt（完整模板：`"Review: {text}\nParaphrase of the review:"`）
  4. 调用 vLLM 或 LLM API 生成改写文本（温度采样，参考 dp-prompt 的 generation 配置）
  5. 收集合成改写文本，写到 `{output_root}/synthetic/`
  6. **评估独立实现**：参考 PrE-Text/round23 eval 模式，自行调用下游任务评估，写出 `{output_root}/eval/downstream_eval_summary.json`，字段含 `best_top1`、`best_top3`、`best_top5`、`best_top10`
  7. 写出 sidecar JSON：`{output_root}/{experiment_id}_dpprompt_runtime.json`，格式：
     ```json
     {
       "budget_policy_type": "dpprompt",
       "dataset_name": "...",
       "experiment_id": "...",
       "synthetic_count": ...,
       "runtime_artifacts": {
         "runtime_output_root": "<output_root 绝对路径>",
         "eval_summary_path": "<downstream_eval_summary.json 绝对路径>",
         "eval_summary": {
           "metrics": {
             "best_top1": <float>,
             "best_top3": <float>,
             "best_top5": <float>,
             "best_top10": <float>
           }
         }
       }
     }
     ```
- [ ] 关键接口对齐：sidecar 中 `eval_summary.metrics.best_top1` 可被 `extract_eval_metric()` 正确读取（该函数先查顶层键、再查 `metrics` 子字典），**无需额外扁平化**

### 1.3 为 E1 新增 DP-Prompt 专用 config YAML

DP-Prompt 不走 round23/round19 的 selector 流水线，而是走独立的 `dp_prompt` pipeline。因此它的 config 格式与 round23 configs 不同，采用 dp-prompt 已有的 YAML 格式。

**注意：现有 4 个数据集的 config 已经存在**（`pretext_congressional.yaml`、`pretext_forums.yaml`、`pretext_microblog.yaml`、`pretext_jobs.yaml`），位于 `dp-prompt/configs/datasets/`，可以直接复用，无需新建。

- [ ] 新建 base config：`/Users/apple/Desktop/code_from_paper/dp-prompt/configs/base/e1_dpprompt_e1_base.yaml`
  - 内容：从同目录的 `pretext_style_base.yaml` 继承（`inherits` 路径相对于本文件所在目录解析，写 `pretext_style_base.yaml` 即可），并设定实验专用参数：
    ```yaml
    inherits:
      - pretext_style_base.yaml

    experiment:
      pipeline_mode: pretext_style

    generation:
      num_documents: 256

    privacy:
      temperature: 0.7
    ```
  - 注：temperature=0.7 是 DP-Prompt 论文中文档级改写的推荐参数；如需 DP 约束，可在后续加入 logits_clipping
- [ ] 无需新建 4 个数据集 config，直接复用现有文件：
  - `dp-prompt/configs/datasets/pretext_congressional.yaml`
  - `dp-prompt/configs/datasets/pretext_forums.yaml`
  - `dp-prompt/configs/datasets/pretext_microblog.yaml`
  - `dp-prompt/configs/datasets/pretext_jobs.yaml`

### 1.4 新建 DP-Prompt 实验 config 生成器

- [ ] 新建文件：`/Users/apple/Desktop/code_from_paper/paper-new-round23/scripts/generate_e1_dpprompt_experiment_configs.py`
- [ ] 文件结构仿照 `generate_round23_experiment_configs.py`，但产出格式为 dp-prompt YAML（不继承 round23 base）：
  - `CONFIG_ROOT` 指向 `paper-new-round23/configs/experiments/single_node_tuning_round23_dynamic`
  - 支持两个 mode：`e1_dpprompt_seen_smoke`（seed=[42]）、`e1_dpprompt_seen_repeat30`（SEEDS_REPEAT30）
  - 每个实验 config 格式（**注意 `inherits` 路径必须是相对于生成的 config 文件位置的相对路径**；config 保存在 `configs/experiments/single_node_tuning_round23_dynamic/e1_dpprompt_seen_*/`，需要 6 层 `../` 才能到达 `code_from_paper/dp-prompt/`，建议写绝对路径或在生成器中用 `Path` 计算正确的相对路径）：
    ```yaml
    inherits:
      - ../../../../../../dp-prompt/configs/base/e1_dpprompt_e1_base.yaml
      - ../../../../../../dp-prompt/configs/datasets/pretext_<dataset>.yaml

    experiment:
      id: e1_dpprompt_<dataset>_seed<N>

    runtime:
      seed: <N>
      output_root: outputs/e1_dpprompt_seen_repeat30/<dataset>/seed<N>
    ```
    注意：`runtime.output_root` 是相对路径（相对于 CWD，即 `paper-new-round23`），runner 传给脚本的 `--output-root` 是绝对路径，脚本用 CLI 参数覆盖 config 中的 output_root（见 1.2 步骤 4），config 中的值仅作为备用默认值
  - manifest TSV 字段：`experiment_id`, `dataset`, `seed`, `config_path`, `output_root`, `method`（固定为 `e1_dpprompt`）
  - manifest 保存到：`single_node_tuning_round23_dynamic/e1_dpprompt_seen_repeat30/round23_e1_dpprompt_seen_repeat30_manifest.tsv`
  - manifest 的 `output_root` 字段使用相对路径（如 `outputs/e1_dpprompt_seen_repeat30/jobs/seed42`），runner 会将其转换为绝对路径后传给脚本

### 1.5 验证 DP-Prompt 单实验端到端可跑

- [ ] 手动运行一个 smoke 实验，确认脚本可在 GPU 上跑通：
  ```bash
  cd paper-new-round23
  python scripts/generate_e1_dpprompt_experiment_configs.py --mode e1_dpprompt_seen_smoke
  python scripts/run_e1_dpprompt_baseline.py \
    --config configs/experiments/single_node_tuning_round23_dynamic/e1_dpprompt_seen_smoke/e1_dpprompt_jobs_seed42.yaml \
    --output-root outputs/e1_dpprompt_seen_smoke/jobs/seed42
  ```
- [ ] 确认 output 目录结构：
  - `outputs/e1_dpprompt_seen_smoke/jobs/seed42/e1_dpprompt_jobs_seed42_dpprompt_runtime.json` 存在
  - runtime JSON 中 `runtime_artifacts.eval_summary.best_top1` 有数值

### 1.6 验证 DP-Prompt 通过 runner 批量可跑

- [ ] 运行 smoke batch：
  ```bash
  python paper-new-round23/scripts/round23_dynamic_experiment_runner.py \
    --mode e1_dpprompt_seen_smoke \
    --dry-run
  ```
  确认 pending 实验列表正确（4个，每个数据集1个seed）
- [ ] 去掉 `--dry-run` 实际跑 smoke，确认 summary TSV 写出正确

---

## 阶段 2：C4-only（最简单）

> 独立构建完整 pipeline：去掉 Stage1 的 DP histogram，随机从 C4 pool 采样，然后 LLaMA 扩充 + 评估。参考 PrE-Text 代码的实现模式，不 import pretext_platform 内部模块。

### 2.1 阅读 PrE-Text 代码理解实现模式

- [ ] 阅读 `/Users/apple/Desktop/code_from_paper/PrE-Text/pretext_platform/algorithms/stage1.py`：了解 Stage1 的输入/输出数据格式（`initialization.json` 读取方式、`surviving_text_it0.json` 写出格式）
- [ ] 阅读 `/Users/apple/Desktop/code_from_paper/PrE-Text/pretext_platform/algorithms/bootstrap.py`：了解 Stage2 的输入（surviving seeds list）、LLaMA few-shot prompt 构造方式、输出文件格式
- [ ] 阅读 PrE-Text 中 c4-only 相关脚本（如有），确认 C4 pool 的文件路径和读取方式

**目标**：理解数据流和文件格式，然后在 `run_e1_c4only_baseline.py` 中从零独立实现，不 import pretext_platform 任何模块。

### 2.2 新建 `run_e1_c4only_baseline.py` 入口脚本

- [ ] 新建文件：`/Users/apple/Desktop/code_from_paper/paper-new-round23/scripts/run_e1_c4only_baseline.py`
- [ ] CLI 接口（**必须包含 `--reference-budget`**，因为 `run_single_experiment()` 始终追加此参数）：
  ```
  --config <path>
  --output-root <path>
  --timeout-seconds <int>
  --reference-budget <int>      # runner 始终传入，脚本必须接受；此处用作 seed_top_k 默认值
  --seed-top-k <int>            # default=20
  ```
- [ ] 核心逻辑（独立实现，参考 PrE-Text 模式但不 import 其模块）：
  1. 自行实现 YAML 加载（`PyYAML` + `inherits` 递归合并），解析 `meta.experiment_id`、`meta.seed`、`meta.dataset_name`
  2. 从 config 中读取 `data.initialization_path`，用 `json.load()` 读取 C4 initialization pool，按 min_words 过滤短文本
  3. `random.seed(meta.seed)`，然后 `selected = random.choices(init_pool, k=nsyn)`（nsyn 按 PrE-Text 的 batch_size × multiplier 约定设置，具体值阅读 PrE-Text 代码确认）
  4. 将 `selected` 写出为 `{output_root}/stage1/surviving_text_it0.json`（`json.dump(list_of_str, ...)`）
  5. 构造伪 `stage1_summary.json`（写到 `{output_root}/stage1_summary.json`）：
     ```json
     {
       "stage_name": "stage1",
       "output_dir": "<output_root>/stage1",
       "artifacts": {
         "surviving_files": ["<output_root>/stage1/surviving_text_it0.json"]
       },
       "metrics": {"epsilon": "inf", "c4only": true}
     }
     ```
  6. **Stage2 独立实现**：参考 PrE-Text `bootstrap.py` 的 LLaMA few-shot 逻辑，自行实现：读取 surviving seeds → 构造 3-shot prompt → 调用 vLLM API 生成 → 写出合成样本到 `{output_root}/stage2/`
  7. **评估独立实现**：参考 PrE-Text eval 脚本，自行调用下游任务评估，写出 `{output_root}/eval/downstream_eval_summary.json`，字段含 `best_top1`、`best_top3`、`best_top5`、`best_top10`
  8. 写出 sidecar：`{output_root}/{experiment_id}_c4only_runtime.json`，格式：
     ```json
     {
       "budget_policy_type": "c4only",
       "seed_top_k": 20,
       "epsilon": "inf",
       "privacy_note": "no DP, random sample from public C4 pool",
       "runtime_artifacts": {
         "runtime_output_root": "...",
         "stage1_summary_path": "...",
         "eval_summary_path": "...",
         "eval_summary": {
           "metrics": {
             "best_top1": <float>,
             "best_top3": <float>,
             "best_top5": <float>,
             "best_top10": <float>
           }
         }
       }
     }
     ```

### 2.3 新建 C4-only 实验 config 生成器

- [ ] 新建文件：`/Users/apple/Desktop/code_from_paper/paper-new-round23/scripts/generate_e1_c4only_experiment_configs.py`
- [ ] 结构仿照 `generate_round23_experiment_configs.py`，但生成的 config 不继承 round23 base，而是继承 round23 的 `_base_selector_tuning_round23_dynamic.yaml`（数据路径复用）或者直接继承 round19 的 base：
  ```yaml
  inherits:
    - ../_base_selector_tuning_round23_dynamic.yaml
    - ../_data_<dataset>.yaml

  meta:
    experiment_id: e1_c4only_<dataset>_seed<N>
    seed: <N>
    dataset_name: <dataset>

  paths:
    output_root: outputs/e1_c4only_seen_repeat30/<dataset>/seed<N>

  c4only_baseline:
    seed_top_k: 20
    source_env: e1_c4only_seen_repeat30
  ```
- [ ] 支持 mode：`e1_c4only_seen_smoke`（seed=[42]）、`e1_c4only_seen_repeat30`（SEEDS_REPEAT30）
- [ ] 数据集固定为 `SEEN_DATASETS = ["jobs", "congressional", "forums", "microblog"]`
- [ ] manifest 字段：`experiment_id`, `dataset`, `seed`, `config_path`, `output_root`, `method`（固定 `e1_c4only`）
- [ ] experiment prefix：`e1_c4only_<dataset>_seed<N>`

### 2.4 验证 C4-only 单实验端到端可跑

- [ ] 手动运行 smoke 实验，确认采样逻辑正确（20 条来自 C4 pool，不是私有数据）
- [ ] 检查 Stage2 是否正常衔接（`llama7b_text_syn.json` 是否生成）
- [ ] 检查 eval 是否输出 `best_top1` 数值
- [ ] 检查 sidecar JSON 格式完整性

### 2.5 验证 C4-only 通过 runner 批量可跑

- [ ] `--dry-run` 确认 manifest 加载正确
- [ ] 实跑 smoke batch（4 实验），确认 summary TSV 写出

---

## 阶段 3：Aug-PE（最复杂，需要外部代码）

> 独立构建完整 pipeline，Stage1 使用 Aug-PE 的 PE 算法，Stage2/评估参考 PrE-Text 模式自行实现。以 pretext conda 环境为主，若依赖冲突则修改 Aug-PE 源码。

### 3.1 获取 Aug-PE 代码并分析

- [ ] 在服务器上执行（先在本地验证 repo 名正确）：
  ```bash
  cd /mnt/public/caiqiyue_file/code_from_paper
  git clone https://github.com/AI-secure/aug-pe aug-pe
  ```
- [ ] 阅读 `aug-pe/README.md`，了解：依赖列表、核心调用方式
- [ ] 阅读 Aug-PE 核心算法文件，重点找：
  - **DP histogram 选种子的核心函数**（`private_evolution` 或 `pe_step` 类似函数）
  - **embedding 计算函数**（私有数据与 synthetic population 的最近邻距离）
  - 函数输入：私有文本列表、合成候选文本列表、epsilon/delta 参数
  - 函数输出：选出的高质量 seed 文本列表
- [ ] 检查 `requirements.txt` / `setup.py`，找出可能与 pretext 环境冲突的依赖

### 3.2 解决依赖冲突（以 pretext 环境为主）

- [ ] 在服务器 pretext 环境中尝试 import Aug-PE 核心模块，记录报错
- [ ] **若有冲突**：直接修改 Aug-PE 源码中冲突的依赖调用，替换为 pretext 环境中等效的 API（保持核心算法逻辑不变）：
  - 常见冲突：`sentence-transformers` 版本差异 → 替换为 pretext 环境中已有的 embedding 调用方式
  - `transformers` 版本差异 → 替换为兼容 API
  - 修改范围：仅限于依赖调用层，不改动 PE 算法本身的数学逻辑
- [ ] 确认在 pretext 环境中 Aug-PE 核心函数可正常 import 和调用

### 3.3 创建 `aug_pe_adapter.py` 适配层

- [ ] 新建文件：`/Users/apple/Desktop/code_from_paper/paper-new-round23/scripts/aug_pe_adapter.py`
- [ ] 提供统一接口：
  ```python
  def run_augpe_seed_selection(
      *,
      private_texts: list[str],
      init_pool: list[str],
      seed_top_k: int,
      epsilon: float,
      delta: float,
      seed: int,
      augpe_repo_root: str | Path,
  ) -> list[str]:
      """
      调用 Aug-PE 的 PE 核心，从 init_pool 中选出 seed_top_k 条高质量 seeds。
      返回值：selected_seeds，list[str]，长度为 seed_top_k。
      """
      ...
  ```
- [ ] 通过 `sys.path.insert()` 动态导入 Aug-PE 代码（代码改动后应可正常 import）
- [ ] embedding 模型：指定使用与 PrE-Text 一致的 minilm（路径从 pretext 环境确认），确保比较公平
- [ ] 做参数范围检查（`epsilon > 0`，`seed_top_k >= 1`）

### 3.4 新建 `run_e1_augpe_baseline.py` 入口脚本

- [ ] 新建文件：`/Users/apple/Desktop/code_from_paper/paper-new-round23/scripts/run_e1_augpe_baseline.py`
- [ ] CLI 接口（**必须包含 `--reference-budget`**，runner 始终追加此参数）：
  ```
  --config <path>
  --output-root <path>
  --augpe-repo-root <path>      (Aug-PE repo 路径，默认 ../aug-pe)
  --epsilon <float>             (default=1.0，与其他方法对齐)
  --timeout-seconds <int>
  --reference-budget <int>      (runner 始终传入，脚本必须接受但可忽略)
  ```
- [ ] 核心流程（独立实现，参考 PrE-Text 模式但不 import 其模块）：
  1. 自行实现 YAML 加载，解析 `meta.experiment_id`、`meta.seed`、`meta.dataset_name`
  2. 自行加载私有训练数据（`json.load(train_path)`，参考 PrE-Text 数据加载方式）
  3. 自行加载 C4 initialization pool（`json.load(initialization_path)`，按 min_words 过滤）
  4. 调用 `aug_pe_adapter.run_augpe_seed_selection()` 获取 seed 列表
  5. 将 seeds 写出为 `{output_root}/stage1/augpe_selected_seeds.json` 和 `surviving_text_it0.json`
  6. 构造伪 `stage1_summary.json`（写到 `{output_root}/stage1_summary.json`，格式同 C4-only）
  7. **Stage2 独立实现**：同 C4-only，参考 PrE-Text bootstrap 模式自行实现 LLaMA few-shot 扩充
  8. **评估独立实现**：同 C4-only，自行调用下游任务评估，写出 `downstream_eval_summary.json`
  9. 写出 sidecar：`{output_root}/{experiment_id}_augpe_runtime.json`，格式：
     ```json
     {
       "budget_policy_type": "augpe",
       "seed_top_k": 20,
       "epsilon": 1.0,
       "augpe_repo_root": "...",
       "selected_seed_count": 20,
       "runtime_artifacts": {
         "runtime_output_root": "...",
         "stage1_summary_path": "...",
         "eval_summary_path": "...",
         "eval_summary": {
           "metrics": {
             "best_top1": <float>,
             "best_top3": <float>,
             "best_top5": <float>,
             "best_top10": <float>
           }
         }
       }
     }
     ```

### 3.5 新建 Aug-PE 实验 config 生成器

- [ ] 新建文件：`/Users/apple/Desktop/code_from_paper/paper-new-round23/scripts/generate_e1_augpe_experiment_configs.py`
- [ ] 结构与 `generate_e1_c4only_experiment_configs.py` 完全一致，只替换 method 名和 prefix：
  - experiment prefix：`e1_augpe_<dataset>_seed<N>`
  - method：`e1_augpe`
  - mode：`e1_augpe_seen_smoke`、`e1_augpe_seen_repeat30`
  - 额外 config 字段：
    ```yaml
    augpe_baseline:
      seed_top_k: 20
      epsilon: 1.0
      delta: 1e-5
      source_env: e1_augpe_seen_repeat30
    ```

### 3.6 验证 Aug-PE 端到端可跑

- [ ] 手动跑 smoke 实验（1 个实验）
- [ ] 确认 Aug-PE PE 选种子步骤正常（check `augpe_selected_seeds.json` 内容）
- [ ] 确认 Stage2 扩充正常（`llama7b_text_syn.json` 生成）
- [ ] 确认 eval 输出 `best_top1` 数值
- [ ] 通过 runner 批量 `--dry-run` 验证

---

## 阶段 4：正式实验运行

### 4.1 新建 E1 新基线完整运行脚本

- [ ] 新建文件：`/Users/apple/Desktop/code_from_paper/paper-new-round23/scripts/run_e1_new_baselines_repeat30_sequential.sh`
- [ ] 格式仿照 `run_e7_all_modes.sh`：
  ```bash
  #!/usr/bin/env bash
  # E1 新增基线实验顺序启动脚本 — C4-only / Aug-PE / DP-Prompt，共 3x4x30=360 个实验
  # GPU: A6000 (index 1), 环境: pretext

  BASE="/mnt/public/caiqiyue_file/code_from_paper"
  PYTHON_BIN="/home/k8smaster/anaconda3/envs/pretext/bin/python"
  RUNNER="$BASE/paper-new-round23/scripts/round23_dynamic_experiment_runner.py"

  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1

  cd "$BASE"

  COMMON_ARGS="--target-gpu-index 1 --min-free-gb-for-vllm 26 --gpu-wait-poll-seconds 60 --gpu-wait-timeout-seconds 43200 --max-attempts 3 --retry-delay-seconds 10"

  run_mode() {
      local mode=$1
      echo "========================================"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 mode: $mode"
      echo "========================================"
      "$PYTHON_BIN" "$RUNNER" --mode "$mode" $COMMON_ARGS
      local exit_code=$?
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] mode $mode 完成，exit_code=$exit_code"
      return $exit_code
  }

  # 先生成 configs
  "$PYTHON_BIN" "$BASE/paper-new-round23/scripts/generate_e1_dpprompt_experiment_configs.py" --mode e1_dpprompt_seen_repeat30
  "$PYTHON_BIN" "$BASE/paper-new-round23/scripts/generate_e1_c4only_experiment_configs.py" --mode e1_c4only_seen_repeat30
  "$PYTHON_BIN" "$BASE/paper-new-round23/scripts/generate_e1_augpe_experiment_configs.py" --mode e1_augpe_seen_repeat30

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] E1 新基线实验启动，共 360 个实验"

  run_mode e1_dpprompt_seen_repeat30
  run_mode e1_c4only_seen_repeat30
  run_mode e1_augpe_seen_repeat30

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部 E1 新基线实验完成！"
  ```

### 4.2 新建 smoke 验证脚本

- [ ] 新建文件：`/Users/apple/Desktop/code_from_paper/paper-new-round23/scripts/run_e1_new_baselines_smoke_sequential.sh`
- [ ] 格式同上，只跑 smoke mode（每方法 4 实验），用于上线前验证

### 4.3 注意：新方法汇总不使用 `merge_thesis_e1_main_results.py`

新方法的结果汇总在阶段 5 的 `merge_thesis_e1_extended_results.py` 中完成（见下方）。`merge_thesis_e1_main_results.py` 的 `--mode` 接口与新方法的 mode 名称不兼容，**不在此步骤修改该文件**。

---

## 阶段 5：结果汇总与集成

### 5.1 新建 E1 扩展结果合并脚本

- [ ] 新建文件：`/Users/apple/Desktop/code_from_paper/paper-new-round23/scripts/merge_thesis_e1_extended_results.py`
- [ ] 该脚本从以下 TSV 读取结果并合并输出（所有 TSV 均为 round23 runner 写出的标准 summary TSV，字段包含 `experiment_id`, `method`, `dataset_name`, `meta_seed`, `status`, `best_top1` ~ `best_top10`）：
  - 原有 round19 summary TSV（PrE-Text / WASP / DPGA-TextSyn / round19）：用现有的 `_normalize_round19()` 函数处理
  - 原有 round23 summary TSV（round23）：用现有的 `_normalize_round23()` 函数处理
  - 新增 c4only summary TSV（`logs/round23_e1_c4only_seen_repeat30_summary.tsv`）：新建 `_normalize_c4only(row)` 函数，读取 `row["best_top1"]` 等字段，设 `method="C4-only"`, `Method="C4-only"`
  - 新增 augpe summary TSV（`logs/round23_e1_augpe_seen_repeat30_summary.tsv`）：新建 `_normalize_augpe(row)` 函数，同上，`Method="Aug-PE"`
  - 新增 dpprompt summary TSV（`logs/round23_e1_dpprompt_seen_repeat30_summary.tsv`）：新建 `_normalize_dpprompt(row)` 函数，同上，`Method="DP-Prompt"`
  - 注：三个新方法的 summary TSV 由 `round23_dynamic_experiment_runner.py` 统一格式写出，可直接读取顶层 `best_top1` 等字段（runner 的 `build_summary_row()` 已调用 `extract_eval_metric()` 将指标提升到顶层）
- [ ] CLI 参数（全部可选，有默认路径）：
  ```
  --mode                  (默认 thesis_main_seen_repeat30)
  --round19-summary       (默认 paper-new-round19/logs/thesis_e1_main_seen_repeat30_summary.tsv)
  --round23-summary       (默认 paper-new-round23/logs/round23_thesis_main_seen_repeat30_summary.tsv)
  --c4only-summary        (默认 paper-new-round23/logs/round23_e1_c4only_seen_repeat30_summary.tsv)
  --augpe-summary         (默认 paper-new-round23/logs/round23_e1_augpe_seen_repeat30_summary.tsv)
  --dpprompt-summary      (默认 paper-new-round23/logs/round23_e1_dpprompt_seen_repeat30_summary.tsv)
  --output-dir            (默认 paper-new-round23/logs)
  ```
- [ ] 输出两个文件：
  - `{log_dir}/e1_extended_method_dataset_summary.tsv`：每行一个 (method, dataset) 对，包含均值指标
  - `{log_dir}/e1_extended_paper_table.tsv`：论文格式的对比表（方法为行，数据集为列）
- [ ] `METHOD_ORDER` 固定为：`("PrE-Text", "round19", "WASP", "DPGA-TextSyn", "round23", "C4-only", "Aug-PE", "DP-Prompt")`
- [ ] `_aggregate()` 和 `_build_paper_table()` 逻辑直接复用 `merge_thesis_e1_main_results.py` 中的同名函数，无需重写

### 5.2 验证汇总结果正确

- [ ] 运行合并脚本，检查所有 8 个方法都出现在输出表中
- [ ] 检查 C4-only 和 Aug-PE 的 `best_top1` 数值合理（C4-only 应低于其他有 DP 算法的方法）
- [ ] 检查 DP-Prompt 的数值在合理范围内

---

## 关键接口约定（工程师必读）

### Sidecar JSON 格式统一规范

每个新方法的 `run_e1_*.py` 脚本都必须写出 sidecar JSON，格式要求：

C4-only 和 Aug-PE（使用 `collect_runtime_artifacts()`）的 sidecar 格式：
```json
{
  "budget_policy_type": "<method_name>",
  "runtime_artifacts": {
    "runtime_output_root": "<绝对路径>",
    "stage1_summary_path": "<绝对路径>",
    "eval_summary_path": "<绝对路径>",
    "eval_summary": {
      "metrics": {
        "best_top1": <float>,
        "best_top3": <float>,
        "best_top5": <float>,
        "best_top10": <float>
      }
    }
  }
}
```

DP-Prompt（手动构造 artifacts）的 sidecar 格式：
```json
{
  "budget_policy_type": "dpprompt",
  "runtime_artifacts": {
    "runtime_output_root": "<pipeline_output_dir 绝对路径>",
    "eval_summary_path": "<eval_dir/eval_small_summary.json 绝对路径>",
    "eval_summary": {
      "metrics": {
        "best_top1": <float>,
        "best_top3": <float>,
        "best_top5": <float>,
        "best_top10": <float>
      }
    }
  }
}
```

这个格式是 `build_summary_row()` 函数（在 `round23_dynamic_experiment_runner.py` 中）读取的契约。具体提取逻辑在 `extract_eval_metric()` 函数（`round23_runtime_utils.py` 第 117 行）：它先找顶层字段，再找 `metrics` 嵌套字段。**`best_top1` 放在 `metrics` 子字典中可被正常提取**。不要将 `best_top1` 等字段放在 sidecar 顶层（那是 round23 controller 特有字段）。

### Manifest TSV 字段规范

新方法的 manifest 与现有 manifest 格式完全一致，但 `method` 字段使用新名称：

```
experiment_id\tdataset\tseed\tconfig_path\toutput_root\tmethod
e1_dpprompt_jobs_seed42\tjobs\t42\t<相对路径>\t<相对路径>\te1_dpprompt
```

`config_path` 用相对于 `paper-new-round23` 根目录的 POSIX 路径（参照现有 manifest 格式）。

### Config YAML 命名规范

新方法的 experiment config 文件命名：
- `e1_<method>_<dataset>_seed<N>.yaml`
- 保存在：`configs/experiments/single_node_tuning_round23_dynamic/e1_<method>_seen_<scale>/`

例：`e1_c4only_jobs_seed42.yaml`，保存在 `e1_c4only_seen_repeat30/`。

### 评估流水线差异说明

三个新方法的评估文件路径**不完全相同**：

- **C4-only / Aug-PE**：通过 `pretext_platform.core.pipeline.run_eval_small()` 评估，写出 `{output_root}/{experiment_id}/eval_small/` 或类似路径；随后调用 `collect_runtime_artifacts(args.output_root)` 自动定位 `downstream_eval_summary.json` 或 `summary.json`
- **DP-Prompt**：pipeline 写出 `{pipeline_output_dir}/eval/eval_small_summary.json`；**不调用 `collect_runtime_artifacts()`**，而是手动构造 artifacts dict，直接引用 `eval_small_summary.json`

两条路径最终都通过 sidecar 的 `runtime_artifacts.eval_summary.metrics.best_top1` 向 runner 暴露指标，`extract_eval_metric()` 可正常读取。

### 随机种子隔离

- C4-only 的 `random.sample()` 必须先 `random.seed(meta.seed)` 再采样，确保跨种子可重复
- Aug-PE 的 `run_augpe_seed_selection()` 必须传入 `seed` 参数，内部初始化所有 RNG
- DP-Prompt 的生成温度通过 config 控制，不额外设种子（与原始 DP-Prompt 论文一致）

---

## Aug-PE 移植特别说明

Aug-PE 的 GitHub repo（`AI-secure/aug-pe`）是独立项目。接入时需注意以下几点：

1. **以 pretext conda 环境为主**：所有实验在服务器 pretext 虚拟环境中运行，Aug-PE 必须在该环境中可运行。**不使用 subprocess 隔离**。

2. **若有依赖冲突，直接修改 Aug-PE 源码**：若 `sentence-transformers`、`transformers` 等版本与 pretext 环境不兼容，直接修改 Aug-PE 源码中冲突的调用（替换为 pretext 环境中已有的等效 API）。修改原则：只改依赖调用层，保证 PE 算法的核心数学逻辑不变（DP histogram 选种子的逻辑不能改）。

3. **适配逻辑放在 `aug_pe_adapter.py`**：在适配层通过 `sys.path.insert()` 动态导入修改后的 Aug-PE 代码，对外提供统一接口。

4. **embedding 模型对齐**：在 `aug_pe_adapter.py` 中明确指定使用与 PrE-Text 一致的 minilm 模型，保证评估公平。

5. **ε 值对齐**：Aug-PE 实验的 ε 与 PrE-Text 等方法保持一致，在 config 中设置 `augpe_baseline.epsilon: 1.0`（以项目标准值为准），并在 `aug_pe_adapter.py` 中验证参数传递正确。

---

## 文件路径速查表

| 文件 | 类型 | 位置 |
|------|------|------|
| `run_e1_dpprompt_baseline.py` | 新建 | `paper-new-round23/scripts/` |
| `run_e1_c4only_baseline.py` | 新建 | `paper-new-round23/scripts/` |
| `run_e1_augpe_baseline.py` | 新建 | `paper-new-round23/scripts/` |
| `aug_pe_adapter.py` | 新建 | `paper-new-round23/scripts/` |
| `generate_e1_dpprompt_experiment_configs.py` | 新建 | `paper-new-round23/scripts/` |
| `generate_e1_c4only_experiment_configs.py` | 新建 | `paper-new-round23/scripts/` |
| `generate_e1_augpe_experiment_configs.py` | 新建 | `paper-new-round23/scripts/` |
| `merge_thesis_e1_extended_results.py` | 新建 | `paper-new-round23/scripts/` |
| `run_e1_new_baselines_smoke_sequential.sh` | 新建 | `paper-new-round23/scripts/` |
| `run_e1_new_baselines_repeat30_sequential.sh` | 新建 | `paper-new-round23/scripts/` |
| `round23_dynamic_experiment_runner.py` | 修改 | `paper-new-round23/scripts/` |
| `merge_thesis_e1_main_results.py` | **不修改** | `paper-new-round23/scripts/` |
| `e1_dpprompt_e1_base.yaml` | 新建 | `dp-prompt/configs/base/` |
| `pretext_congressional.yaml` | **已存在，复用** | `dp-prompt/configs/datasets/` |
| `pretext_forums.yaml` | **已存在，复用** | `dp-prompt/configs/datasets/` |
| `pretext_microblog.yaml` | **已存在，复用** | `dp-prompt/configs/datasets/` |
| `pretext_jobs.yaml` | **已存在，复用** | `dp-prompt/configs/datasets/` |
| config YAMLs (x30x4 per method) | 生成 | `paper-new-round23/configs/experiments/single_node_tuning_round23_dynamic/e1_*/` |
| manifest TSVs (6 个) | 生成 | 同上 |

---

## 依赖关系图

```
阶段0（runner 扩展：RUN_SCRIPTS / MODE_PATHS / sidecar_suffix / resolve_model_dir_for_spec）
    ↓
阶段1（DP-Prompt）← 依赖: dp-prompt 现有代码, thesis_platform 评估链;
                     不调用 collect_runtime_artifacts()，手动构造 artifacts
    ↓
阶段2（C4-only）← 依赖: pretext_platform.core.pipeline.run_bootstrap/run_eval_small;
                   写 stage1_summary.json 后可调用 collect_runtime_artifacts()
    ↓
阶段3（Aug-PE）← 依赖: aug-pe clone, aug_pe_adapter.py;
                  同 C4-only，调用 pretext_platform.core.pipeline.run_bootstrap/run_eval_small
    ↓
阶段4（实验运行）← 依赖: 阶段0 完成 + 阶段1-3 全部完成
    ↓
阶段5（结果汇总）← 依赖: 阶段4 运行完毕
```

阶段1/2/3 之间无强依赖，但推荐按顺序开发（DP-Prompt 最快反馈，C4-only 最简单作为框架验证，Aug-PE 留最后处理外部依赖问题）。
