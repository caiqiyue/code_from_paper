# Round 5 方向 1 实施计划：gate 网格扩展（g5/g6/g7）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `paper-new/` 内新增 3 组 gate 配置（g5/g6/g7）共 12 个实验，纯 config 扩展，不改算法代码；通过自动化 runner 串行执行 12 个实验并把结果整理回 Round 4 文档第十一节。

**Architecture:** 完全复用 Round 4 已验证的 selector pipeline（`paper_new_selector/run_selector_single_node`）和 yaml 继承体系。新建独立目录 `single_node_tuning_round4_ext/` 存放 1 个 base + 3 个组 + 12 个叶子配置；新建 `old_automation/run_round4_ext_queue.py` 作为串行 runner（基于 Round 4 runner 模板）。

**Tech Stack:** YAML config inheritance, Python subprocess runner, tmux session management, A6000 GPU on remote server (CUDA_VISIBLE_DEVICES=1)。

**前置依赖：**
- Spec：`paper-new/docs/2026-04-27-round5-dual-track-design.md` §2
- Round 4 base 配置：`paper-new/configs/experiments/single_node_tuning_round4/_base_selector_tuning_round4.yaml`
- 已有 runner 模板：`old_automation/run_round4_queue.py`（在远端服务器，本地无副本）
- 远端服务器路径：`/mnt/public/caiqiyue_file/code_from_paper/`
- conda 环境：`pretext`，Python 路径 `/home/k8smaster/anaconda3/envs/pretext/bin/python`

---

## Task 1: 新建目录与基础配置文件

**Files:**
- Create: `paper-new/configs/experiments/single_node_tuning_round4_ext/_base_selector_tuning_round4_ext.yaml`

- [ ] **Step 1: 新建目录**

```bash
mkdir -p /Users/apple/Desktop/code_from_paper/paper-new/configs/experiments/single_node_tuning_round4_ext
```

- [ ] **Step 2: 写 base 配置**

文件 `paper-new/configs/experiments/single_node_tuning_round4_ext/_base_selector_tuning_round4_ext.yaml`：

```yaml
inherits:
  - ../single_node_tuning_round4/_base_selector_tuning_round4.yaml

meta:
  stage: single_node_tuning_round4_ext
```

只覆盖 `meta.stage` 字段，其余全部继承自 Round 4 base（包括 256/256/1024 数据规模、6 epochs、gpt2 small、所有 selector 参数）。

- [ ] **Step 3: 验证 base 配置可解析**

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new
python -c "import yaml; print(yaml.safe_load(open('configs/experiments/single_node_tuning_round4_ext/_base_selector_tuning_round4_ext.yaml')))"
```

Expected: 输出 `{'inherits': [...], 'meta': {'stage': 'single_node_tuning_round4_ext'}}`，无 yaml 错误。

- [ ] **Step 4: Commit**

```bash
git add paper-new/configs/experiments/single_node_tuning_round4_ext/_base_selector_tuning_round4_ext.yaml
git commit -m "feat(round5): add round4_ext base config inheriting round4 base"
```

---

## Task 2: 写 g5 / g6 / g7 三组组件配置

**Files:**
- Create: `paper-new/configs/experiments/single_node_tuning_round4_ext/_g5_compromise_low_high_mid.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4_ext/_g6_low_only_early.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4_ext/_g7_mid_softer_lite.yaml`

- [ ] **Step 1: 写 g5（折中带）**

文件 `_g5_compromise_low_high_mid.yaml`：

```yaml
inherits:
  - ./_base_selector_tuning_round4_ext.yaml

selector:
  genericity_gate_low: 0.76
  genericity_gate_high: 0.88
  genericity_gate_mid_scale: 0.35
```

- [ ] **Step 2: 写 g6（仅 low 提前）**

文件 `_g6_low_only_early.yaml`：

```yaml
inherits:
  - ./_base_selector_tuning_round4_ext.yaml

selector:
  genericity_gate_low: 0.75
```

注：仅覆盖 gate_low；gate_high 和 mid_scale 保持继承的 0.90 / 0.45。

- [ ] **Step 3: 写 g7（mid 略软）**

文件 `_g7_mid_softer_lite.yaml`：

```yaml
inherits:
  - ./_base_selector_tuning_round4_ext.yaml

selector:
  genericity_gate_mid_scale: 0.40
```

注：仅覆盖 mid_scale；gate_low/high 保持 0.78/0.90。

- [ ] **Step 4: 验证三组配置参数对照**

```bash
cd /Users/apple/Desktop/code_from_paper/paper-new
for g in g5 g6 g7; do
  echo "=== $g ==="
  python -c "
import yaml, sys
sys.path.insert(0, '.')
from thesis_platform.core.config import load_config
cfg = load_config('configs/experiments/single_node_tuning_round4_ext/_${g}_*.yaml')
sel = cfg['selector']
print(f'gate_low={sel[\"genericity_gate_low\"]}, gate_high={sel[\"genericity_gate_high\"]}, low_scale={sel[\"genericity_gate_low_scale\"]}, mid_scale={sel[\"genericity_gate_mid_scale\"]}')
" 2>/dev/null || echo "(用 grep 兜底)"
done
```

Expected:
- g5: `gate_low=0.76, gate_high=0.88, low_scale=0.10, mid_scale=0.35`
- g6: `gate_low=0.75, gate_high=0.90, low_scale=0.10, mid_scale=0.45`
- g7: `gate_low=0.78, gate_high=0.90, low_scale=0.10, mid_scale=0.40`

如果 `load_config` 路径不可用（本地未运行环境），改用纯文本 grep 检查：

```bash
grep -E "genericity_gate" paper-new/configs/experiments/single_node_tuning_round4_ext/_g{5,6,7}*.yaml
```

- [ ] **Step 5: Commit**

```bash
git add paper-new/configs/experiments/single_node_tuning_round4_ext/_g5_*.yaml \
        paper-new/configs/experiments/single_node_tuning_round4_ext/_g6_*.yaml \
        paper-new/configs/experiments/single_node_tuning_round4_ext/_g7_*.yaml
git commit -m "feat(round5): add g5/g6/g7 group configs (gate grid extension)"
```

---

## Task 3: 写 g5 的 4 个数据集叶子配置

**Files:**
- Create: `paper-new/configs/experiments/single_node_tuning_round4_ext/ns_tune4_g5_jobs.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4_ext/ns_tune4_g5_congressional.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4_ext/ns_tune4_g5_forums.yaml`
- Create: `paper-new/configs/experiments/single_node_tuning_round4_ext/ns_tune4_g5_microblog.yaml`

模板基于 `single_node_tuning_round4/ns_tune4_g3_forums.yaml`（已验证可跑），仅改 `inherits`、`experiment_id`、`output_root` 与数据路径。

- [ ] **Step 1: 写 ns_tune4_g5_jobs.yaml**

```yaml
inherits:
  - ./_g5_compromise_low_high_mid.yaml

meta:
  experiment_id: ns_tune4_g5_jobs

paths:
  output_root: paper-new/outputs/ns_tune4_g5_jobs

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 2: 写 ns_tune4_g5_congressional.yaml**

```yaml
inherits:
  - ./_g5_compromise_low_high_mid.yaml

meta:
  experiment_id: ns_tune4_g5_congressional

paths:
  output_root: paper-new/outputs/ns_tune4_g5_congressional

data:
  dataset_name: congressional
  train_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_train.json
  eval_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 3: 写 ns_tune4_g5_forums.yaml**

```yaml
inherits:
  - ./_g5_compromise_low_high_mid.yaml

meta:
  experiment_id: ns_tune4_g5_forums

paths:
  output_root: paper-new/outputs/ns_tune4_g5_forums

data:
  dataset_name: forums
  train_path: thesis_platform/datasets/pretext_forums/formatted/forums_train.json
  eval_path: thesis_platform/datasets/pretext_forums/formatted/forums_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 4: 写 ns_tune4_g5_microblog.yaml**

```yaml
inherits:
  - ./_g5_compromise_low_high_mid.yaml

meta:
  experiment_id: ns_tune4_g5_microblog

paths:
  output_root: paper-new/outputs/ns_tune4_g5_microblog

data:
  dataset_name: microblog
  train_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_train.json
  eval_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 5: 验证 4 个叶子文件路径与原 Round 4 g3 模板对齐**

```bash
diff <(cat paper-new/configs/experiments/single_node_tuning_round4/ns_tune4_g3_forums.yaml | sed 's|_g3_conditional_genericity_early_high|_g5_compromise_low_high_mid|; s|ns_tune4_g3_forums|ns_tune4_g5_forums|') \
     paper-new/configs/experiments/single_node_tuning_round4_ext/ns_tune4_g5_forums.yaml
```

Expected: 仅 `inherits` 路径差异（一个是 `./` 一个是同目录）；其他字段语义等同。

- [ ] **Step 6: Commit**

```bash
git add paper-new/configs/experiments/single_node_tuning_round4_ext/ns_tune4_g5_*.yaml
git commit -m "feat(round5): add g5 leaf configs (jobs/congressional/forums/microblog)"
```

---

## Task 4: 写 g6 的 4 个数据集叶子配置

模板与 Task 3 相同，只把 `_g5_compromise_low_high_mid` 全部换成 `_g6_low_only_early`，把 `g5` 全部换成 `g6`。

**Files:**
- Create: `ns_tune4_g6_jobs.yaml` / `ns_tune4_g6_congressional.yaml` / `ns_tune4_g6_forums.yaml` / `ns_tune4_g6_microblog.yaml`（均在 `paper-new/configs/experiments/single_node_tuning_round4_ext/`）

- [ ] **Step 1: 写 ns_tune4_g6_jobs.yaml**

```yaml
inherits:
  - ./_g6_low_only_early.yaml

meta:
  experiment_id: ns_tune4_g6_jobs

paths:
  output_root: paper-new/outputs/ns_tune4_g6_jobs

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 2: 写 ns_tune4_g6_congressional.yaml**

```yaml
inherits:
  - ./_g6_low_only_early.yaml

meta:
  experiment_id: ns_tune4_g6_congressional

paths:
  output_root: paper-new/outputs/ns_tune4_g6_congressional

data:
  dataset_name: congressional
  train_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_train.json
  eval_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 3: 写 ns_tune4_g6_forums.yaml**

```yaml
inherits:
  - ./_g6_low_only_early.yaml

meta:
  experiment_id: ns_tune4_g6_forums

paths:
  output_root: paper-new/outputs/ns_tune4_g6_forums

data:
  dataset_name: forums
  train_path: thesis_platform/datasets/pretext_forums/formatted/forums_train.json
  eval_path: thesis_platform/datasets/pretext_forums/formatted/forums_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 4: 写 ns_tune4_g6_microblog.yaml**

```yaml
inherits:
  - ./_g6_low_only_early.yaml

meta:
  experiment_id: ns_tune4_g6_microblog

paths:
  output_root: paper-new/outputs/ns_tune4_g6_microblog

data:
  dataset_name: microblog
  train_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_train.json
  eval_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 5: Commit**

```bash
git add paper-new/configs/experiments/single_node_tuning_round4_ext/ns_tune4_g6_*.yaml
git commit -m "feat(round5): add g6 leaf configs (jobs/congressional/forums/microblog)"
```

---

## Task 5: 写 g7 的 4 个数据集叶子配置

模板同 Task 3-4，把 `_g7_mid_softer_lite` 替换 inherits，把 `g7` 替换组名。

**Files:**
- Create: `ns_tune4_g7_jobs.yaml` / `ns_tune4_g7_congressional.yaml` / `ns_tune4_g7_forums.yaml` / `ns_tune4_g7_microblog.yaml`（均在 `paper-new/configs/experiments/single_node_tuning_round4_ext/`）

- [ ] **Step 1: 写 ns_tune4_g7_jobs.yaml**

```yaml
inherits:
  - ./_g7_mid_softer_lite.yaml

meta:
  experiment_id: ns_tune4_g7_jobs

paths:
  output_root: paper-new/outputs/ns_tune4_g7_jobs

data:
  dataset_name: jobs
  train_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json
  eval_path: thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 2: 写 ns_tune4_g7_congressional.yaml**

```yaml
inherits:
  - ./_g7_mid_softer_lite.yaml

meta:
  experiment_id: ns_tune4_g7_congressional

paths:
  output_root: paper-new/outputs/ns_tune4_g7_congressional

data:
  dataset_name: congressional
  train_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_train.json
  eval_path: thesis_platform/datasets/pretext_congressional/formatted/congressional_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 3: 写 ns_tune4_g7_forums.yaml**

```yaml
inherits:
  - ./_g7_mid_softer_lite.yaml

meta:
  experiment_id: ns_tune4_g7_forums

paths:
  output_root: paper-new/outputs/ns_tune4_g7_forums

data:
  dataset_name: forums
  train_path: thesis_platform/datasets/pretext_forums/formatted/forums_train.json
  eval_path: thesis_platform/datasets/pretext_forums/formatted/forums_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 4: 写 ns_tune4_g7_microblog.yaml**

```yaml
inherits:
  - ./_g7_mid_softer_lite.yaml

meta:
  experiment_id: ns_tune4_g7_microblog

paths:
  output_root: paper-new/outputs/ns_tune4_g7_microblog

data:
  dataset_name: microblog
  train_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_train.json
  eval_path: thesis_platform/datasets/pretext_microblog/formatted/microblog_eval.json
  initialization_path: thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json
```

- [ ] **Step 5: 验证全部 12 个叶子文件存在**

```bash
ls -1 /Users/apple/Desktop/code_from_paper/paper-new/configs/experiments/single_node_tuning_round4_ext/ns_tune4_g*_*.yaml | wc -l
```

Expected: `12`

- [ ] **Step 6: Commit**

```bash
git add paper-new/configs/experiments/single_node_tuning_round4_ext/ns_tune4_g7_*.yaml
git commit -m "feat(round5): add g7 leaf configs (jobs/congressional/forums/microblog)"
```

---

## Task 6: 写自动化串行 runner（run_round4_ext_queue.py）

**Files:**
- Create: `old_automation/run_round4_ext_queue.py`（本地副本；远端服务器同步路径 `/mnt/public/caiqiyue_file/code_from_paper/old_automation/run_round4_ext_queue.py`）

模板基于已验证的 `run_round4_queue.py`，改 EXPERIMENTS 列表、config 路径、log 路径与日志前缀。

- [ ] **Step 1: 写 runner 文件**

```python
#!/usr/bin/env python3
"""Sequential runner for Round 4 extension experiments (12 total: g5/g6/g7 × 4 datasets) on A6000 GPU."""
import datetime
import os
import subprocess
import sys

REPO = "/mnt/public/caiqiyue_file/code_from_paper"
PAPER_NEW = REPO + "/paper-new"
AUTOMATION = REPO + "/old_automation"
LOG_PATH = AUTOMATION + "/run_round4_ext_queue.log"

EXPERIMENTS = [
    ("g5", "jobs"), ("g5", "congressional"), ("g5", "forums"), ("g5", "microblog"),
    ("g6", "jobs"), ("g6", "congressional"), ("g6", "forums"), ("g6", "microblog"),
    ("g7", "jobs"), ("g7", "congressional"), ("g7", "forums"), ("g7", "microblog"),
]

ENV = {
    **os.environ,
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "1",
    "PYTHONUNBUFFERED": "1",
    "VLLM_HOST_IP": "127.0.0.1",
    "HOST_IP": "127.0.0.1",
}

PYTHON = "/home/k8smaster/anaconda3/envs/pretext/bin/python"


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_experiment(group, dataset):
    exp_id = f"ns_tune4_{group}_{dataset}"
    config = f"configs/experiments/single_node_tuning_round4_ext/{exp_id}.yaml"
    remote_log = f"{AUTOMATION}/NS-T4-EXT-{group.upper()}-{dataset.upper()}.remote.log"
    log(f"Starting {exp_id}")
    log(f"Config: {config}")
    with open(remote_log, "w") as out:
        result = subprocess.run(
            [PYTHON, "-m", "paper_new_selector.run_selector_single_node",
             "--config", config],
            cwd=PAPER_NEW,
            env=ENV,
            stdout=out,
            stderr=out,
        )
    if result.returncode == 0:
        log(f"SUCCESS: {exp_id}")
        return True
    else:
        log(f"FAILED (exit {result.returncode}): {exp_id} -- see {remote_log}")
        return False


def main():
    total = len(EXPERIMENTS)
    done = 0
    failed = 0
    log(f"=== Round 4 EXT Queue Start: {total} experiments on A6000 (CUDA_VISIBLE_DEVICES=1) ===")
    exp_ids = [f"ns_tune4_{g}_{d}" for g, d in EXPERIMENTS]
    log(f"Queue: {exp_ids}")

    for i, (group, dataset) in enumerate(EXPERIMENTS, 1):
        log(f"--- [{i}/{total}] ---")
        ok = run_experiment(group, dataset)
        if ok:
            done += 1
        else:
            failed += 1

    log(f"=== Round 4 EXT Done: {done} success, {failed} failed out of {total} ===")


if __name__ == "__main__":
    main()
```

注意：相比 Round 4 原 runner 的差异——
- `LOG_PATH` 改为 `run_round4_ext_queue.log`
- `EXPERIMENTS` 改为 g5/g6/g7 × 4 datasets = 12 项
- config 路径改为 `single_node_tuning_round4_ext/`
- `remote_log` 前缀改为 `NS-T4-EXT-`
- 启动/结束日志改为 "Round 4 EXT"

- [ ] **Step 2: 本地语法检查**

```bash
cd /Users/apple/Desktop/code_from_paper
python -c "import ast; ast.parse(open('old_automation/run_round4_ext_queue.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 验证 EXPERIMENTS 列表数量**

```bash
python -c "
import ast
tree = ast.parse(open('old_automation/run_round4_ext_queue.py').read())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(t.id == 'EXPERIMENTS' for t in node.targets if hasattr(t, 'id')):
        print(f'EXPERIMENTS count: {len(node.value.elts)}')
"
```

Expected: `EXPERIMENTS count: 12`

- [ ] **Step 4: Commit**

```bash
git add old_automation/run_round4_ext_queue.py
git commit -m "feat(round5): add run_round4_ext_queue.py for g5/g6/g7 sequential execution"
```

---

## Task 7: 同步代码到远端服务器并启动实验队列

**Files:** （远端）
- Sync: 13 个 yaml 文件 + 1 个 runner 到 `/mnt/public/caiqiyue_file/code_from_paper/`

服务器连接信息：
- Host: `1u72c85740.zicp.fun`（备用 IP `58.217.205.244`）
- Port: `54360`
- User: `k8smaster`
- Password: `k8s`
- 已连接成功的方式：macOS `sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun '...'`

- [ ] **Step 1: 上传 13 个 yaml 文件**

```bash
cd /Users/apple/Desktop/code_from_paper
sshpass -p 'k8s' scp -P 54360 -r \
    paper-new/configs/experiments/single_node_tuning_round4_ext \
    k8smaster@1u72c85740.zicp.fun:/mnt/public/caiqiyue_file/code_from_paper/paper-new/configs/experiments/
```

- [ ] **Step 2: 上传 runner**

```bash
sshpass -p 'k8s' scp -P 54360 \
    old_automation/run_round4_ext_queue.py \
    k8smaster@1u72c85740.zicp.fun:/mnt/public/caiqiyue_file/code_from_paper/old_automation/
```

- [ ] **Step 3: 远端验证文件齐全**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'ls /mnt/public/caiqiyue_file/code_from_paper/paper-new/configs/experiments/single_node_tuning_round4_ext/ | wc -l && ls /mnt/public/caiqiyue_file/code_from_paper/old_automation/run_round4_ext_queue.py'
```

Expected: 第一行输出 `16`（1 base + 3 group + 12 leaf），第二行输出 runner 路径，无 "No such file" 错误。

- [ ] **Step 4: 在远端 tmux session 启动队列**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'tmux new-session -d -s round4_ext "/home/k8smaster/anaconda3/envs/pretext/bin/python /mnt/public/caiqiyue_file/code_from_paper/old_automation/run_round4_ext_queue.py"'
```

- [ ] **Step 5: 验证队列已启动**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'tmux list-sessions && tail -5 /mnt/public/caiqiyue_file/code_from_paper/old_automation/run_round4_ext_queue.log'
```

Expected: 看到 `round4_ext` session；log 显示 `=== Round 4 EXT Queue Start: 12 experiments ...` 与 `Starting ns_tune4_g5_jobs`。

---

## Task 8: 监控实验进度（每 10-15 分钟一次轮询）

整个队列预计 ~50 分钟，每个实验 ~4 分钟。**不要忙等**——每 10-15 分钟拉一次远端 log 即可。

- [ ] **Step 1: 周期性查看进度**

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'tail -30 /mnt/public/caiqiyue_file/code_from_paper/old_automation/run_round4_ext_queue.log'
```

观察：
- `[N/12]` 当前进度
- `SUCCESS: ns_tune4_*` 已完成的数量
- 任何 `FAILED (exit ...)` 立即排查

- [ ] **Step 2: 检测全部完成的标志**

队列结束后 log 会写：
```
=== Round 4 EXT Done: X success, Y failed out of 12 ===
```

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
    'grep "Round 4 EXT Done" /mnt/public/caiqiyue_file/code_from_paper/old_automation/run_round4_ext_queue.log'
```

Expected: `=== Round 4 EXT Done: 12 success, 0 failed out of 12 ===`

如果 `failed > 0`，进入排查：查 `NS-T4-EXT-*-*.remote.log` 找具体错误，修复后重跑该单个实验。

---

## Task 9: 拉取实验结果并整理到 Round 4 文档第十一节

**Files:**
- Modify: `paper-new/docs/2026-04-26-round4-algorithm-and-experiment-design.md`（追加第十一节）

- [ ] **Step 1: 拉取 12 个 downstream_eval_summary.json**

```bash
mkdir -p /tmp/round4_ext_results
for g in g5 g6 g7; do
  for d in jobs congressional forums microblog; do
    sshpass -p 'k8s' scp -P 54360 \
      "k8smaster@1u72c85740.zicp.fun:/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_tune4_${g}_${d}/eval/downstream_eval_summary.json" \
      "/tmp/round4_ext_results/ns_tune4_${g}_${d}.json" 2>&1 | grep -v "100%" || true
  done
done
ls /tmp/round4_ext_results/ | wc -l
```

Expected: `12`

- [ ] **Step 2: 拉取每个实验的合成句数**

```bash
for g in g5 g6 g7; do
  for d in jobs congressional forums microblog; do
    sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
      "python -c 'import json; d=json.load(open(\"/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_tune4_${g}_${d}/eval/stage2/llama7b_text_syn.json\")); print(len(d))'"
  done
done
```

输出 12 行整数，每行对应一个实验的合成句数；记下来用于结果表的"合成句数"列。

- [ ] **Step 3: 解析 best_top1/3/5/10**

```bash
for f in /tmp/round4_ext_results/*.json; do
  exp=$(basename "$f" .json)
  python -c "
import json
d = json.load(open('$f'))
b = d.get('best', {})
print(f'$exp\t{b.get(\"top1\", b.get(\"best_top1\", \"?\"))}\t{b.get(\"top3\", b.get(\"best_top3\", \"?\"))}\t{b.get(\"top5\", b.get(\"best_top5\", \"?\"))}\t{b.get(\"top10\", b.get(\"best_top10\", \"?\"))}'
"
done
```

注：实际 json 字段名以 Round 4 文档 §8.1 的取值方式为准（如有差异，参照 Round 4 整理时使用的脚本）。

- [ ] **Step 4: 在 Round 4 文档末尾追加第十一节**

定位插入点：在文件 `paper-new/docs/2026-04-26-round4-algorithm-and-experiment-design.md` 现有第十节（`## 十、下一步路线分叉前的硬约束检查`）之后追加。

追加模板（实际数值由 Step 3 的输出填入）：

```markdown

---

## 十一、Round 4 扩展（gate 网格 g5/g6/g7）

### 11.1 三组扩展配置

| 配置 | gate_low | gate_high | low_scale | mid_scale | 假设 |
|---|---|---|---|---|---|
| **g5（折中带）**| 0.76 | 0.88 | 0.10 | 0.35 | g3 的 early-low + g2 的 soft-mid 是否能同时见效 |
| **g6（仅 low 提前）**| 0.75 | 0.90 | 0.10 | 0.45 | gate_high 提前是否是 forums 退步元凶 |
| **g7（mid 略软）**| 0.78 | 0.90 | 0.10 | 0.40 | g2 的 0.30 太软；0.40 是 g1 与 g2 的安全中间值 |

### 11.2 完整结果表

| 实验 ID | 组别 | 数据集 | 合成句数 | best_top1 | best_top3 | best_top5 | best_top10 |
|---------|------|--------|---------|-----------|-----------|-----------|------------|
| ns_tune4_g5_jobs | g5 | jobs | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |
| ns_tune4_g5_congressional | g5 | congressional | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |
| ns_tune4_g5_forums | g5 | forums | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |
| ns_tune4_g5_microblog | g5 | microblog | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |
| ns_tune4_g6_jobs | g6 | jobs | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |
| ns_tune4_g6_congressional | g6 | congressional | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |
| ns_tune4_g6_forums | g6 | forums | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |
| ns_tune4_g6_microblog | g6 | microblog | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |
| ns_tune4_g7_jobs | g7 | jobs | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |
| ns_tune4_g7_congressional | g7 | congressional | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |
| ns_tune4_g7_forums | g7 | forums | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |
| ns_tune4_g7_microblog | g7 | microblog | <Step 2> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> | <Step 3 数值> |

### 11.3 与 PrE-Text 基线对比（best_top1）

| 数据集 | PrE-Text | g5 | g6 | g7 | 是否有任一组超过 PrE-Text |
|---|---|---|---|---|---|
| jobs | 0.2732 | <填> | <填> | <填> | <Y/N> |
| congressional | 0.2950 | <填> | <填> | <填> | <Y/N> |
| forums | 0.2501 | <填> | <填> | <填> | <Y/N> |
| microblog | 0.2763 | <填> | <填> | <填> | <Y/N> |

### 11.4 关键发现

（基于实际数据，写 3-5 条 finding：哪一组最优、forums 是否突破、g5/g6/g7 各自验证或证伪了什么假设）

### 11.5 与 Round 4 g1-g4 的合并最优

把 g1-g7 共 7 组的 best_top1 合并，找每个数据集的最优值，确定是否有"per-dataset 全胜"组合。

| 数据集 | 7 组中最高 best_top1 | 来自配置 | 是否超过 PrE-Text |
|---|---|---|---|
| jobs | <填> | <填> | <Y/N> |
| congressional | <填> | <填> | <Y/N> |
| forums | <填> | <填> | <Y/N> |
| microblog | <填> | <填> | <Y/N> |

### 11.6 方向 1 成功判据评估

按 spec §2.4 的三档判据：
- **强成功**：某一组 g5/g6/g7 在 4 个数据集 best_top1 全部超过 PrE-Text → <Y/N>
- **弱成功**：forums 最佳值（不限组）≥ 0.2501，且其他三个数据集至少各有一组维持 Round 4 胜利 → <Y/N>
- **失败**：forums 仍 < 0.2501 → <Y/N>，依赖方向 2a
```

- [ ] **Step 5: 填入实际数据并校对**

把 Step 2-3 的输出对照模板填入；double-check 每个实验 ID 与数据对应。

- [ ] **Step 6: Commit**

```bash
git add paper-new/docs/2026-04-26-round4-algorithm-and-experiment-design.md
git commit -m "docs(round5): add section 11 with g5/g6/g7 extension results"
```

---

## Task 10: 方向 1 整体收尾验证

- [ ] **Step 1: 复盘成功判据**

打开第十一节 §11.6，确认：
- 是否触发"强成功"？如是 → 论文主推该全局配置
- 是否触发"弱成功"？如是 → 至少 forums 已破，可放论文 per-dataset 表
- 是否仍是"失败"？→ 完全依赖方向 2a 的结果

- [ ] **Step 2: 总结 Round 4 + Round 4 EXT 共 7 组（28 实验）的结论一行总结**

在 commit message 或者口头汇报里写一句话：例如 "g6 在 forums 上 0.2515 终于破 PrE-Text 基线，但 congressional 退步至 0.2945"。

- [ ] **Step 3: 决定是否仍需要进入方向 2a**

如果方向 1 已强成功 → 方向 2a 仍跑（作为论文 ablation/对照），但优先级降级
如果方向 1 弱成功或失败 → 方向 2a 是主线，按照 direction2a plan 推进

---

## Self-Review

- [ ] **Spec coverage**：本 plan 是否覆盖了 spec §2 全部要求？
  - §2.1 目录与文件 → Task 1, 3, 4, 5
  - §2.2 三组配置参数 → Task 2
  - §2.3 自动化执行 → Task 6, 7
  - §2.4 成功判据 → Task 9 §11.6, Task 10
  - §2.5 文档输出 → Task 9
  - **覆盖完整。**
- [ ] **Placeholder scan**：搜索 `TODO`、`TBD`、`<填>`等占位符
  - 在 Task 9 §11.2/11.3/11.4/11.5/11.6 中有 `<填>` `<Y/N>` `<Step 3 数值>` 占位 — 这些是**模板占位**（需运行后由实际数据填入），不是 plan 的缺漏。Plan 已注明每个占位的来源。
- [ ] **Type consistency**：函数签名/参数名一致 — Plan 仅涉及 yaml + 1 个独立 runner，没有跨任务的 API 一致性问题。

---

## 完成判定

本 plan 完成的标志：
1. 12 个新实验全部 SUCCESS
2. Round 4 文档第十一节填入完整数据
3. 已在 §11.6 给出方向 1 三档判据的明确评估（Y/N）
4. 决策：方向 2a 是主线（继续）还是 ablation（降级优先级）

预计总时长：**1-1.5 小时**（其中 ~50 分钟是 GPU 实验时间，可与方向 2a 的 plan 准备工作并行）。
