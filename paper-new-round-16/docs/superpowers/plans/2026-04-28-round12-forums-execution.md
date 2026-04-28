# Round12 Forums Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the Round12 forums conservative sweep on the old server with the `pretext` environment and A6000 GPU.

**Architecture:** No algorithm code changes are needed. Execution consists of remote validation, old-output archival, sequential config execution, and metric summarization.

**Tech Stack:** SSH, conda `pretext`, Python module `paper_new_selector.run_selector_single_node`, NVIDIA A6000 via `CUDA_VISIBLE_DEVICES=1`.

---

### Task 1: Validate Remote State

**Files:**
- Read: `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round11/configs/experiments/single_node_tuning_round12/*.yaml`

- [ ] **Step 1: Check branch and config presence**

Run:

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
  'cd /mnt/public/caiqiyue_file/code_from_paper/paper-new-round11 &&
   git symbolic-ref --short HEAD &&
   ls configs/experiments/single_node_tuning_round12/ns_tune12_*.yaml'
```

Expected: branch is `paper-2-genereic`, and eight `ns_tune12_*.yaml` files are listed.

- [ ] **Step 2: Validate YAML matrix**

Run:

```bash
sshpass -p 'k8s' ssh -p 54360 k8smaster@1u72c85740.zicp.fun \
  'cd /mnt/public/caiqiyue_file/code_from_paper/paper-new-round11 &&
   source /home/k8smaster/anaconda3/etc/profile.d/conda.sh &&
   conda activate pretext &&
   python - <<PY
from pathlib import Path
from paper_new_selector.thesis_bridge import load_yaml_config
base = Path("configs/experiments/single_node_tuning_round12")
for p in sorted(base.glob("ns_tune12_*.yaml")):
    cfg = load_yaml_config(p)
    sel = cfg["selector"]
    print(p.name, cfg["meta"]["seed"], sel["_forums_seed_top_k"], sel["_forums_max_tokens"])
PY'
```

Expected: f1-f8 match the design matrix.

### Task 2: Start Background Execution

**Files:**
- Create remote: `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round11/logs/run_round12_a6000.sh`
- Create remote: `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round11/logs/run_round12_a6000.out`

- [ ] **Step 1: Create and start remote runner**

Run a remote shell script that:

```bash
set -euo pipefail
cd /mnt/public/caiqiyue_file/code_from_paper/paper-new-round11
source /home/k8smaster/anaconda3/etc/profile.d/conda.sh
conda activate pretext
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

Then archive existing Round12 outputs and run configs f1-f8 sequentially with:

```bash
python -m paper_new_selector.run_selector_single_node --config "$cfg"
```

Expected: a background PID is printed and the log starts with `Started round12 A6000 suite`.

- [ ] **Step 2: Confirm GPU use**

Run:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Expected: a Python process appears on A6000 after model loading begins.

### Task 3: Monitor and Summarize

**Files:**
- Read remote: `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round11/logs/run_round12_a6000.out`
- Read remote: `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/ns_tune12_*/eval/downstream_eval_summary.json`

- [ ] **Step 1: Poll log**

Run:

```bash
tail -120 /mnt/public/caiqiyue_file/code_from_paper/paper-new-round11/logs/run_round12_a6000.out
```

Expected: each config prints `RESULT <name> best_top1= ... synthetic_train_count= ...`.

- [ ] **Step 2: Report result table**

After completion, parse summaries and report:

```text
config | seed_top_k | max_tokens | meta.seed | best_top1 | synthetic_train_count | vs PrE-Text
```

Expected: identify whether any run exceeds `0.2501448715`.

## Self-Review

- Spec coverage: covers remote validation, A6000 execution, archival, and result summary.
- Placeholder scan: no TODO/TBD placeholders.
- Scope: focused on running experiments only; no algorithm changes.
