# Round13 Unified Seed Top-K Sweep Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all 20 Round13 unified seed_top_k sweep experiments on the old server with the `pretext` environment and A6000 GPU.

**Architecture:** No code changes. Validate synced configs, archive stale outputs, run all configs sequentially, and summarize whether any seed_top_k value beats PrE-Text on all four datasets.

**Tech Stack:** SSH, conda `pretext`, `paper_new_selector.run_selector_single_node`, NVIDIA A6000 via `CUDA_VISIBLE_DEVICES=1`.

---

### Task 1: Validate Remote Sync

**Files:**
- Read: `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round11/configs/experiments/single_node_tuning_round13/*.yaml`

- [ ] Confirm branch is `paper-2-genereic`.
- [ ] Confirm exactly 20 Round13 configs exist.
- [ ] Parse all configs and verify each group `18/19/20/21/22` has jobs, congressional, forums, microblog.
- [ ] Verify all configs use `selector.seed_top_k`, `bootstrap.max_tokens=85`, and `meta.seed=42`.

### Task 2: Start Remote Runner

**Files:**
- Create remote: `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round11/logs/run_round13_unified_a6000.sh`
- Create remote: `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round11/logs/run_round13_unified_a6000.out`

- [ ] Archive existing Round13 outputs.
- [ ] Run all 20 configs sequentially with `CUDA_VISIBLE_DEVICES=1`.
- [ ] Print `RESULT <name> best_top1= ... synthetic_train_count= ...` after each config.

### Task 3: Monitor and Summarize

**Files:**
- Read remote: `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round11/logs/run_round13_unified_a6000.out`

- [ ] Confirm first experiment starts and vLLM loads on A6000.
- [ ] After completion, build a table by `seed_top_k` and mark whether all four datasets beat PrE-Text.

## Self-Review

- Spec coverage: covers validation, execution, archival, monitoring, and summary.
- Placeholder scan: no TODO/TBD placeholders.
- Scope: execution only; no algorithm or config changes.
