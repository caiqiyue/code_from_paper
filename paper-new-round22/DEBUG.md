## 2026-05-11 Round22 Smoke Runtime OOM

### Observations

- `round22 smoke` 的第一条实验 `r22dc_jobs_seed42_k20` 已经越过了之前的跨平台 `config_path` 问题，成功进入真实的 `round19` 执行路径。
- 当前失败日志来自服务器：
  - `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round22/logs/round22_bandit_smoke_logs/r22dc_jobs_seed42_k20.log`
- 关键运行事实：
  - 启动前 vLLM precheck 通过：`free=18.29 GiB required=2.00 GiB`
  - vLLM 成功初始化：`# GPU blocks: 1871, # CPU blocks: 512`
  - 失败不是在初始化时，而是在第一轮真实 `generate()` 时：
    - `torch.cuda.OutOfMemoryError: Tried to allocate 20.00 MiB`
    - `GPU 0 ... 5.88 MiB is free`
    - `Process 26131 has 18.05 GiB memory in use`
    - `Including non-PyTorch memory, this process has 27.66 GiB memory in use`
    - `Process 28554 has 1.80 GiB memory in use`
- 当前 `round22` bandit 采集配置继承的是 `round19` base，对应运行 profile 为：
  - `gpu_memory_utilization = 0.35`
  - `startup_required_free_gb = 2`
  - `max_model_len = 512`
  - `candidate_count = 24`
  - `generated_per_round = 8`
- 这说明：
  - 不是路径/runner 逻辑又错了
  - 也不是典型的 `GPU blocks = 0`
  - 而是当前 vLLM profile 在共享 A6000 环境下，于“真实生成阶段”触发了 OOM

### Hypotheses

#### H1: `round22` 直接沿用了 `round19` 的高占用 vLLM profile，导致共享 A6000 上在生成阶段 OOM（ROOT HYPOTHESIS）
- Supports:
  - `round22` bandit collection base 继承 `round19` base。
  - 当前 profile 为 `gpu_memory_utilization=0.35`、`max_model_len=512`，这正是之前在 `round19/20` 中会产生较高单进程显存占用的一档。
  - 这次失败不是启动前门槛问题，也不是路径问题，而是在生成阶段分配额外 `20 MiB` 时只剩 `5.88 MiB`。
- Conflicts:
  - 当次外部还有一个 `1.8 GiB` 的进程占卡，但这个量本身不太可能单独解释从 `18.29 GiB free` 到完全无余量的全部变化。
- Test:
  - 对比 `round22` 当前 profile 与之前已经为共享 A6000 压过显存的 `round20` 低占用 profile。

#### H2: 主要是服务器上瞬时外部进程抢卡，`round22` profile 本身没问题
- Supports:
  - 日志里确实存在额外 `1.80 GiB` 外部进程。
- Conflicts:
  - 这次本进程自身非 PyTorch + PyTorch 合计已到 `27.66 GiB`，说明即使没有额外那 `1.8 GiB`，当前 profile 也已接近高水位。
  - 当前 precheck 阈值只有 `2 GiB`，不足以保证“启动后 + 生成时”仍有安全余量。
- Test:
  - 仅提高 `startup_required_free_gb` 无法降低本进程占用；真正需要的是更低 footprint profile。

#### H3: runner 的 reward/state 抽取逻辑导致了额外的 GPU 占用
- Supports:
  - `round22` runner 在实验结束后会做额外特征抽取。
- Conflicts:
  - 当前失败发生在 `paper_new_selector.run_selector_single_node` 内部的第一轮生成阶段，runner 还没进入后处理/汇总阶段。
- Test:
  - 无需额外实验；栈已经足以排除。

### Experiments

#### E1: 读取失败日志并定位失败阶段
- Change: 无代码改动，直接读取 `r22dc_jobs_seed42_k20.log`。
- Expected confirm: 如果失败发生在 `generate()` 而不是路径加载或汇总阶段，则 runner 逻辑不是当前主因。
- Result: Confirmed.

#### E2: 对比 `round22` 采集 base 与已知共享 GPU 调优经验
- Change: 无代码改动，检查 `round22` base 和过往 `round20` 低占用 profile。
- Expected confirm: 如果 `round22` 仍沿用 `round19` 的 `0.35 / 2 / 512` 运行窗口，则它尚未针对共享 A6000 做 profile 收缩。
- Result: Confirmed.

### Root Cause

- `round22 smoke` 当前失败的根因不是 runner 再次出错，而是 **bandit 数据采集实验直接继承了 `round19` 的高占用 vLLM 运行窗口**。在共享 A6000 环境下，这个 profile 能通过启动前的低门槛检查，但在第一轮真实生成时会把显存推到极限，并在一次额外的 `20 MiB` 分配上触发 OOM。

### Fix Direction

- 不需要继续改 runner 逻辑。
- 需要为 `round22 bandit data collection` 单独定义一套 **共享 GPU 低占用 profile**，而不是直接沿用 `round19` 的默认 Stage1/Stage2 vLLM 配置。
- 修复原则：
  1. 只改 `round22` 采集实验的 vLLM profile，不动算法逻辑。
  2. 优先收缩 `gpu_memory_utilization`，保留 `max_model_len=512` 作为第一优先口径。
  3. 如仍不稳，再评估是否温和调整 `max_model_len`。

### Applied Fix

- 针对共享 GPU 环境，曾短暂测试过把 `round22` bandit collection profile 收到 `0.30 / 2 / 512`。
- 但在当前用户确认“A6000 当前没有其他进程、显存环境干净”后，最终决定把 `round22` 数据采集 profile 恢复到：
  - `llm.generator.gpu_memory_utilization: 0.35`
  - `llm.generator.startup_required_free_gb: 2`
  - `bootstrap.gpu_memory_utilization: 0.35`
  - `bootstrap.startup_required_free_gb: 2`
- 这样做的理由是：
  - 当前目标是优先保证 `GPU blocks` 充足和生成稳定性
  - 在 A6000 空卡条件下，`0.35` 是此前已经多次跑通的熟悉运行窗口
  - 先保留 `max_model_len = 512`

## 2026-05-13 Learned Runtime Output Root Mismatch

### Observations

- The learned-policy wrapper writes reference artifacts and sidecar files under the CLI `--output-root`, for example:
  - `_k20_reference_features.json`
  - `*_learned_override.yaml`
  - `*_learned_budget_policy_runtime.json`
- The current `generate_override_config(...)` implementation only overrides:
  - `meta.learned_budget_runtime`
  - `selector.seed_top_k`
  - `selector.seed_budget_rule.enabled`
- It does **not** override `paths.output_root`.
- Therefore, when the wrapper is invoked with a base config like `r22dc_jobs_seed42_k20.yaml`, the final round19 runtime still inherits the original config output root such as:
  - `paper-new-round22/outputs/bandit_data_collection/jobs/seed42/k20`
  rather than the wrapper runtime root such as:
  - `.../outputs/learned_policy_smoke/r22_jobs_seed42`
- This explains the observed symptom:
  - learned-policy metadata exists under the wrapper output root,
  - but no downstream eval artifacts appear there,
  - even if the round19 subprocess may have succeeded.
- The current sidecar also does not verify or record:
  - actual runtime `stage1_summary.json`
  - actual runtime downstream eval summary path

### Hypotheses

#### H1: The final round19 runtime actually ran, but wrote results to the original config output root because `paths.output_root` was not overridden (ROOT HYPOTHESIS)
- Supports:
  - Code inspection confirms `generate_override_config(...)` does not set `paths.output_root`.
  - User observed reference/sidecar files under learned-policy output root but no final runtime artifacts there.
  - Wrapper prints `Done` only after subprocess return code `0`, so a zero-exit subprocess with outputs elsewhere would create exactly this symptom.
- Conflicts:
  - If the subprocess actually failed with nonzero exit, this hypothesis alone would not explain the missing artifacts.
- Test:
  - Add a unit test that asserts override config rewrites `paths.output_root` to the wrapper runtime root.
  - After fix, require wrapper to verify expected runtime artifact files after subprocess success.

#### H2: The final round19 runtime subprocess still failed before writing outputs, and the missing files are due to a hidden runtime error
- Supports:
  - The user observed no downstream eval output at the expected wrapper location.
- Conflicts:
  - Current wrapper would return nonzero and print stderr on a subprocess failure; the observed "Done" symptom is more consistent with return code `0`.
- Test:
  - Persist runtime stdout/stderr or verify required artifact files after success; if missing, treat as failure.

#### H3: The wrapper output root is correct, but the expected artifact names/locations were wrong
- Supports:
  - The user looked for a `k=18` subdirectory, while the current wrapper does not create per-budget subdirectories for final runtime.
- Conflicts:
  - Even without a per-budget subdir, a successful runtime should still produce `stage1_summary.json` and `eval/...` under some deterministic location.
- Test:
  - Explicitly define and verify the final runtime artifact locations in the wrapper.

### Experiments

#### E1: Inspect override generation logic
- Change: No production fix. Code inspection only.
- Expected confirm: If `paths.output_root` is not rewritten, the output-root mismatch theory is valid.
- Result: Confirmed.

### Root Cause

- The learned-policy wrapper generated a valid override config for budget prediction, but it **did not redirect `paths.output_root` to the wrapper runtime root**. As a result, the final round19 runtime could write outputs to the original base-config location instead of the learned-policy output directory, making the experiment appear incomplete even when the subprocess itself may have succeeded.

### Fix Direction

- Update override generation so the final round19 runtime always writes to the wrapper-managed runtime root.
- After subprocess success, explicitly verify the presence of:
  - `stage1_summary.json`
  - `eval/downstream_eval_summary.json` (or equivalent eval summary artifact if produced)
- Write the verified runtime artifact paths back into the learned runtime sidecar for later inspection.
