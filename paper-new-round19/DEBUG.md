## Observations

- 从仓库根目录运行 `conda run -n pretext python paper-new/scripts/run_selector_single_node.py --config paper-new/configs/single_node_jobs_selector.yaml --validate-only` 会报 `ModuleNotFoundError: No module named 'paper_new_selector'`。
- 从 `paper-new` 目录运行 `conda run -n pretext python -m unittest discover -s tests -p "test_*.py" -v` 时，多数测试会把 `paper-new/configs/...` 解析成 `paper-new/paper-new/configs/...`，导致 `FileNotFoundError`。
- `selector.py` 里拒绝样本只在 `accept_scores[index] == 0.0` 时才会重算冗余惩罚；如果样本在最后一次选择前已经算过分数，那么最终 selected set 扩大后，它的 `accept_score` 和 `redundancy_penalty` 会保持陈旧值。
- 最小复现：
  - `candidate_vectors=[[1,0],[0,1],[0,0.99]]`
  - `private_support=[1.0,0.95,0.94]`
  - `lambda_redundancy=1.0, seed_top_k=2`
  - 结果中第 3 个候选和第 2 个 seed 高相似，但 `redundancy_penalty == 0.0`。
- `stage1_runner.py` 每轮都固定使用 `init_samples[:exemplars_per_prompt]`，没有对 `D_init` 做轮换或采样，候选池多样性受到固定前缀样本支配。
- 真实包内调用 `from paper_new_selector.pipeline import run_pipeline` 仍然可以跑通完整测试级流水线，说明核心算法和真实模型接线基本可用，失败主要集中在入口包装与一处 selector 边界逻辑。

## Hypotheses

### H1: 配置路径缺少统一解析规则，导致不同工作目录下同一个相对路径行为不一致
- Supports: `load_yaml_config()` 直接 `Path(config_path).resolve()`；测试把路径写死为 `paper-new/configs/...`，在 `paper-new` 目录中会变成双重前缀。
- Conflicts: 直接传入绝对路径时不会失败。
- Test: 给 `load_yaml_config()` 增加“按 cwd / worktree root / resource root”解析的最小候选列表，验证两种工作目录都能加载同一配置。

### H2: 脚本入口没有把项目根加入 `sys.path`，所以只能依赖调用方的当前工作目录碰巧正确
- Supports: `scripts/run_selector_single_node.py` 在顶层直接导入 `paper_new_selector.pipeline`；从仓库根目录按脚本路径执行时立刻报 `ModuleNotFoundError`。
- Conflicts: 在 `paper-new` 目录内运行 `python -c "from paper_new_selector..."` 可以成功，因为 cwd 恰好是包根。
- Test: 在脚本最前面插入项目根到 `sys.path`，再从仓库根目录复跑脚本。

### H3: `selector.py` 的拒绝样本排序使用了“最后一次迭代时的 accept_score 缓存”，没有基于最终 selected set 全量重算
- Supports: 代码只在 `accept_scores[index] == 0.0` 时重算；最小复现里第 3 个候选的冗余惩罚保持为 0。
- Conflicts: 如果 `seed_top_k` 只选 1 个，或者拒绝样本此前从未参与评分，这个 bug 不会暴露。
- Test: 增加回归测试，要求 near-duplicate rejected candidate 在最终结果里有显著正的 `redundancy_penalty`，再修改实现让所有 rejected candidates 基于最终 selected set 重算。

### H4: 固定使用初始化集前缀样本不是“随机性不足”，而是缺少一个显式的 exemplar 选择策略
- Supports: 当前实现每轮都取前 `k` 个，没有轮换索引，也没有基于 seed 的随机采样。
- Conflicts: 如果初始化集本身已经洗牌且前缀足够多样，问题会被掩盖。
- Test: 提取一个 `_select_seed_samples()` 帮助函数，按 `meta.seed + round_id` 做稳定采样；加测试验证不同 round_id 会选到不同样本组合，同时同一 round_id 可复现。

## Experiments

### E1: 复现 selector 最终重评分缺失
- Change: 无代码改动，直接执行最小复现命令。
- Expected confirm: 被拒绝的近重复候选 `redundancy_penalty` 为 0，说明最终 selected set 没有回流到 rejected scoring。
- Result: Confirmed.

### E2: 复现脚本入口导入失败
- Change: 无代码改动，直接从仓库根目录执行脚本路径。
- Expected confirm: `ModuleNotFoundError: paper_new_selector`。
- Result: Confirmed.

### E3: 复现相对配置路径在不同 cwd 下不一致
- Change: 无代码改动，在 `paper-new` 目录执行测试。
- Expected confirm: `paper-new/configs/...` 被解析成 `paper-new/paper-new/configs/...`。
- Result: Confirmed.

### E4: 复现 Stage 1 候选生成因 prompt 过长崩溃
- Change: 修复 exemplar 采样后，执行完整小规模流水线。
- Expected confirm: `distilgpt2` 因输入长度超出上下文窗口报错。
- Result: Confirmed. 报错为 `Token indices sequence length is longer than the specified maximum sequence length ... IndexError: index out of range in self`。

## Root Cause

- 入口相关失败的根因是：`paper-new` 缺少统一的配置路径解析和脚本导入引导，导致运行是否成功依赖当前工作目录。
- selector 边界失真的根因是：拒绝候选沿用了选择过程中的旧 `accept_score` 缓存，没有按最终 selected seed 集重新计算冗余惩罚。
- Stage 1 候选多样性偏窄的根因是：缺少显式的 exemplar 采样策略，代码退化成了每轮重复使用初始化集前缀样本。
- 完整小规模流水线崩溃的根因是：引入更随机的 exemplar 采样后，初始化文本长度波动变大，而测试配置没有显式限制 Stage 1 prompt/exemplar 长度，导致 `distilgpt2` 输入越界。

## Fix

- 增加统一的配置路径解析函数，兼容 `configs/...`、`paper-new/configs/...` 和绝对路径。
- 给脚本入口补上项目根 `sys.path` 注入，并新增包内模块入口，和旧实验队列的 `python -m ...` 风格对齐。
- 在 selector 完成 seed 选择后，对所有 rejected candidates 基于最终 selected set 重新计算 `redundancy_penalty` 和 `accept_score`。
- 为 Stage 1 加入稳定可复现的 exemplar 采样函数，避免每轮固定使用初始化集前缀样本。
- 在测试配置里显式加入 `generator.max_prompt_chars` 和 `generator.max_exemplar_chars`，把 Stage 1 prompt 长度控制纳入实验超参数，而不是依赖隐式截断或随机幸运值。
## 2026-04-24 Formal Eval Bug

### Observations

- Server-side `NS-C1` did not hang. It exited with a Python traceback after the second checkpoint load finished.
- The traceback points to `paper_new_selector/eval_bridge.py` inside `_build_thesis_eval_config()`.
- The exact failure is: `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'`.
- Local reproduction matches the server failure:
  - `_build_thesis_eval_config('configs/experiments/single_node_formal/ns_c1_jobs_base.yaml')`
  - raises the same `TypeError`.
- All 9 formal configs define:
  - `train_limit: null`
  - `eval_limit: null`
  - `initialization_limit: null`
- `thesis_bridge.load_text_samples()` already treats `(None, "")` as "no limit", so Stage 1 can run with the formal configs.
- `eval_bridge._build_thesis_eval_config()` is inconsistent with that contract because it forces all three limits through `int(...)`.

### Hypotheses

#### H1: `_build_thesis_eval_config()` is the only production path that mishandles formal `null` limits (ROOT HYPOTHESIS)
- Supports: stack trace points exactly to `int(selector_cfg["data"].get("train_limit", 64))`.
- Supports: every formal config sets the three limits to `null`.
- Supports: Stage 1 passed, which means the data-loading path already tolerates `None`.
- Conflicts: none found.
- Test: preserve `None` for the three limits in `_build_thesis_eval_config()` and add a regression test over all 9 formal configs.

#### H2: formal configs themselves are wrong and should use numeric defaults instead of `null`
- Supports: `eval_bridge` currently expects numeric values.
- Conflicts: the formal intent is "full dataset", and `thesis_bridge` already models that as `None`.
- Conflicts: changing all configs to arbitrary integers would silently change experiment semantics.
- Test: compare the existing Stage 1 loader contract with formal config design; if `None` already means "full dataset", keep configs unchanged.

#### H3: there are additional `None -> int()` conversions in the formal experiment path beyond `eval_bridge`
- Supports: once one bridge drifts from config semantics, others may too.
- Conflicts: repo search shows the only formal-limit conversions are in `thesis_bridge` and `eval_bridge`, and `thesis_bridge` already guards `None`.
- Test: grep `paper_new_selector` for `train_limit|eval_limit|initialization_limit` and inspect every conversion site.

### Experiments

#### E1: Reproduce the failure locally with one formal config
- Change: no code change; directly call `_build_thesis_eval_config()` for `ns_c1_jobs_base`.
- Expected confirm: same `TypeError`.
- Result: Confirmed.

#### E2: Check whether this is a one-off config issue or a formal-config family issue
- Change: no code change; inspect `train_limit`, `eval_limit`, `initialization_limit` across all 9 formal configs.
- Expected confirm: all 9 use `None`.
- Result: Confirmed.

#### E3: Inspect all limit conversion sites in `paper_new_selector`
- Change: no code change; grep all `train_limit|eval_limit|initialization_limit` uses.
- Expected confirm: only `eval_bridge` mishandles `None`.
- Result: Confirmed.

### Root Cause

- The formal experiment configs intentionally use `null` to mean "no dataset cap", but `eval_bridge._build_thesis_eval_config()` violates that contract by forcing those `None` values through `int(...)`, so every formal experiment fails as soon as it enters the eval-config construction path.

### Fix Plan

- Add one shared helper in `eval_bridge` that preserves `None` for optional integer limits.
- Use that helper for `train_limit`, `eval_limit`, and `initialization_limit`.
- Add regression tests that:
  - reproduce the old failure path through `_build_thesis_eval_config()`
  - verify all 9 formal configs now build successfully
  - verify the three limits remain `None` in the derived thesis eval config.

## 2026-04-25 Screening Eval Limit Drift

### Observations

- `PrE-Text` screening runs report `eval_count = 256`, which matches the screening configs.
- `paper-new` screening runs report full-dataset `eval_count` values such as `1000` and `28632`, even though the screening configs also set `data.eval_limit = 256`.
- `paper_new_selector.eval_bridge._build_thesis_eval_config()` preserves `train_limit`, `eval_limit`, and `initialization_limit` in `thesis_config.data`.
- `thesis_platform.evaluation.downstream_eval._build_pretext_raw()` rebuilds a `PrE-Text` config, but its `data` section omits all three limit fields.
- `PrE-Text/pretext_platform/data/loaders.py` applies `train_limit`, `eval_limit`, and `initialization_limit` when those fields are present.

### Hypotheses

#### H1: `_build_pretext_raw()` drops the three data limits, so `paper-new` screening eval silently falls back to full datasets (ROOT HYPOTHESIS)
- Supports: `thesis_config.data` already contains the limit fields before the call.
- Supports: `_build_pretext_raw()` copies dataset paths and other data fields, but not the limits.
- Supports: `PrE-Text` loaders honor the limits if present.
- Conflicts: none found.
- Test: assert that `_build_pretext_raw()` output contains the three limits for a screening config.

#### H2: the limits are present in `_build_pretext_raw()` but later lost when building `PretextExperimentConfig`
- Supports: there is an additional config conversion layer after `_build_pretext_raw()`.

## 2026-05-07 Round19 Genericity Length-Modulation Crash

### Observations

- Server-side `round19` quick comparison runs all failed inside `paper_new_selector.stage1_runner.run_stage1_with_runtime()`.
- The exact error is `TypeError: compute_genericity_penalties() got an unexpected keyword argument 'candidate_lengths'`.
- The failing call site is [paper_new_selector/stage1_runner.py](paper_new_selector/stage1_runner.py), which passes `candidate_lengths`, `length_modulation_enabled`, `length_alpha`, `length_factor_min`, and `length_factor_max`.
- The current implementation in [paper_new_selector/genericity.py](paper_new_selector/genericity.py) does not accept any of those keyword arguments.
- Local `paper-new-round19` reproduces the same mismatch, so this is not a server-only sync problem.
- Existing test coverage already assumes the upgraded call contract:
  - `tests/test_stage1_runner.py::test_stage1_runner_passes_length_modulation_config_to_genericity`
  - design docs under `docs/2026-04-27-round5-*` also specify the intended `genericity.py` extension.
- Existing tests did not catch the crash because `stage1_runner` tests patch `compute_genericity_penalties` with a mock instead of exercising the real implementation.

### Hypotheses

#### H1: `stage1_runner.py` was upgraded, but `genericity.py` was never finished to match the new keyword-only interface (ROOT HYPOTHESIS)
- Supports: direct signature mismatch between caller and callee.
- Supports: design docs and runner tests both describe the extended interface, but the production implementation does not.
- Supports: local and server copies fail the same way.
- Conflicts: none found.
- Test: add a regression test that calls `compute_genericity_penalties()` with the new length-modulation kwargs and verify it fails before the fix, then implement the documented interface and rerun.

#### H2: the server sync was incomplete, and only the server copy is inconsistent
- Supports: server was the first place the bug surfaced during actual quick experiments.
- Conflicts: local `genericity.py` has the same missing parameters, so the mismatch already exists before sync.
- Test: compare local call site and local implementation signatures.

#### H3: the intended fix is to remove the new kwargs from `stage1_runner.py`, not to extend `genericity.py`
- Supports: removing kwargs would avoid the crash with minimal code change.
- Conflicts: runner tests explicitly assert those kwargs are forwarded.
- Conflicts: round5 design docs define the length-modulated genericity behavior as part of the algorithm, so stripping the kwargs would regress intended functionality.
- Test: inspect tests and design docs for the expected public contract.

### Experiments

#### E1: Local signature comparison
- Change: no code change; inspect `stage1_runner.py`, `genericity.py`, tests, and docs.
- Expected confirm: caller and tests require length-modulation kwargs, callee does not implement them.
- Result: Confirmed.

#### E2: Server/local sync falsification
- Change: no code change; inspect the local `genericity.py`.
- Expected confirm: if local is already missing the kwargs, then this is not only a deployment sync issue.
- Result: Confirmed.

#### E3: Regression test for the real implementation
- Change: add a test that calls the real `compute_genericity_penalties()` with `candidate_lengths` and length-modulation kwargs.
- Expected confirm: current code fails before any production fix is applied.
- Result: Confirmed. Before the fix, both new regression tests failed with `TypeError: compute_genericity_penalties() got an unexpected keyword argument 'candidate_lengths'`.

### Root Cause

- `paper_new_selector/stage1_runner.py` had already been upgraded to forward the round5/round19 length-modulation genericity parameters, but `paper_new_selector/genericity.py` was still on the older interface and implementation.
- This is not merely a local/server sync issue: the same caller/callee mismatch exists in the local repository, and the server only surfaced it first because the quick-comparison batch exercised the real implementation path.
- Existing tests missed the problem because the `stage1_runner` coverage mocked `compute_genericity_penalties()` instead of executing the concrete function with the forwarded kwargs.

### Fix

- Extend `compute_genericity_penalty()` to accept the documented length-modulation parameters and apply modulation only when `length_modulation_enabled=True`, `candidate_length` is present, `l_ref` is present, and `length_alpha != 0.0`.
- Extend `compute_genericity_penalties()` to:
  - accept `candidate_lengths` plus the new modulation kwargs
  - validate vector/length alignment
  - compute a batch median `l_ref`
  - preserve exact old behavior when modulation is disabled
- Add regression tests that:
  - prove the new kwargs are accepted without changing disabled behavior
  - prove non-zero `length_alpha` modulates penalties against the batch median length
- Verification:
  - targeted regression tests pass
  - full suite passes: `python -m unittest discover -s tests -p "test_*.py" -v` -> `Ran 98 tests ... OK`
- Conflicts: inspecting `_build_pretext_raw()` already shows the keys are absent at the raw-mapping stage.
- Test: compare `thesis_config.data` with raw output before `PretextExperimentConfig.from_mapping(...)`.

#### H3: `PrE-Text` small eval ignores `eval_limit` even when the config carries it
- Supports: the user-visible symptom is a wrong `eval_count`.
- Conflicts: `pretext_platform.data.loaders.load_dataset_bundle()` clearly applies `_apply_limit(..., eval_limit, ...)`.
- Test: verify the raw config passed into `PrE-Text` actually carries the field; if it does, the loader path should be correct.

### Experiments

#### E1: Compare `paper-new` eval config with `PrE-Text` raw config for one screening run
- Change: no code change; inspect `_build_thesis_eval_config()` output and `_build_pretext_raw()` output for `ns_s_jobs_screening.yaml`.
- Expected confirm: `thesis_config.data.eval_limit == 256`, but `_build_pretext_raw(...)[\"data\"].get(\"eval_limit\") is None`.
- Result: Confirmed.

### Root Cause

- `paper-new` correctly stores screening dataset caps in `thesis_config.data`, but `thesis_platform.evaluation.downstream_eval._build_pretext_raw()` drops `train_limit`, `eval_limit`, and `initialization_limit` when translating that config into the `PrE-Text` eval config, so downstream eval runs on full datasets instead of the intended screening subset.

### Fix Plan

- Add `train_limit`, `eval_limit`, and `initialization_limit` to `_build_pretext_raw(...)[\"data\"]`.
- Add one regression test in `thesis_platform` for the raw bridge.
- Add one end-to-end regression test in `paper-new` that proves a screening config keeps its limits all the way into the generated `PrE-Text` raw config.

## 2026-04-25 vLLM Rendezvous Host Drift

### Observations

- The restarted remote `NS-S-JOBS` run failed in `Stage 1` before any output directory was created.
- The traceback is not an OOM. It fails inside `torch.distributed.init_process_group(...)` while `vllm.LLM(...)` is starting.
- The exact error is a rendezvous connect timeout to `node03:45937`, then to `192.168.1.113:45937`.
- On the server, `env | grep -E 'HOST_IP|VLLM_HOST_IP|MASTER_ADDR'` returns nothing, so there is no explicit override.
- On the server, `/etc/hosts` maps `node03` to `192.168.1.113`.
- On the same server, `hostname -I` reports live interface addresses `10.168.1.100`, `172.26.0.1`, `172.17.0.1`.
- Current code in both `thesis_platform.models.backends` and `paper_new_selector.pretext_bridge` resolves the fallback host as:
  - first explicit `VLLM_HOST_IP/HOST_IP`
  - otherwise `socket.gethostbyname(socket.gethostname())`
- The patch layer currently uses `os.environ.setdefault(...)`, so it preserves any pre-existing bad values instead of forcing a safe rendezvous address.

### Hypotheses

#### H1: fallback host resolution is using `/etc/hosts` hostname mapping (`192.168.1.113`), which is not a valid local rendezvous address for the current vLLM/torch.distributed startup path (ROOT HYPOTHESIS)
- Supports: the error connects exactly to `node03` / `192.168.1.113`.
- Supports: the server has no explicit `HOST_IP/VLLM_HOST_IP`, so the fallback path is active.
- Supports: `/etc/hosts` maps `node03` to `192.168.1.113`, while the live host IPs are different.
- Conflicts: previous runs succeeded, so the bug is environment-sensitive rather than deterministic.
- Test: make the default fallback `127.0.0.1`, force-export that value into `VLLM_HOST_IP`, `HOST_IP`, and `MASTER_ADDR`, then verify the patch helper returns loopback when no override is present.

#### H2: the monkey-patch to `vllm.utils.get_ip` / `vllm.engine.llm_engine.get_ip` is correct, but `torch.distributed` is reading `MASTER_ADDR` from somewhere else and bypassing the patch
- Supports: the failing stack is inside `torch.distributed.init_process_group`.
- Conflicts: there is no `MASTER_ADDR` in the environment, so without an explicit set this still depends on vLLM's chosen address.
- Test: set `MASTER_ADDR` alongside `VLLM_HOST_IP/HOST_IP` and verify the patch helper exports all three consistently.

#### H3: the problem is only in `paper-new`, while `thesis_platform` Stage 1 startup is fine
- Supports: the visible failure happened in `paper-new`.
- Conflicts: `paper-new` Stage 1 uses `thesis_platform.models.backends.VllmTextBackend`, so both code paths share the same fallback logic.
- Test: fix the shared helper in `thesis_platform` and mirror the same rule in `paper-new/pretext_bridge`.

### Experiments

#### E1: Inspect remote runtime environment and hostname mapping
- Change: no code change; read remote `env`, `/etc/hosts`, and `hostname -I`.
- Expected confirm: no explicit host override vars, but hostname resolution path points at `192.168.1.113`.
- Result: Confirmed.

### Root Cause

- When no explicit `VLLM_HOST_IP/HOST_IP` is set, the current vLLM patch code falls back to `socket.gethostbyname(socket.gethostname())`, and on the server that resolves `node03` to `/etc/hosts` entry `192.168.1.113`; vLLM then feeds that address into `torch.distributed` rendezvous, which times out. The fallback is therefore not a safe default for these single-host experiments.

### Fix Plan

- Change the no-override fallback host to `127.0.0.1` in both `thesis_platform.models.backends` and `paper_new_selector.pretext_bridge`.
- Force-export the resolved host into `VLLM_HOST_IP`, `HOST_IP`, and `MASTER_ADDR` instead of using `setdefault(...)`.
- Add regression tests that verify:
  - no-env fallback chooses loopback
  - explicit env override is still honored
  - the patch helper exports the resolved host consistently.

## 2026-05-07 Round19 Jobs vLLM Cache-Block Startup Failure

### Observations

- `round19` quick comparison finished with `9/10` success; the only failure was `r19_full_jobs`.
- The failing server log ends with:
  - `# GPU blocks: 0, # CPU blocks: 512`
  - `No available memory for the cache blocks. Try increasing gpu_memory_utilization when initializing the engine.`
- `r19_full_jobs`, `r19_full_congressional`, `r19_full_forums`, and `r19_full_microblog` all inherit the same Stage 1 vLLM config from the same base. The dataset leaf files only change dataset paths and output roots.
- `paper-new-round19`, `paper-new`, and `PrE-Text` all intentionally use local `vllm` + `llama_2_7b_hf` for synthetic generation, with the same shared budget shape:
  - `max_model_len: 512`
  - `gpu_memory_utilization: 0.35`
  - `startup_required_free_gb: 2`
  - `enforce_eager: true`
- Successful `round19` runs immediately before/after the failure used the same vLLM config and succeeded:
  - `r19_full_congressional`: `free=34.39 GiB`, `# GPU blocks: 457`
  - `r19_full_forums`: `free=34.38 GiB`, `# GPU blocks: 457`
  - `r19_full_microblog`: `free=34.39 GiB`, `# GPU blocks: 457`
- The failed `r19_full_jobs` run reported even more free memory before startup (`free=43.87 GiB`) but still got `# GPU blocks: 0`, which argues against a deterministic config insufficiency.
- Stage 1 candidate generation in `paper-new-round19` currently calls `round_ctx.text_backend.generate(...)` directly through `PretextPromptLLMGenerator.generate()` with no local retry path around transient vLLM startup failures.
- `PrE-Text` has an explicit startup-memory precheck for Stage 2 vLLM, but `paper-new-round19` Stage 1 has no equivalent retry/recovery wrapper for a transient `cache blocks` startup failure.

### Hypotheses

#### H1: this is a transient vLLM startup failure on a shared GPU, and `paper-new-round19` incorrectly treats it as a permanent configuration error because Stage 1 has no release-and-retry wrapper (ROOT HYPOTHESIS)
- Supports: identical vLLM config succeeds for the other `full_run` datasets.
- Supports: the failed run had *more* free memory than the successful runs, so the failure does not fit a stable "budget too small" explanation.
- Supports: the failure happens exactly at engine startup before generation progresses, which is the place where a transient retry is feasible.
- Conflicts: none found so far.
- Test: add a regression test where Stage 1 generation raises `ValueError("No available memory for the cache blocks")` on the first call and succeeds on the second; verify `run_stage1()` releases runtime memory, sleeps/backs off, and retries once.

#### H2: the `jobs` dataset leaf overrides some hidden vLLM parameter not present in the other datasets
- Supports: only `r19_full_jobs` failed.
- Conflicts: local config comparison shows the dataset leaf only changes dataset paths/output root; Stage 1 vLLM settings come from the same inherited base.
- Test: compare `r19_full_jobs.yaml`, `r19_full_congressional.yaml`, and the shared base config.

#### H3: `gpu_memory_utilization=0.35` is fundamentally too low for all `round19 full_run` experiments, and the other three only succeeded by luck
- Supports: the raw vLLM message suggests increasing `gpu_memory_utilization`.
- Conflicts: the same budget is intentionally used in `paper-new` and `PrE-Text`, and the other `round19` full runs completed under that same budget.
- Conflicts: a deterministic budget insufficiency would not explain `jobs` failing with *higher* observed free memory than the succeeding runs.
- Test: compare inherited configs and successful logs before changing any budget parameter.

### Experiments

#### E1: Compare dataset leaf configs and inherited vLLM settings
- Change: no code change; inspect `r19_full_jobs`, `r19_full_congressional`, and the shared base config.
- Expected confirm: the leaf files do not override the Stage 1 vLLM budget.
- Result: Confirmed.

#### E2: Compare failed and successful server startup logs
- Change: no code change; inspect `r19_full_jobs.log` and the successful `full_run` logs.
- Expected confirm: identical vLLM constructor shape, but only `jobs` hits `# GPU blocks: 0`.
- Result: Confirmed.

#### E3: Compare with `paper-new` / `PrE-Text` vLLM generation setup
- Change: no code change; inspect local config and runtime guard code.
- Expected confirm: the projects intentionally share the same local-model + vLLM budget, so changing only the `jobs` dataset config would be a weak explanation.
- Result: Confirmed.

#### E4: Regression test for retryable Stage 1 cache-block startup failures
- Change: add a Stage 1 regression test where generator startup raises `ValueError("No available memory for the cache blocks.")` once and succeeds on the second attempt.
- Expected confirm: current code fails without a local recovery path; fixed code should release the backend and retry once.
- Result: Confirmed. The new test failed before the fix and now passes.

### Root Cause

- `r19_full_jobs` did not fail because the `jobs` config uses a different vLLM/local-model setup. The `round19` full-run datasets share the same inherited Stage 1 vLLM budget, and the same budget also exists in `paper-new` and `PrE-Text`.
- The failure is a transient vLLM startup condition on the shared GPU path: the engine sometimes comes up with `# GPU blocks: 0` and raises `No available memory for the cache blocks`, even though the same configuration succeeds immediately before/after on the same A6000.
- `paper-new-round19` Stage 1 treated that specific startup failure as a permanent configuration/runtime error and had no local release-and-retry path, so `r19_full_jobs` aborted instead of self-recovering.

### Fix

- Add a narrow local recovery path in `paper_new_selector.stage1_runner`:
  - detect the retryable vLLM startup message `No available memory for the cache blocks`
  - release the Stage 1 text backend runtime
  - wait briefly
  - retry candidate generation once
- Keep all other exceptions non-retryable so configuration errors and unrelated runtime failures still surface immediately.
- Add a regression test in `tests/test_stage1_runner.py` that proves Stage 1 now retries exactly once after this retryable startup failure.
- Verification:
  - targeted regression test passes
  - full suite passes: `python -m unittest discover -s tests -p "test_*.py" -v` -> `Ran 99 tests ... OK`
