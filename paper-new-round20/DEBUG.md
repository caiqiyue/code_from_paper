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

## 2026-05-08 Round20 A6000 30GiB+ GPU Spike

### Observations

- `round20` 的 merged config 没有把 vLLM/Stage2 显存参数调大。
  - `r19_full_microblog.yaml` 与 `r20_microblog_arbitration.yaml` 的关键参数一致：
  - `llm.generator.gpu_memory_utilization = 0.35`
  - `bootstrap.gpu_memory_utilization = 0.35`
  - `max_model_len = 512`
  - `tensor_parallel_size = 1`
- 服务器上的 `jobs` 两项失败不是算法逻辑崩溃，而是启动期 OOM：
  - `r20_jobs_baseline_fallback.log` 和 `r20_jobs_arbitration.log` 都报
    `vllm_runtime_gpu_oom`
  - 日志显示当时 A6000 上同时还有别的进程：
    - `Process 13576 ... 6.21 GiB`
    - `Process 22744 ... 10.40/11.51 GiB`
    - `Process 29592 ... 16.66/18.96 GiB`
- `microblog` 两项都能完整跑完，说明 `round20` 配置本身不是“必然 OOM”。
  - `r20_microblog_baseline_fallback` 完成并写出结果。
  - `r20_microblog_arbitration` 也完成并写出结果。
- 运行中的 `round20` Python 进程 `PID 11279` 在 A6000 上一度占用约 `34.9 GiB`。
- `r20_microblog_baseline_fallback` 的产物表明：
  - `stage2.generation_path = "shared_session"`
  - 也就是 Stage 2 不是单独新起 bootstrap vLLM，而是复用 Stage 1 的 shared vLLM backend。
- `pipeline.py` 的执行顺序是：
  1. `run_stage1_with_runtime(...)`
  2. 若有 `shared_session`，用它完成 Stage 2 generation
  3. 在 `finally:` 中调用 `release_runtime_memory(stage1_text_backend, embedder)`
  4. 然后继续做 `run_eval(...)`
- `VllmTextBackend.release()` 只是：
  - 把 `self._llm = None`
  - 查找 `llm.release`
  - 仅当 `llm.release` 可调用时才真正执行 release
- 服务器 `pretext` 环境实测：
  - `from vllm import LLM`
  - `hasattr(LLM, "release") == False`
- 这意味着当前 `release_runtime_memory(text_backend)` 对 vLLM engine 实际上不会触发真正的 unload/shutdown。

### Hypotheses

#### H1: `round20` 的 30GiB+ 峰值来自 Stage 1 vLLM session 没有被真正释放，随后又叠加了后续阶段的 GPU 占用（ROOT HYPOTHESIS）
- Supports:
  - `stage2.generation_path = "shared_session"`，说明 Stage 1 的 vLLM engine 会跨到 Stage 2。
  - `pipeline.py` 只在 Stage 2 之后才尝试 release Stage 1 backend。
  - `VllmTextBackend.release()` 依赖 `llm.release()`。
  - 服务器实测 `vllm.LLM` 没有 `release()`。
  - 因此 Stage 1 vLLM engine 很可能一直活到进程退出，而不是在 eval 前释放。
- Conflicts:
  - 同样的 backend/release 逻辑在 `round19` 里也存在，所以这不是“只在 round20 新引入的 bug”。
  - 更准确地说，这是一个被 `round20` 运行现象清楚暴露出来的 inherited runtime-lifetime bug。
- Test:
  - 确认 `round20` 的 Stage 2 走的是 shared session。
  - 确认服务器 vLLM 类没有 `release()`，使当前 `release()` 路径失效。

#### H2: `round20` 的显存暴涨是因为配置继承错误，把 `gpu_memory_utilization` 或模型长度调大了
- Supports:
  - 这是最直观的怀疑方向。
- Conflicts:
  - merged config 对比显示 `round19` 和 `round20` 的 vLLM / bootstrap 核心显存参数完全一致。
- Test:
  - 比较 merged config 中的 `llm.generator` 与 `bootstrap` 关键字段。

#### H3: `round20` arbitration 代码额外实例化了第二个 Stage 1 generator/backend
- Supports:
  - `round20` 新增了 uncertain arbitration 逻辑，理论上可能重复构建策略或 runtime。
- Conflicts:
  - 本地代码检查显示 arbitration 只复用 `metrics_by_budget` 和 policy summaries，不会新建第二个 text backend。
  - `hierarchical_budget.py` 只做 budget resolution，不接触 vLLM backend 生命周期。
- Test:
  - 审查 `hierarchical_budget.py`、`uncertainty_arbitration.py`、`stage1_runner.py` 的对象构建路径。

#### H4: 30GiB+ 主要是外部进程抢卡导致的错觉，不是 `round20` 自身生命周期问题
- Supports:
  - `jobs` 两项确实因为别的进程占卡而 OOM。
- Conflicts:
  - `microblog` 两项在相同外部环境下能跑完，但本进程仍冲到 30GiB+。
  - 这说明“外部抢卡”解释了 `jobs` 的失败，但解释不了 `round20` 单进程为何长期占到 30GiB+。
- Test:
  - 将 `jobs` 的 OOM 日志与 `microblog` 的成功日志分开分析。

### Experiments

#### E1: 对比 `round19` 与 `round20` merged config 的显存参数
- Change: 无代码改动，只加载 merged config。
- Expected confirm:
  - 若参数一致，则排除“round20 配置变大”。
- Result: Confirmed.
  - `gpu_memory_utilization/max_model_len/tensor_parallel_size` 等关键参数一致。

#### E2: 检查 `microblog` 成功运行时的 Stage 2 generation path
- Change: 无代码改动，只读 `r20_microblog_baseline_fallback` / `r20_microblog_arbitration` 产物。
- Expected confirm:
  - 若 `generation_path == shared_session`，则 Stage 1 vLLM engine 会跨到 Stage 2。
- Result: Confirmed.

#### E3: 检查服务器 `pretext` 环境里 `vllm.LLM` 是否有 `release()`
- Change: 无代码改动，只在服务器环境执行：
  - `from vllm import LLM`
  - `hasattr(LLM, "release")`
- Expected confirm:
  - 若为 `False`，则当前 `VllmTextBackend.release()` 的真正 unload 路径失效。
- Result: Confirmed.
  - `has_release False`

#### E4: 检查 `jobs` 失败与 `microblog` 成功是否属于同一类问题
- Change: 无代码改动，只读远端日志。
- Expected confirm:
  - `jobs` 如果是启动期 OOM，而 `microblog` 成功完成，则“30GiB+ 占用”与 `jobs` 失败不是同一个根因。
- Result: Confirmed.
  - `jobs` 是启动期 OOM。
  - `microblog` 成功跑完但仍暴露出大显存峰值。

### Root Cause

- `round20` 出现 30GiB+ A6000 占用的直接根因，不是算法配置变大，也不是 arbitration 本身额外创建了第二个 Stage 1 runtime，而是 **Stage 1 的 vLLM engine 在进入后续阶段后没有被真正释放**。
- 具体机制是：
  - Stage 2 走 `shared_session`，所以 Stage 1 的 vLLM backend 会继续存活到 Stage 2 结束。
  - `pipeline.py` 虽然在 Stage 2 后调用了 `release_runtime_memory(stage1_text_backend, ...)`，
    但 `VllmTextBackend.release()` 依赖 `llm.release()`。
  - 服务器 `pretext` 环境中的 `vllm.LLM` 并没有 `release()` 方法，因此这条 release 路径实际上不执行真正的 vLLM teardown。
  - 结果是 Stage 1 的 vLLM 显存常驻，随后再叠加 eval 等后续 GPU 开销，就把单进程峰值推到了 30GiB+。
- 这解释了为什么：
  - `jobs` 在外部进程抢卡时更容易直接 OOM；
  - `microblog` 即使能跑完，也会表现出远高于单个 0.35-utilization vLLM 的显存峰值。

### Fix Plan

- 不应继续假设 `VllmTextBackend.release()` 能靠 `llm.release()` 完成卸载。
- 下一步修复需要选择一种“可证实真正释放 GPU”的策略，而不能拍脑袋：
  1. 先确认 vLLM 是否存在别的正式 shutdown API；
  2. 如果没有正式 API，则应考虑用**进程边界**隔离 Stage 1/Stage 2 与 eval，而不是继续在同一 Python 进程里堆叠生命周期。
- 在找到可证实的释放方案前，不应贸然写一个“看起来会释放”的假清理逻辑。

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

## 2026-05-07 Round19 Repeat15 Runner Wrong Project Root On Server

### Observations

- The local `repeat15` driver was synced to the server and launched from `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round19`.
- The repeat15 master log on the server recorded config paths under the parent directory instead of the repo:
  - `/mnt/public/caiqiyue_file/code_from_paper/tmp_round19_repeat15/...`
  - not `/mnt/public/caiqiyue_file/code_from_paper/paper-new-round19/tmp_round19_repeat15/...`
- The first failed run log reported:
  - `/home/k8smaster/anaconda3/bin/python: Error while finding module specification for 'paper_new_selector.run_selector_single_node' (ModuleNotFoundError: No module named 'paper_new_selector')`
- `repeat15_runner.py` currently derives `root` from `resolve_worktree_root()`.
- `resolve_worktree_root()` is designed around the larger thesis resource layout, not specifically the `paper-new-round19` repo root. On the server that larger root resolves to `/mnt/public/caiqiyue_file/code_from_paper`, which contains `thesis_platform` but not the `paper_new_selector` package for this repo.
- The bug only appeared in the repeat15 driver. Existing single-run `round19` commands succeeded because they were launched manually from the repo root and did not re-derive root through this path.

### Hypotheses

#### H1: `repeat15_runner.py` is using the thesis resource root instead of the `paper-new-round19` repo root, so it writes logs/configs one level too high and launches subprocesses with the wrong `cwd` (ROOT HYPOTHESIS)
- Supports: server master log paths are rooted at `/mnt/public/caiqiyue_file/code_from_paper/tmp_round19_repeat15/...`.
- Supports: the failing subprocess error is `ModuleNotFoundError: No module named 'paper_new_selector'`, which is exactly what happens when `cwd` is set above the repo that contains the package.
- Supports: the runner currently calls `resolve_worktree_root()`, which can legitimately return the larger thesis resource root.
- Conflicts: none.
- Test: add a regression test that simulates the server path `/mnt/public/.../paper-new-round19/paper_new_selector/repeat15_runner.py` and requires the runner root to be `/mnt/public/.../paper-new-round19`; add a second test that requires `run_repeat15_batch()` to use that repo root for `cwd`, logs, and temp configs.

#### H2: the failure comes from the server not activating the `pretext` environment correctly
- Supports: the failing subprocess is a Python module import error.
- Conflicts: `python -m paper_new_selector.repeat15_runner` itself started correctly under `pretext`, which means the environment activation was successful.
- Conflicts: the failing import is inside the child process launched by the driver, which depends on `cwd`/module path more than on the already-activated environment.
- Test: compare parent-process success with child-process failure and verify the child `cwd` is wrong.

#### H3: the generated temp YAML paths are wrong, but the subprocess `cwd` is correct
- Supports: the master log showed the temp config path rooted at the parent directory.
- Conflicts: the child error is package import failure, which points to a bad `cwd` or module search root, not only a bad config path.
- Test: add a regression test that captures both generated config path and subprocess `cwd`; require both to stay under the repo root.

### Experiments

#### E1: server-side failure inspection
- Change: no code change; inspect the server repeat15 nohup/master logs and the first failed experiment log.
- Expected confirm: runner paths resolve to the parent thesis directory and child imports fail from there.
- Result: Confirmed.

#### E2: local regression tests for repo-root resolution
- Change: add two tests in `tests/test_repeat15_runner.py`:
  - `resolve_repeat15_project_root()` must return the repo parent of `paper_new_selector/repeat15_runner.py`
  - `run_repeat15_batch(temp_root)` must use `temp_root` for `cwd`, logs, summary, and temp configs
- Expected confirm: current code lacks that behavior and the new tests fail before the fix.
- Result: Confirmed. The new test module fails before the fix because `resolve_repeat15_project_root` does not exist yet and `run_repeat15_batch` has no repo-root override path.

### Root Cause

- `repeat15_runner.py` used the thesis resource-oriented `resolve_worktree_root()` to derive its runtime root. On the server that resolves to `/mnt/public/caiqiyue_file/code_from_paper`, not the actual `paper-new-round19` repository. The driver therefore wrote temp configs/logs one directory too high and launched child runs with the wrong `cwd`, causing `paper_new_selector` imports to fail.

### Fix Plan

- Add a dedicated `resolve_repeat15_project_root()` that anchors to the repo containing `paper_new_selector/repeat15_runner.py`, not the broader thesis resource root.
- Let `run_repeat15_batch()` accept an optional explicit project root so tests can pin the runtime root and the CLI path can still default safely.
- Keep the fix local to `repeat15_runner.py` instead of weakening `thesis_bridge.resolve_worktree_root()` for the rest of the codebase.
- Verify with:
  - targeted repeat15 tests first
  - then full `python -m unittest discover -s tests -p "test_*.py" -v`

## 2026-05-07 Round19 Repeat15 OOM While A6000 Still Has Free Memory

### Observations

- The repeat15 batch was relaunched from `paper-new-round19` with `pretext` active and `CUDA_VISIBLE_DEVICES=1`.
- The first several experiments all failed quickly with the same error family:
  - `thesis_platform.models.backends.VllmGenerationError: vllm_runtime_gpu_oom`
- The failing logs report:
  - `vLLM generation memory precheck | free=10.58 GiB required=2.00 GiB gpu=0 visible=1`
  - `torch.cuda.OutOfMemoryError ... GPU 0 has a total capacty of 10.75 GiB`
- `10.75 GiB` matches the server's `NVIDIA GeForce RTX 2080 Ti`, not the `NVIDIA RTX A6000` (49 GiB).
- At the same time, the user-provided `nvidia-smi` snapshot showed the A6000 still had large free memory, so the OOM cannot be explained by true exhaustion of the intended target card.
- `VllmTextBackend` does not pass a device ordinal into vLLM; the effective device selection is driven by the launcher environment.
- The current `scripts/run_round19_full_repeat15.sh` script does not export `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- Historical `paper-new` / `paper-new-round19` vLLM docs and run notes repeatedly pair:
  - `export CUDA_DEVICE_ORDER=PCI_BUS_ID`
  - `export CUDA_VISIBLE_DEVICES=1`
- CUDA ordinal order is not guaranteed to match `nvidia-smi` index order unless `CUDA_DEVICE_ORDER=PCI_BUS_ID` is fixed.

### Hypotheses

#### H1: repeat15 relies on `CUDA_VISIBLE_DEVICES=1` but does not force `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so CUDA ordinal `1` resolves to the 2080 Ti instead of the A6000 (ROOT HYPOTHESIS)
- Supports: the failing process clearly sees a `10.75 GiB` GPU, which matches the 2080 Ti.
- Supports: the log still reports `visible=1`, so the environment variable exists but may be interpreted under the wrong device order.
- Supports: earlier successful experiment docs explicitly include `CUDA_DEVICE_ORDER=PCI_BUS_ID` alongside `CUDA_VISIBLE_DEVICES=1`.
- Conflicts: none.
- Test: add a regression test that requires repeat15 child processes to run with `CUDA_DEVICE_ORDER=PCI_BUS_ID` whenever `CUDA_VISIBLE_DEVICES` is present; confirm current code does not set that env and the test fails.

#### H2: vLLM startup memory precheck is querying the wrong GPU, but the actual model load still targets the A6000
- Supports: precheck and runtime both mention `gpu=0`, which could be logical index only.
- Conflicts: runtime OOM total capacity is still `10.75 GiB`, so the actual model load also landed on the smaller card.
- Test: inspect runtime traceback and compare total capacity with server GPU inventory.

#### H3: the batch really is running on A6000, but another hidden process already consumed ~38 GiB before vLLM startup
- Supports: A6000 did have other processes on it.
- Conflicts: if that were true, the runtime total capacity would still be ~49 GiB rather than `10.75 GiB`.
- Test: compare reported total device capacity in the traceback with `nvidia-smi`.

### Experiments

#### E1: compare failure capacity with server inventory
- Change: no code change; compare the traceback's total capacity with the user-provided `nvidia-smi`.
- Expected confirm: the failing process is on the 2080 Ti, not the A6000.
- Result: Confirmed. `10.75 GiB` matches the 2080 Ti.

#### E2: compare repeat15 launch path with earlier successful launch conventions
- Change: no code change; inspect local scripts/docs for prior vLLM launches.
- Expected confirm: earlier successful runs explicitly set `CUDA_DEVICE_ORDER=PCI_BUS_ID`, while repeat15 does not.
- Result: Confirmed.

#### E3: regression test for repeat15 child env
- Change: add a repeat15 test that captures `subprocess.run(..., env=...)` and requires `CUDA_DEVICE_ORDER=PCI_BUS_ID` to be present when `CUDA_VISIBLE_DEVICES` is set.
- Expected confirm: current runner does not enforce this, so the test fails before the fix.
- Result: Confirmed. The new test failed before the fix because repeat15 child runs inherited no explicit `CUDA_DEVICE_ORDER`.

### Root Cause

- The repeat15 execution path depended on `CUDA_VISIBLE_DEVICES=1` but did not also pin `CUDA_DEVICE_ORDER=PCI_BUS_ID`. On this server, CUDA ordinal order is not matching the `nvidia-smi` display order, so ordinal `1` resolved to the smaller 2080 Ti. That is why the failing logs reported `visible=1` but a total GPU capacity of only `10.75 GiB`. The OOM was therefore not a real A6000 exhaustion event; it was a wrong-device selection bug in the new repeat15 launch path.

### Fix Plan

- Make repeat15 explicitly stabilize CUDA ordinal mapping for child runs by exporting/passing `CUDA_DEVICE_ORDER=PCI_BUS_ID` alongside the inherited `CUDA_VISIBLE_DEVICES`.
- Keep the fix local to `paper_new_selector.repeat15_runner` and `scripts/run_round19_full_repeat15.sh`, since the failure was introduced by this new repeat15 execution path.
- Add a regression test that inspects the child process env passed by `run_repeat15_batch()`.

### Fix

- Added `build_repeat15_child_env()` in `paper_new_selector.repeat15_runner`:
  - inherit the parent environment
  - if `CUDA_VISIBLE_DEVICES` is set and `CUDA_DEVICE_ORDER` is absent, force `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  - pass that env explicitly to every `subprocess.run(...)` child launch
- Hardened `scripts/run_round19_full_repeat15.sh` with the same guard so shell-based launches also pin PCI bus ordering before Python starts.
- Added a regression test in `tests/test_repeat15_runner.py` that captures the child env and requires:
  - `CUDA_VISIBLE_DEVICES=1`
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- Verification:
  - `python -m unittest tests.test_repeat15_runner -v` -> pass
  - `python -m unittest discover -s tests -p "test_*.py" -v` -> `Ran 105 tests ... OK`

## 2026-05-08 Round20 A6000 30GiB+ GPU Spike

### Observations

- `round20` on the server showed a single Python process on the A6000 occupying `34GiB+`, which is much higher than earlier quick-compare runs usually observed.
- `jobs` failed during startup with `vllm_runtime_gpu_oom`, but `microblog` completed, so the spike was not caused by a universal config regression.
- Comparing `round19` and `round20` quick-compare configs showed no material Stage 1 / Stage 2 memory knob increase:
  - same `gpu_memory_utilization`
  - same `max_model_len`
  - same `tensor_parallel_size`
- Successful `round20 microblog` summaries reported `stage2.generation_path = "shared_session"`, so Stage 2 reused the Stage 1 vLLM backend.
- Local `pipeline.py` released runtime memory only after Stage 2 and then called downstream eval in the **same Python process**.
- `thesis_platform.models.backends.VllmTextBackend.release()` only does a hard release if the underlying `vllm.LLM` object exposes `release()`.
- The server `pretext` environment's `vllm.LLM` does **not** expose `release()`:
  - `hasattr(LLM, "release") == False`
  - inspecting `vllm/entrypoints/llm.py` confirmed no release/close/shutdown API exists on that class.

### Hypotheses

#### H1: round20 accidentally increased generation/eval memory settings, so the 30GiB+ spike is simply expected behavior
- Supports: the observed process memory was much larger than the user expected.
- Conflicts: merged config inspection showed the relevant vLLM knobs match round19.
- Test: compare merged `round19` and `round20` config values for Stage 1 / Stage 2 runtime settings.

#### H2: the arbitration algorithm launches two Stage 1 vLLM engines or doubles model residency (ROOT HYPOTHESIS REJECTED)
- Supports: the new innovation is in the `uncertain` path, so it is a natural first suspect.
- Conflicts: local code inspection showed arbitration only reuses in-memory budget metrics and does not construct another generator backend.
- Test: inspect `hierarchical_budget.py`, `uncertainty_arbitration.py`, and the runtime pipeline for extra backend creation.

#### H3: Stage 1 vLLM survives too long because eval runs in the same process after a non-effective `release()` path, so memory residency persists into downstream eval (ROOT HYPOTHESIS)
- Supports: Stage 2 uses `shared_session`, so Stage 1 backend remains alive through synthetic generation.
- Supports: the server vLLM build does not provide `LLM.release()`, so `release_runtime_memory(...)` cannot actually tear down the engine.
- Supports: eval was being called in the same Python process immediately after the weak release path.
- Test: verify the installed vLLM API, then isolate eval behind a subprocess boundary and require pipeline tests to observe subprocess eval instead of in-process eval.

### Experiments

#### E1: compare round19 and round20 merged configs
- Change: inspect merged configs locally and on the server.
- Expected confirm: if memory knobs are identical, the spike is not caused by a larger round20 config.
- Result: Confirmed. Round19 and round20 use the same relevant Stage 1 / Stage 2 vLLM settings.

#### E2: inspect vLLM release behavior
- Change: inspect local bridge code and query the server environment for `hasattr(vllm.LLM, "release")`.
- Expected confirm: if `release()` is absent, the current runtime cleanup cannot truly unload the Stage 1 engine.
- Result: Confirmed. The server vLLM build has no `release()` method on `LLM`.

#### E3: force a red test for eval isolation
- Change: add a pipeline test that patches `paper_new_selector.pipeline.run_eval` to raise if called directly, while expecting subprocess execution instead.
- Expected confirm: the current code should fail because eval still runs in-process.
- Result: Confirmed. The new test failed before the fix.

#### E4: move eval behind a subprocess boundary and rerun tests
- Change: add `paper_new_selector.eval_subprocess_runner`, serialize Stage 2 synthetic outputs to disk, and invoke eval via `subprocess.run(...)` after runtime cleanup.
- Expected confirm: if eval happens in a child process, the parent no longer carries Stage 1 vLLM residency into the downstream evaluation stage.
- Result: Confirmed locally. The new subprocess test passed, the pipeline smoke tests passed, and the full suite passed.

### Root Cause

- The 30GiB+ spike was **not** caused by round20 making the algorithm itself larger. The direct issue was runtime lifetime: Stage 1 vLLM stayed resident through Stage 2 shared-session generation, and the parent process then proceeded into downstream eval without a reliable way to truly unload that vLLM engine first. Because the deployed `vllm.LLM` implementation has no `release()` API, the existing cleanup path could only null references and empty caches, not force engine teardown. That allowed Stage 1 model residency to overlap with later stages and inflated the single-process GPU footprint.

### Fix Plan

- Keep round20 algorithm/config behavior unchanged.
- Isolate downstream eval behind a subprocess boundary so the parent process can drop Stage 1 runtime references before eval begins.
- Preserve all existing output artifacts and eval summary contracts so experiment harnesses do not need redesign.
- Add a regression test that fails if eval runs directly inside the parent process again.

### Fix

- Added `paper_new_selector.eval_subprocess_runner`:
  - reads serialized synthetic texts
  - executes `run_eval(...)`
  - prints the JSON summary to stdout for the parent process
- Updated `paper_new_selector.pipeline.run_pipeline(...)`:
  - keep Stage 1 / Stage 2 behavior unchanged
  - serialize downstream synthetic texts to `eval_synthetic_texts.json`
  - call `release_runtime_memory(...)`
  - run eval via `subprocess.run([sys.executable, "-m", "paper_new_selector.eval_subprocess_runner", ...])`
  - parse the child JSON summary back into the pipeline return value
- Added a regression test in `tests/test_pipeline_smoke.py` that:
  - fails if `run_eval(...)` is called in-process
  - requires a subprocess eval launch instead

### Verification

- `python -m unittest tests.test_pipeline_smoke.PipelineSmokeTests.test_pipeline_runs_eval_in_subprocess_after_runtime_release -v` -> pass
- `python -m unittest tests.test_pipeline_smoke -v` -> pass
- `python -m unittest discover -s tests -p "test_*.py" -v` -> `Ran 115 tests ... OK`

## 2026-05-08 Round20 Still Spikes To 30GiB+ During Jobs Arbitration

### Observations

- After the eval-subprocess fix, the user still observed `round20` reaching `33GiB+` on the A6000 during `r20_jobs_arbitration`.
- The offending process was the main experiment process itself, not an eval subprocess.
- Remote `r20_jobs_arbitration.log` showed:
  - `vLLM generation memory precheck | free=25.01 GiB required=2.00 GiB`
  - `# GPU blocks: 2504`
- The large cache allocation happened before downstream eval finished, so the remaining spike was not explained by the already-fixed eval overlap bug.
- `round20` configs still inherited `round19`'s full quick-compare vLLM profile:
  - `llm.generator.gpu_memory_utilization = 0.35`
  - `bootstrap.gpu_memory_utilization = 0.35`
  - `startup_required_free_gb = 2`
- For the user’s shared-GPU scenario, this profile is too aggressive for the "jobs + microblog uncertain-path validation" purpose of round20.

### Hypotheses

#### H1: the 33GiB+ spike is still caused by eval overlap, so the eval-subprocess fix did not actually work
- Supports: the earlier bug also produced unexpectedly large end-to-end memory residency.
- Conflicts: the observed high-memory process during `jobs_arbitration` was the main experiment process while Stage 1 / Stage 2 generation was still active.
- Test: inspect the running PID / cmdline and correlate with the experiment log phase.

#### H2: round20 arbitration creates extra GPU models or duplicate vLLM engines (ROOT HYPOTHESIS REJECTED)
- Supports: the new algorithm branch is in round20, so it is a natural suspect.
- Conflicts: arbitration only compares already-computed policy metrics and does not instantiate extra generator backends.
- Test: inspect local round20 arbitration code paths and remote log stage ordering.

#### H3: round20 quick-compare configs still use the inherited high-footprint vLLM profile, so the remaining 30GiB+ spike is an experiment-profile issue rather than an algorithm bug (ROOT HYPOTHESIS)
- Supports: remote logs show high cache-block count (`2504`) already during Stage 1 generation.
- Supports: local config inheritance confirms round20 had not yet introduced a low-footprint shared-GPU profile.
- Supports: the round20 first validation pass only targets uncertain-path behavior on `jobs` and `microblog`, so it does not require the same aggressive vLLM footprint as round19 full quick-compare.
- Test: lower round20's compare-profile vLLM utilization / startup gate locally, add config regression tests, and keep the algorithm unchanged.

### Experiments

#### E1: inspect the live high-memory process phase
- Change: inspect the remote PID, batch logs, and `r20_jobs_arbitration.log`.
- Expected confirm: if the log is still inside Stage 1 / Stage 2 generation while the process is already at `33GiB+`, the spike is intrinsic to the current runtime profile rather than a downstream eval overlap.
- Result: Confirmed.

#### E2: compare round20 config inheritance with its intended validation scope
- Change: inspect `configs/experiments/single_node_tuning_round20/_base_selector_tuning_round20.yaml`.
- Expected confirm: if round20 simply inherits round19’s full vLLM profile, it lacks a dedicated low-footprint shared-GPU compare profile.
- Result: Confirmed.

#### E3: add a dedicated low-footprint round20 compare profile
- Change: override round20 base config with:
  - `llm.generator.gpu_memory_utilization: 0.18`
  - `llm.generator.startup_required_free_gb: 20`
  - `bootstrap.gpu_memory_utilization: 0.18`
  - `bootstrap.startup_required_free_gb: 20`
- Expected confirm: round20 compare experiments keep the same algorithmic logic while reducing the steady-state vLLM cache footprint and refusing to start when the shared GPU is too crowded.
- Result: Confirmed locally via merged-config tests.

### Root Cause

- The remaining `33GiB+` spike was not another lifecycle leak. After the eval isolation fix, the high memory came from the round20 compare profile itself: it still inherited round19’s high-footprint vLLM settings (`gpu_memory_utilization=0.35`, `startup_required_free_gb=2`) even though round20’s first purpose was only lightweight uncertain-path validation on a shared A6000. In other words, the algorithm was fine, but the round20 experiment profile was still too expensive for the intended shared-GPU operating mode.

### Fix Plan

- Keep round20 arbitration logic unchanged.
- Introduce a dedicated low-footprint vLLM profile only for `single_node_tuning_round20`.
- Raise the startup free-memory gate so these shared-GPU validation runs fail early instead of competing for nearly all remaining memory.
- Add config regression coverage so future edits cannot silently restore the high-footprint inherited profile.

### Fix

- Updated `configs/experiments/single_node_tuning_round20/_base_selector_tuning_round20.yaml`:
  - `llm.generator.gpu_memory_utilization: 0.18`
  - `llm.generator.startup_required_free_gb: 20`
  - `bootstrap.gpu_memory_utilization: 0.18`
  - `bootstrap.startup_required_free_gb: 20`
- Added `tests.test_config.PaperNewSelectorConfigTests.test_round20_uncertain_compare_uses_low_footprint_shared_gpu_profile` to lock this profile in.

### Verification

- `python -m unittest tests.test_config.PaperNewSelectorConfigTests.test_round20_uncertain_compare_uses_low_footprint_shared_gpu_profile -v` -> pass
- `python -m unittest tests.test_config -v` -> pass
- `python -m unittest discover -s tests -p "test_*.py" -v` -> `Ran 116 tests ... OK`

## 2026-05-08 Round20 Probe Threshold Retune After `GPU blocks: 0`

### Observations

- The first natural probe `r20_probe_jobs_seed42` failed even with abundant free memory:
  - `free=45.23 GiB required=20.00 GiB`
  - `# GPU blocks: 0`
- This happened under the newly introduced low-footprint round20 profile:
  - `llm.generator.gpu_memory_utilization = 0.18`
  - `bootstrap.gpu_memory_utilization = 0.18`
- The failure happened during Stage 1 startup, before any bootstrap/eval behavior mattered.

### Root Cause

- `round20`'s Stage 1 vLLM utilization floor was pushed too low. On this server/runtime combination, `gpu_memory_utilization=0.18` can pass the free-memory gate but still yield `# GPU blocks: 0`, so vLLM cannot construct a usable cache.

### Fix

- Retuned only the Stage 1 probe/compare profile in `single_node_tuning_round20`:
  - `llm.generator.gpu_memory_utilization: 0.18 -> 0.22`
- Kept bootstrap unchanged at `0.18`.
- Kept `startup_required_free_gb` unchanged at `20`.

### Verification

- Updated `tests.test_config.PaperNewSelectorConfigTests.test_round20_uncertain_compare_uses_low_footprint_shared_gpu_profile` to assert:
  - Stage 1 `gpu_memory_utilization == 0.22`
  - bootstrap `gpu_memory_utilization == 0.18`

## 2026-05-08 Round20 Probe Retune After Six Natural Probes All Failed

### Observations

- After the first retune (`Stage1 gpu_memory_utilization=0.22`, `startup_required_free_gb=20`), all 6 natural probe runs still failed.
- Most failures had the same pattern:
  - free memory comfortably above threshold
  - `# GPU blocks: 0`
  - `No available memory for the cache blocks`
- One retry (`jobs_seed456`) then failed a second time because free memory dropped below the raised startup gate:
  - `free=18.23 GiB required=20.00 GiB`

### Root Cause

- The first retune fixed neither side of the problem cleanly:
  - `gpu_memory_utilization=0.22` was still too low for reliable Stage 1 cache-block construction on this server/runtime
  - `startup_required_free_gb=20` was too high for retry behavior under normal shared-GPU fluctuation

### Fix

- Retuned round20 probe/compare profile again:
  - `llm.generator.gpu_memory_utilization: 0.22 -> 0.28`
  - `llm.generator.startup_required_free_gb: 20 -> 2`
  - `bootstrap.startup_required_free_gb: 20 -> 2`
- Kept bootstrap `gpu_memory_utilization` at `0.18`.

### Verification

- Updated config regression expectations to:
  - Stage 1 `gpu_memory_utilization == 0.28`
  - Stage 1 `startup_required_free_gb == 2`
  - bootstrap `gpu_memory_utilization == 0.18`
  - bootstrap `startup_required_free_gb == 2`

## 2026-05-07 Round19 Repeat15 Summary Crash After Successful First Run

### Observations

- After the CUDA device-order fix, `r19_repeat15_round01_jobs_seed01` executed successfully on the A6000 path:
  - startup log showed `free=33.47 GiB`
  - vLLM initialized with `# GPU blocks: 451`
  - the experiment log includes a completed downstream eval summary with valid metrics (`best_top1=0.2783...`).
- However, the repeat15 driver itself crashed immediately after the first run, before writing `END ... status=0` into the master log.
- The failing stack trace in `round19_full_repeat15_nohup.log` is:
  - `FileNotFoundError: Missing required artifact: /mnt/.../paper-new-round19/paper-new-round19/outputs/repeat15_rounds/.../stage1_budget_calibration.json`
- The missing path contains a duplicated `paper-new-round19/` segment.
- `Repeat15RunSpec.relative_output_root` currently stores the config-facing string `paper-new-round19/outputs/repeat15_rounds/<exp>`.
- `run_repeat15_batch()` constructs the local artifact path as `root / spec.relative_output_root`, which doubles the repo directory when `root` is already the `paper-new-round19` repo root.

### Hypotheses

#### H1: the repeat15 driver is reusing the config-facing output root string as a local filesystem subpath, so artifact lookup doubles the repo directory and fails during summary collection (ROOT HYPOTHESIS)
- Supports: the thrown path is exactly `root / "paper-new-round19/outputs/..."`.
- Supports: the actual eval artifacts already exist under the single-repo path, proving the experiment itself completed.
- Supports: `relative_output_root` is intentionally shaped for YAML config, not necessarily for local path joins.
- Conflicts: none.
- Test: add a regression test that resolves the runtime artifact directory for one spec and requires `/.../paper-new-round19/outputs/repeat15_rounds/<exp>` instead of `/.../paper-new-round19/paper-new-round19/outputs/...`.

#### H2: `load_yaml_config()` resolved `paths.output_root` differently at runtime than the driver expected
- Supports: output paths are config-driven.
- Conflicts: the actual eval summary path in the experiment log is already at the correct single-repo location, so the config resolution path itself worked.
- Test: compare the actual artifact path in the completed log with the path used by `append_repeat15_summary_row`.

#### H3: the experiment never wrote `stage1_budget_calibration.json`
- Supports: the immediate exception is "missing required artifact".
- Conflicts: the path in the exception is visibly malformed with duplicated repo name, so a path bug explains the miss more directly.
- Test: inspect the completed experiment output tree path from the log.

### Experiments

#### E1: inspect the successful first-run log and the driver crash stack
- Change: no code change; compare `r19_repeat15_round01_jobs_seed01.log` with `round19_full_repeat15_nohup.log`.
- Expected confirm: the experiment writes outputs successfully, then the driver looks in the wrong doubled path.
- Result: Confirmed.

#### E2: regression test for runtime output directory resolution
- Change: add a repeat15 test that resolves the runtime artifact directory for one spec under a temp repo root and requires no duplicated `paper-new-round19/`.
- Expected confirm: current code fails this test before the fix.
- Result: pending.

### Root Cause

- pending

### Fix Plan

- Split "config-facing output root" from "runtime filesystem output dir" in `repeat15_runner.py`.
- Keep `paper-new-round19/outputs/...` in the generated YAML so config loading stays unchanged.
- Use a separate local helper for artifact lookup and cleanup under the actual repo root.
