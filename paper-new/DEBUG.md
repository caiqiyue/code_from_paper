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
