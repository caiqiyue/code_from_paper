## Observations

- The reported failure is `RuntimeError: CUDA error: invalid device ordinal`.
- The launch path uses `CUDA_VISIBLE_DEVICES=1`, which hides physical GPU 0 from the process.
- Current formal Linux configs still contain `device: cuda:1` in several thesis_platform experiment files, which is incompatible with a process that only sees one GPU.
- `PrE-Text` formal configs currently use `device: cuda`, so they are already aligned with a single visible GPU.
- `thesis_platform` code paths for the formal runners use direct `model.to(device)` semantics for non-auto devices, so the config value itself controls the ordinal that gets requested.

## Hypotheses

### H1: The root cause is an invalid ordinal caused by `CUDA_VISIBLE_DEVICES=1` plus `device: cuda:1` (ROOT HYPOTHESIS)
- Supports: the exact runtime error is `invalid device ordinal`, and `cuda:1` cannot exist if the process only sees one GPU.
- Conflicts: none so far.
- Test: change formal Linux configs to `device: cuda`, then verify no formal config still requests `cuda:1`.

### H2: A backend is still auto-sharding across both GPUs via `device_map="auto"`
- Supports: the repo historically had auto device-map paths.
- Conflicts: current formal config paths do not select `auto`, and the direct error shown is ordinal-related, not shard-related.
- Test: scan formal config/code paths for remaining `device_map="auto"` and confirm they are not used by the formal runners.

### H3: The visible-device mapping is correct, but one of the launch scripts clears it before execution
- Supports: launch-time environment changes can override config assumptions.
- Conflicts: the reported failure text specifically attributes the error to the `CUDA_VISIBLE_DEVICES=1` and `cuda:1` combination.
- Test: inspect the launch wrapper and verify the env is preserved through the formal experiment entry point.

## Experiments

- Re-scanned the formal experiment directories after changing the configs back to `device: cuda`.
- Result: no `cuda:1` entries remain in the formal Linux / formal PrE-Text config paths.
- Ran `python -m unittest thesis_platform.tests.test_thesis_platform_config`.
- Result: passed.
- Ran `python -m unittest discover -s tests -p "test_formal_config_paths.py"` inside `PrE-Text`.
- Result: passed.

## Root Cause

The failure was caused by writing `device: cuda:1` in configurations that were launched with `CUDA_VISIBLE_DEVICES=1`, so the process only had one logical CUDA device and ordinal 1 did not exist.

## Fix

- Formal Linux and formal PrE-Text configs now use `device: cuda` instead of `cuda:1`.
- The formal downstream defaults remain `run_small_eval: true` and `run_large_eval: false`.
- The code paths for the formal runners continue to use single-device loads, so the process stays on the one visible A6000 and does not shard across both GPUs.
