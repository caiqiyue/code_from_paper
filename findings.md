# Findings

## Current Known Facts

- `execute/new/单节点实验记录表.md` currently lists `SN-C1`, `SN-C2`, `SN-C3`, `SN-C4`, `SP-C1`, `SP-C2`, `SP-C3`, `SP-C4`, `GA1`, `GA3`.
- The queue state in `old_automation/old_experiment_queue_state.json` has previously been updated to include later experiments, but those additions still need config/path verification before relying on them.
- `PrE-Text/pretext_platform/core/gpu_memory.py` contains the vLLM startup memory precheck using `startup_required_free_gb`.
- Remote verification confirmed these completed experiments are actually finished and their key metrics match the record table:
  - `SN-C1` -> `best_top1=0.28755274602687014`, `synthetic_train_count=919`
  - `SN-C2` -> `best_top1=0.31428494901642934`, `synthetic_train_count=1006`
  - `SN-C3` -> `best_top1=0.25086451316846126`, `synthetic_train_count=1040`
  - `SP-C1` -> `best_top1=0.285903674841964`, `synthetic_train_count=1408`
  - `SP-C2` -> `best_top1=0.29726286467576873`, `synthetic_train_count=1409`
  - `SP-C3` -> `best_top1=0.2562400642443909`, `synthetic_train_count=1407`
  - `GA1` -> `best_top1=0.27935589219601314`, `synthetic_train_count=30`
  - `GA3` -> `best_top1=0.2707548542512085`, `synthetic_train_count=307`
- The queue config audit passed for `SN-C6~C9` and `SP-C6~C9`; their config files exist and inherit from the intended base jobs.

## Questions To Answer

- Which experiments in the record table are actually complete and correctly reported?
- Do the queued `SN-C6~C9 / SP-C6~C9` config paths point at real files?
- Do those configs match the expected environment split: `SN -> caiqiyue-vllm`, `SP -> pretext`?
