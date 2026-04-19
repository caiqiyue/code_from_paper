## Observations

- User question: determine whether `PT-P3` (`jobs_real_eps758`) failed on the new server.
- Remote output directory exists: `/root/autodl-tmp/caiqiyue/code_from_paper/PrE-Text/outputs/pretext_platform/jobs_real_eps758`.
- Remote files present:
  - `stage1_summary.json`
  - `stage2_summary.json`
  - `metrics_summary.json`
- Remote file missing:
  - `eval_small_summary.json`
- `metrics_summary.json` contains `status: "SUCCESS"` and includes successful `stage1` and `stage2` summaries.
- `stage2_summary.json` shows `synthetic_corpus_path` exists and `generated_count: 50000`.
- No current remote process matches `pretext_platform.scripts.run_pipeline`, `pretext_platform.scripts.run_eval_small`, or `jobs_real_eps758`.
- Recent file timestamps for `jobs_real_eps758` stop at `2026-04-06 18:57:29 +0800`.
- Remote log search found no `jobs_real_eps758` log file to indicate a crash during `run_eval_small`.
- Local record already marks `PT-P3` as `阶段完成`, with note: `已有 stage1/stage2，缺 eval_small_summary，不能算完整正式完成`.
- Final formal design requires two separate commands for PrE-Text formal runs: `run_pipeline`, then `run_eval_small`.
- Current local config `PrE-Text/configs/experiments/jobs_real_eps758.yaml` uses `bootstrap.num_prompts: 10000`, but remote `stage2_summary.json` shows `prompt_count: 50000`, indicating the remote artifacts were produced by an older config/run.

## Hypotheses

### H1: `PT-P3` pipeline succeeded, but `run_eval_small` was never launched (ROOT HYPOTHESIS)
- Supports:
  - `metrics_summary.json` says `SUCCESS`.
  - `stage1` and `stage2` outputs are complete.
  - `eval_small_summary.json` is missing.
  - No current process exists.
  - No failure log was found.
  - Remote artifacts use old parameters (`prompt_count: 50000`), consistent with an older pre-formal run that stopped after pipeline.
- Conflicts:
  - None found.
- Test:
  - Inspect remote summaries/logs and compare artifact parameters against current formal config.

### H2: `run_eval_small` was launched and crashed before writing `eval_small_summary.json`
- Supports:
  - Missing `eval_small_summary.json` could happen after a crash.
- Conflicts:
  - No matching log or traceback found.
  - No `eval_small` output directory found.
  - No process is running now.
  - Existing artifacts look like a clean pipeline-only completion, not a mid-eval crash.
- Test:
  - Search remote logs for `jobs_real_eps758` and traceback markers.

### H3: `PT-P3` was treated as complete by an older workflow that only required pipeline artifacts
- Supports:
  - Remote artifact timestamps are `2026-04-06`, earlier than the `2026-04-09` final formal design.
  - Remote `prompt_count: 50000` conflicts with the current formal design's `10000`.
  - Local records explicitly call it `阶段完成` rather than formally complete.
- Conflicts:
  - None strong; this is compatible with H1.
- Test:
  - Compare remote artifact metadata with current config and formal design.

## Experiments

### E1
- Change:
  - No code changes. Read remote summaries for `jobs_real_eps758`.
- Expected if H1 is true:
  - `stage1`/`stage2` summaries exist, `metrics_summary.json` is successful, `eval_small_summary.json` is missing.
- Result:
  - Confirmed.

### E2
- Change:
  - No code changes. Search remote logs for `jobs_real_eps758`.
- Expected if H2 is true:
  - A log or traceback from `run_eval_small` should exist.
- Result:
  - Rejected. No matching log evidence was found.

### E3
- Change:
  - Compare remote `stage2_summary.json` against current local config and formal design.
- Expected if H3 is true:
  - Remote artifacts should reflect old parameters.
- Result:
  - Confirmed. Remote `prompt_count` is `50000`; current config/formal design expects `10000`.

## Root Cause

`PT-P3` does not show evidence of a runtime failure in `run_pipeline`; instead, the server contains an older pipeline-only run that completed `stage1` and `stage2`, but never completed the separate `run_eval_small` step required for formal completion.

## Fix

- Do not classify `PT-P3` as failed.
- Classify it as `incomplete / pipeline-only`.
- If formal completion is needed, run:
  - `python -m pretext_platform.scripts.run_eval_small --config configs/experiments/jobs_real_eps758.yaml` only if the existing stage2 artifact is still valid for the current formal design.
- Because the existing artifacts were generated with old parameters (`prompt_count: 50000`), the stricter fix is to rerun `PT-P3` under the current formal config, then run `run_eval_small`.
