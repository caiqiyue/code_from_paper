## Observations

- User reported remote runs for `ns_tune11_f1_forums`, `ns_tune11_f2_forums`, and
  `ns_tune11_f3_forums` all produced `best_top1 = 0.24789131414590174`.
- Remote eval summaries show all three forums runs used `synthetic_train_count = 93`
  and the same top-k metrics.
- Remote SHA256 for all three forums synthetic corpora is identical:
  `8fbaa03dc6f8b0a01d26a3d4c1c747483a4a291584c39e17e7c81f456a2711d7`.
- The identical eval result is therefore caused upstream of evaluation: stage2 received
  byte-identical synthetic text for f1/f2/f3.
- Round11 forums configs vary `_forums_lambda_generic`,
  `_forums_lambda_redundancy`, and genericity gate values, but all inherit
  `_forums_seed_top_k: 23`, `bootstrap.num_prompts: 100`, and
  `bootstrap.max_tokens: 85`.
- Stage2 prompt construction samples three examples from `selected_texts` with the
  same meta seed. If the selected seed list is unchanged, stage2 prompts and outputs
  are deterministic and identical.
- Locally, `load_yaml_config("configs/...")` fails before checking the existing cwd
  path because `resolve_config_path()` calls `resolve_worktree_root()` and
  `resolve_repo_root()` eagerly. This local checkout has `thesis_platform/datasets`
  but no `thesis_platform/open_model`.
- The current forums override in `stage1_runner.py` runs before genericity penalty
  computation, so gate overrides are no longer being applied too late.

## Hypotheses

### H1: Forums experiments select the same seed list, so stage2 receives identical prompts (ROOT HYPOTHESIS)

- Supports: f1/f2/f3 synthetic corpus files are byte-identical; all configs use the
  same `_forums_seed_top_k: 23` while only 24 candidates are generated, so penalty
  changes can easily leave the selected set unchanged.
- Conflicts: without persisted stage1 summary, selected indices are not directly
  observable from the completed remote runs.
- Test: compare remote stage2 corpus hashes and inspect the config dimensions that
  control deterministic stage2 generation.

### H2: Forums override values are not applied before scoring

- Supports: original task history mentioned moving the override before genericity
  calculation.
- Conflicts: current `stage1_runner.py` applies `_forums_*` overrides before
  `compute_genericity_penalties()` and `greedy_select_candidates()`.
- Test: unit-test that a forums config passes overridden gate and selector values to
  both genericity and greedy selection.

### H3: Evaluation is reusing stale or shared output files

- Supports: all configs resolve outputs under the sibling `paper-new/outputs`, not
  inside `paper-new-round11`.
- Conflicts: f1/f2/f3 use distinct experiment output directories, and each directory
  contains its own eval files.
- Test: hash each run's `eval/stage2/llama7b_text_syn.json`.

### H4: Local failures are caused by an over-strict config path resolver

- Supports: local `load_yaml_config()` fails with `Could not locate the repo root`
  even when the relative config file exists under cwd.
- Conflicts: remote has a full resource root, so this is not the remote metric cause.
- Test: call `load_yaml_config()` from the round11 worktree on an existing relative
  config.

## Experiments

### E1: Hash remote forums corpora

- Change: no code change; read remote stage2 corpus files and compute SHA256.
- Result: confirmed. f1/f2/f3 all have identical size `29289` and identical SHA256
  `8fbaa03dc6f8b0a01d26a3d4c1c747483a4a291584c39e17e7c81f456a2711d7`.
- Conclusion: identical scores come from identical stage2 input, not downstream eval
  noise.

### E2: Inspect merged round11 config values without resolver

- Change: no production code; used private `_load_with_includes()` directly.
- Result: f1/f2/f3 have different `_forums_*` penalty/gate values, but the inherited
  generic `lambda_*` values remain unchanged until the runtime override. All three
  inherit `_forums_seed_top_k: 23`, `num_prompts: 100`, and `max_tokens: 85`.
- Conclusion: the experiment matrix varies only the selector penalty shape. With
  23/24 seeds selected and deterministic stage2 generation, this can collapse to
  the same synthetic corpus.

### E3: Reproduce local config resolver failure

- Change: no production code; called `load_yaml_config()` on an existing local
  round11 config.
- Result: confirmed. The resolver raises before checking the cwd candidate because
  it eagerly resolves missing resource roots.
- Conclusion: local debugging and validation need a resolver fix independent of the
  remote metric issue.

## Root Cause

Round11 forums f1/f2/f3 produced the same metric because the experiment matrix did
not change the deterministic stage2 input: each run selected effectively the same
large seed set (`_forums_seed_top_k: 23` out of 24 candidates) and used identical
bootstrap settings, producing byte-identical synthetic corpora.

## Fix

## Debug: Round17 extended batch summary crash

### Observations

- The failing batch was the ad hoc Round17 extended batch that should run:
  - `r17_microblog_r099`
  - `r17_microblog_r098`
  - `r17_microblog_r097`
  - `r17_jobs_r098`
  - `r17_congressional_r098`
- On the server, `r17_microblog_r099` completed and wrote both:
  - `stage1_budget_calibration.json`
  - `eval/downstream_eval_summary.json`
- The batch stopped immediately after the first run, before `r17_microblog_r098` started.
- The remote `round17_extended_batch_nohup.out` showed:
  - `SyntaxError: invalid syntax`
  - offending line rendered as `feasible = ,.join(...)`
- The same log also showed `date: extra operand '%T'`, which indicates the date command string was being passed through quoting incorrectly.
- The local repo did not contain a checked-in copy of this batch script; it had only been created ad hoc on the server.

### Hypotheses

#### H1: Inline Python in the shell script was mangled by nested shell quoting (ROOT HYPOTHESIS)
- Supports: the logged Python line lost the quotes around `","`, which is classic nested-quote damage; the crash happened in the inline Python summary block after the first experiment finished.
- Conflicts: none.
- Test: replace the inline Python block with a checked-in standalone helper script and re-run syntax checks locally.

#### H2: The failure was caused by malformed experiment output JSON
- Supports: the crash happened while summarizing experiment output.
- Conflicts: the error is a Python parse-time `SyntaxError`, not a runtime JSON parsing error.
- Test: inspect the failing code path and confirm the exception occurs before any JSON is loaded.

#### H3: The `date` failure killed the batch
- Supports: `date` emitted an error in the same nohup log.
- Conflicts: the batch had already started and completed the first experiment; the fatal error happened later in the Python summary step.
- Test: isolate the date call and confirm it only affects logging text, not control flow.

### Experiments

- Reviewed the remote nohup log and confirmed the first fatal error was the inline Python `SyntaxError`.
- Compared the failure site against the generated shell and observed that both the Python snippet and the `date` call were vulnerable to nested quote expansion.
- Replaced the inline Python summary logic with a dedicated checked-in helper script:
  - `scripts/append_round17_summary.py`
- Added a checked-in batch launcher:
  - `scripts/run_round17_extended_batch.sh`
- Kept the shell layer simple:
  - shell only launches experiments and passes plain arguments
  - Python helper owns JSON parsing and TSV row construction

### Root Cause

The Round17 extended batch failed because its ad hoc shell script embedded a Python summary snippet inside nested SSH/shell quoting, which stripped required quotes from `','.join(...)` and turned the post-run summary step into a `SyntaxError`; the same quoting pattern also corrupted the `date` logging command.

### Fix

- Added [scripts/append_round17_summary.py](/Users/apple/Desktop/code_from_paper/paper-new-round-16/scripts/append_round17_summary.py) to generate summary rows without inline Python quoting.
- Added [scripts/run_round17_extended_batch.sh](/Users/apple/Desktop/code_from_paper/paper-new-round-16/scripts/run_round17_extended_batch.sh) as the checked-in launcher for the 5-experiment Round17 extended batch.
- Standardized logging calls to:
  - `date '+%F %T'`
- Removed nested inline Python from the shell path entirely so `feasible_budgets` formatting and metric extraction happen in normal Python code.

- Make local relative config loading check existing paths before requiring a full
  resource root.
- Make round11 forums variants vary bootstrap max tokens as well as penalties, so
  their stage2 corpora cannot silently collapse when selector seeds are unchanged.
- Add unit coverage that forums overrides are applied before genericity and greedy
  selection.

## Round16 Metrics-Missing Investigation

### Observations

- The Round16 batch summary TSV reported `best_top1=NA` for all experiments even
  when the process exit status was `0`.
- Remote experiment logs for completed runs such as `r16_c1_forums` and
  `r16_no_budget_cost_forums` contain a full `eval` payload with:
  - `best_top1`
  - `best_top3`
  - `best_top5`
  - `best_top10`
  - `summary_path = .../eval/downstream_eval_summary.json`
- Remote filesystem checks confirm that both
  `.../eval/downstream_eval_summary.json` and
  `.../eval/pretext_small_eval_summary.json` do exist for completed Round16 runs.
- The downstream summary schema stores metrics under the nested
  `metrics` object, not at the top level. `DownstreamEvalManager.run()` writes:
  `summary["metrics"] = primary_stage.get("metrics", {})`.
- The batch script `logs/run_round16_all50.sh` extracts `best_top1` using:
  `print(obj.get('best_top1', 'NA'))`, which reads the wrong JSON level.
- The same script never extracts `best_top3`, `best_top5`, or `best_top10` at all,
  so those fields can never appear in the TSV even when evaluation succeeds.

### Hypotheses

#### H1: The experiments never ran downstream eval, so there were no top-k metrics

- Supports: early manual checks looked like some eval summaries were missing.
- Conflicts: remote logs show completed `small_eval` sections with concrete top-k
  metrics and explicit `summary_path` values.
- Test: inspect a completed experiment log and locate the final `eval` payload.

#### H2: The output path used by the batch summary script is wrong

- Supports: Round16 outputs live in sibling `paper-new/outputs`, which is easy to
  mismatch with `paper-new-round-16`.
- Conflicts: remote `find` shows the expected summary files exactly under
  `/mnt/public/caiqiyue_file/code_from_paper/paper-new/outputs/<exp>/eval/`.
- Test: locate `downstream_eval_summary.json` for a known completed experiment.

#### H3: The batch summary script reads the wrong JSON shape (ROOT HYPOTHESIS)

- Supports: `downstream_eval_summary.json` stores metrics under
  `summary["metrics"]`, while the script reads `obj.get("best_top1")`.
- Supports: this exactly explains why runs complete successfully, logs show metrics,
  files exist, yet the TSV still records `NA`.
- Conflicts: none after confirming the on-disk summary schema and the script body.
- Test: compare a real `downstream_eval_summary.json` payload with the extraction
  code in `run_round16_all50.sh`.

#### H4: The top-k metrics exist, but only in `pretext_small_eval_summary.json`

- Supports: the stage-level small-eval file definitely contains the metrics.
- Conflicts: the final `downstream_eval_summary.json` also contains them nested
  under `metrics`, so the final summary is not missing data.
- Test: inspect both summary files for the same experiment.

### Experiments

#### E1: Inspect a completed experiment log

- Change: no code change; read the tail of `r16_c1_forums.log`.
- Result: confirmed. The printed final JSON includes:
  - `metrics.best_top1 = 0.2487283497521087`
  - `metrics.best_top3 = 0.38439250531195673`
  - `metrics.best_top5 = 0.4511621917455412`
  - `metrics.best_top10 = 0.5360247247440603`
  - `summary_path = .../eval/downstream_eval_summary.json`
- Conclusion: evaluation did run, and the top-k metrics do exist.

#### E2: Locate summary files on disk for completed runs

- Change: no code change; used remote `find` for `r16_c1_forums`,
  `r16_no_budget_cost_congressional`, and `r16_no_budget_cost_forums`.
- Result: confirmed. Both `downstream_eval_summary.json` and
  `pretext_small_eval_summary.json` exist under the expected output directories.
- Conclusion: the issue is not a missing output file.

#### E3: Inspect the batch summary extraction script

- Change: no code change; read `logs/run_round16_all50.sh`.
- Result: confirmed. The script checks for
  `$out_root/eval/downstream_eval_summary.json`, then loads it and executes:
  `print(obj.get('best_top1', 'NA'))`.
- Conclusion: `best_top1` is being read from the wrong JSON level; it should be
  read from `obj["metrics"]["best_top1"]`. The script also omits extraction of
  `best_top3`, `best_top5`, and `best_top10` entirely.

### Root Cause

Round16 top-k metrics were not actually missing from the experiment outputs; they
were missing only from the batch summary because the aggregation script read
`downstream_eval_summary.json` using the wrong JSON shape, looking for `best_top1`
at the top level instead of under `metrics`, and it never extracted `best_top3`,
`best_top5`, or `best_top10` at all.

### Fix

- Update the batch-summary extraction logic to read:
  - `obj["metrics"]["best_top1"]`
  - `obj["metrics"]["best_top3"]`
  - `obj["metrics"]["best_top5"]`
  - `obj["metrics"]["best_top10"]`
- Extend the TSV/header to include all four top-k metrics instead of only
  `best_top1`.
