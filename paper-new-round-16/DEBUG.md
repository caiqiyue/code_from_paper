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

- Make local relative config loading check existing paths before requiring a full
  resource root.
- Make round11 forums variants vary bootstrap max tokens as well as penalties, so
  their stage2 corpora cannot silently collapse when selector seeds are unchanged.
- Add unit coverage that forums overrides are applied before genericity and greedy
  selection.
