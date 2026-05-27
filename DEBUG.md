## Observations

### O1. Historical "successful" exact configs do not currently beat PrE-Text on the server

Server replay results under:

- branch: `paper-2-genereic`
- env: `pretext`
- GPU: `A6000` via `CUDA_VISIBLE_DEVICES=1`

Observed replay outputs:

- `r182_forums_fixed22`
  - `resolved_seed_top_k = 22`
  - `best_top1 = 0.24988732212993367`
  - lower than `PrE-Text forums = 0.2501448715`
- `r181_congressional_g1`
  - `resolved_seed_top_k = 18`
  - `best_top1 = 0.29237150819884633`
  - lower than `PrE-Text congressional = 0.2950`

This means the failure is not unique to `Round18.3 hybrid`.

### O2. Local and server code/config state match for the key replay paths

Local:

- branch: `paper-2-genereic`
- commit: `a291013494d292a906654d245e86f602b6822bf7`

Server:

- branch: `paper-2-genereic`
- commit: `a291013494d292a906654d245e86f602b6822bf7`

Key file hashes matched between local and server:

- `paper_new_selector/stage1_runner.py`
- `paper_new_selector/budget_calibration.py`
- `paper_new_selector/selector.py`
- `paper_new_selector/support.py`
- `paper_new_selector/genericity.py`
- `configs/experiments/single_node_tuning_round181/diagnostics/r181_congressional_g1.yaml`
- `configs/experiments/single_node_tuning_round182/fixed_budget/r182_forums_fixed22.yaml`
- `configs/experiments/single_node_tuning_round181/_base_selector_tuning_round181.yaml`
- `configs/experiments/single_node_tuning_round181/_g1_loose.yaml`

So "server code was not synced" is currently not supported.

### O3. The current Round18.3 failure is not explained by obvious config drift

Compared `r182_forums_fixed22` vs `r183_guard_forums`:

- `bootstrap.max_tokens = 85` in both
- `lambda_generic = 0.35` in both
- `lambda_redundancy = 0.25` in both
- `top_q = 4` in both
- `rank_weights = [1.0, 0.6, 0.3, 0.15]` in both
- `reference_top_k = 6` in both
- `reference_rank_weights = [1.0, 0.8, 0.6, 0.4, 0.25, 0.1]` in both
- `genericity_gate_*` values match

Therefore the sharp drop from prior reported `r182_forums_fixed22 = 0.2514` to current replay `0.249887...` is not explained by `max_tokens` or an obvious selector-parameter mismatch.

### O4. The server environment is concrete; the local runtime environment is incomplete

Server `pretext` environment:

- Python `3.10.20`
- `torch 2.1.2+cu121`
- `transformers 4.38.2`
- `sentence_transformers 2.5.1`
- `numpy 1.26.4`
- `yaml 6.0.1`

Local machine:

- no local conda env named `pretext`
- local shell Python from the project tooling was `3.13.9`

Therefore:

- end-to-end historical replay is currently only runnable on the server
- local project investigation can verify code/config state, but not runtime equivalence

### O5. Server datasets currently used by replay are present and concrete

Current server dataset file hashes:

- `thesis_platform/datasets/pretext_forums/formatted/forums_train.json`
  - sha256 `e02bc1ee2e8650a3ec9249a333c8a1b12e8942494ec2269a1e888a1d1ede6006`
- `thesis_platform/datasets/pretext_forums/formatted/forums_eval.json`
  - sha256 `842a32a9f7d1fb765f4e48ad5a8e1d7dda8c48ccdf2a44a8a882d54e7c0349de`
- `thesis_platform/datasets/congressional/formatted/congressional_train.json`
  - sha256 `fb84f49f16a96e644a444f9a380135229a213deee575c060a0c6d2102a8623b5`
- `thesis_platform/datasets/congressional/formatted/congressional_eval.json`
  - sha256 `e180c525d957a9a767997aad1e27ad6ae5bfd5308070bb919cc9e64224bb0edb`
- `thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json`
  - sha256 `69b5d9009464ee2f136336a1228983fe5a27945686428cc1955b2508709d1a6e`

We do not yet have historical dataset hashes from the runs that produced `0.2514` and `0.2971`, so dataset drift across time remains unresolved.

### O6. Model/backend pointers are consistent across the replay configs

For both replay configs:

- `embedding.model_path = thesis_platform/open_model/all_minilm_l6_v2`
- `generator.backend = thesis_pretext_prompt`
- no explicit `generator.model_path`
- no explicit `llm.generator_model`

So the configs point at the same logical embedding path and generator backend on both local config inspection and server config inspection.

### O7. Local project structure differs from server execution root assumptions

On local:

- `resolve_repo_root()` in `paper_new_selector/thesis_bridge.py` failed because the local workspace layout does not satisfy the expected `.git + thesis_platform` root signature used by that helper
- the concrete dataset files used on the server are not present under local `code_from_paper/thesis_platform/datasets/...`

This means local code inspection is valid, but local data/runtime parity with the server is not currently available.

### O8. Server has a persistent bitsandbytes CUDA override, but this is not obviously new

On the server:

- `BNB_CUDA_VERSION=121`
- `bitsandbytes = 0.43.1`
- `vllm = 0.3.3`
- `torch = 2.1.2+cu121`
- `LD_LIBRARY_PATH` contains `/usr/local/cuda-12.1/lib64`

The variable source is:

- `/home/k8smaster/.bashrc: export BNB_CUDA_VERSION=121`

Important nuance:

- existing historical automation logs in the workspace already contain the same
  `BNB_CUDA_VERSION=121` warning text

So this is a real hidden runtime state, but it is not yet evidence of a *new* drift.

### O9. Re-running `r182_forums_fixed22` changes both downstream accuracy and synthetic corpus size

Current server observations for the exact same config:

- replay run 1:
  - `best_top1 = 0.24988732212993367`
  - `synthetic_train_count = 91`
- replay run 2:
  - `best_top1 = 0.24647479235078232`
  - `synthetic_train_count = 89`

Both runs used:

- `resolved_seed_top_k = 22`
- same config id: `r182_forums_fixed22`
- same branch/env/GPU target

This shows that the instability is not only in downstream scoring; it already appears in Stage2 output volume.

### O10. `expand_only` and `expand_private` fail before eval when Stage2 requires a shared vLLM backend

Local minimal reproduction using `run_pipeline()` with mocked Stage1/Stage2 contracts:

- `stage1_summary["skip_bootstrap"] = False`
- `stage1_runtime["shared_session"] = None`
- `prepare_bootstrap_runtime()` exposes both:
  - `generate_bootstrapped_samples(...)`
  - `generate_with_shared_session(...)`

Observed result:

- pipeline calls `generate_with_shared_session(...)` unconditionally
- `pretext_bridge.generate_with_shared_vllm_session()` raises
  - `ValueError: shared_session must expose a backend for Stage 2 shared generation.`

This matches the server failures for:

- `eo_jobs_single_run`
- `ep_forums_single_run`

### O11. `c4_only` succeeds because it skips Stage2 entirely

Server single-run bug check:

- `c4_congressional_single_run` completed
- `expand_only` / `expand_private` did not

The working/broken boundary is:

- `skip_bootstrap=True` path works
- `skip_bootstrap=False` path without `shared_session.backend` does not

### O12. `WASP` and `DPGA-TextSyn` single-run configs currently fail on missing source artifacts, not eval logic

`validate-only` results for current config-driven external single-run entries:

- `WASP/outputs/paper_new_screening/jobs/train.jsonl`
  - `source_exists = False`
- `DPGA-TextSyn/outputs/paper_new_screening/jobs/epoch_all.json`
  - `source_exists = False`

Server failures matched those exact missing files. This means the current blocker is:

- missing four-dataset artifact preparation/organization flow

not a downstream evaluator crash.

### O13. On the server, `CUDA_VISIBLE_DEVICES=0` is the A6000 and `CUDA_VISIBLE_DEVICES=1` is the 2080Ti

Server probe under the `pretext` environment showed:

- unset:
  - device 0 = `NVIDIA RTX A6000`
  - device 1 = `NVIDIA GeForce RTX 2080 Ti`
- `CUDA_VISIBLE_DEVICES=0`:
  - visible device 0 = `NVIDIA RTX A6000`
- `CUDA_VISIBLE_DEVICES=1`:
  - visible device 0 = `NVIDIA GeForce RTX 2080 Ti`

So earlier "run on A6000 with `CUDA_VISIBLE_DEVICES=1`" was incorrect for this host.

### O14. Raw `jobs` / `forums` training texts are far longer than the Stage1 seeds used by already-completed experiments

Local length comparison:

- raw `jobs_train` sample slice:
  - mean words ≈ `329.7`
  - median words ≈ `164.5`
  - max words = `3808`
- raw `forums_train` sample slice:
  - mean words ≈ `286.4`
  - median words ≈ `209.0`
  - max words = `2070`
- completed `PrE-Text` Stage1 surviving seeds:
  - mean words ≈ `52.8`
  - median words ≈ `53.5`
  - max words = `56`

This is a major structural difference between:

- already-working pipelines (`PrE-Text` / selector-based paper-new)
- new `expand_only` / `expand_private` baselines

because the new baselines were feeding raw public/private documents directly into Stage2 prompt construction.

### O15. On the server, `expand_only` now reaches Stage2 but writes an empty downstream synthetic corpus

After the shared-session fix and the first round of single-run config tightening:

- `eo_jobs_single_run` exits with process status `0`
- `eval_small` fails with:
  - `Columns ['input_ids', 'attention_mask', 'labels'] not in the dataset. Current columns in the dataset: []`
- the exported corpus at
  - `outputs/single_run_baseline_screening/expand_only/jobs/eval/stage2/llama7b_text_syn.json`
  is an empty list (`count = 0`)

This means the failure is not in the trainer first; the downstream training corpus is already empty.

### O16. `expand_private` now produces synthetic texts, but the same process still holds large-model GPU memory when small eval starts

Server rerun after the shared-session fix:

- `ep_forums_single_run` writes
  - `outputs/single_run_baseline_screening/expand_private/forums/eval/stage2/llama7b_text_syn.json`
  with `28` non-empty strings
- then `eval_small` fails with CUDA OOM on the A6000
- the failure message reports the same process already holds about `44 GiB`

GPT-2 / DistilGPT2 small eval should not need that much memory by itself, so the most plausible explanation is that the standalone Stage2 vLLM engine is still resident when eval begins.

### O17. The bootstrap prompt token lengths still exceed or sit too close to the configured single-run `max_model_len`

Server-side tokenizer check using the exact `llama_2_7b_hf` tokenizer and the actual `stage1_summary.json` outputs:

- `expand_only`
  - seed words: all `56`
  - prompt token length:
    - mean ≈ `285.2`
    - min = `266`
    - max = `304`
- `expand_private`
  - seed words: all `56`
  - prompt token length:
    - mean ≈ `262.1`
    - min = `239`
    - max = `280`

But the single-run config currently sets:

- `bootstrap.max_model_len = 256`
- `bootstrap.max_tokens = 32`

So `expand_only` is definitively over budget, and `expand_private` is partially over budget.

## Hypotheses

### H1. Hidden runtime/model-state drift on the server changed outputs even though code/config files are identical (ROOT HYPOTHESIS)

- Supports:
  - key code/config commit and hashes match across local and server
  - exact replay configs still fail
  - current server replay uses a concrete `pretext` environment, but we do not yet know whether this exactly matches the environment from the earlier successful runs
  - embedding path and generator backend are logical pointers; underlying installed packages or model artifacts may have drifted over time
- Conflicts:
  - we have not yet identified a direct version mismatch against the historical success environment
  - `BNB_CUDA_VERSION=121` appears in older automation logs too, so that specific warning is not sufficient evidence by itself
- Test:
  - inspect whether repeated replay of the exact same config is stable; if it varies materially, runtime/model-state nondeterminism becomes more likely
  - search for old environment records / logs / manifests that capture package or model versions from the successful runs

### H2. Historical success numbers came from a different dataset snapshot than the one currently on the server

- Supports:
  - local machine does not carry the replay datasets, so there is no local cross-check
  - we only know current server dataset hashes, not the historical hashes from the earlier successful runs
  - if data was reformatted or regenerated in place, exact same config could degrade without any code diff
- Conflicts:
  - no direct evidence yet that the server datasets changed
- Test:
  - inspect old experiment logs/artifacts/docs for dataset paths, sizes, or hashes; compare against current files

### H3. Generation/evaluation is materially nondeterministic even with the same config and seed

- Supports:
  - current replay values differ from earlier reported values while using the same config ids
  - the pipeline depends on LLM generation and GPU runtime, both of which may be nondeterministic depending on backend behavior
  - `r182_forums_fixed22` already changed from `best_top1=0.249887...` to `0.246474...` across two runs
  - `synthetic_train_count` also changed from `91` to `89`, so the drift starts before downstream training
- Conflicts:
  - only one config has been rerun twice so far
- Test:
  - rerun `r182_forums_fixed22` again in the same server environment and compare `best_top1`; if it moves materially, nondeterminism is real

### H4. Round results were historically summarized from different outputs than the named experiment IDs suggest

- Supports:
  - current config names match, but earlier reported metrics may have been copied from a neighboring run or a temporary branch
  - exact replay failed for both configs, which raises the possibility that earlier success was attributed to the wrong run ids
- Conflicts:
  - we do have detailed round notes stating those exact ids and values
- Test:
  - inspect historical docs/logs around `Round18.2` and `Round18.1` for raw output references, not just summary prose

### H5. `expand_only` / `expand_private` assume a Stage1 shared generator session that those baseline modes never create (ROOT HYPOTHESIS for single-run internal baseline bug)

- Supports:
  - both modes return `shared_session = None` from `run_stage1_with_runtime()`
  - pipeline currently calls `generate_with_shared_session(...)` whenever `skip_bootstrap=False`
  - the raised error message exactly matches `pretext_bridge.generate_with_shared_vllm_session()`
  - `c4_only` works because it avoids the shared-session-dependent Stage2 path
- Conflicts:
  - none found
- Test:
  - reproduce locally with mocked pipeline contracts and confirm the failure is triggered solely by `shared_session=None`

### H6. `WASP` / `DPGA-TextSyn` single-run configs fail because standardized per-dataset source artifacts are never materialized under `outputs/paper_new_screening/<dataset>/...` (ROOT HYPOTHESIS for external baseline blocker)

- Supports:
  - `validate-only` reports `source_exists=False`
  - server failures are direct `FileNotFoundError` on the YAML-declared artifact paths
  - current repos have adapter/export scripts, but no four-dataset artifact preparation script targeting those standardized paths
- Conflicts:
  - none found
- Test:
  - add dataset-aware artifact preparation scripts that materialize the exact expected paths, then verify the configs resolve to existing sources

### H7. Even after the shared-session bug is fixed, `expand_only` / `expand_private` remain fragile because their raw sampled seed texts are much longer than the synthetic Stage1 seeds used in successful experiments

- Supports:
  - raw `jobs` / `forums` texts are 5x-6x longer on average than completed Stage1 seeds
  - Stage2 prompt builder concatenates 3 seed texts verbatim into each bootstrap prompt
  - server retry on the correct A6000 no longer fails on shared session; it fails inside vLLM cache sizing / context budgeting
- Conflicts:
  - none found
- Test:
  - cap the baseline seed text length to match the already-working Stage1 seed scale, and give single-run expand baselines a more appropriate A6000 bootstrap config

### H8. `expand_only` writes an empty corpus because its bootstrap prompts exceed the configured `max_model_len`, leaving no effective room for generation (ROOT HYPOTHESIS for the new `expand_only` failure)

- Supports:
  - exported downstream corpus is empty
  - server tokenizer check shows `expand_only` prompt tokens are always above `256`
  - Stage2 prompt builder concatenates 3 seed texts plus scaffolding before generation
- Conflicts:
  - none found
- Test:
  - reduce the expand-baseline seed budget further until prompt token lengths comfortably fit beneath the single-run `max_model_len`

### H9. `expand_private` small-eval OOM is caused by the standalone Stage2 vLLM engine not being explicitly released before downstream eval begins (ROOT HYPOTHESIS for the new `expand_private` failure)

- Supports:
  - `expand_private` already produced `28` non-empty synthetic texts, so generation itself succeeded
  - the OOM message shows the process already holds about `44 GiB`, which is inconsistent with GPT-2-scale eval alone
  - the standalone Stage2 path currently calls `generate_bootstrapped_samples_vllm(...)` from PrE-Text, which does not explicitly release its local `LLM` object
- Conflicts:
  - none found
- Test:
  - implement a local standalone vLLM bootstrap path in `paper_new_selector.pretext_bridge` that creates the `LLM`, generates outputs, and explicitly releases GPU memory before returning

## Experiments

### E1. Compare local vs server commit/config/code state

- Change:
  - none; read-only inspection
- Result:
  - local/server commit ids matched
  - key file hashes matched
- Conclusion:
  - rejected the hypothesis that the server is simply running unsynced code/config for the replayed paths

### E2. Replay the two historically successful configs on the current server

- Change:
  - none to code; reran `r182_forums_fixed22` and `r181_congressional_g1`
- Result:
  - `r182_forums_fixed22 = 0.24988732212993367`
  - `r181_congressional_g1 = 0.29237150819884633`
- Conclusion:
  - confirmed that the regression reproduces under the current server environment
  - rejected the hypothesis that only `Round18.3 hybrid` is at fault

### E3. Inspect current server environment versions

- Change:
  - none; read-only inspection
- Result:
  - server `pretext` versions recorded in `O4`

### E4. Reproduce the `expand_only` / `expand_private` failure without heavy runtime dependencies

- Change:
  - none to production code; ran `run_pipeline()` with mocked Stage1/Stage2 contracts
- Result:
  - reproduced `ValueError: shared_session must expose a backend for Stage 2 shared generation.`
  - reproduction required only:
    - `skip_bootstrap=False`
    - `shared_session=None`
- Conclusion:
  - confirms `H5`
  - rejects the idea that the failure depends on dataset paths or eval code

### E5. Validate external single-run configs against the current filesystem

- Change:
  - none to production code; ran `run_external_single_run_from_config(..., validate_only=True)`
- Result:
  - `WASP jobs`: `source_exists=False`
  - `DPGA jobs`: `source_exists=False`
- Conclusion:
  - confirms `H6`
  - rejects the idea that common eval itself is the primary blocker for the external baselines
  - local machine has no `pretext` env
- Conclusion:
  - local runtime cannot currently falsify or confirm server-side environment drift
  - environment mismatch remains plausible across time, but not yet proven

### E4. Inspect current server dataset files

- Change:
  - none; read-only inspection
- Result:
  - current server dataset hashes recorded in `O5`
- Conclusion:
  - current dataset snapshot is known
  - historical snapshot equivalence remains inconclusive

### E5. Inspect server-side hidden runtime state

- Change:
  - none; read-only inspection
- Result:
  - server uses `bitsandbytes 0.43.1`, `vllm 0.3.3`, `torch 2.1.2+cu121`
  - shell environment exports `BNB_CUDA_VERSION=121`
  - older automation logs already show the same `BNB_CUDA_VERSION=121` warning
- Conclusion:
  - there is hidden runtime state worth tracking
  - but the currently discovered bitsandbytes override is not yet enough to explain the regression by itself

### E6. Repeat `r182_forums_fixed22` on the same server environment

- Change:
  - rerun the exact same config again on `paper-2-genereic + pretext + A6000`
- Result:
  - first replay: `best_top1 = 0.24988732212993367`, `synthetic_train_count = 91`
  - second replay: `best_top1 = 0.24647479235078232`, `synthetic_train_count = 89`
- Conclusion:
  - confirmed material run-to-run instability for `r182_forums_fixed22`
  - instability is visible before downstream scoring, because synthetic corpus size also changed

### E7. Implement and verify a Stage2 fallback path for baseline modes without a shared Stage1 session

- Change:
  - updated `paper-new-round19/paper_new_selector/pipeline.py`
  - when `skip_bootstrap=False` but `shared_session.backend` is absent, pipeline now calls
    `generate_bootstrapped_samples(prompt_list, model_path, bootstrap_cfg)`
    instead of `generate_with_shared_session(...)`
- Result:
  - `tests.test_pipeline_smoke` passes under `unittest`
  - new regression test confirms `expand_only`/`expand_private` style contracts use
    `generation_path = "standalone_bootstrap"`
- Conclusion:
  - confirmed the internal single-run blocker was the unconditional shared-session Stage2 call

### E8. Add and verify standardized artifact preparation scripts for `WASP` and `DPGA-TextSyn`

- Change:
  - added `WASP/src/prepare_paper_new_artifacts.py`
  - added `DPGA-TextSyn/main/prepare_paper_new_artifacts.py`
  - both scripts normalize user-provided per-dataset raw artifacts into:
    - `WASP/outputs/paper_new_screening/<dataset>/train.jsonl`
    - `DPGA-TextSyn/outputs/paper_new_screening/<dataset>/epoch_all.json`
- Result:
  - direct CLI checks succeeded for both scripts
  - after preparing `jobs` artifacts, config-driven `validate-only` reports:
    - `wasp_jobs_single_run`: `source_exists=True`
    - `dpga_jobs_single_run`: `source_exists=True`
- Conclusion:
  - confirmed the external single-run blocker was the missing artifact-organization flow, not the evaluator

### E9. Re-run internal single-run expand baselines on the server using the actual A6000 mapping

- Change:
  - no code change; reran with `CUDA_VISIBLE_DEVICES=0`
- Result:
  - the old `shared_session must expose a backend` error disappeared
  - new failure became:
    - `The model's max seq len (512) is larger than the maximum number of tokens that can be stored in KV cache ...`
- Conclusion:
  - confirms the shared-session bug is fixed
  - identifies a new runtime/config/input-shape blocker

### E10. Compare raw baseline seed lengths against already-working Stage1 seeds

- Change:
  - read-only local comparison
- Result:
  - raw `jobs` / `forums` texts are dramatically longer than working Stage1 seeds
- Conclusion:
  - confirms `H7`
  - indicates the new baselines need seed-shape normalization, not just GPU selection fixes

### E11. Rerun the internal single-run expand baselines on the server after the first local seed-cap/config adjustments

- Change:
  - reran `eo_jobs_single_run` and `ep_forums_single_run` on the actual A6000 (`CUDA_VISIBLE_DEVICES=0`)
- Result:
  - `expand_only`
    - no longer fails at Stage2 startup
    - but exports an empty synthetic corpus
  - `expand_private`
    - produces 28 synthetic texts
    - then fails in `eval_small` with CUDA OOM
- Conclusion:
  - the original shared-session bug is gone
  - the two internal baselines now fail for different reasons

### E12. Measure actual server-side tokenizer lengths for the expand-baseline bootstrap prompts

- Change:
  - read actual `stage1_summary.json` outputs and tokenized prompts with the server's local `llama_2_7b_hf` tokenizer
- Result:
  - `expand_only` prompts are always over `256` tokens
  - `expand_private` prompts are frequently near or above `256` tokens
- Conclusion:
  - confirms `H8`
  - strengthens `H7`

## Root Cause

`expand_only` and `expand_private` were written as Stage2-expanding baselines but their Stage1 paths never create a shared generator backend, while the pipeline unconditionally required one; `WASP` and `DPGA-TextSyn` already had export/eval adapters but lacked any dataset-aware script that materialized the exact source artifact paths declared by the four-dataset single-run YAMLs.

For the next layer of failures after that fix:

- `expand_only` still produced an empty corpus because its public-seed bootstrap prompts exceeded the configured single-run `max_model_len`, so the standalone Stage2 vLLM path had no practical generation budget.
- `expand_private` then failed in `eval_small` because the standalone Stage2 vLLM engine was not explicitly released before downstream eval began, leaving the process with large-model GPU memory still resident.

## Fix

- `paper-new-round19/paper_new_selector/pipeline.py`
  - add shared-session detection
  - use shared-session Stage2 generation when available
  - otherwise fall back to standalone bootstrap generation with the configured bootstrap model
- `paper-new-round19/paper_new_selector/baseline_modes.py`
  - cap `expand_only` / `expand_private` seed texts to a configurable word limit before Stage2 prompt building
- `paper-new-round19/paper_new_selector/stage1_runner.py`
  - default the expand-baseline seed text cap to `56` words, matching the scale seen in completed `PrE-Text` Stage1 seeds
- `paper-new-round19/configs/experiments/single_run_baseline_screening/_base_single_run_expand_only.yaml`
  - keep the Stage2 vLLM bootstrap parameters aligned with `PrE-Text / paper-new` defaults
  - only add `seed_text_max_words: 40` to normalize the expand-baseline seed shape
- `paper-new-round19/configs/experiments/single_run_baseline_screening/_base_single_run_expand_private.yaml`
  - keep the Stage2 vLLM bootstrap parameters aligned with `PrE-Text / paper-new` defaults
  - only add `seed_text_max_words: 40`
- `paper-new-round19/paper_new_selector/pretext_bridge.py`
  - replace the imported standalone Stage2 vLLM call with a local wrapper that:
    - runs the same `vllm + local model` generation path
    - explicitly releases the temporary `LLM` runtime in `finally`
- `paper-new-round19/tests/test_pretext_bridge.py`
  - add regression coverage that the standalone Stage2 vLLM helper releases runtime memory
- `paper-new-round19/tests/test_pipeline_smoke.py`
  - add regression coverage for the no-shared-session Stage2 path
- `WASP/src/prepare_paper_new_artifacts.py`
  - add dataset/manifest-driven artifact normalization to `outputs/paper_new_screening/<dataset>/train.jsonl`
- `DPGA-TextSyn/main/prepare_paper_new_artifacts.py`
  - add dataset/manifest-driven artifact normalization to `outputs/paper_new_screening/<dataset>/epoch_all.json`
- `paper-new-round19/configs/experiments/single_run_baseline_screening/_base_single_run_wasp.yaml`
  - record `prepare_entry`
- `paper-new-round19/configs/experiments/single_run_baseline_screening/_base_single_run_dpga.yaml`
  - record `prepare_entry`

## Observations (2026-05-08 repeat10 GPU routing)

- The repeat10 batch runner exported both `CUDA_VISIBLE_DEVICES=0` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- On the server:
  - `CUDA_VISIBLE_DEVICES=0` with no `CUDA_DEVICE_ORDER` made `torch.cuda.get_device_name(0)` report `NVIDIA RTX A6000`.
  - `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID` made `torch.cuda.get_device_name(0)` report `NVIDIA GeForce RTX 2080 Ti`.
  - `CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID` made `torch.cuda.get_device_name(0)` report `NVIDIA RTX A6000`.
- The failed repeat10 logs for `expand_only`, `expand_private`, and `WASP` showed `GPU 0 has a total capacity of 10.75 GiB`, which matches the 2080Ti, not the A6000.
- `c4` runs succeeded because they do not load the Stage2 `vllm` generator, while every baseline that needed `vllm` failed immediately with 2080Ti-sized OOM conditions.

## Hypotheses (2026-05-08 repeat10 GPU routing)

### H1: `CUDA_DEVICE_ORDER=PCI_BUS_ID` reorders the devices so `CUDA_VISIBLE_DEVICES=0` points at the 2080Ti (ROOT HYPOTHESIS)
- Supports: direct server-side `torch.cuda.get_device_name(0)` flips from A6000 to 2080Ti when only `CUDA_DEVICE_ORDER=PCI_BUS_ID` is added.
- Conflicts: none after the direct comparison experiment.
- Test: compare `torch.cuda.get_device_name(0)` under the three environment combinations above.

### H2: `nohup` or `conda activate` dropped the intended GPU environment variables
- Supports: the first server run behaved differently from a simple foreground `torch` check.
- Conflicts: the logs showed the exact memory signature expected from the `PCI_BUS_ID + visible=0` combination; the issue reproduced without `nohup`.
- Test: reproduce the device-name check in a normal shell with the same exported variables.

### H3: The batch runner was actually using the A6000, but another process inside the same environment lowered the visible memory reading
- Supports: A6000 already had other workloads on it.
- Conflicts: the reported total capacity was 10.75 GiB, not 49 GiB, so the device identity itself was wrong.
- Test: compare reported total capacity in failure logs against direct `torch` device-name checks.

## Experiments (2026-05-08 repeat10 GPU routing)

### E13. Compare `torch.cuda.get_device_name(0)` across the exact env combinations used by repeat10

- Change:
  - no code change; three read-only server checks
- Result:
  - `CUDA_VISIBLE_DEVICES=0` => `NVIDIA RTX A6000`
  - `CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID` => `NVIDIA GeForce RTX 2080 Ti`
  - `CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID` => `NVIDIA RTX A6000`
- Conclusion:
  - confirms `H1`
  - rejects `H2` and `H3` as primary causes

## Root Cause (2026-05-08 repeat10 GPU routing)

The repeat10 launcher forced `CUDA_DEVICE_ORDER=PCI_BUS_ID` while still pinning `CUDA_VISIBLE_DEVICES=0`, and on this server that combination selects the 2080Ti rather than the A6000.

## Fix (2026-05-08 repeat10 GPU routing)

- `paper-new-round19/paper_new_selector/repeat10_baseline_runner.py`
  - default repeat10 runs to `CUDA_VISIBLE_DEVICES=1` when using `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  - allow an explicit override via `REPEAT10_CUDA_VISIBLE_DEVICES`
- `paper-new-round19/scripts/run_repeat10_baseline_screening.sh`
  - keep `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  - export `CUDA_VISIBLE_DEVICES=${REPEAT10_CUDA_VISIBLE_DEVICES:-1}` so the default server target is the A6000
- `paper-new-round19/tests/test_repeat10_baseline_runner.py`
  - update regression coverage to assert the repeat10 child env defaults to the A6000 slot and still honors explicit overrides

## Observations (2026-05-08 repeat10 external baseline seed04 failures)

- The repeat10 resume run completed `ep_microblog_repeat10_seed04` and `wasp_jobs_repeat10_seed04`, then failed the remaining external runs for `seed04`:
  - `wasp_congressional_repeat10_seed04`
  - `wasp_forums_repeat10_seed04`
  - `wasp_microblog_repeat10_seed04`
  - `dpga_congressional_repeat10_seed04`
  - `dpga_forums_repeat10_seed04`
  - `dpga_microblog_repeat10_seed04`
- Representative success logs (`wasp_jobs_repeat10_seed04`, `dpga_jobs_repeat10_seed04`) show the full eval path completed normally.
- Representative failure logs (`wasp_congressional_repeat10_seed04`, `dpga_congressional_repeat10_seed04`) both fail before eval, inside external artifact generation.
- The shared error signature is `thesis_platform.models.backends.VllmGenerationError: vllm_runtime_gpu_oom`.
- The failure logs explicitly report the A6000 as the target device (`visible=1`, total capacity `47.54 GiB`), so this is not another wrong-GPU-selection bug.
- The same failure logs also show another process already holding roughly `34 GiB` on the A6000:
  - `Process 1228 has 34.23 GiB memory in use.` in the WASP failure
  - `Process 11279 has 34.05 GiB memory in use.` in the DPGA failure
- The repeat10 runner currently retries only `# GPU blocks: 0` / `No available memory for the cache blocks`, and it launches every baseline immediately without waiting for a safe A6000 free-memory window.

## Hypotheses (2026-05-08 repeat10 external baseline seed04 failures)

### H1: External baseline failures are caused by transient A6000 memory contention from other processes (ROOT HYPOTHESIS)
- Supports:
  - both failing baselines die in `vllm_runtime_gpu_oom`
  - both logs show a different large foreign process consuming ~34 GiB on the A6000
  - `jobs` succeeds for the same seed when the card is freer
- Conflicts:
  - none in the collected evidence
- Test:
  - compare success vs failure logs and verify the failure path coincides with low free memory on the A6000

### H2: Congressional / forums / microblog configs are intrinsically wrong for WASP and DPGA
- Supports:
  - only non-`jobs` runs failed in this batch
- Conflicts:
  - the same error appears in both methods and points to GPU OOM, not config or path errors
  - the runner continued successfully into later `c4` work, so the batch loop itself was healthy
- Test:
  - inspect the failure stack and check whether the exception appears before config-dependent eval logic

### H3: The repeat10 runner is correctly using A6000 but treats resource contention as a permanent failure
- Supports:
  - the runner only classifies cache-block startup failures as retryable
  - the actual error is `vllm_runtime_gpu_oom`, which it currently records as hard failure
- Conflicts:
  - none
- Test:
  - inspect retry classification logic and confirm it lacks runtime-OOM handling and any pre-launch wait for safe free memory

## Experiments (2026-05-08 repeat10 external baseline seed04 failures)

### E14. Compare successful and failed external-baseline repeat10 logs for the same seed

- Change:
  - read-only inspection of:
    - `logs/wasp_jobs_repeat10_seed04.log`
    - `logs/wasp_congressional_repeat10_seed04.log`
    - `logs/dpga_jobs_repeat10_seed04.log`
    - `logs/dpga_congressional_repeat10_seed04.log`
- Result:
  - `jobs` succeeds end-to-end for both baselines
  - `congressional` fails in external artifact generation with `vllm_runtime_gpu_oom`
  - the failing logs show another process consuming ~34 GiB on the A6000
- Conclusion:
  - confirms `H1`
  - confirms `H3`
  - rejects `H2` as the primary cause

## Root Cause (2026-05-08 repeat10 external baseline seed04 failures)

The repeat10 runner treats `vllm_runtime_gpu_oom` as a permanent failure and launches external `vllm` generation without first waiting for the A6000 to have a safe free-memory buffer, so transient contention from other users' large A6000 jobs gets written into the summary as baseline failures.

## Fix (2026-05-08 repeat10 external baseline seed04 failures)

- `paper-new-round19/paper_new_selector/repeat10_baseline_runner.py`
  - classify `vllm_runtime_gpu_oom`, `CUDA out of memory`, and the standalone Stage2 OOM wrapper message as retryable resource failures
  - add `spec_requires_vllm(...)` so only non-`c4` baselines wait on GPU capacity
  - query `nvidia-smi` for the A6000 free memory before each `vllm` run
  - wait until the A6000 has a safe free-memory threshold before launching a `vllm`-dependent experiment
  - raise the repeat10 retry budget beyond a single cache-block retry so transient resource contention can recover
- `paper-new-round19/tests/test_repeat10_baseline_runner.py`
  - add regression coverage for runtime-OOM classification
  - add regression coverage for parsing the A6000 free-memory report
  - add regression coverage for `spec_requires_vllm(...)`

## Observations (2026-05-27 E6 round14 GPU routing)

- `E6 formal180` was launched from:
  - `paper-new-round-14/scripts/run_round14_lineage_e6_budget_sweep_repeat2_180_sequential.sh`
- The launcher exported:
  - `CUDA_VISIBLE_DEVICES=1`
  - but **did not** export `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- The downstream failure logs repeatedly showed:
  - `vLLM generation memory precheck | free=10.58 GiB required=26.00 GiB gpu=0 visible=1`
- At the same time, direct server-side `nvidia-smi` showed:
  - physical GPU 0 = `RTX 2080 Ti`
  - physical GPU 1 = `RTX A6000`
  - A6000 free memory was actually about `39 GiB`
- Therefore the `10.58 GiB` free-memory reading matched the 2080Ti, not the A6000.
- The same server had already established earlier that:
  - with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=1` maps to the A6000
  - without that ordering guarantee, visible-device mapping on this host does not match the physical GPU index assumptions used by the E4/E5 launchers

## Hypotheses (2026-05-27 E6 round14 GPU routing)

### H1: E6 launcher selected the wrong visible GPU because it exported `CUDA_VISIBLE_DEVICES=1` without `CUDA_DEVICE_ORDER=PCI_BUS_ID` (ROOT HYPOTHESIS)
- Supports:
  - failure precheck reads `free=10.58 GiB`, consistent with the 2080Ti
  - direct `nvidia-smi` shows A6000 still has plenty of free memory
  - E4/E5 scripts explicitly export `CUDA_DEVICE_ORDER=PCI_BUS_ID`, while E6 did not
- Conflicts:
  - none after comparing the launcher scripts and the observed free-memory numbers
- Test:
  - align the E6 launcher with the E4/E5 GPU-routing contract by exporting `CUDA_DEVICE_ORDER=PCI_BUS_ID` and deriving `TARGET_GPU_INDEX` from the same visible-device value

### H2: E6 still waited on the correct physical GPU, but the inner PyTorch/vLLM precheck ignored `CUDA_VISIBLE_DEVICES`
- Supports:
  - outer wait logic uses physical index `1`
- Conflicts:
  - the inner log still reported `visible=1`, so it did inherit the variable; the mismatch is more consistent with device-order interpretation than with env loss
- Test:
  - keep the env inheritance path unchanged and only add `CUDA_DEVICE_ORDER=PCI_BUS_ID`; if the mismatch disappears, H2 is rejected

## Experiments (2026-05-27 E6 round14 GPU routing)

### E15. Compare E6 launcher GPU env against the established E4/E5 launcher contract

- Change:
  - read-only script comparison
- Result:
  - `E4/E5` scripts export both:
    - `CUDA_DEVICE_ORDER=PCI_BUS_ID`
    - `CUDA_VISIBLE_DEVICES=...`
  - `E6` script exported only:
    - `CUDA_VISIBLE_DEVICES=1`
- Conclusion:
  - confirms `H1`
  - rejects the idea that E6 was already using the same GPU-routing contract as the successful experimental pipelines

## Root Cause (2026-05-27 E6 round14 GPU routing)

The `round14` E6 launcher pinned `CUDA_VISIBLE_DEVICES=1` but omitted `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so on this server the visible-device mapping resolved to the 2080Ti rather than the intended A6000; the outer wait logic checked physical GPU 1, while the inner PyTorch/vLLM startup precheck effectively ran against the wrong visible device.

## Fix (2026-05-27 E6 round14 GPU routing)

- `paper-new-round-14/scripts/run_round14_lineage_e6_budget_sweep_smoke90_sequential.sh`
  - export `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  - export `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}`
  - derive `TARGET_GPU_INDEX` from the same visible-device pin
- `paper-new-round-14/scripts/run_round14_lineage_e6_budget_sweep_repeat2_180_sequential.sh`
  - export `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  - export `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}`
  - derive `TARGET_GPU_INDEX` from the same visible-device pin
- `paper-new-round-14/tests/test_round14_lineage_e6_budget_sweep.py`
  - add regression coverage that the E6 sequential launchers include the same `CUDA_DEVICE_ORDER=PCI_BUS_ID` contract as E4/E5
