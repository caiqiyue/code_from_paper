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
