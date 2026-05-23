# DEBUG: E1 controller-dev extra repeat15 failures

## Observations

- Server run: `thesis_main_controller_dev_extra_repeat15`, 150 experiments total.
- As of 2026-05-23 08:23 CST:
  - Round23 was running normally.
  - Baseline summary showed systematic failures for `pretext`, `wasp`, and `dpga` on `imdb/openreview`.
  - Failures were quick and not OOM.
- Failing generated configs inherited missing dataset-specific templates:
  - `pretext_imdb_seed42.yaml` -> `../single_run_baseline_screening/ep_imdb_single_run.yaml`
  - `wasp_imdb_seed42.yaml` -> `../single_run_baseline_screening/wasp_imdb_single_run.yaml`
  - `dpga_imdb_seed42.yaml` -> `../single_run_baseline_screening/dpga_imdb_single_run.yaml`
- Those dataset-specific templates do not exist for `imdb/openreview`.
- Existing data configs do exist:
  - `single_node_tuning_round19/_data_imdb.yaml`
  - `single_node_tuning_round19/_data_openreview.yaml`
- E2 pretext configs for new datasets already use generic base + dataset data config and succeeded.

## Hypotheses

### H1: Controller-dev extra baseline config generation incorrectly reused original four-dataset template logic. (ROOT HYPOTHESIS)
- Supports: generated configs referenced missing `ep_imdb_single_run.yaml`, `wasp_imdb_single_run.yaml`, `dpga_imdb_single_run.yaml`.
- Conflicts: round19 and round23 use `_data_imdb/_data_openreview` and mostly/fully work, so datasets are usable.
- Test: assert generated `pretext/wasp/dpga` imdb configs inherit generic base configs plus `_data_imdb.yaml`.

### H2: WASP/DPGA external artifacts are missing and might still block execution after config inheritance is fixed.
- Supports: source artifacts currently absent locally.
- Conflicts: external baseline runner has `generator_entry` support and can materialize missing artifacts at runtime if config loads.
- Test: validate external contract resolves `generator_entry` and expected source artifact path.

### H3: Two round19 failures are stochastic runtime failures unrelated to systematic config bug.
- Supports: 28 round19 runs succeeded under same mode.
- Conflicts: not relevant to pretext/WASP/DPGA systematic failures.
- Test: rerun failed experiments after config fix; runner skips prior successes and retries failures.

## Experiments

### E1: Regression test for pretext inheritance
- Before fix: failed; actual inherit was missing `ep_imdb_single_run.yaml`.
- After fix: passed.

### E2: Regression test for WASP/DPGA inheritance
- After fix: passed; `wasp_imdb_seed42.yaml` and `dpga_imdb_seed42.yaml` inherit generic external base configs plus `_data_imdb.yaml`.

### E3: External contract validation
- Result: `run_external_single_run_from_config(..., validate_only=True)` resolves:
  - WASP `stage1_mode=wasp_external`, `generator_entry=WASP/src/generate_paper_new_artifacts.py`
  - DPGA `stage1_mode=dpga_external`, `generator_entry=DPGA-TextSyn/main/generate_paper_new_artifacts.py`
- Source artifacts are absent before run, but runner can invoke generators.

## Root Cause

`thesis_main_controller_dev_extra_repeat15` generated pretext/WASP/DPGA configs for `imdb/openreview` using dataset-specific single-run templates that only exist for the original four E1 datasets.

## Fix

- For `thesis_main_controller_dev_extra_repeat15`, `pretext` now inherits:
  - `_base_single_run_expand_private.yaml`
  - `_data_imdb.yaml` / `_data_openreview.yaml`
- For `thesis_main_controller_dev_extra_repeat15`, `wasp` and `dpga` now inherit:
  - `_base_single_run_wasp.yaml` / `_base_single_run_dpga.yaml`
  - `_data_imdb.yaml` / `_data_openreview.yaml`
- Regenerated local 120 baseline configs.
- Added regression tests for this mode.

## Verification

- `D:\anconda\envs\pretext\python.exe -m unittest paper-new-round19.tests.test_thesis_e1_main_runner` -> 10 tests OK.
- `D:\anconda\envs\pretext\python.exe -m pytest paper-new-round23/scripts/test_round23_configs_and_runner.py -q` -> 11 passed.
- Config audit: 120 baseline configs, 0 bad dataset-specific inherits.
