# Findings

## Anchor Robustness Audit

- `paper-new-round19/configs/experiments/single_node_tuning_round19/round23_collection_repeat40` contains generated fixed-budget configs spanning:
  - datasets across the round19/round23 collection set
  - 40 seeds
  - budgets `k=18,19,20,21,22`
- This implies the nominal 1200 count matches `6 datasets * 40 seeds * 5 k values = 1200` if all six datasets are present in the manifest/results.
- Early `round23` search shows two kinds of `k0=20` dependency:
  - configurable defaults such as `--reference-budget 20` and generated YAML `reference_budget: 20`
  - stronger schema/artifact coupling such as `support_mean_at_k20` and bundle metadata `reference_budget: 20`
- `paper-new-round19/tests/test_round19_round23_collection_runner.py` asserts the collection manifest contains exactly 1200 specs and spans from `jobs ... k18` to `openreview ... k22`.
- The local workspace currently has the 1200-config manifest, but does not have materialized `paper-new-round19/outputs/round23_collection_repeat40/...` or `paper-new-round19/logs/round19_round23_collection_repeat40_summary.tsv`. So the 1200 local items are currently manifest/config records, not completed collection result records.
- `paper-new-round19/paper_new_selector/round19_round23_collection_runner.py` validates a completed collection run by requiring:
  - `collection/context_summary.json`
  - `collection/final_result_summary.json`
  - `collection/budget_table.jsonl`
  - `eval/downstream_eval_summary.json`
- `paper-new-round23` runtime is partially parameterized for alternate `k0`:
  - `run_round23_with_dynamic_controller.py` accepts `--reference-budget`
  - `round23_reference_stage0_features.py` actually rewrites `selector.seed_top_k = reference_budget`
  - but helper filenames/dirs and feature keys remain hardcoded as `_k20_reference*` and `*_at_k20`
- The formal `round23` batch runner does not expose `reference_budget`:
  - `round23_dynamic_experiment_runner.py` only forwards config/model/output/timeout
  - generated YAMLs and base config also pin `seed_top_k: 20` and `reference_budget: 20`
- The current exported local bundles under `paper-new-round23/artifacts/controller_bundle/round23_controller_*_local` are not built from the 1200 collection set:
  - metadata says `training_data_version=round22_full500_round23_controller_local_v1`
  - bundle schema one-hot order only covers `jobs/congressional/forums/microblog`
- `model-train` already contains a formal collection-to-training pipeline:
  - `build_round23_controller_dataset.py` can build from `--collection-manifest`
  - `split_round23_controller_dataset.py` enforces `4 train datasets + 2 unseen test datasets`
  - `train_round23_controller.py` and `scripts/export_round23_controller_bundle.py` can train/export a new bundle
- But that `model-train` pipeline is also structurally centered on `REFERENCE_BUDGET = 20`:
  - `common.py` sets `REFERENCE_BUDGET = 20`
  - dataset builder names state features as `support_mean_at_k20`, `coverage_*_at_k20`, etc.
  - bundle export writes `reference_budget: 20`
- A local prototype dataset already exists under `model-train/artifacts/round23_datasets_local`, and its build report shows it was built from old `full-500` prototype files rather than the new collection manifest.

## Experiment Scope From Round23 Docs

- The formal thesis scope is fixed to a shared family axis plus method axis plus dataset axis:
  - methods: `PrE-Text`, `round19`, `round23`
  - seen datasets: `jobs`, `congressional`, `forums`, `microblog`
  - unseen datasets: `imdb`, `openreview`
  - target family presentation in the paper: `C4`, `EP`
- The required evidence chain includes:
  - main result table
  - unseen generalization table
  - controller decision quality table
  - mechanism ablations
  - end-to-end runtime experiments
  - motivation/diagnosis experiments for early selector innovation and fixed-`k` sweep

## Repo Coverage Notes

- The workspace contains separate directories for `PrE-Text`, `paper-new`, `paper-new-round19`, `paper-new-round23`, `WASP`, `DPGA-TextSyn`, `dp-fedavg`, `dp-ftrl`, and `dp-prompt`.
- The workspace already contains thesis planning docs and prior debug notes, including `DEBUG.md`, which likely records historical blockers for external baselines and replay issues.
- `thesis_platform/datasets` contains all six target dataset assets in local formatted form:
  - seen: `pretext_jobs`, `congressional`, `pretext_forums`, `pretext_microblog`
  - unseen: `pretext_imdb`, `pretext_openreview`
  - shared init asset present locally: `pretext_initialization_c4_en`
- No local `initialization_ep`-style dataset asset was found under `thesis_platform/datasets`.
- `PrE-Text` formal configs resolve correctly to the four seen datasets plus `pretext_initialization_c4_en` in both `single_node_formal` and `federated_formal`.
- `paper-new` early selector configs cover only the four seen datasets across `single_node_screening` and tuning rounds; no `imdb/openreview` configs were found there.
- `paper-new-round19` has:
  - adaptive base configs for the four seen datasets in `full_run`
  - repeat manifests for seen quick-compare
  - a large `round23_collection_repeat40` fixed-budget collection manifest spanning six datasets and multiple `k` values
  - external baseline screening configs for `WASP` / `DPGA-TextSyn` on the four seen datasets
- `paper-new-round23` has generated config trees for:
  - `real_smoke` on the four seen datasets
  - `quick_compare_repeat30` on the four seen datasets
  - `unseen_dataset_final_eval_repeat40` on `imdb/openreview`
  - local controller bundles under `artifacts/controller_bundle`
- The current local `round23_controller_lightgbm_local` bundle metadata says `training_data_version=round22_full500_round23_controller_local_v1`.
- The current local `round23_controller_lightgbm_local/feature_schema.json` uses `onehot_order = ['jobs', 'congressional', 'forums', 'microblog']`.
- Direct verification of `round23_context_features.build_feature_vector(...)` shows:
  - `dataset_name='jobs'` succeeds with the local lightgbm bundle schema
  - `dataset_name='imdb'` raises `ValueError` because `imdb` is absent from `onehot_order`
- `paper-new-round23/logs/round23_real_smoke_summary.tsv` currently contains only the header row.
- `WASP` and `DPGA-TextSyn` each provide:
  - a generator/export adapter
  - a `prepare_paper_new_artifacts.py` normalizer
  - a `run_paper_new_screening.py` summarizer
  - tests for the normalization/summarization flow
- Local standardized external baseline artifacts are currently missing:
  - `WASP/outputs/paper_new_screening/...`
  - `DPGA-TextSyn/outputs/paper_new_screening/...`
- `dp-fedavg` is incomplete in the local workspace:
  - experiment YAMLs inherit `configs/datasets/*.yaml`
  - the `configs/datasets` directory does not exist
  - direct config loading fails with `FileNotFoundError` for `dp-fedavg/configs/datasets/jobs.yaml`
- `dp-prompt` configs load successfully, but they split into two different protocols:
  - `p1_*_pretext_style.yaml` for four seen pretext-style datasets
  - `r1_imdb_*.yaml` for an `imdb_document` pipeline using `thesis_platform/datasets/imdb/formatted/train_len256.jsonl`
  - no `openreview` config exists
- `dp-ftrl` contains only `dp-ftrl.docx`; no runnable code or configs were found.
- The local Python test environment is currently broken for all repos at `pytest` import time because installed `pytest` tries to import `typing.assert_never` and fails before project tests run.

## Questions To Answer

- Which formal experiments are runnable now from code plus local assets?
- Which experiment families have implementation gaps or metric/asset mismatches?
- Which baselines should stay out of the main thesis table?
- Are all six datasets and the seen/unseen split concretely represented in code and data?
