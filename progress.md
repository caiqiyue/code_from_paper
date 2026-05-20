# Progress

- Reset the on-disk plan to focus on the `anchor robustness` readiness audit.
- Confirmed `paper-new-round19` includes a generated `round23_collection_repeat40` config set with budgets `k18..k22`, so `k0=19/20/21` exists at least at manifest/config level.
- Confirmed `paper-new-round23` exposes `reference_budget` as a parameter in scripts/configs, but several feature names and bundle artifacts are explicitly tied to `k20`.
- Confirmed the 1200-count collection is present locally as manifest/config coverage over `6 datasets x 40 seeds x 5 budgets`, but materialized collection outputs/log summaries are not present under `paper-new-round19/outputs` or `paper-new-round19/logs`.
- Confirmed `model-train` already has build/split/train/eval/export code for `round23`, including a `--collection-manifest` path, but its constants and exported schema still treat `k0=20` as the fixed reference budget.
- Confirmed the currently exported local runtime bundle in `paper-new-round23/artifacts/controller_bundle/round23_controller_lightgbm_local` comes from `round22_full500_round23_controller_local_v1`, not from the 1200 collection manifest.
