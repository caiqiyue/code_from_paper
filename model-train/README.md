# model-train

This directory contains the training-side implementation for:

- `round22` static learned budget policy
- `round23` two-round dynamic `delta_k` controller

## Round22 workflow

1. Build training tables from `round22_bandit_full_summary.jsonl`
2. Create context-level train/validation/test splits
3. Train five action-specific regressors over `k ∈ {18,19,20,21,22}`
4. Evaluate against oracle, fixed-k baselines, and `round19` replay

Primary scripts:

- `build_round22_bandit_dataset.py`
- `split_round22_bandit_dataset.py`
- `train_round22_bandit.py`
- `round19_replay.py`
- `eval_round22_bandit.py`

## Round23 workflow

`round23` reuses `round22 full-500` artifacts and reframes them as a two-round controller problem:

- Round 0 fixed at `k0 = 20`
- controller predicts `delta_k ∈ {-2,-1,0,+1,+2}`
- Round 1 runs at `k1 = 20 + delta_k`

Primary scripts:

- `build_round23_controller_dataset.py`
- `split_round23_controller_dataset.py`
- `train_round23_controller.py`
- `eval_round23_controller.py`
- `scripts/multi_model_experiment_round23.py`
- `scripts/export_round23_controller_bundle.py`

Primary round23 outputs:

- `artifacts/round23_datasets/`
- `artifacts/round23_splits/`
- `artifacts/round23_models/`
- `artifacts/round23_reports/`

## Environment note

The round23 training/evaluation code expects working installations of:

- `lightgbm`
- `scikit-learn`
- optional `xgboost`

Dataset build and split scripts are pure-Python. Model-training scripts will fail fast with dependency errors if the local environment is missing the required compiled packages.
