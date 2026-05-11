# round22 model-train

This directory contains the training-side implementation for the `round22` contextual-bandit learned budget policy.

Planned workflow:

1. Build training tables from `round22_bandit_full_summary.jsonl`
2. Create context-level train/validation/test splits
3. Train five action-specific `LightGBMRegressor` models
4. Evaluate against oracle, fixed-k baselines, and `round19` replay

Primary scripts:

- `build_round22_bandit_dataset.py`
- `split_round22_bandit_dataset.py`
- `train_round22_bandit.py`
- `round19_replay.py`
- `eval_round22_bandit.py`

Related docs:

- `paper-new/docs/2026-05-11-round22-contextual-bandit-training-and-evaluation-design.md`
- `paper-new/docs/2026-05-11-round22-contextual-bandit-training-implementation-plan.md`

