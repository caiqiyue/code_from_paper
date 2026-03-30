from __future__ import annotations

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.experiment_runner import ExperimentRunner


def run_pipeline(config_path: str, *, resume: bool = False, resume_dir: str | None = None) -> dict:
    """Load an experiment config and execute the full platform pipeline."""

    config = load_experiment_config(config_path)
    runner = ExperimentRunner(config)
    return runner.run(resume=resume, resume_dir=resume_dir)

