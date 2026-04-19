from __future__ import annotations

import thesis_platform.adapters  # noqa: F401 - populate registry via import side effects.

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.experiment_runner import ExperimentRunner
from thesis_platform.core.registry import create
from thesis_platform.core.single_node_runner import SingleNodeRunner


def _build_single_node_runner(config) -> SingleNodeRunner:
    """Instantiate the single-node pipeline components from one resolved config."""

    repo_root = config.repo_root()
    generator = create(
        "generator",
        config.generator.get("name", "pretext_prompt_llm"),
        config.generator,
        repo_root,
    )
    scorer = create(
        "scorer",
        config.scorer.get("name", "datainf_real"),
        config.scorer,
        repo_root,
    )
    retriever = create(
        "retriever",
        config.retriever.get("name", "knn"),
        config.retriever,
        repo_root,
    )
    critic = create(
        "critic",
        config.critic.get("name", "fedtextgrad_llm"),
        config.critic,
        repo_root,
    )
    aggregator = create(
        "aggregator",
        config.aggregator.get("name", "dbscan_attn_tsgdm"),
        config.aggregator,
        repo_root,
    )
    return SingleNodeRunner(
        generator=generator,
        scorer=scorer,
        retriever=retriever,
        critic=critic,
        aggregator=aggregator,
        config=config,
    )


def run_pipeline(config_path: str, *, resume: bool = False, resume_dir: str | None = None) -> dict:
    """Load one experiment config and execute the matching pipeline."""

    config = load_experiment_config(config_path)
    if str(config.execution.get("mode", "federated")).strip().lower() == "single_node":
        runner = _build_single_node_runner(config)
        return runner.run()
    runner = ExperimentRunner(config)
    return runner.run(resume=resume, resume_dir=resume_dir)
