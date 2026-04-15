"""Entry point for single-node fine experiments."""

from __future__ import annotations

import argparse
import json
import os

# Set PyTorch CUDA allocator config BEFORE importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import thesis_platform.adapters  # noqa: F401 - Populate the adapter registry via import side effects.

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.logging_utils import get_logger
from thesis_platform.core.registry import create
from thesis_platform.core.single_node_runner import SingleNodeRunner


def _build_components(config):
    """Instantiate all required components via the registry.

    Args:
        config: ExperimentConfig object

    Returns:
        Tuple of (generator, scorer, retriever, critic, aggregator)
    """
    repo_root = config.repo_root()

    # Generator
    generator = create(
        "generator",
        config.generator.get("name", "pretext_prompt_llm"),
        config.generator,
        repo_root,
    )

    # Scorer
    scorer = create(
        "scorer",
        config.scorer.get("name", "datainf"),
        config.scorer,
        repo_root,
    )

    # Retriever
    retriever = create(
        "retriever",
        config.retriever.get("name", "knn"),
        config.retriever,
        repo_root,
    )

    # Critic
    critic = create(
        "critic",
        config.critic.get("name", "fedtextgrad_llm"),
        config.critic,
        repo_root,
    )

    # Aggregator
    aggregator = create(
        "aggregator",
        config.aggregator.get("name", "dbscan_attn_tsgdm"),
        config.aggregator,
        repo_root,
    )

    return generator, scorer, retriever, critic, aggregator


def main() -> None:
    """Run a single-node fine experiment."""

    parser = argparse.ArgumentParser(
        description="Run a single-node fine-branch experiment."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment YAML config file.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and components without running the pipeline.",
    )
    args = parser.parse_args()

    # Load configuration
    logger = get_logger()
    logger.info("Loading configuration from: %s", args.config)
    config = load_experiment_config(args.config)
    logger.info("Experiment ID: %s", config.meta.get("experiment_id"))
    logger.info("Stage: %s", config.meta.get("stage"))

    # Build components
    logger.info("Instantiating components...")
    try:
        generator, scorer, retriever, critic, aggregator = _build_components(config)
        logger.info("Components instantiated successfully")
        for kind, name in [
            ("generator", config.generator.get("name", "pretext_prompt_llm")),
            ("scorer", config.scorer.get("name", "datainf")),
            ("retriever", config.retriever.get("name", "knn")),
            ("critic", config.critic.get("name", "fedtextgrad_llm")),
            ("aggregator", config.aggregator.get("name", "dbscan_attn_tsgdm")),
        ]:
            logger.info("  %s: %s", kind, name)
    except Exception as e:
        logger.error("Failed to instantiate components: %s", e)
        raise

    if args.validate_only:
        logger.info("Validation complete (--validate-only was set)")
        return

    # Create and run the single-node runner
    runner = SingleNodeRunner(
        generator=generator,
        scorer=scorer,
        retriever=retriever,
        critic=critic,
        aggregator=aggregator,
        config=config,
    )

    try:
        summary = runner.run()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()
