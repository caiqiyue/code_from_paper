from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_yaml_config
from .runners import build_experiment_runtime, run_federated, run_single_node


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    runtime = build_experiment_runtime(config_path, config=config)
    if runtime.runner_mode == "federated":
        run_federated(runtime)
        return
    if runtime.runner_mode == "single_node":
        run_single_node(runtime)
        return
    raise ValueError(f"Unsupported runner_mode: {runtime.runner_mode}")


if __name__ == "__main__":
    main()
