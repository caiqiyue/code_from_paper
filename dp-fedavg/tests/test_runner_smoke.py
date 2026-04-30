from pathlib import Path

from dp_fedavg.config import load_yaml_config
from dp_fedavg.runners import build_experiment_runtime


def test_build_experiment_runtime_from_real_yaml() -> None:
    config_path = Path("configs/experiments/smoke/single_node_jobs_smoke.yaml").resolve()
    config = load_yaml_config(config_path)
    runtime = build_experiment_runtime(config_path, config=config)
    assert runtime.dataset_name == "jobs"
    assert runtime.runner_mode == "single_node"
    assert runtime.output_root.name == "single_node_jobs_smoke"
