from pathlib import Path

from dp_fedavg.config import load_yaml_config
from dp_fedavg.paths import resolve_project_root


def test_resolve_project_root_points_at_dp_fedavg() -> None:
    root = resolve_project_root()
    assert root.name == "dp-fedavg"
    assert (root / "docs").exists()


def test_load_yaml_config_merges_inherits(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("runtime:\n  seed: 42\npaths:\n  output_root: outputs/base\n", encoding="utf-8")
    child.write_text("inherits:\n  - ./base.yaml\nruntime:\n  device: cuda\n", encoding="utf-8")

    cfg = load_yaml_config(child)

    assert cfg["runtime"]["seed"] == 42
    assert cfg["runtime"]["device"] == "cuda"
    assert cfg["paths"]["output_root"] == "outputs/base"
