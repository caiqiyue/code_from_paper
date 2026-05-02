from pathlib import Path

from dp_prompt.config import load_experiment_config


def test_load_experiment_config_merges_inherits(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "runtime:\n  seed: 42\nmodel:\n  name: base\nprivacy:\n  temperature: 1.0\n",
        encoding="utf-8",
    )
    child.write_text(
        f"inherits:\n  - {base}\nprivacy:\n  temperature: 1.5\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(child)

    assert cfg["runtime"]["seed"] == 42
    assert cfg["model"]["name"] == "base"
    assert cfg["privacy"]["temperature"] == 1.5


def test_load_experiment_config_records_source_paths(tmp_path: Path):
    cfg_file = tmp_path / "standalone.yaml"
    cfg_file.write_text("runtime:\n  seed: 7\n", encoding="utf-8")

    cfg = load_experiment_config(cfg_file)

    assert str(cfg_file) in cfg["_meta"]["config_chain"]
