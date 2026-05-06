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


def test_all_checked_in_experiment_configs_resolve():
    root = Path(__file__).resolve().parents[1]
    config_paths = [
        "configs/experiments/r1_imdb_base.yaml",
        "configs/experiments/r1_imdb_temp_low.yaml",
        "configs/experiments/r1_imdb_temp_mid.yaml",
        "configs/experiments/r1_imdb_temp_high.yaml",
        "configs/experiments/p1_jobs_pretext_style.yaml",
        "configs/experiments/p1_forums_pretext_style.yaml",
        "configs/experiments/p1_microblog_pretext_style.yaml",
        "configs/experiments/p1_congressional_pretext_style.yaml",
    ]

    for config_path in config_paths:
        cfg = load_experiment_config(root / config_path)
        assert cfg["experiment"]["id"]
