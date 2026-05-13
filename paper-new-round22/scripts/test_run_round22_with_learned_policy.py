"""Targeted tests for run_round22_with_learned_policy.py."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_round22_with_learned_policy import (  # noqa: E402
    build_round19_subprocess_env,
    generate_override_config,
    load_yaml_with_inherits,
)


def test_build_round19_subprocess_env_adds_round19_root():
    env = build_round19_subprocess_env()
    assert "PYTHONPATH" in env
    assert "paper-new-round19" in env["PYTHONPATH"]


def test_build_round19_subprocess_env_sets_cuda_order_when_needed():
    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    old_order = os.environ.get("CUDA_DEVICE_ORDER")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ.pop("CUDA_DEVICE_ORDER", None)
        env = build_round19_subprocess_env()
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    finally:
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
        if old_order is None:
            os.environ.pop("CUDA_DEVICE_ORDER", None)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = old_order


def test_generate_override_config_disables_budget_rule_and_pins_seed_top_k():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config_path = root / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "meta:",
                    "  experiment_id: test_exp",
                    "selector:",
                    "  seed_top_k: 20",
                    "  seed_budget_rule:",
                    "    enabled: true",
                    "    mode: hierarchical_shape_routing",
                    "paths:",
                    "  output_root: ./outputs/test_exp",
                ]
            ),
            encoding="utf-8",
        )

        override_path, experiment_id = generate_override_config(
            original_config_path=config_path,
            predicted_k=19,
            model_dir=root / "bundle",
            output_root=root / "runtime",
        )
        merged = load_yaml_with_inherits(override_path)
        assert experiment_id == "test_exp"
        assert int(merged["selector"]["seed_top_k"]) == 19
        assert bool(merged["selector"]["seed_budget_rule"]["enabled"]) is False
        assert str(merged["meta"]["learned_budget_runtime"]["model_dir"]).endswith("bundle")


if __name__ == "__main__":
    tests = [
        ("env.adds_round19_root", test_build_round19_subprocess_env_adds_round19_root),
        ("env.cuda_order", test_build_round19_subprocess_env_sets_cuda_order_when_needed),
        ("override_config", test_generate_override_config_disables_budget_rule_and_pins_seed_top_k),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  {name}: OK")
        except Exception as exc:  # pragma: no cover - direct-script harness
            failures += 1
            print(f"  {name}: FAILED - {exc}")
    if failures:
        raise SystemExit(1)
    print("\nALL LEARNED RUNTIME TESTS PASSED")
