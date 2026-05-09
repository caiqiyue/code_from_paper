import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from paper_new_selector.external_baselines.common_eval import (
    load_direct_synthetic_summary,
    run_external_stage1_summary_eval,
)


def test_load_direct_synthetic_summary_requires_direct_mode(tmp_path: Path):
    path = tmp_path / "stage1_summary.json"
    path.write_text(
        json.dumps({"skip_bootstrap": True, "direct_synthetic_texts": ["alpha beta"]}),
        encoding="utf-8",
    )
    payload = load_direct_synthetic_summary(path)
    assert payload["direct_synthetic_texts"] == ["alpha beta"]


def test_run_external_stage1_summary_eval_forwards_texts_to_common_eval(tmp_path: Path):
    path = tmp_path / "stage1_summary.json"
    path.write_text(
        json.dumps(
            {
                "skip_bootstrap": True,
                "direct_synthetic_texts": ["alpha beta", "gamma delta"],
            }
        ),
        encoding="utf-8",
    )
    with patch(
        "paper_new_selector.external_baselines.common_eval.run_eval",
        return_value={"status": "completed"},
    ) as run_eval:
        result = run_external_stage1_summary_eval(
            summary_path=path,
            config_path="configs/experiments/single_node_screening/c4_s_jobs_screening.yaml",
        )
    assert result["status"] == "completed"
    run_eval.assert_called_once()
    assert run_eval.call_args.kwargs["synthetic_texts"] == ["alpha beta", "gamma delta"]


def test_load_direct_synthetic_summary_deduplicates_and_filters(tmp_path: Path):
    path = tmp_path / "stage1_summary.json"
    path.write_text(
        json.dumps(
            {
                "skip_bootstrap": True,
                "direct_synthetic_texts": ["dup enough words", "dup enough words", "x", "unique text here"],
            }
        ),
        encoding="utf-8",
    )
    payload = load_direct_synthetic_summary(path)
    assert payload["direct_synthetic_texts"] == ["dup enough words", "unique text here"]


def test_run_external_baseline_eval_cli_runs_as_script():
    result = subprocess.run(
        [sys.executable, "paper_new_selector/run_external_baseline_eval.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
