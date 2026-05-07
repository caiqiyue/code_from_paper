from __future__ import annotations

import json
from pathlib import Path

from paper_new_selector.external_baselines.single_run_runner import (
    build_external_stage1_summary_from_config,
    run_external_single_run_from_config,
)


def test_build_external_stage1_summary_from_config_for_wasp(tmp_path, monkeypatch):
    repo_root = tmp_path
    source_path = repo_root / "WASP" / "outputs" / "paper_new_screening" / "jobs" / "train.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        '\n'.join(
            [
                json.dumps({"X": "alpha beta"}),
                json.dumps({"C": "gamma delta"}),
                json.dumps({"X": "alpha beta"}),
            ]
        ),
        encoding="utf-8",
    )
    config_path = repo_root / "paper-new-round19" / "configs" / "experiments" / "single_run_baseline_screening" / "wasp_jobs_single_run.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "meta": {"experiment_id": "wasp_jobs_single_run"},
                "pipeline": {"stage1_mode": "wasp_external"},
                "paths": {"output_root": "paper-new-round19/outputs/single_run_baseline_screening/wasp/jobs"},
                "external_baseline": {
                    "source_artifact_path": "WASP/outputs/paper_new_screening/jobs/train.jsonl",
                    "summary_output_path": "paper-new-round19/outputs/single_run_baseline_screening/wasp/jobs/stage1_summary.json",
                    "expected_budget": 100,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "paper_new_selector.external_baselines.single_run_runner.load_yaml_config",
        lambda _: json.loads(config_path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        "paper_new_selector.external_baselines.single_run_runner.resolve_repo_root",
        lambda: repo_root,
    )

    summary_path, payload = build_external_stage1_summary_from_config(config_path)
    assert summary_path.exists()
    assert payload["mode"] == "wasp_adapter"
    assert payload["direct_synthetic_texts"] == ["alpha beta", "gamma delta"]


def test_run_external_single_run_from_config_for_dpga_validate_only(tmp_path, monkeypatch):
    repo_root = tmp_path
    source_path = repo_root / "DPGA-TextSyn" / "outputs" / "paper_new_screening" / "jobs" / "epoch_all.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            [
                {"text": "delta epsilon"},
                {"text": "zeta eta"},
                {"text": "delta epsilon"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config_path = repo_root / "paper-new-round19" / "configs" / "experiments" / "single_run_baseline_screening" / "dpga_jobs_single_run.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "meta": {"experiment_id": "dpga_jobs_single_run"},
                "pipeline": {"stage1_mode": "dpga_external"},
                "paths": {"output_root": "paper-new-round19/outputs/single_run_baseline_screening/dpga/jobs"},
                "external_baseline": {
                    "source_artifact_path": "DPGA-TextSyn/outputs/paper_new_screening/jobs/epoch_all.json",
                    "summary_output_path": "paper-new-round19/outputs/single_run_baseline_screening/dpga/jobs/stage1_summary.json",
                    "expected_budget": 100,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "paper_new_selector.external_baselines.single_run_runner.load_yaml_config",
        lambda _: json.loads(config_path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        "paper_new_selector.external_baselines.single_run_runner.resolve_repo_root",
        lambda: repo_root,
    )

    result = run_external_single_run_from_config(config_path, validate_only=True)
    assert result["experiment_id"] == "dpga_jobs_single_run"
    assert result["stage1_mode"] == "dpga_external"
    assert result["source_exists"] is True
    assert result["expected_budget"] == 100
