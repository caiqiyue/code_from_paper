import json
from pathlib import Path

from paper_new_selector.repeat10_baseline_runner import (
    REPEAT10_SEEDS,
    build_repeat10_child_env,
    build_repeat10_command,
    build_repeat10_run_specs,
    classify_retryable_failure,
    materialize_repeat10_configs,
    reset_repeat10_output_dir,
    resolve_repeat10_effective_status,
)


def test_repeat10_reuses_the_first_ten_round19_seeds():
    assert REPEAT10_SEEDS == list(range(1, 11))


def test_repeat10_builds_expected_200_specs():
    specs = build_repeat10_run_specs()
    assert len(specs) == 200
    assert specs[0].experiment_id == "c4_jobs_repeat10_seed01"
    assert specs[-1].experiment_id == "dpga_microblog_repeat10_seed10"


def test_repeat10_external_source_artifacts_are_seed_specific():
    specs = {spec.experiment_id: spec for spec in build_repeat10_run_specs()}
    assert specs["wasp_jobs_repeat10_seed01"].relative_source_artifact.as_posix().endswith(
        "WASP/outputs/paper_new_screening/repeat10/jobs/seed01/train.jsonl"
    )
    assert specs["dpga_jobs_repeat10_seed01"].relative_source_artifact.as_posix().endswith(
        "DPGA-TextSyn/outputs/paper_new_screening/repeat10/jobs/seed01/epoch_all.json"
    )


def test_repeat10_build_repeat10_child_env_pins_a6000_slot():
    env = build_repeat10_child_env({"CUDA_VISIBLE_DEVICES": "1"})
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_repeat10_retryable_failure_classifier_handles_vllm_cache_issue():
    log_text = "some text\n# GPU blocks: 0\nNo available memory for the cache blocks\n"
    assert classify_retryable_failure(log_text) == "retryable_vllm_cache"
    assert classify_retryable_failure("different failure") is None


def test_repeat10_command_routing_uses_external_runner_for_external_baselines():
    specs = {spec.experiment_id: spec for spec in build_repeat10_run_specs()}
    internal = build_repeat10_command(
        specs["eo_jobs_repeat10_seed01"],
        Path("configs/experiments/repeat10_baseline_screening/eo_jobs_repeat10_seed01.yaml"),
    )
    external = build_repeat10_command(
        specs["wasp_jobs_repeat10_seed01"],
        Path("configs/experiments/repeat10_baseline_screening/wasp_jobs_repeat10_seed01.yaml"),
    )
    assert internal[2] == "paper_new_selector.run_selector_single_node"
    assert external[2] == "paper_new_selector.run_external_baseline_single_run"


def test_repeat10_materialize_configs_returns_200_paths():
    generated = materialize_repeat10_configs(Path(__file__).resolve().parents[1])
    assert len(generated) == 200


def test_repeat10_effective_status_requires_completed_downstream_eval(tmp_path: Path):
    output_dir = tmp_path / "seed01"
    (output_dir / "eval").mkdir(parents=True)
    blocked_payload = {"status": "blocked", "metrics": {"best_top1": 0.1}}
    (output_dir / "eval" / "downstream_eval_summary.json").write_text(
        json.dumps(blocked_payload),
        encoding="utf-8",
    )
    assert resolve_repeat10_effective_status(0, output_dir) == 87

    completed_payload = {"status": "completed", "metrics": {"best_top1": 0.2}}
    (output_dir / "eval" / "downstream_eval_summary.json").write_text(
        json.dumps(completed_payload),
        encoding="utf-8",
    )
    assert resolve_repeat10_effective_status(0, output_dir) == 0


def test_repeat10_reset_output_dir_removes_stale_eval_artifacts(tmp_path: Path):
    output_dir = tmp_path / "seed02"
    stale = output_dir / "eval" / "downstream_eval_summary.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    reset_repeat10_output_dir(output_dir)
    assert output_dir.exists()
    assert not stale.exists()
