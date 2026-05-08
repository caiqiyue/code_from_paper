from pathlib import Path

from paper_new_selector.repeat10_baseline_runner import build_repeat10_run_specs
from paper_new_selector.thesis_bridge import load_yaml_config


def test_all_internal_screening_baseline_configs_resolve():
    config_paths = [
        Path("configs/experiments/single_node_screening/c4_s_jobs_screening.yaml"),
        Path("configs/experiments/single_node_screening/c4_s_forums_screening.yaml"),
        Path("configs/experiments/single_node_screening/c4_s_congressional_screening.yaml"),
        Path("configs/experiments/single_node_screening/c4_s_microblog_screening.yaml"),
        Path("configs/experiments/single_node_screening/eo_s_jobs_screening.yaml"),
        Path("configs/experiments/single_node_screening/eo_s_forums_screening.yaml"),
        Path("configs/experiments/single_node_screening/eo_s_congressional_screening.yaml"),
        Path("configs/experiments/single_node_screening/eo_s_microblog_screening.yaml"),
        Path("configs/experiments/single_node_screening/ep_s_jobs_screening.yaml"),
        Path("configs/experiments/single_node_screening/ep_s_forums_screening.yaml"),
        Path("configs/experiments/single_node_screening/ep_s_congressional_screening.yaml"),
        Path("configs/experiments/single_node_screening/ep_s_microblog_screening.yaml"),
        Path("configs/experiments/single_node_screening/wasp_s_jobs_screening.yaml"),
        Path("configs/experiments/single_node_screening/wasp_s_forums_screening.yaml"),
        Path("configs/experiments/single_node_screening/wasp_s_congressional_screening.yaml"),
        Path("configs/experiments/single_node_screening/wasp_s_microblog_screening.yaml"),
        Path("configs/experiments/single_node_screening/dpga_s_jobs_screening.yaml"),
        Path("configs/experiments/single_node_screening/dpga_s_forums_screening.yaml"),
        Path("configs/experiments/single_node_screening/dpga_s_congressional_screening.yaml"),
        Path("configs/experiments/single_node_screening/dpga_s_microblog_screening.yaml"),
    ]

    for path in config_paths:
        config = load_yaml_config(path)
        assert config["data"]["train_limit"] == 256
        assert config["data"]["eval_limit"] == 256
        assert config["data"]["initialization_limit"] == 1024
        assert config["bootstrap"]["num_prompts"] == 100
        assert config["eval"]["max_samples_per_client"] == 16


def test_single_run_expand_baselines_apply_compact_bootstrap_overrides():
    for path in [
        Path("configs/experiments/single_run_baseline_screening/eo_jobs_single_run.yaml"),
        Path("configs/experiments/single_run_baseline_screening/ep_forums_single_run.yaml"),
    ]:
        config = load_yaml_config(path)
        assert config["bootstrap"]["max_tokens"] == 85
        assert config["bootstrap"]["max_model_len"] == 512
        assert config["bootstrap"]["gpu_memory_utilization"] == 0.35
        assert config["bootstrap"]["seed_text_max_words"] == 40


def test_repeat10_configs_exist_for_all_200_runs():
    config_paths = list(Path("configs/experiments/repeat10_baseline_screening").glob("*.yaml"))
    assert len(config_paths) == 200
    assert len(build_repeat10_run_specs()) == 200


def test_repeat10_external_artifacts_are_seed_specific():
    config = load_yaml_config("configs/experiments/repeat10_baseline_screening/wasp_jobs_repeat10_seed01.yaml")
    assert config["external_baseline"]["source_artifact_path"].endswith("repeat10/jobs/seed01/train.jsonl")
    assert config["external_baseline"]["summary_output_path"].endswith(
        "repeat10_baseline_screening/wasp/jobs/seed01/stage1_summary.json"
    )

    config = load_yaml_config("configs/experiments/repeat10_baseline_screening/dpga_jobs_repeat10_seed01.yaml")
    assert config["external_baseline"]["source_artifact_path"].endswith("repeat10/jobs/seed01/epoch_all.json")
    assert config["external_baseline"]["summary_output_path"].endswith(
        "repeat10_baseline_screening/dpga/jobs/seed01/stage1_summary.json"
    )


def test_repeat10_internal_expand_baselines_keep_server_valid_bootstrap_defaults():
    for path in [
        Path("configs/experiments/repeat10_baseline_screening/eo_jobs_repeat10_seed01.yaml"),
        Path("configs/experiments/repeat10_baseline_screening/ep_forums_repeat10_seed01.yaml"),
    ]:
        config = load_yaml_config(path)
        assert config["bootstrap"]["max_tokens"] == 85
        assert config["bootstrap"]["max_model_len"] == 512
        assert config["bootstrap"]["gpu_memory_utilization"] == 0.35
        assert config["bootstrap"]["seed_text_max_words"] == 40
