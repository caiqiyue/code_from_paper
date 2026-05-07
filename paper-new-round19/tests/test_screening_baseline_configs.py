from pathlib import Path

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
