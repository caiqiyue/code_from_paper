from dp_prompt.config import load_experiment_config


def test_pretext_style_experiment_config_sets_pipeline_mode():
    cfg = load_experiment_config("configs/experiments/p1_jobs_pretext_style.yaml")
    assert cfg["experiment"]["pipeline_mode"] == "pretext_style"
    assert cfg["llm"]["server"]["engine"] == "vllm"
    assert cfg["data"]["dataset_name"] == "jobs"
