from dp_prompt.pretext_bridge import build_thesis_eval_config


def test_build_thesis_eval_config_sets_small_eval_paths():
    cfg = build_thesis_eval_config(
        experiment_id="demo",
        dataset_name="jobs",
        train_path="thesis_platform/datasets/pretext_jobs/formatted/jobs_train.json",
        eval_path="thesis_platform/datasets/pretext_jobs/formatted/jobs_eval.json",
        initialization_path="thesis_platform/datasets/pretext_initialization_c4_en/formatted/initialization.json",
        output_root="outputs/demo",
    )

    assert cfg.data["dataset_name"] == "jobs"
    assert cfg.downstream_eval["run_small_eval"] is True
    assert cfg.downstream_eval["linux_small_eval_mode"] == "gpt2"
