import json
from pathlib import Path

from dp_prompt.config import load_experiment_config
from dp_prompt.runners.pretext_pipeline import run_pretext_style_pipeline


def test_pretext_style_experiment_config_sets_pipeline_mode():
    root = Path(__file__).resolve().parents[1]
    cfg = load_experiment_config(root / "configs/experiments/p1_jobs_pretext_style.yaml")
    assert cfg["experiment"]["pipeline_mode"] == "pretext_style"
    assert cfg["llm"]["server"]["engine"] == "vllm"
    assert cfg["data"]["dataset_name"] == "jobs"


def test_pretext_style_pipeline_writes_stage2_and_eval_artifacts(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    cfg = load_experiment_config(root / "configs/experiments/p1_jobs_pretext_style.yaml")
    cfg["runtime"]["output_root"] = str(tmp_path / "outputs")
    cfg["generation"]["num_documents"] = 2
    cfg["data"]["train_limit"] = 2
    cfg["data"]["eval_limit"] = 3
    cfg["data"]["initialization_limit"] = 4
    cfg["data"]["max_samples_per_client"] = 5

    monkeypatch.setattr(
        "dp_prompt.runners.pretext_pipeline._load_pretext_texts",
        lambda config: (["private text 1", "private text 2"], ["eval a", "eval b", "eval c"]),
    )

    class DummyBackend:
        def __init__(self):
            self.released = False

        def generate_batch(self, prompts, max_new_tokens, temperature):
            return [f"synthetic::{index}" for index, _ in enumerate(prompts)]

        def release(self):
            self.released = True

    backend = DummyBackend()
    monkeypatch.setattr(
        "dp_prompt.runners.pretext_pipeline._build_vllm_backend",
        lambda config: backend,
    )

    captured = {}

    def fake_build_thesis_eval_config(**kwargs):
        captured["build_thesis_eval_config"] = kwargs
        return {"kind": "thesis-config"}

    def fake_export_and_run_small_eval(thesis_eval_config, *, synthetic_texts, stage2_dir, eval_dir):
        stage2_dir.mkdir(parents=True, exist_ok=True)
        (stage2_dir / "llama7b_text_syn.json").write_text(
            json.dumps(synthetic_texts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        captured["synthetic_texts"] = list(synthetic_texts)
        return {"metrics": {"best_top1": 0.5, "best_top3": 0.7}}

    monkeypatch.setattr(
        "dp_prompt.runners.pretext_pipeline.build_thesis_eval_config",
        fake_build_thesis_eval_config,
    )
    monkeypatch.setattr(
        "dp_prompt.runners.pretext_pipeline.export_and_run_small_eval",
        fake_export_and_run_small_eval,
    )

    summary = run_pretext_style_pipeline(cfg)

    experiment_dir = Path(cfg["runtime"]["output_root"]) / cfg["experiment"]["id"]
    assert (experiment_dir / "sanitized_corpus.json").exists()
    assert (experiment_dir / "stage2" / "llama7b_text_syn.json").exists()
    assert (experiment_dir / "eval" / "eval_small_summary.json").exists()
    assert summary["synthetic_count"] == 2
    assert captured["synthetic_texts"] == ["synthetic::0", "synthetic::1"]
    assert captured["build_thesis_eval_config"]["train_limit"] == 2
    assert captured["build_thesis_eval_config"]["eval_limit"] == 3
    assert captured["build_thesis_eval_config"]["initialization_limit"] == 4
    assert captured["build_thesis_eval_config"]["max_samples_per_client"] == 5
    assert backend.released is True
