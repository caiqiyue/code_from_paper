from paper_new_selector.external_baselines.wasp_adapter import (
    build_wasp_stage1_summary,
    clamp_text_budget,
)


def test_clamp_text_budget_truncates_to_screening_budget():
    texts = [f"text {i} with enough words" for i in range(120)]
    assert len(clamp_text_budget(texts, budget=100)) == 100


def test_build_wasp_stage1_summary_marks_direct_synthetic_outputs():
    summary = build_wasp_stage1_summary(
        texts=[f"text {i} with enough words" for i in range(100)],
        budget=100,
    )
    assert summary["mode"] == "wasp_adapter"
    assert summary["skip_bootstrap"] is True
    assert len(summary["direct_synthetic_texts"]) == 100

