from paper_new_selector.external_baselines.dpga_adapter import (
    build_dpga_stage1_summary,
)


def test_build_dpga_stage1_summary_marks_direct_synthetic_outputs():
    summary = build_dpga_stage1_summary(
        texts=[f"sample {i} with enough words" for i in range(100)],
        budget=100,
    )
    assert summary["mode"] == "dpga_adapter"
    assert summary["skip_bootstrap"] is True
    assert len(summary["direct_synthetic_texts"]) == 100

