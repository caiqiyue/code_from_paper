from paper_new_selector.baseline_modes import build_c4_only_summary


def test_c4_only_uses_public_texts_and_skips_bootstrap():
    summary = build_c4_only_summary(
        init_texts=[f"public text {i} with enough words" for i in range(130)],
        final_budget=100,
        seed=42,
    )

    assert summary["mode"] == "c4_only"
    assert summary["skip_bootstrap"] is True
    assert len(summary["direct_synthetic_texts"]) == 100
    assert summary["selected_texts"] == []
    assert summary["hard_negative_texts"] == []
