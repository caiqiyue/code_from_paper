from paper_new_selector.baseline_modes import build_expand_only_summary


def test_expand_only_uses_public_seeds_and_keeps_bootstrap():
    summary = build_expand_only_summary(
        init_texts=[f"public seed {i} with enough words" for i in range(40)],
        seed_top_k=6,
        seed=7,
    )

    assert summary["mode"] == "expand_only"
    assert summary["skip_bootstrap"] is False
    assert len(summary["selected_texts"]) == 6
    assert summary["hard_negative_texts"] == []


def test_expand_only_truncates_public_seed_texts_for_bootstrap():
    summary = build_expand_only_summary(
        init_texts=[" ".join(f"w{i}" for i in range(80))],
        seed_top_k=1,
        seed=7,
        seed_text_max_words=56,
    )

    assert len(summary["selected_texts"][0].split()) == 56
    assert summary["seed_budget"]["seed_text_max_words"] == 56
