from paper_new_selector.baseline_modes import build_expand_private_summary


def test_expand_private_uses_private_train_as_seed_source():
    summary = build_expand_private_summary(
        private_texts=[f"private text {i} with enough words" for i in range(30)],
        seed_top_k=6,
        seed=9,
    )

    assert summary["mode"] == "expand_private"
    assert summary["skip_bootstrap"] is False
    assert len(summary["selected_texts"]) == 6
    assert summary["selected_texts"][0].startswith("private text")


def test_expand_private_truncates_private_seed_texts_for_bootstrap():
    summary = build_expand_private_summary(
        private_texts=[" ".join(f"w{i}" for i in range(90))],
        seed_top_k=1,
        seed=9,
        seed_text_max_words=56,
    )

    assert len(summary["selected_texts"][0].split()) == 56
    assert summary["seed_budget"]["seed_text_max_words"] == 56
