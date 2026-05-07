from paper_new_selector.synthetic_contract import resolve_downstream_synthetic_texts


def test_resolve_downstream_synthetic_texts_prefers_direct_outputs():
    stage1_summary = {
        "mode": "c4_only",
        "selected_texts": [],
        "skip_bootstrap": True,
        "direct_synthetic_texts": ["alpha", "beta", "gamma"],
    }

    result = resolve_downstream_synthetic_texts(
        stage1_summary=stage1_summary,
        bootstrap_outputs=["should", "not", "be", "used"],
    )

    assert result == ["alpha", "beta", "gamma"]


def test_resolve_downstream_synthetic_texts_uses_bootstrap_outputs_by_default():
    stage1_summary = {
        "mode": "selector_seed_search",
        "selected_texts": ["seed a", "seed b"],
    }

    result = resolve_downstream_synthetic_texts(
        stage1_summary=stage1_summary,
        bootstrap_outputs=["boot a", "boot b"],
    )

    assert result == ["boot a", "boot b"]
