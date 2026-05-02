from dp_prompt.decoding.privacy import build_privacy_controls_summary
from dp_prompt.prompting.templates import render_document_prompt


def test_render_document_prompt_uses_review_template():
    prompt = render_document_prompt("hello world", template_name="review_paraphrase")
    assert "hello world" in prompt
    assert "Paraphrase" in prompt


def test_build_privacy_controls_summary_contains_reproducible_fields():
    summary = build_privacy_controls_summary(
        {
            "temperature": 1.25,
            "logits_clipping": {"enabled": True, "lower_bound": -3.0, "upper_bound": 3.0},
            "max_generated_tokens": 96,
            "stop_sequences": ["\n\n"],
        }
    )

    assert summary["temperature"] == 1.25
    assert summary["logits_clipping"]["enabled"] is True
    assert summary["max_generated_tokens"] == 96
