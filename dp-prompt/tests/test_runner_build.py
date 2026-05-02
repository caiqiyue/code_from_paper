from dp_prompt.generation.backend import LocalTransformersGenerator, build_generation_request
from dp_prompt.runners.document_pipeline import build_pipeline_components


def test_build_generation_request_includes_sampling_controls():
    request = build_generation_request(
        prompt="Document: a\nParaphrase:",
        cfg={
            "temperature": 1.5,
            "max_generated_tokens": 80,
            "stop_sequences": ["\n\n"],
            "logits_clipping": {
                "enabled": True,
                "lower_bound": -2.5,
                "upper_bound": 2.5,
            },
        },
    )

    assert request["temperature"] == 1.5
    assert request["max_new_tokens"] == 80
    assert request["logits_clipping"]["enabled"] is True
    assert request["logits_clipping"]["lower_bound"] == -2.5


def test_build_pipeline_components_exposes_required_sections():
    cfg = {
        "dataset": {"name": "imdb"},
        "model": {"backend": "local_transformers"},
        "privacy": {"temperature": 1.0, "max_generated_tokens": 32},
    }

    components = build_pipeline_components(cfg)

    assert "privacy_controls" in components
    assert components["privacy_controls"]["temperature"] == 1.0


def test_from_config_treats_unset_tokenizer_env_placeholder_as_none():
    generator = LocalTransformersGenerator.from_config(
        {
            "model_path": "/tmp/model",
            "tokenizer_path": "${DP_PROMPT_LOCAL_TOKENIZER_PATH}",
            "batch_size": 2,
        }
    )

    assert generator.tokenizer_path is None
