import pandas as pd

from dp_prompt.attacks.text_attacks import build_attack_views


def test_build_attack_views_matches_static_semantics():
    df = pd.DataFrame(
        [
            {"sample_id": "1", "text": "a", "sanitized_text": "sa", "label": 1, "author_id": "u1", "split": "train"},
            {"sample_id": "2", "text": "b", "sanitized_text": "sb", "label": 0, "author_id": "u2", "split": "validation"},
            {"sample_id": "3", "text": "c", "sanitized_text": "sc", "label": 1, "author_id": "u3", "split": "test"},
        ]
    )

    views = build_attack_views(df, attack_mode="static")

    assert views["train"]["text_field"] == "text"
    assert views["test"]["text_field"] == "sanitized_text"


def test_build_attack_views_matches_adaptive_semantics():
    df = pd.DataFrame(
        [
            {"sample_id": "1", "text": "a", "sanitized_text": "sa", "label": 1, "author_id": "u1", "split": "train"},
            {"sample_id": "2", "text": "b", "sanitized_text": "sb", "label": 0, "author_id": "u2", "split": "validation"},
            {"sample_id": "3", "text": "c", "sanitized_text": "sc", "label": 1, "author_id": "u3", "split": "test"},
        ]
    )

    views = build_attack_views(df, attack_mode="adaptive")

    assert views["train"]["text_field"] == "sanitized_text"
    assert views["validation"]["text_field"] == "sanitized_text"
    assert views["test"]["text_field"] == "sanitized_text"
