from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline


def _evaluate_split(model, frame: pd.DataFrame, text_field: str) -> dict[str, Any]:
    labels = frame["label"].tolist()
    predictions = model.predict(frame[text_field].tolist())
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "num_examples": int(len(frame)),
    }


def run_utility_evaluation(dataframe: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    train_df = dataframe[dataframe["split"] == "train"]
    validation_df = dataframe[dataframe["split"] == "validation"]
    test_df = dataframe[dataframe["split"] == "test"]

    text_field = config.get("evaluation", {}).get("utility_text_field", "sanitized_text")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000),
    )
    model.fit(train_df[text_field].tolist(), train_df["label"].tolist())

    return {
        "task": config.get("evaluation", {}).get("task_name", "document_classification"),
        "backend": "sklearn_tfidf_logreg",
        "validation": _evaluate_split(model, validation_df, text_field),
        "test": _evaluate_split(model, test_df, text_field),
    }
