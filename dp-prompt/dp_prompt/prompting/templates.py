from __future__ import annotations

TEMPLATE_REGISTRY = {
    "review_paraphrase": "Review: {text}\nParaphrase of the review:",
    "document_paraphrase": "Document: {text}\nParaphrase of the document:",
}


def render_document_prompt(text: str, template_name: str = "review_paraphrase") -> str:
    try:
        template = TEMPLATE_REGISTRY[template_name]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt template: {template_name}") from exc
    return template.format(text=text)
