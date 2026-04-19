from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from thesis_platform.core.context import ClientContext
from thesis_platform.core.schemas import Sample


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _append_bias(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.ones((features.shape[0], 1), dtype=np.float64)], axis=1)


def _response_text(sample: Sample) -> str:
    return sample.response or sample.render_text()


def _instruction_text(sample: Sample) -> str:
    return sample.instruction or ""


def _embed_texts(embedder: Any, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float64)
    return np.asarray(embedder.embed_texts(texts), dtype=np.float64)


def _pair_alignment_features(samples: list[Sample], embedder: Any) -> np.ndarray:
    instructions = [_instruction_text(sample) for sample in samples]
    responses = [_response_text(sample) for sample in samples]
    instruction_vectors = _embed_texts(embedder, instructions)
    response_vectors = _embed_texts(embedder, responses)
    if instruction_vectors.size == 0 or response_vectors.size == 0:
        return np.zeros((0, 1), dtype=np.float64)
    abs_delta = np.abs(instruction_vectors - response_vectors)
    return np.concatenate([instruction_vectors, response_vectors, abs_delta], axis=1)


def sample_features(samples: list[Sample], embedder: Any, *, objective: str) -> np.ndarray:
    """Embed samples into the feature space expected by the probe objective."""

    objective = objective.lower()
    if objective == "pair_alignment":
        return _pair_alignment_features(samples, embedder)
    return _embed_texts(embedder, [sample.rendered_text() for sample in samples])


def _synthetic_negative_samples(samples: list[Sample]) -> list[Sample]:
    negatives: list[Sample] = []
    if len(samples) < 2:
        return negatives
    responses = [sample.response for sample in samples]
    shifted = responses[1:] + responses[:1]
    for sample, response in zip(samples, shifted):
        negatives.append(
            Sample(
                sample_id=f"{sample.sample_id}_neg",
                client_id=sample.client_id,
                round_id=sample.round_id,
                source=sample.source,
                dataset_name=sample.dataset_name,
                task_type=sample.task_type,
                text=f"Instruction: {sample.instruction or ''}\nResponse: {response or ''}".strip(),
                instruction=sample.instruction,
                response=response,
                meta=dict(sample.meta),
            )
        )
    return negatives


def _collect_probe_sets(client_ctx: ClientContext, objective: str) -> tuple[list[Sample], list[int], list[Sample], list[int]]:
    """Assemble train/validation sample sets and binary labels for one client."""

    objective = objective.lower()
    positives_train = list(client_ctx.train_samples)
    positives_val = list(client_ctx.validation_samples or client_ctx.train_samples)
    if objective == "pair_alignment":
        positives_train = [sample for sample in positives_train if sample.instruction and sample.response]
        positives_val = [sample for sample in positives_val if sample.instruction and sample.response]
        negatives_train = _synthetic_negative_samples(positives_train)
        negatives_val = _synthetic_negative_samples(positives_val)
    else:
        negatives_train = list(client_ctx.negative_samples)
        negatives_val = list(client_ctx.negative_samples)
        # When negative_samples is empty (single-node case), fall back to synthetic negatives
        if not negatives_train and positives_train:
            negatives_train = _synthetic_negative_samples(positives_train)
        if not negatives_val and positives_val:
            negatives_val = _synthetic_negative_samples(positives_val)
        negatives_train = negatives_train[: max(1, len(positives_train))]
        negatives_val = negatives_val[: max(1, len(positives_val))]

    train_samples = positives_train + negatives_train
    train_labels = [1] * len(positives_train) + [0] * len(negatives_train)
    val_samples = positives_val + negatives_val
    val_labels = [1] * len(positives_val) + [0] * len(negatives_val)
    return train_samples, train_labels, val_samples, val_labels


def _binary_cross_entropy(features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    logits = features @ weights
    probs = np.clip(_sigmoid(logits), 1e-6, 1.0 - 1e-6)
    losses = -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    return float(np.mean(losses))


def _fit_binary_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    epochs: int,
    lr: float,
    damping: float,
) -> np.ndarray:
    weights = np.zeros(train_features.shape[1], dtype=np.float64)
    for _ in range(max(1, epochs)):
        probs = _sigmoid(train_features @ weights)
        grad = (train_features.T @ (probs - train_labels)) / max(len(train_labels), 1)
        grad[:-1] += damping * weights[:-1]
        weights -= lr * grad
    return weights


def sample_gradient(feature: np.ndarray, label: float, weights: np.ndarray) -> np.ndarray:
    """Compute the per-example last-layer gradient for one binary probe sample."""

    prob = float(_sigmoid(feature @ weights))
    return feature * (prob - label)


def hessian_inverse(features: np.ndarray, weights: np.ndarray, *, damping: float) -> np.ndarray:
    """Compute the damped inverse Hessian of the binary probe."""

    probs = _sigmoid(features @ weights)
    diag = probs * (1.0 - probs)
    weighted = features * diag[:, None]
    hessian = (features.T @ weighted) / max(features.shape[0], 1)
    hessian += np.eye(hessian.shape[0], dtype=np.float64) * damping
    return np.linalg.pinv(hessian)


@dataclass(slots=True)
class ProbeBundle:
    """Cached client-local probe state reused by research-mode scorers."""

    objective: str
    weights: np.ndarray
    train_features: np.ndarray
    train_labels: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray
    val_gradient: np.ndarray
    positive_reference_gradient: np.ndarray
    h_inv: np.ndarray
    val_loss_before: float
    learning_rate: float
    embedder_name: str

    def simulate_update_loss(self, synthetic_features: np.ndarray) -> float:
        if synthetic_features.size == 0:
            return self.val_loss_before
        mean_feature = np.mean(synthetic_features, axis=0)
        update = sample_gradient(mean_feature, 1.0, self.weights)
        updated_weights = self.weights - self.learning_rate * update
        return _binary_cross_entropy(self.val_features, self.val_labels, updated_weights)


def build_probe_bundle(
    client_ctx: ClientContext,
    *,
    objective: str,
    probe_epochs: int,
    probe_lr: float,
    damping: float,
) -> ProbeBundle:
    """Fit a client-local binary probe and cache all matrices needed by scorers."""

    train_samples, train_labels, val_samples, val_labels = _collect_probe_sets(client_ctx, objective)
    if not train_samples:
        raise ValueError(f"Cannot build probe for client {client_ctx.client_id}: empty training sample set.")

    train_features = sample_features(train_samples, client_ctx.embedder, objective=objective)
    val_features = sample_features(val_samples, client_ctx.embedder, objective=objective)
    if train_features.ndim != 2 or train_features.shape[0] == 0:
        raise ValueError(f"Cannot build probe for client {client_ctx.client_id}: empty feature matrix.")
    train_aug = _append_bias(train_features)
    val_aug = _append_bias(val_features)
    train_y = np.asarray(train_labels, dtype=np.float64)
    val_y = np.asarray(val_labels, dtype=np.float64)

    weights = _fit_binary_probe(train_aug, train_y, epochs=probe_epochs, lr=probe_lr, damping=damping)
    val_grads = np.asarray([sample_gradient(feature, label, weights) for feature, label in zip(val_aug, val_y)], dtype=np.float64)
    positive_mask = val_y > 0.5
    positive_reference = val_grads[positive_mask] if positive_mask.any() else val_grads
    return ProbeBundle(
        objective=objective,
        weights=weights,
        train_features=train_aug,
        train_labels=train_y,
        val_features=val_aug,
        val_labels=val_y,
        val_gradient=np.mean(val_grads, axis=0),
        positive_reference_gradient=np.mean(positive_reference, axis=0),
        h_inv=hessian_inverse(train_aug, weights, damping=damping),
        val_loss_before=_binary_cross_entropy(val_aug, val_y, weights),
        learning_rate=probe_lr,
        embedder_name=getattr(client_ctx.embedder, "backend_name", type(client_ctx.embedder).__name__),
    )
