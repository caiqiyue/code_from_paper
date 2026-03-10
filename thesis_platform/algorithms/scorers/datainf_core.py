from __future__ import annotations

from thesis_platform.algorithms.math_utils import add, dot, mean_vector, scale, subtract


def compute_datainf_scores(
    train_vectors: list[list[float]],
    val_vectors: list[list[float]],
    *,
    lambda_const_param: float,
) -> list[float]:
    """Compute a light-weight DataInf-style influence score over vectorized samples."""

    if not train_vectors:
        return []
    if not val_vectors:
        val_vectors = train_vectors

    val_avg = mean_vector(val_vectors)  # Treat the validation mean vector as the anchor gradient.
    sq_means = [sum(value * value for value in vector) / max(len(vector), 1) for vector in train_vectors]
    lambda_const = max(sum(sq_means) / max(len(sq_means), 1) / max(lambda_const_param, 1e-6), 1e-6)

    hvp = [0.0] * len(val_avg)
    n_train = float(len(train_vectors))
    for grad in train_vectors:
        coeff = dot(val_avg, grad) / (lambda_const + dot(grad, grad))  # Proposed HVP closed-form coefficient.
        update = scale(subtract(val_avg, scale(grad, coeff)), 1.0 / (n_train * lambda_const))
        hvp = add(hvp, update)

    return [-dot(hvp, grad) for grad in train_vectors]
