from __future__ import annotations

from collections.abc import Sequence


def exact_match(prediction: str, target: str) -> float:
    return float(prediction.strip() == target.strip())


def retrieval_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        return 0.0
    hits = sum(int(pred.strip() == tgt.strip()) for pred, tgt in zip(predictions, targets))
    return hits / len(predictions)


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
