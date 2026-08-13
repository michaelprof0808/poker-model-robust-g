"""Pure, deterministic Poker44 v2 session reward."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class RewardResult:
    reward: float
    average_precision: float
    average_precision_skill: float
    recall_at_fpr_05: float
    false_positive_rate: float
    brier_skill: float
    accuracy: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _recall_at_fpr(
    scores: np.ndarray,
    labels: np.ndarray,
    max_fpr: float,
    sample_weights: np.ndarray,
) -> tuple[float, float]:
    positives = float(np.sum(sample_weights[labels == 1]))
    negatives = float(np.sum(sample_weights[labels == 0]))
    if positives <= 0.0 or negatives <= 0.0:
        return 0.0, 1.0
    order = np.argsort(-scores, kind="mergesort")
    tp = 0.0
    fp = 0.0
    best_recall = 0.0
    observed_fpr = 0.0
    cursor = 0
    while cursor < len(order):
        threshold = scores[order[cursor]]
        end = cursor
        while end < len(order) and scores[order[end]] == threshold:
            if labels[order[end]] == 1:
                tp += float(sample_weights[order[end]])
            else:
                fp += float(sample_weights[order[end]])
            end += 1
        fpr = fp / negatives
        if fpr <= max_fpr:
            best_recall = max(best_recall, tp / positives)
            observed_fpr = fpr
        cursor = end
    return float(best_recall), float(observed_fpr)


def reward(
    predictions: list[float],
    labels: list[int],
    *,
    allow_single_class: bool = False,
    sample_weights: list[float] | None = None,
) -> RewardResult:
    scores = np.asarray(predictions, dtype=float)
    truth = np.asarray(labels, dtype=int)
    if scores.shape != truth.shape or scores.size == 0:
        raise ValueError("Predictions and labels must be non-empty and aligned")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Predictions must be finite values in [0, 1]")
    if set(np.unique(truth).tolist()) - {0, 1}:
        raise ValueError("Labels must be binary")
    weights = (
        np.ones(scores.shape, dtype=float)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=float)
    )
    if weights.shape != scores.shape:
        raise ValueError("Sample weights must align with predictions")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Sample weights must be finite positive values")
    weights = weights / float(np.sum(weights))
    if len(np.unique(truth)) < 2:
        if not allow_single_class:
            return RewardResult(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        # Operational bot-only E2E windows cannot measure ranking skill. They
        # can still verify that miners classify the sole observed class. Keep
        # this opt-in and report no AP/recall skill to avoid conflating the two.
        brier = float(np.average(np.square(scores - truth), weights=weights))
        class_score = float(np.clip(1.0 - brier, 0.0, 1.0))
        accuracy = float(
            np.average((scores >= 0.5).astype(int) == truth, weights=weights)
        )
        return RewardResult(
            class_score,
            0.0,
            0.0,
            0.0,
            1.0,
            class_score,
            accuracy,
        )

    average_precision = float(
        average_precision_score(truth, scores, sample_weight=weights)
    )
    prevalence = float(np.average(truth, weights=weights))
    average_precision_skill = max(
        0.0,
        (average_precision - prevalence) / max(1e-12, 1.0 - prevalence),
    )
    recall, observed_fpr = _recall_at_fpr(
        scores, truth, max_fpr=0.05, sample_weights=weights
    )
    brier = float(np.average(np.square(scores - truth), weights=weights))
    baseline_brier = float(
        np.average(np.square(prevalence - truth), weights=weights)
    )
    brier_skill = max(0.0, 1.0 - brier / max(1e-12, baseline_brier))
    accuracy = float(
        np.average((scores >= 0.5).astype(int) == truth, weights=weights)
    )
    final = float(
        np.clip(
            0.50 * average_precision_skill + 0.30 * recall + 0.20 * brier_skill,
            0.0,
            1.0,
        )
    )
    return RewardResult(
        reward=final,
        average_precision=average_precision,
        average_precision_skill=average_precision_skill,
        recall_at_fpr_05=recall,
        false_positive_rate=observed_fpr,
        brier_skill=brier_skill,
        accuracy=accuracy,
    )
