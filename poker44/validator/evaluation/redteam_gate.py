"""Validator-local canary for miner-visible capture-shape leakage."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from poker44.validator.evaluation.reward import reward


@dataclass(frozen=True)
class RedTeamGateResult:
    passed: bool
    skipped: bool
    threshold: float
    reward: float
    feature: str | None
    split_value: float | None
    bot_when_below: bool | None
    balanced_accuracy: float
    metrics: dict[str, float]
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _decisions(session: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        decision
        for decision in (session.get("decisions") or [])
        if isinstance(decision, dict)
    ]


def _context_signature(session: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{decision.get('phase')}|{decision.get('pressure')}"
            for decision in _decisions(session)
        )
    )


def _position_distribution(
    sessions: list[dict[str, Any]], labels: list[int], label: int
) -> dict[str, float]:
    counts = Counter(
        str(decision.get("position_group"))
        for session, session_label in zip(sessions, labels)
        if session_label == label
        for decision in _decisions(session)
    )
    total = sum(counts.values())
    return {
        position: counts[position] / total if total else 0.0
        for position in ("early", "late", "blinds")
    }


STRATEGIC_FEATURES: dict[str, Callable[[dict[str, Any]], float]] = {
    "decision_count": lambda session: float(len(_decisions(session))),
    "postflop_decisions": lambda session: float(
        sum(
            1
            for decision in _decisions(session)
            if decision.get("phase") != "preflop"
        )
    ),
    "facing_bet_decisions": lambda session: float(
        sum(
            1
            for decision in _decisions(session)
            if decision.get("pressure") == "facing_bet"
        )
    ),
    "context_variant_count": lambda session: float(
        len(set(_context_signature(session)))
    ),
}


def _balanced_accuracy(predictions: list[int], labels: list[int]) -> float:
    truth = np.asarray(labels, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    bot_recall = float(np.mean(predicted[truth == 1] == 1))
    human_recall = float(np.mean(predicted[truth == 0] == 0))
    return (bot_recall + human_recall) / 2.0


def audit_redteam_leakage(
    sessions: list[dict[str, Any]],
    labels: list[int],
    *,
    threshold: float = 0.15,
) -> RedTeamGateResult:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Red-team threshold must be within [0, 1]")
    if len(sessions) != len(labels) or not sessions:
        raise ValueError("Sessions and labels must be non-empty and aligned")
    if set(labels) - {0, 1}:
        raise ValueError("Labels must be binary")
    if len(set(labels)) < 2:
        return RedTeamGateResult(
            passed=False,
            skipped=True,
            threshold=threshold,
            reward=0.0,
            feature=None,
            split_value=None,
            bot_when_below=None,
            balanced_accuracy=0.5,
            metrics={},
            reason="single_class_window_cannot_audit_leakage",
        )

    schema_versions = {
        str(session.get("schema_version") or "") for session in sessions
    }
    if schema_versions != {"4.1"}:
        return RedTeamGateResult(
            passed=False,
            skipped=False,
            threshold=threshold,
            reward=1.0,
            feature="mixed_schema_versions",
            split_value=None,
            bot_when_below=None,
            balanced_accuracy=1.0,
            metrics={},
            reason="strategic_window_mixes_payload_contracts",
        )
    context_by_class = {
        label: Counter(
            _context_signature(session)
            for session, session_label in zip(sessions, labels)
            if session_label == label
        )
        for label in (0, 1)
    }
    if context_by_class[0] != context_by_class[1]:
        return RedTeamGateResult(
            passed=False,
            skipped=False,
            threshold=threshold,
            reward=1.0,
            feature="strategic_context_signature",
            split_value=None,
            bot_when_below=None,
            balanced_accuracy=1.0,
            metrics={},
            reason="strategic_context_distribution_differs",
        )
    human_positions = _position_distribution(sessions, labels, 0)
    bot_positions = _position_distribution(sessions, labels, 1)
    position_gap = max(
        abs(human_positions[position] - bot_positions[position])
        for position in human_positions
    )
    if position_gap > 0.15:
        return RedTeamGateResult(
            passed=False,
            skipped=False,
            threshold=threshold,
            reward=1.0,
            feature="strategic_position_distribution",
            split_value=0.15,
            bot_when_below=None,
            balanced_accuracy=1.0,
            metrics={"position_distribution_max_gap": position_gap},
            reason="strategic_position_distribution_differs",
        )

    best: tuple[
        float,
        str | None,
        float | None,
        bool | None,
        float,
        dict[str, float],
    ] = (0.0, None, None, None, 0.5, {})
    for feature_name, feature in STRATEGIC_FEATURES.items():
        values = [feature(session) for session in sessions]
        unique = sorted(set(values))
        splits = (
            unique
            if len(unique) == 1
            else [
                (left + right) / 2.0
                for left, right in zip(unique[:-1], unique[1:])
            ]
        )
        for split in splits:
            for bot_when_below in (True, False):
                binary = [
                    int(value <= split if bot_when_below else value > split)
                    for value in values
                ]
                scores = [0.99 if prediction else 0.01 for prediction in binary]
                result = reward(scores, labels)
                balanced_accuracy = _balanced_accuracy(binary, labels)
                candidate = (
                    result.reward,
                    feature_name,
                    split,
                    bot_when_below,
                    balanced_accuracy,
                    result.to_dict(),
                )
                if candidate[0] > best[0]:
                    best = candidate

    redteam_reward, feature, split, bot_when_below, balanced_accuracy, metrics = best
    passed = redteam_reward <= threshold
    return RedTeamGateResult(
        passed=passed,
        skipped=False,
        threshold=threshold,
        reward=redteam_reward,
        feature=feature,
        split_value=split,
        bot_when_below=bot_when_below,
        balanced_accuracy=balanced_accuracy,
        metrics=metrics,
        reason=None if passed else "redteam_reward_above_threshold",
    )
