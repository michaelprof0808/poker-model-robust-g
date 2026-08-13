"""Deterministic transition-period emission allocation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def winner_fraction(burn_fraction: float, funding_fraction: float) -> float:
    fractions = np.asarray([burn_fraction, funding_fraction], dtype=np.float64)
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0.0):
        raise ValueError("emission fractions must be finite and non-negative")
    remaining = 1.0 - float(fractions.sum())
    if remaining <= 0.0:
        raise ValueError("burn and funding fractions must leave a positive winner reward")
    return remaining


def winner_uid(evaluations: Sequence[Any]) -> int | None:
    """Return the highest positive finite score, breaking exact ties by UID."""

    candidates = [
        (float(item.quality_score), int(item.uid))
        for item in evaluations
        if np.isfinite(float(item.quality_score)) and float(item.quality_score) > 0.0
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (-row[0], row[1]))[1]


def ranked_score_rows(evaluations: Sequence[Any]) -> list[dict[str, Any]]:
    """Serialize every miner score and its binary round reward."""

    clean = sorted(
        evaluations,
        key=lambda item: (
            -float(item.quality_score)
            if np.isfinite(float(item.quality_score))
            else 0.0,
            int(item.uid),
        ),
    )
    winner = winner_uid(evaluations)
    return [
        {
            "uid": int(item.uid),
            "hotkey": str(item.hotkey),
            "quality_score": (
                float(item.quality_score)
                if np.isfinite(float(item.quality_score))
                else 0.0
            ),
            "rank": index + 1,
            "round_reward": 1.0 if int(item.uid) == winner else 0.0,
            "is_winner": int(item.uid) == winner,
            "metrics": dict(item.metrics),
            "model_version": item.model_version,
            "error": item.error,
        }
        for index, item in enumerate(clean)
    ]


def emission_scores(
    size: int,
    *,
    winner_uid: int,
    owner_uid: int,
    funding_uid: int,
    burn_fraction: float,
    funding_fraction: float,
) -> np.ndarray:
    """Allocate burn, tournament funding and the remaining winner reward."""

    winner_share = winner_fraction(burn_fraction, funding_fraction)
    targets = {
        "winner": int(winner_uid),
        "owner": int(owner_uid),
        "funding": int(funding_uid),
    }
    for role, uid in targets.items():
        if uid < 0 or uid >= size:
            raise ValueError(f"{role} UID is outside the metagraph")
    if owner_uid == funding_uid:
        raise ValueError("funding hotkey must be different from the subnet owner")

    scores = np.zeros(size, dtype=np.float32)
    scores[owner_uid] += float(burn_fraction)
    scores[funding_uid] += float(funding_fraction)
    scores[winner_uid] += winner_share
    return scores


def weight_rows(prepared_weights) -> list[dict[str, float | int]]:
    """Serialize the exact SDK-processed vector used for chain submission."""

    if prepared_weights is None:
        return []
    uids, weights, _uint_uids, _uint_weights = prepared_weights
    return [
        {"uid": int(uid), "weight": float(weight)}
        for uid, weight in zip(uids, weights)
        if float(weight) > 0.0
    ]
