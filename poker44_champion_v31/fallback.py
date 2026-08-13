"""Independent, dependency-free safe fallback scorer.

Used both as the runtime fallback when a trained artifact is missing, corrupt or
incompatible, and as the promotion baseline that the supervised champion must
beat out-of-release. It relies only on the leakage-safe features, so it is
deterministic, metadata-invariant and item-permutation equivariant. It does not
import sklearn and never touches an artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from poker44_champion_v31 import features

__all__ = ["VERSION", "heuristic_score", "heuristic_scores"]

VERSION = "champion-v31-fallback-1"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def heuristic_score(payload: Mapping[str, Any]) -> float:
    """Return a bot-likelihood in [0, 1] from strategic-line aggression."""
    f = features.extract_features(payload)
    # Mechanical, escalating, big-sizing aggression reads as bot-like; mixed
    # passive lines and folding read as human-like.
    score = (
        0.40 * f["aggression_frac"]
        + 0.20 * f["big_bet_frac"]
        + 0.15 * f["escalation_frac"]
        + 0.15 * f["monotone_aggression"]
        + 0.10 * (1.0 - f["distinct_action_frac"])
        - 0.20 * f["passive_frac"]
    )
    return round(_clamp(0.5 + (score - 0.35)), 6)


def heuristic_scores(payloads: Sequence[Mapping[str, Any]]) -> list[float]:
    return [heuristic_score(p) for p in payloads]
