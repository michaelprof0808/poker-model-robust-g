"""Leakage-safe feature extraction for Poker44 micro-sessions.

Only the four miner-visible strategic decisions are used. Transport metadata
(``item_id``, ``window_id``, ``schema_version``), the release identity, the
raw array position of an item, and ``decision_number`` as a numeric value are
NEVER used as features. ``decision_number`` is used exclusively to reconstruct
the play sequence for transition/repetition features, never as a value.

The feature vocabulary is derived purely from the schema enums, so the feature
dimension is fixed and independent of any particular dataset. This keeps the
extractor deterministic and reproducible across releases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from poker44_champion_v31 import FEATURE_VERSION

__all__ = ["FEATURE_NAMES", "FEATURE_VERSION", "extract_features", "vectorize"]

# Fixed enum orderings taken from contracts/subject-session.v4.1.schema.json.
PHASES = ("preflop", "flop", "turn", "river")
POSITIONS = ("early", "late", "blinds")
PRESSURES = ("facing_bet", "no_call")
ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
SIZES = (
    "not_applicable",
    "unknown",
    "third_pot_or_less",
    "half_pot",
    "three_quarter_pot",
    "pot",
    "overbet",
    "all_in",
)
POSTFLOP = frozenset({"flop", "turn", "river"})
AGGRESSIVE = frozenset({"bet", "raise", "all_in"})
PASSIVE = frozenset({"check", "call"})

# Coarse aggression class used for transition/repetition features.
_AGG_CLASSES = ("F", "P", "A")  # fold, passive, aggressive


def _agg_class(action: str) -> str:
    if action in AGGRESSIVE:
        return "A"
    if action in PASSIVE:
        return "P"
    return "F"  # fold (or anything unexpected) -> fold class


def _ordered_decisions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("decisions")
    decisions = [d for d in raw if isinstance(d, Mapping)] if isinstance(raw, Sequence) else []
    # Order by decision_number for the play sequence. decision_number is used
    # ONLY for ordering here, never emitted as a feature value. Ties fall back
    # to the stable input order.
    indexed = list(enumerate(decisions))

    def sort_key(pair: tuple[int, Mapping[str, Any]]) -> tuple[float, int]:
        index, decision = pair
        number = decision.get("decision_number")
        rank = float(number) if isinstance(number, (int, float)) else float(index)
        return (rank, index)

    return [dict(decision) for _, decision in sorted(indexed, key=sort_key)]


def extract_features(payload: Mapping[str, Any]) -> dict[str, float]:
    """Return the ordered feature dict for one micro-session payload."""
    decisions = _ordered_decisions(payload)
    n = max(1, len(decisions))

    actions = [str(d.get("action_type") or "") for d in decisions]
    phases = [str(d.get("phase") or "") for d in decisions]
    positions = [str(d.get("position_group") or "") for d in decisions]
    pressures = [str(d.get("pressure") or "") for d in decisions]
    sizes = [str(d.get("size_bucket") or "") for d in decisions]
    all_ins = [bool(d.get("is_all_in")) for d in decisions]
    agg_seq = [_agg_class(a) for a in actions]

    feats: dict[str, float] = {}

    # --- Main effects (fraction over decisions) ---
    for p in PHASES:
        feats[f"phase_frac__{p}"] = phases.count(p) / n
    for p in POSITIONS:
        feats[f"pos_frac__{p}"] = positions.count(p) / n
    for p in PRESSURES:
        feats[f"pressure_frac__{p}"] = pressures.count(p) / n
    for a in ACTIONS:
        feats[f"action_frac__{a}"] = actions.count(a) / n
    for s in SIZES:
        feats[f"size_frac__{s}"] = sizes.count(s) / n
    feats["allin_frac"] = sum(all_ins) / n

    # --- Aggregate derived main effects ---
    feats["aggression_frac"] = sum(a in AGGRESSIVE for a in actions) / n
    feats["passive_frac"] = sum(a in PASSIVE for a in actions) / n
    feats["postflop_frac"] = sum(p in POSTFLOP for p in phases) / n
    feats["sized_bet_frac"] = (
        sum(s not in {"not_applicable", "unknown"} for s in sizes) / n
    )
    feats["big_bet_frac"] = sum(s in {"pot", "overbet", "all_in"} for s in sizes) / n

    # --- Interactions (fraction over decisions) ---
    for pr in PRESSURES:
        for a in ACTIONS:
            feats[f"px__{pr}__{a}"] = (
                sum(1 for i in range(len(decisions)) if pressures[i] == pr and actions[i] == a)
                / n
            )
    for ph in PHASES:
        for a in ACTIONS:
            feats[f"ph__{ph}__{a}"] = (
                sum(1 for i in range(len(decisions)) if phases[i] == ph and actions[i] == a)
                / n
            )
    for po in POSITIONS:
        for a in ACTIONS:
            feats[f"po__{po}__{a}"] = (
                sum(1 for i in range(len(decisions)) if positions[i] == po and actions[i] == a)
                / n
            )

    # --- Transitions over the ordered aggression-class sequence ---
    pairs = list(pairwise(agg_seq))
    denom = max(1, len(pairs))
    for c in _AGG_CLASSES:
        for d in _AGG_CLASSES:
            feats[f"tr__{c}__{d}"] = pairs.count((c, d)) / denom
    escalate = {("P", "A"), ("F", "P"), ("F", "A")}
    deescalate = {("A", "P"), ("A", "F"), ("P", "F")}
    feats["escalation_frac"] = sum(p in escalate for p in pairs) / denom
    feats["deescalation_frac"] = sum(p in deescalate for p in pairs) / denom

    # --- Repetition / diversity features ---
    feats["distinct_action_frac"] = len(set(actions)) / n
    feats["distinct_size_frac"] = len(set(sizes)) / n
    feats["distinct_position_frac"] = len(set(positions)) / n
    feats["repeated_action_frac"] = (n - len(set(actions))) / n

    longest = 1 if actions else 0
    run = 1
    for i in range(1, len(actions)):
        run = run + 1 if actions[i] == actions[i - 1] else 1
        longest = max(longest, run)
    feats["max_action_run_frac"] = longest / n

    feats["same_position_all"] = 1.0 if len(set(positions)) == 1 and positions else 0.0
    monotone = all(
        _AGG_CLASSES.index(agg_seq[i]) <= _AGG_CLASSES.index(agg_seq[i + 1])
        for i in range(len(agg_seq) - 1)
    )
    feats["monotone_aggression"] = 1.0 if monotone else 0.0

    return feats


# A canonical payload used only to freeze the feature ordering at import time.
_TEMPLATE_PAYLOAD = {
    "decisions": [
        {
            "decision_number": i + 1,
            "phase": PHASES[i],
            "position_group": "late",
            "pressure": "no_call",
            "action_type": "check",
            "size_bucket": "not_applicable",
            "is_all_in": False,
        }
        for i in range(4)
    ]
}

FEATURE_NAMES: tuple[str, ...] = tuple(extract_features(_TEMPLATE_PAYLOAD).keys())


def vectorize(payloads: Sequence[Mapping[str, Any]]):
    """Return an ``(n_items, n_features)`` float array aligned with FEATURE_NAMES."""
    import numpy as np

    matrix = np.zeros((len(payloads), len(FEATURE_NAMES)), dtype=np.float64)
    for row, payload in enumerate(payloads):
        feats = extract_features(payload)
        for col, name in enumerate(FEATURE_NAMES):
            matrix[row, col] = feats[name]
    return matrix
