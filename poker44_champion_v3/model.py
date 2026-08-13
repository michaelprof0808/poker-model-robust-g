"""V3-compatible Poker44 micro-session detector.

This owner-controlled compatibility model intentionally does not load legacy
hand-chunk artifacts: the live v3 validator sends schema-v4.1 micro-session
items. Until labelled v3 micro-session training releases are public, this model
uses deterministic, rank-stable behavioural features and conservative
calibration designed to avoid hard-zero behaviour from over-flagging humans.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from poker44.miner.config import MinerModelConfig

_ACTION_RISK = {
    "fold": 0.34,
    "check": 0.18,
    "call": 0.27,
    "bet": 0.62,
    "raise": 0.74,
    "all_in": 0.86,
}
_SIZE_RISK = {
    "not_applicable": 0.18,
    "unknown": 0.38,
    "third_pot_or_less": 0.39,
    "half_pot": 0.48,
    "three_quarter_pot": 0.58,
    "pot": 0.68,
    "overbet": 0.82,
    "all_in": 0.90,
}
_PRESSURE_RISK = {"no_call": 0.46, "facing_bet": 0.55}
_POSITION_RISK = {"early": 0.51, "late": 0.46, "blinds": 0.57}
_PHASE_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, float(value)))


def _decision_list(session: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = session.get("decisions")
    if not isinstance(decisions, list):
        return []
    return [d for d in decisions if isinstance(d, dict)]


@dataclass
class ChampionV3MicroSessionModel:
    """Small deterministic model implementing Poker44's v3 model protocol."""

    config: MinerModelConfig

    def __post_init__(self) -> None:
        self.version = self.config.version or "champion-v3-ms-1"

    def load(self) -> None:
        return None

    @staticmethod
    def _score_session(session: dict[str, Any]) -> float:
        decisions = _decision_list(session)
        if len(decisions) != 4:
            # Official service validates before calling the model. This fallback
            # keeps direct calls safe and avoids accidental bot-heavy outputs.
            return 0.25

        actions = Counter(str(d.get("action_type") or "") for d in decisions)
        sizes = Counter(str(d.get("size_bucket") or "") for d in decisions)
        phases = [str(d.get("phase") or "") for d in decisions]

        n = float(len(decisions))
        aggression = (actions["bet"] + actions["raise"] + actions["all_in"]) / n
        passivity = (actions["check"] + actions["call"]) / n
        folds = actions["fold"] / n
        all_in_share = sum(bool(d.get("is_all_in")) for d in decisions) / n
        large_size_share = (
            sizes["pot"] + sizes["overbet"] + sizes["all_in"]
        ) / n
        unknown_size_share = sizes["unknown"] / n
        facing_bet_share = sum(
            str(d.get("pressure") or "") == "facing_bet" for d in decisions
        ) / n
        postflop_aggression = sum(
            str(d.get("phase") or "") in {"flop", "turn", "river"}
            and str(d.get("action_type") or "") in {"bet", "raise", "all_in"}
            for d in decisions
        ) / n

        # Per-decision prior from visible categorical choices.
        categorical = 0.0
        for d in decisions:
            categorical += 0.48 * _ACTION_RISK.get(str(d.get("action_type") or ""), 0.40)
            categorical += 0.24 * _SIZE_RISK.get(str(d.get("size_bucket") or ""), 0.38)
            categorical += 0.14 * _PRESSURE_RISK.get(str(d.get("pressure") or ""), 0.50)
            categorical += 0.14 * _POSITION_RISK.get(str(d.get("position_group") or ""), 0.50)
        categorical /= n

        # Sequence-shape features that survive v3 sanitization.
        phase_nums = [_PHASE_ORDER.get(p, 0) for p in phases]
        monotone_phase = all(b >= a for a, b in zip(phase_nums, phase_nums[1:]))
        repeated_action_penalty = max(actions.values()) / n
        river_pressure = any(
            str(d.get("phase") or "") == "river"
            and str(d.get("pressure") or "") == "facing_bet"
            for d in decisions
        )
        late_large_aggression = any(
            str(d.get("phase") or "") in {"turn", "river"}
            and str(d.get("action_type") or "") in {"raise", "all_in"}
            and str(d.get("size_bucket") or "") in {"pot", "overbet", "all_in"}
            for d in decisions
        )

        raw = (
            0.36 * categorical
            + 0.16 * aggression
            + 0.12 * postflop_aggression
            + 0.10 * large_size_share
            + 0.08 * all_in_share
            + 0.06 * folds
            + 0.05 * facing_bet_share
            + 0.04 * unknown_size_share
            + 0.03 * repeated_action_penalty
        )
        if monotone_phase:
            raw += 0.025
        if river_pressure and actions["fold"] == 0:
            raw += 0.035
        if late_large_aggression:
            raw += 0.075
        if passivity >= 0.75 and all_in_share == 0.0:
            raw -= 0.10

        # Conservative calibration: preserve ranking but keep ordinary sessions
        # below 0.5 unless multiple bot-like signals stack together.
        score = 1.0 / (1.0 + math.exp(-5.2 * (raw - 0.70)))
        return round(_clamp(score, 0.02, 0.98), 6)

    def predict(self, sessions: list[dict[str, Any]]) -> list[float]:
        scores: list[float] = []
        for session in sessions or []:
            try:
                scores.append(self._score_session(session if isinstance(session, dict) else {}))
            except Exception:
                scores.append(0.25)
        return scores


def create_model(config: MinerModelConfig) -> ChampionV3MicroSessionModel:
    return ChampionV3MicroSessionModel(config)
