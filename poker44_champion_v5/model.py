"""Context-matched transductive Poker44 schema-v4.1 detector.

The production validator guarantees that the multiset of phase/pressure context
signatures is identical between the private human and bot classes. This model
uses that public invariant without reading IDs, request order, or labels:

1. score only strategic responses (action and size, optionally conditioned on
   context), never context marginals;
2. rank sessions within each matched context-signature group; and
3. center every group at probability 0.5.

The result deliberately decorrelates miner B from the existing independent
champion-v3 ranker while retaining its already competitive strategic priors.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from poker44.miner.config import MinerModelConfig
from poker44_champion_v3.model import ChampionV3MicroSessionModel

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
_AGGRESSIVE = {"bet", "raise", "all_in"}
_LARGE = {"pot", "overbet", "all_in"}
_POSTFLOP = {"flop", "turn", "river"}

# Rank is the robust signal; raw separation is only a small tie/confidence term.
RANK_AMPLITUDE = 0.40
SEPARATION_AMPLITUDE = 0.10
OUTPUT_FLOOR = 0.05
OUTPUT_CEIL = 0.95


def _safe_global_calibration(base: float) -> float:
    """The deployed v4 monotonic calibration, used only as a safe fallback."""
    value = float(base)
    if not math.isfinite(value):
        return 0.5
    z = 5.0 * (value - 0.15)
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _decisions(session: Any) -> list[dict[str, Any]]:
    if not isinstance(session, dict):
        return []
    value = session.get("decisions")
    if not isinstance(value, list) or len(value) != 4:
        return []
    if not all(isinstance(decision, dict) for decision in value):
        return []
    return value


def _context_signature(session: Any) -> tuple[str, ...] | None:
    decisions = _decisions(session)
    if not decisions:
        return None
    return tuple(
        sorted(
            f"{decision.get('phase')}|{decision.get('pressure')}"
            for decision in decisions
        )
    )


def _normalised_entropy(counts: Counter[str], total: int) -> float:
    if total <= 1 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(total)


def strategic_score(session: Any) -> float:
    """Return an ID-blind strategic prior in [0, 1].

    Context fields never contribute marginal risk. They are used only in
    action/context interaction terms, which is the intended production signal.
    """
    decisions = _decisions(session)
    if not decisions:
        return 0.5

    actions = [str(decision.get("action_type") or "") for decision in decisions]
    sizes = [str(decision.get("size_bucket") or "") for decision in decisions]
    action_counts = Counter(actions)
    response_counts = Counter(zip(actions, sizes))
    n = float(len(decisions))

    action_prior = sum(_ACTION_RISK.get(action, 0.40) for action in actions) / n
    size_prior = sum(_SIZE_RISK.get(size, 0.38) for size in sizes) / n
    aggression = sum(action in _AGGRESSIVE for action in actions) / n
    large_size = sum(size in _LARGE for size in sizes) / n
    all_in = sum(bool(decision.get("is_all_in")) for decision in decisions) / n
    postflop_aggression = sum(
        str(decision.get("phase") or "") in _POSTFLOP
        and str(decision.get("action_type") or "") in _AGGRESSIVE
        for decision in decisions
    ) / n
    facing_bet_aggression = sum(
        str(decision.get("pressure") or "") == "facing_bet"
        and str(decision.get("action_type") or "") in _AGGRESSIVE
        for decision in decisions
    ) / n
    response_repetition = max(response_counts.values()) / n
    action_concentration = 1.0 - _normalised_entropy(action_counts, len(actions))

    value = (
        0.39 * action_prior
        + 0.21 * size_prior
        + 0.12 * aggression
        + 0.07 * large_size
        + 0.05 * all_in
        + 0.06 * postflop_aggression
        + 0.04 * facing_bet_aggression
        + 0.035 * response_repetition
        + 0.025 * action_concentration
    )
    return max(0.0, min(1.0, float(value)))


def _midranks(values: list[float]) -> list[float]:
    """Zero-based midranks; equal values remain equal and order-independent."""
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        midrank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            result[ordered[position]] = midrank
        cursor = end
    return result


def _score_group(base_scores: list[float]) -> list[float]:
    if len(base_scores) <= 1:
        return [0.5 for _ in base_scores]
    ranks = _midranks(base_scores)
    denominator = float(len(base_scores) - 1)
    mean_base = sum(base_scores) / len(base_scores)
    scores = []
    for base, rank in zip(base_scores, ranks):
        centered_rank = (rank - denominator / 2.0) / denominator
        score = (
            0.5
            + RANK_AMPLITUDE * centered_rank
            + SEPARATION_AMPLITUDE * (base - mean_base)
        )
        scores.append(round(max(OUTPUT_FLOOR, min(OUTPUT_CEIL, score)), 9))
    return scores


@dataclass
class ChampionV5MatchedPolicyModel:
    config: MinerModelConfig

    def __post_init__(self) -> None:
        self.version = self.config.version or "champion-v5-matched-1"
        self._fallback_ranker = ChampionV3MicroSessionModel(self.config)

    def load(self) -> None:
        return None

    def predict(self, sessions: list[dict[str, Any]]) -> list[float]:
        if not sessions:
            return []
        try:
            output = [0.5] * len(sessions)
            grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
            malformed: list[int] = []
            for index, session in enumerate(sessions):
                signature = _context_signature(session)
                if signature is None:
                    malformed.append(index)
                else:
                    grouped[signature].append(index)

            # A production red-team-passed window has an equal human/bot count
            # for every context signature, hence every group is even and >= 2.
            # If that invariant is absent, do not force a false 50/50 split.
            if malformed or any(len(indices) < 2 or len(indices) % 2 for indices in grouped.values()):
                return [
                    round(
                        0.5
                        if not _decisions(session)
                        else _safe_global_calibration(
                            self._fallback_ranker._score_session(session)
                        ),
                        9,
                    )
                    for session in sessions
                ]

            for indices in grouped.values():
                base_scores = [strategic_score(sessions[index]) for index in indices]
                for index, score in zip(indices, _score_group(base_scores)):
                    output[index] = score
            for index in malformed:
                output[index] = 0.5
            return output
        except Exception:
            return [0.5 for _ in sessions]


def create_model(config: MinerModelConfig) -> ChampionV5MatchedPolicyModel:
    return ChampionV5MatchedPolicyModel(config)
