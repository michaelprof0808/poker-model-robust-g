"""Round-7 robust bot-evidence scorer (v6).

Design goals (see ``CHAMPION_V6_NOTES.md``):

* **Context-agnostic by construction.** Only ``action_type``, ``size_bucket`` and
  ``is_all_in`` and their *chronological* structure feed the score. ``phase``,
  ``pressure`` and ``position_group`` are the exact channels the validator
  red-team gate class-matches, so they carry no legitimate marginal signal; v6
  never reads them. This makes v6 invariant to ids, window, schema AND the whole
  context channel -- it cannot exploit a position leak like the failed-audit
  preview has.

* **Orientation-guarded.** Every feature is oriented so "more bot-like" only ever
  *increases* it, and every weight is non-negative, so the aggregate evidence is
  monotone non-decreasing in each bot-ward direction. No input combination can
  invert the production-validated orientation (mechanical / aggressive / coarse
  play reads as bot).

* **Rank-sharp, magnitude-conservative.** Evidence is mapped through a bounded
  ``tanh`` so the *ranking* is fine-grained and tie-free (protects AP and
  recall@FPR and the validator tie-diagnostics gate) while the *probability* is
  held inside a conservative band around 0.5 (bounds the Brier downside if the
  private orientation ever disagrees). Calibration constants (``SLOPE``,
  ``AMPL``) are selected offline for worst-case robustness across synthetic
  policy worlds -- never fit to the 10-item preview.

The scorer is pure Python (no numpy/sklearn), so per-item latency is microseconds
and there is no code-bearing artifact to trust or overfit.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "VERSION",
    "NEUTRAL",
    "OUTPUT_LO",
    "OUTPUT_HI",
    "SLOPE",
    "AMPL",
    "WEIGHTS",
    "FEATURE_NAMES",
    "extract_features",
    "bot_evidence",
    "score_session",
    "score_sessions",
    "sample_reference_items",
]

VERSION = "champion-v6-robust-1"
NEUTRAL = 0.5

# --- legitimate per-response risk atoms -------------------------------------
# action_type and size_bucket marginals are NOT class-matched by the red-team
# gate (unlike phase|pressure|position), so their levels are legitimate signal.
ACTION_RISK = {
    "fold": 0.35,
    "check": 0.15,
    "call": 0.28,
    "bet": 0.60,
    "raise": 0.70,
    "all_in": 0.92,
}
SIZE_RISK = {
    "not_applicable": 0.15,
    "unknown": 0.30,
    "third_pot_or_less": 0.34,
    "half_pot": 0.50,
    "three_quarter_pot": 0.62,
    "pot": 0.72,
    "overbet": 0.85,
    "all_in": 0.93,
}
_DEFAULT_ACTION_RISK = 0.40
_DEFAULT_SIZE_RISK = 0.35

_AGG = frozenset({"bet", "raise", "all_in"})
_PASSIVE = frozenset({"check", "call"})
_BIG = frozenset({"pot", "overbet", "all_in"})
_EXTREME_SIZE = frozenset({"overbet", "all_in"})

# Calibration band. Because p = 0.5 + AMPL*tanh(SLOPE*evidence) is a MONOTONE map
# of the evidence, SLOPE and AMPL do NOT change the ranking (AP-skill and
# recall@FPR are identical for any positive SLOPE/AMPL) -- they are purely a Brier
# calibration choice. They are frozen conservatively: on a realistic, imperfectly
# separated window (a reversed minority) a lower AMPL keeps a positive Brier skill
# where a bold one collapses it, and stays near 0.5 on class-independent noise.
# Selected on the robust frontier in scripts/miner/round7_diagnostic.py. tanh
# already bounds the score to 0.5 +/- AMPL; the hard clamp is a finite-range guard.
SLOPE = 1.1
AMPL = 0.30
OUTPUT_LO = 0.02
OUTPUT_HI = 0.98

# Fixed, sign-consistent (all >= 0) feature weights. Grouped by the policy-world
# channel each covers. Chosen by domain priors and held constant; only SLOPE/AMPL
# are tuned, and only for robustness. See CHAMPION_V6_NOTES.md.
WEIGHTS: dict[str, float] = {
    # backbone: response levels (production-validated orientation)
    "action_level": 1.00,
    "size_level": 0.70,
    "allin_frac": 0.50,
    # consistency: mechanicalness (orientation-agnostic; catches repetitive bots
    # whatever their aggression level)
    "action_repeat": 0.90,
    "response_repeat": 0.70,
    "action_entropy_low": 0.80,
    # chronological dynamics
    "longest_run": 0.50,
    "net_escalation": 0.50,
    # coarseness / extremity of sizing
    "big_size_frac": 0.70,
    "extreme_frac": 0.90,
}
FEATURE_NAMES: tuple[str, ...] = tuple(WEIGHTS)


def _agg_ordinal(action: str) -> int:
    if action in _AGG:
        return 2
    if action in _PASSIVE:
        return 1
    return 0  # fold / unexpected


def _normalised_entropy(items: list[str]) -> float:
    n = len(items)
    if n <= 1:
        return 0.0
    counts = Counter(items)
    if len(counts) <= 1:
        return 0.0
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    return entropy / math.log(n)


def _ordered_responses(session: Any) -> tuple[list[str], list[str], list[bool]] | None:
    """Return (actions, sizes, all_ins) ordered by decision_number, or None.

    ``decision_number`` is used ONLY to reconstruct the chronological order; its
    numeric value is never emitted as a feature. A boolean decision_number (a
    ``bool`` is an ``int``) is ignored for ordering and falls back to input order.
    """
    if not isinstance(session, Mapping):
        return None
    raw = session.get("decisions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    decisions = [d for d in raw if isinstance(d, Mapping)]
    if not decisions:
        return None

    def sort_key(pair: tuple[int, Mapping[str, Any]]) -> tuple[float, int]:
        index, decision = pair
        number = decision.get("decision_number")
        rank = float(number) if isinstance(number, (int, float)) and not isinstance(number, bool) else float(index)
        return (rank, index)

    ordered = [d for _, d in sorted(enumerate(decisions), key=sort_key)]
    actions = [str(d.get("action_type") or "") for d in ordered]
    sizes = [str(d.get("size_bucket") or "") for d in ordered]
    all_ins = [bool(d.get("is_all_in")) for d in ordered]
    return actions, sizes, all_ins


def extract_features(actions: list[str], sizes: list[str], all_ins: list[bool]) -> dict[str, float]:
    """Pure, context-free features from the chronological response channel."""
    n = len(actions)
    feats: dict[str, float] = {}

    feats["action_level"] = sum(ACTION_RISK.get(a, _DEFAULT_ACTION_RISK) for a in actions) / n
    feats["size_level"] = sum(SIZE_RISK.get(s, _DEFAULT_SIZE_RISK) for s in sizes) / n
    feats["allin_frac"] = sum(1 for b in all_ins if b) / n

    feats["action_repeat"] = (n - len(set(actions))) / n
    responses = Counter(zip(actions, sizes))
    feats["response_repeat"] = max(responses.values()) / n
    feats["action_entropy_low"] = 1.0 - _normalised_entropy(actions)

    longest = run = 1 if actions else 0
    for i in range(1, n):
        run = run + 1 if actions[i] == actions[i - 1] else 1
        longest = max(longest, run)
    feats["longest_run"] = longest / n

    ordinals = [_agg_ordinal(a) for a in actions]
    pairs = list(zip(ordinals, ordinals[1:]))
    denom = max(1, len(pairs))
    escalations = sum(1 for a, b in pairs if b > a) / denom
    deescalations = sum(1 for a, b in pairs if b < a) / denom
    feats["net_escalation"] = escalations - deescalations

    feats["big_size_frac"] = sum(1 for s in sizes if s in _BIG) / n
    feats["extreme_frac"] = sum(
        1 for a, s, b in zip(actions, sizes, all_ins) if b or a == "all_in" or s in _EXTREME_SIZE
    ) / n
    return feats


# Feature centres. Because ``bot_evidence`` subtracts a per-feature centre and
# then sums with fixed weights, the centres add a single GLOBAL constant to every
# item's evidence: they do NOT change the ranking (AP / recall are invariant to
# them) -- they only place where 0.5 falls and thus the Brier scale. We therefore
# centre on a broad, label-free "typical play" reference population (4 i.i.d.
# decisions from a documented neutral prior) so that ordinary human-ish variety
# and normal repetition map near 0.5, and only genuinely extreme mechanical /
# aggressive / coarse lines push confidently above it. This prior is fixed and is
# NOT derived from the preview labels.
_REFERENCE_ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
_REFERENCE_ACTION_WEIGHTS = (0.18, 0.22, 0.22, 0.14, 0.20, 0.04)
_REFERENCE_SIZES = ("third_pot_or_less", "half_pot", "three_quarter_pot", "pot", "overbet", "all_in")
_REFERENCE_SIZE_WEIGHTS = (0.34, 0.32, 0.16, 0.12, 0.04, 0.02)


def _sample_reference_responses(rng: random.Random) -> tuple[list[str], list[str], list[bool]]:
    actions: list[str] = []
    sizes: list[str] = []
    all_ins: list[bool] = []
    for _ in range(4):
        action = rng.choices(_REFERENCE_ACTIONS, _REFERENCE_ACTION_WEIGHTS)[0]
        if action in ("fold", "check", "call"):
            size, all_in = "not_applicable", False
        elif action == "all_in":
            size, all_in = "all_in", True
        else:
            size = rng.choices(_REFERENCE_SIZES, _REFERENCE_SIZE_WEIGHTS)[0]
            all_in = size == "all_in"
        actions.append(action)
        sizes.append(size)
        all_ins.append(all_in)
    return actions, sizes, all_ins


def _reference_centers(samples: int = 8000, seed: int = 20240807) -> dict[str, float]:
    rng = random.Random(seed)
    totals = {name: 0.0 for name in WEIGHTS}
    for _ in range(samples):
        feats = extract_features(*_sample_reference_responses(rng))
        for name in totals:
            totals[name] += feats[name]
    return {name: totals[name] / samples for name in totals}


_CENTERS: dict[str, float] = _reference_centers()

# Fixed context used only to shape schema-valid reference items; v6 ignores it.
_REFERENCE_CONTEXT = (
    ("preflop", "early", "facing_bet"),
    ("flop", "late", "no_call"),
    ("turn", "blinds", "facing_bet"),
    ("river", "late", "no_call"),
)


def sample_reference_items(n: int, seed: int = 1234) -> list[dict[str, Any]]:
    """Return ``n`` schema-4.1 items drawn from the neutral calibration prior.

    By construction their expected evidence is ~0, so the spread of their scores
    around 0.5 is a direct read of over-confidence (used offline to select the
    conservative SLOPE/AMPL); it is not used at inference time.
    """
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for k in range(int(n)):
        actions, sizes, all_ins = _sample_reference_responses(rng)
        decisions = [
            {
                "decision_number": i + 1,
                "phase": _REFERENCE_CONTEXT[i][0],
                "position_group": _REFERENCE_CONTEXT[i][1],
                "pressure": _REFERENCE_CONTEXT[i][2],
                "action_type": actions[i],
                "size_bucket": sizes[i],
                "is_all_in": all_ins[i],
            }
            for i in range(4)
        ]
        items.append({"schema_version": "4.1", "item_id": f"ref-{k}", "window_id": "reference", "decisions": decisions})
    return items


def bot_evidence(session: Any) -> float | None:
    """Return the raw (uncalibrated) bot-evidence scalar, or None if unscoreable."""
    parsed = _ordered_responses(session)
    if parsed is None:
        return None
    feats = extract_features(*parsed)
    return sum(WEIGHTS[k] * (feats[k] - _CENTERS[k]) for k in WEIGHTS)


def score_session(session: Any) -> float:
    """Return a finite bot-probability in [OUTPUT_LO, OUTPUT_HI]; never raises.

    Unscoreable / malformed input maps to the neutral 0.5 so a fallback item is
    ranking- and Brier-neutral and injects no spurious signal.
    """
    try:
        evidence = bot_evidence(session)
        if evidence is None:
            return NEUTRAL
        score = 0.5 + AMPL * math.tanh(SLOPE * evidence)
        if not math.isfinite(score):
            return NEUTRAL
        return min(OUTPUT_HI, max(OUTPUT_LO, score))
    except Exception:  # noqa: BLE001 -- scoring must never crash the miner
        return NEUTRAL


def score_sessions(sessions: Any) -> list[float]:
    if not sessions:
        return []
    return [score_session(session) for session in sessions]
