"""Validator-valid synthetic policy worlds for robust, adversarial model selection.

PROXY EVIDENCE ONLY. These are stylised generators, not the private generator.
Their sole purpose is to let us *select* a robust model across plausible policy
worlds instead of fitting the failed-audit 10-item preview.

Every world respects the exact invariant a production window satisfies: one fixed
context template shared by both classes, so the phase|pressure signature multiset
and the position-group marginal are class-matched by construction (they pass
``poker44.validator.evaluation.redteam_gate.audit_redteam_leakage``). Class signal
is injected ONLY through the legitimate response channel -- ``action_type`` /
``size_bucket`` / ``is_all_in``.

Worlds:
  aligned      bots aggressive + mechanical + coarse; humans passive + varied
               (production-orientation world -- every channel agrees).
  mechanical   action *level* class-matched; bots repeat one action (low entropy)
               while humans vary -- isolates the consistency channel.
  coarse       actions class-matched exactly; bots size extreme/big, humans
               moderate/small -- isolates the sizing/extremity channel.
  reversal     nit-bot: bots passive+small but still mechanical; humans loose and
               aggressive -- the aggression backbone is reversed.
  adversarial  worst case: bots disguised as varied-moderate-passive humans while
               humans play like textbook bots -- every v6 channel is reversed.
  null         class-independent policy -- no signal at all.
"""

from __future__ import annotations

import random
from collections.abc import Callable

__all__ = ["WORLD_NAMES", "build_world", "context_template"]

NA = "not_applicable"

# One fixed, class-matched context template (mixed phases, >=1 postflop).
_CONTEXT: tuple[tuple[str, str, str], ...] = (
    ("preflop", "early", "facing_bet"),
    ("flop", "late", "no_call"),
    ("turn", "blinds", "facing_bet"),
    ("river", "late", "no_call"),
)


def context_template() -> tuple[tuple[str, str, str], ...]:
    return _CONTEXT


def _resp(action: str, size: str | None = None) -> tuple[str, str, bool]:
    """A schema-consistent (action, size, is_all_in) response."""
    if action in ("fold", "check", "call"):
        return (action, NA, False)
    if action == "all_in":
        return ("all_in", "all_in", True)
    size = size or "half_pot"
    return (action, size, size == "all_in")


def _item(item_id: str, responses: list[tuple[str, str, bool]]) -> dict:
    decisions = []
    for i, (phase, position, pressure) in enumerate(_CONTEXT):
        action, size, all_in = responses[i]
        decisions.append(
            {
                "decision_number": i + 1,
                "phase": phase,
                "position_group": position,
                "pressure": pressure,
                "action_type": action,
                "size_bucket": size,
                "is_all_in": bool(all_in),
            }
        )
    return {
        "schema_version": "4.1",
        "item_id": item_id,
        "window_id": "synthetic",
        "decisions": decisions,
    }


# --- class-conditional policies (label 1 == bot, 0 == human) -----------------
def _aligned(rng: random.Random, label: int) -> list[tuple[str, str, bool]]:
    if label == 1:
        action = rng.choice(["raise", "bet"])
        return [_resp(action, rng.choice(["pot", "overbet"])) for _ in range(4)]
    actions = rng.sample(["fold", "check", "call", "bet"], 4)
    return [_resp(a, rng.choice(["third_pot_or_less", "half_pot"])) for a in actions]


def _mechanical(rng: random.Random, label: int) -> list[tuple[str, str, bool]]:
    # Symmetric-risk alphabet so "repeat one" and "spread over all" have the same
    # expected action level -> the level channel is matched; only entropy differs.
    alphabet = ["check", "call", "bet", "raise"]
    if label == 1:
        action = rng.choice(alphabet)
        return [_resp(action, "half_pot") for _ in range(4)]
    actions = rng.sample(alphabet, 4)
    return [_resp(a, "half_pot") for a in actions]


def _coarse(rng: random.Random, label: int) -> list[tuple[str, str, bool]]:
    actions = ["raise", "bet", "raise", "bet"]  # identical actions -> level matched
    if label == 1:
        sizes = [rng.choice(["overbet", "all_in", "pot"]) for _ in range(4)]
    else:
        sizes = [rng.choice(["third_pot_or_less", "half_pot", "three_quarter_pot"]) for _ in range(4)]
    return [_resp(a, s) for a, s in zip(actions, sizes)]


def _reversal(rng: random.Random, label: int) -> list[tuple[str, str, bool]]:
    if label == 1:  # nit-bot: passive, small, but mechanical (repeated)
        action = rng.choice(["fold", "check", "call"])
        return [_resp(action) for _ in range(4)]
    actions = rng.sample(["bet", "raise", "call", "check"], 4)  # loose, aggressive, varied
    return [_resp(a, rng.choice(["half_pot", "pot"])) for a in actions]


def _adversarial(rng: random.Random, label: int) -> list[tuple[str, str, bool]]:
    if label == 1:  # bot disguised as a varied, moderate, passive human
        actions = rng.sample(["fold", "check", "call", "bet"], 4)
        return [_resp(a, "half_pot") for a in actions]
    action = rng.choice(["raise", "bet"])  # human plays like a textbook bot
    return [_resp(action, rng.choice(["overbet", "all_in"])) for _ in range(4)]


def _null(rng: random.Random, label: int) -> list[tuple[str, str, bool]]:
    actions = rng.sample(["fold", "check", "call", "raise"], 4)  # ignores label
    return [_resp(a, "half_pot") for a in actions]


_POLICIES: dict[str, Callable[[random.Random, int], list[tuple[str, str, bool]]]] = {
    "aligned": _aligned,
    "mechanical": _mechanical,
    "coarse": _coarse,
    "reversal": _reversal,
    "adversarial": _adversarial,
    "null": _null,
}
WORLD_NAMES: tuple[str, ...] = tuple(_POLICIES)


def build_world(name: str, n_per_class: int, seed: int = 44) -> tuple[list[dict], list[int]]:
    """Return (items, labels): a class-balanced, validator-valid synthetic window."""
    if name not in _POLICIES:
        raise KeyError(f"unknown world: {name!r}; choose from {WORLD_NAMES}")
    policy = _POLICIES[name]
    # Deterministic integer seed derived from name + seed (no hash-randomisation).
    rng = random.Random(1000 * int(seed) + sum(ord(c) for c in name))
    items: list[dict] = []
    labels: list[int] = []
    for k in range(int(n_per_class)):
        for label in (1, 0):
            responses = policy(rng, label)
            items.append(_item(f"{name}-{label}-{k}", responses))
            labels.append(label)
    return items, labels
