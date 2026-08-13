"""Robust, adversarial model-selection evidence for v6 (PROXY, not ground truth).

The 10-item public preview failed its quality audit (single tournament; it even
fails the validator red-team gate on position). So instead of fitting those
labels we evaluate v6 across a battery of *validator-valid* synthetic policy
worlds -- phase|pressure multisets and position marginals class-matched exactly
like a production window, with class signal injected ONLY through the legitimate
action/size/all-in channel. Every number here is proxy evidence over stylised
worlds, clearly labelled as such; none of it is a claim about the private
generator.
"""

from __future__ import annotations

import numpy as np

from poker44.validator.evaluation.redteam_gate import audit_redteam_leakage
from poker44.validator.evaluation.reward import reward
from poker44_champion_v6 import scoring, worlds

N_PER_CLASS = 60
SEED = 44

SIGNAL_WORLDS = ("aligned", "mechanical", "coarse")
ALL_WORLDS = worlds.WORLD_NAMES


def _balanced_weights(labels):
    labels = np.asarray(labels, dtype=int)
    w = np.ones(labels.shape, dtype=float)
    for c in (0, 1):
        mask = labels == c
        n = int(mask.sum())
        if n:
            w[mask] = 0.5 / n
    return w.tolist()


def _reward(scores, labels):
    return reward(list(map(float, scores)), list(map(int, labels)), sample_weights=_balanced_weights(labels))


def _v6_scores(items):
    return [scoring.score_session(it) for it in items]


def _raw_aggression(items):
    out = []
    agg = {"bet", "raise", "all_in"}
    for it in items:
        ds = it["decisions"]
        out.append(sum(d["action_type"] in agg for d in ds) / max(1, len(ds)))
    return out


def test_world_registry_is_complete():
    assert set(ALL_WORLDS) == {"aligned", "mechanical", "coarse", "reversal", "adversarial", "null"}


def test_every_world_is_validator_valid_and_class_balanced():
    for name in ALL_WORLDS:
        items, labels = worlds.build_world(name, N_PER_CLASS, SEED)
        assert len(items) == len(labels) == 2 * N_PER_CLASS
        assert labels.count(0) == labels.count(1) == N_PER_CLASS
        for it in items:
            assert it["schema_version"] == "4.1"
            assert len(it["decisions"]) == 4
        gate = audit_redteam_leakage(items, labels)
        # The window respects the exact invariant the validator enforces: matched
        # phase|pressure signatures + position, signal only in the response channel.
        assert gate.passed, (name, gate.reason, gate.to_dict())


def test_world_generation_is_deterministic():
    a_items, a_labels = worlds.build_world("aligned", N_PER_CLASS, SEED)
    b_items, b_labels = worlds.build_world("aligned", N_PER_CLASS, SEED)
    assert a_labels == b_labels
    assert _v6_scores(a_items) == _v6_scores(b_items)


def test_signal_worlds_are_learnable_by_v6():
    # v6 must extract real skill from each legitimate signal channel.
    thresholds = {"aligned": 0.15, "mechanical": 0.05, "coarse": 0.05}
    for name in SIGNAL_WORLDS:
        items, labels = worlds.build_world(name, N_PER_CLASS, SEED)
        r = _reward(_v6_scores(items), labels)
        assert r.reward > thresholds[name], (name, r.to_dict())


def test_null_world_has_no_illusory_skill():
    # A single class-independent window is noisy; the *expected* skill is ~0, so
    # average over seeds to strip finite-sample noise.
    rewards = [
        _reward(_v6_scores(worlds.build_world("null", N_PER_CLASS, s)[0]),
                worlds.build_world("null", N_PER_CLASS, s)[1]).reward
        for s in range(12)
    ]
    assert np.mean(rewards) < 0.05, rewards


def test_reversal_is_rescued_by_the_consistency_channel():
    # Aggression is reversed (nit-bots), so the action-level backbone is worse
    # than useless -- yet v6's orientation-agnostic consistency channel still
    # separates the mechanical bots. This is the core robustness property: a
    # single-orientation aggression bet is floored here; v6 is not.
    items, labels = worlds.build_world("reversal", N_PER_CLASS, SEED)
    v6 = _reward(_v6_scores(items), labels).reward
    agg = _reward(_raw_aggression(items), labels).reward
    assert v6 > 0.15, v6
    assert v6 > agg


def test_adversarial_world_is_floored_not_harmful():
    # The genuine worst case: EVERY v6 channel is reversed (bots disguised as
    # varied/moderate/passive humans; humans play like textbook bots). Per-component
    # flooring plus bounded, centred scores keep this at ~0 -- never a large loss.
    rewards = [
        _reward(_v6_scores(worlds.build_world("adversarial", N_PER_CLASS, s)[0]),
                worlds.build_world("adversarial", N_PER_CLASS, s)[1]).reward
        for s in range(12)
    ]
    assert all(r >= 0.0 for r in rewards)
    assert max(rewards) < 0.06, rewards


def test_v6_dominates_single_orientation_baseline_across_worlds():
    # Robustness claim: adding orientation-agnostic consistency/coarseness channels
    # strictly widens coverage vs a naive action-only aggression bet, without
    # hurting worst-case. v6's mean skill over signal worlds beats raw aggression,
    # and v6 wins the worlds a pure action-level bet is blind to.
    v6_signal, agg_signal = [], []
    for name in SIGNAL_WORLDS:
        items, labels = worlds.build_world(name, N_PER_CLASS, SEED)
        v6_signal.append(_reward(_v6_scores(items), labels).reward)
        agg_signal.append(_reward(_raw_aggression(items), labels).reward)
    assert np.mean(v6_signal) > np.mean(agg_signal)

    for name in ("mechanical", "coarse"):
        items, labels = worlds.build_world(name, N_PER_CLASS, SEED)
        v6 = _reward(_v6_scores(items), labels).reward
        agg = _reward(_raw_aggression(items), labels).reward
        assert v6 > agg + 0.03, (name, v6, agg)


def test_v6_worst_case_over_all_worlds_is_non_negative():
    worst = min(
        _reward(_v6_scores(worlds.build_world(n, N_PER_CLASS, SEED)[0]),
                worlds.build_world(n, N_PER_CLASS, SEED)[1]).reward
        for n in ALL_WORLDS
    )
    assert worst >= 0.0


def test_scores_are_bounded_and_conservative():
    # Magnitude safety: even the most extreme synthetic bot line stays inside a
    # conservative band so a mis-orientation cannot produce a catastrophic Brier.
    items, _ = worlds.build_world("aligned", N_PER_CLASS, SEED)
    scores = _v6_scores(items)
    assert min(scores) >= 0.0 and max(scores) <= 1.0
    assert max(scores) <= scoring.OUTPUT_HI + 1e-12
    assert min(scores) >= scoring.OUTPUT_LO - 1e-12
