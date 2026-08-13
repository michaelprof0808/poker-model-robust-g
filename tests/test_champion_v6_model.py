"""Contract/invariant tests for the Round-7 robust champion (v6).

These tests pin the *structural* guarantees the miner requires, independent of
any tuned magnitudes: finite bounded output, strict metadata AND context
invariance, item-permutation equivariance, malformed whole-request fallback,
fine-grained (tie-free) separation, a sign-fixed orientation guard, determinism
and latency safety. Magnitudes are selected offline against synthetic worlds
(see ``tests/test_champion_v6_worlds.py``), never against the 10-item preview.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

from poker44.miner.config import MinerModelConfig
from poker44.miner.loader import load_model
from poker44.miner.model import BotDetectionModel

REPO = Path(__file__).resolve().parent.parent


def _cfg(version: str = "") -> MinerModelConfig:
    return MinerModelConfig(
        factory="poker44_champion_v6.model:create_model",
        model_path=None,
        device="cpu",
        version=version,
        max_sessions_per_request=256,
    )


def _decision(number, phase, position, pressure, action, size, all_in=None):
    return {
        "decision_number": number,
        "phase": phase,
        "position_group": position,
        "pressure": pressure,
        "action_type": action,
        "size_bucket": size,
        "is_all_in": bool(all_in) if all_in is not None else (action == "all_in" or size == "all_in"),
    }


def _item(item_id, spec, *, phases=None, positions=None, pressures=None, window_id="w"):
    """Build a schema-4.1 item from a list of (action, size) pairs."""
    phases = phases or ("preflop", "flop", "turn", "river")
    positions = positions or ("early", "late", "blinds", "late")
    pressures = pressures or ("no_call", "facing_bet", "no_call", "facing_bet")
    decisions = [
        _decision(i + 1, phases[i], positions[i], pressures[i], action, size)
        for i, (action, size) in enumerate(spec)
    ]
    return {"schema_version": "4.1", "item_id": item_id, "window_id": window_id, "decisions": decisions}


NA = "not_applicable"


def _canonical_bot(item_id="bot"):
    # Mechanical, escalating, coarse/extreme aggression repeated across contexts.
    return _item(item_id, [("raise", "pot"), ("raise", "pot"), ("bet", "overbet"), ("all_in", "all_in")])


def _canonical_human(item_id="hum"):
    # Clearly human-ish: passive, folding, varied, no sizing pressure -- below the
    # neutral reference line.
    return _item(item_id, [("fold", NA), ("check", NA), ("call", NA), ("check", NA)])


def _varied_items():
    specs = [
        [("call", NA), ("check", NA), ("call", NA), ("fold", NA)],
        [("raise", "half_pot"), ("bet", "third_pot_or_less"), ("call", NA), ("raise", "pot")],
        [("raise", "pot"), ("raise", "pot"), ("raise", "pot"), ("bet", "overbet")],
        [("fold", NA), ("call", NA), ("raise", "half_pot"), ("check", NA)],
        [("bet", "third_pot_or_less"), ("raise", "half_pot"), ("all_in", "all_in"), ("call", NA)],
        [("check", NA), ("check", NA), ("bet", "half_pot"), ("call", NA)],
        [("all_in", "all_in"), ("raise", "overbet"), ("bet", "pot"), ("raise", "three_quarter_pot")],
        [("fold", NA), ("fold", NA), ("check", NA), ("call", NA)],
    ]
    return [_item(f"i{k}", s) for k, s in enumerate(specs)]


def _model(version=""):
    model = load_model(_cfg(version))
    return model


def test_conforms_to_bot_detection_protocol_and_versioning():
    from poker44_champion_v6 import scoring

    model = _model()
    assert isinstance(model, BotDetectionModel)
    assert model.version == scoring.VERSION  # default version comes from the module
    assert getattr(model, "mode", None) == "robust"
    # An explicit version override is honoured.
    assert _model("round7-uid-poker").version == "round7-uid-poker"


def test_finite_bounded_on_preview_and_varied():
    model = _model()
    preview_path = REPO / "data" / "benchmark-latest.json"
    preview_items = []
    if preview_path.exists():
        preview = json.loads(preview_path.read_text())
        preview_items = [it["payload"] for it in preview["items"]]
    items = preview_items + _varied_items()
    scores = model.predict(items)
    assert len(scores) == len(items)
    assert all(isinstance(s, float) for s in scores)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_metadata_invariance_ids_and_window():
    model = _model()
    items = [_canonical_human("a"), _canonical_bot("b")]
    base = model.predict(items)
    changed = deepcopy(items)
    changed[0]["item_id"] = "totally-different-id-XXXXXXXX"
    changed[1]["item_id"] = "another"
    for it in changed:
        it["window_id"] = "another-window"
        it["schema_version"] = "4.1"
    assert model.predict(changed) == base


def test_full_context_invariance_phase_pressure_position():
    # v6 must be a pure function of action/size/is_all_in and their chronological
    # order. Rewriting every phase/pressure/position label (the validator-matched
    # context channel) must not move any score.
    model = _model()
    items = _varied_items()
    base = model.predict(items)
    scrambled = deepcopy(items)
    for it in scrambled:
        for j, d in enumerate(it["decisions"]):
            d["phase"] = ("river", "preflop", "river", "flop")[j % 4]
            d["position_group"] = ("blinds", "blinds", "early", "late")[j % 4]
            d["pressure"] = ("facing_bet", "no_call", "facing_bet", "no_call")[j % 4]
    assert model.predict(scrambled) == base


def test_constant_response_makes_context_irrelevant():
    # With a single fixed (action,size) at every decision, context can only be a
    # conditioner; changing it must not add or remove bot risk.
    from poker44_champion_v6 import scoring

    base = _item("a", [("raise", "half_pot")] * 4)
    shifted = deepcopy(base)
    for j, d in enumerate(shifted["decisions"]):
        d["phase"] = ("preflop", "turn", "flop", "river")[j]
        d["pressure"] = "no_call" if d["pressure"] == "facing_bet" else "facing_bet"
        d["position_group"] = ("blinds", "early", "late", "early")[j]
    assert scoring.score_session(base) == scoring.score_session(shifted)


def test_permutation_equivariance():
    model = _model()
    items = _varied_items()
    base = model.predict(items)
    order = [3, 0, 6, 1, 7, 2, 5, 4]
    shuffled = model.predict([items[i] for i in order])
    assert shuffled == [base[i] for i in order]
    # Reversal too (used by the runtime self-check).
    assert model.predict(list(reversed(items))) == list(reversed(base))


def test_malformed_whole_request_fallback_is_neutral_and_never_raises():
    from poker44_champion_v6 import scoring

    model = _model()
    bad = [{}, {"decisions": "bad"}, 7, {"decisions": []}, {"decisions": [1, 2, 3]}]
    scores = model.predict(bad)  # type: ignore[list-item]
    assert scores == [scoring.NEUTRAL] * len(bad)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # A malformed item must not corrupt a valid sibling (per-item isolation).
    mixed = model.predict([_canonical_bot("b"), {"decisions": "bad"}, _canonical_human("h")])  # type: ignore[list-item]
    assert mixed[1] == scoring.NEUTRAL
    assert mixed[0] > 0.5 > mixed[2]
    assert model.predict([]) == []


def test_neutral_fallback_injects_no_signal():
    from poker44_champion_v6 import scoring

    # The neutral score is exactly 0.5 so a fallback item is ranking-neutral and
    # Brier-neutral against a balanced window.
    assert scoring.NEUTRAL == 0.5
    assert scoring.score_session({"decisions": []}) == 0.5


def test_orientation_guard_bot_ranks_above_human():
    from poker44_champion_v6 import scoring

    bot = scoring.score_session(_canonical_bot())
    human = scoring.score_session(_canonical_human())
    assert bot > 0.5 > human
    assert bot > human


def test_monotone_along_bot_evidence_ramp():
    # Making a line progressively more mechanical/aggressive/coarse must never
    # lower the score (sign-fixed orientation guard).
    from poker44_champion_v6 import scoring

    ramp = [
        _item("r0", [("fold", NA), ("check", NA), ("call", NA), ("check", NA)]),
        _item("r1", [("call", NA), ("check", NA), ("raise", "half_pot"), ("call", NA)]),
        _item("r2", [("raise", "half_pot"), ("call", NA), ("raise", "half_pot"), ("bet", "half_pot")]),
        _item("r3", [("raise", "pot"), ("raise", "pot"), ("bet", "pot"), ("raise", "half_pot")]),
        _item("r4", [("raise", "pot"), ("raise", "overbet"), ("bet", "pot"), ("all_in", "all_in")]),
    ]
    scores = [scoring.score_session(it) for it in ramp]
    assert scores == sorted(scores)
    assert scores[-1] > scores[0]


def test_no_coarse_avoidable_ties():
    # Distinct behavioural lines must get distinct scores (protects AP and
    # recall@FPR and the validator tie-diagnostics gate).
    model = _model()
    items = _varied_items()
    scores = model.predict(items)
    assert len(set(scores)) == len(scores)
    # Scores are not quantised to a coarse grid.
    assert len({round(s, 9) for s in scores}) == len(scores)


def test_identical_behaviour_ties_but_context_does_not_break_it():
    # Two items with identical action/size lines *should* tie regardless of ids
    # or context: that is a correct tie, not an avoidable one.
    model = _model()
    a = _item("a", [("raise", "pot")] * 4, phases=("preflop",) * 4)
    b = _item("b", [("raise", "pot")] * 4, phases=("river",) * 4, positions=("blinds",) * 4)
    sa, sb = model.predict([a, b])
    assert sa == sb


def test_determinism():
    model = _model()
    items = _varied_items()
    assert model.predict(items) == model.predict(list(items))


def test_output_is_not_constant():
    model = _model()
    scores = model.predict(_varied_items())
    assert max(scores) - min(scores) > 0.1


def test_latency_safety():
    model = _model()
    items = _varied_items() * 300  # 2400 items
    start = time.perf_counter()
    scores = model.predict(items)
    elapsed = time.perf_counter() - start
    assert len(scores) == len(items)
    assert elapsed < 1.5  # generous headroom for a pure-python per-item scorer


def test_end_to_end_through_miner_inference_service():
    # The real production path: schema/forbidden-key validation then predict.
    import asyncio

    from poker44.miner.service import MinerInferenceService

    cfg = _cfg()
    model = load_model(cfg)
    service = MinerInferenceService(model, cfg)
    items = [_canonical_human("h"), _canonical_bot("b")] + _varied_items()
    scores = asyncio.run(service.predict_micro_sessions(items))
    assert len(scores) == len(items)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[1] > scores[0]  # canonical bot ranks above canonical human
    # End-to-end separation is preserved (no coarse ties introduced by the path).
    assert len(set(scores)) == len(scores)


def test_service_rejects_forbidden_and_wrong_schema_before_scoring():
    # v6 never sees leaked ground-truth; the service guards it. Confirm the guard
    # is active so metadata-invariance is enforced upstream too.
    import asyncio

    from poker44.miner.service import MinerInferenceService

    cfg = _cfg()
    service = MinerInferenceService(load_model(cfg), cfg)
    leaked = _canonical_bot("b")
    leaked["decisions"][0]["label"] = 1  # forbidden ground-truth field
    try:
        asyncio.run(service.predict_micro_sessions([leaked]))
        raised = False
    except ValueError:
        raised = True
    assert raised
