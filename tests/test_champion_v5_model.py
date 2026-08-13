from __future__ import annotations

from copy import deepcopy

from poker44.miner.config import MinerModelConfig


def _cfg():
    return MinerModelConfig(
        factory="poker44_champion_v5.model:create_model",
        model_path=None,
        device="cpu",
        version="champion-v5-matched-1",
        max_sessions_per_request=256,
    )


def _item(item_id: str, actions: tuple[str, str, str, str], sizes: tuple[str, str, str, str]):
    phases = ("preflop", "flop", "turn", "river")
    pressures = ("no_call", "facing_bet", "no_call", "facing_bet")
    positions = ("early", "late", "blinds", "late")
    return {
        "schema_version": "4.1",
        "item_id": item_id,
        "window_id": "window-test",
        "decisions": [
            {
                "decision_number": index + 1,
                "phase": phases[index],
                "position_group": positions[index],
                "pressure": pressures[index],
                "action_type": actions[index],
                "size_bucket": sizes[index],
                "is_all_in": actions[index] == "all_in" or sizes[index] == "all_in",
            }
            for index in range(4)
        ],
    }


def _passive(item_id: str):
    return _item(
        item_id,
        ("call", "check", "call", "fold"),
        ("not_applicable", "not_applicable", "not_applicable", "not_applicable"),
    )


def _scripted_aggressive(item_id: str):
    return _item(
        item_id,
        ("raise", "bet", "raise", "all_in"),
        ("half_pot", "three_quarter_pot", "pot", "all_in"),
    )


def test_matched_pair_is_centered_and_strategic_order_wins():
    from poker44_champion_v5.model import create_model

    model = create_model(_cfg())
    scores = model.predict([_passive("human-like"), _scripted_aggressive("bot-like")])

    assert scores[0] < 0.5 < scores[1]
    assert scores[0] < 0.30 < 0.70 < scores[1]
    assert abs(sum(scores) / 2.0 - 0.5) < 1e-12


def test_prediction_is_permutation_equivariant():
    from poker44_champion_v5.model import create_model

    model = create_model(_cfg())
    items = [
        _passive("a"),
        _scripted_aggressive("b"),
        _item("c", ("fold", "call", "raise", "fold"), ("not_applicable", "not_applicable", "half_pot", "not_applicable")),
        _item("d", ("raise", "check", "bet", "raise"), ("pot", "not_applicable", "half_pot", "overbet")),
    ]
    original = model.predict(items)
    order = [2, 0, 3, 1]
    shuffled = model.predict([items[index] for index in order])

    assert shuffled == [original[index] for index in order]


def test_ids_and_window_metadata_do_not_affect_scores():
    from poker44_champion_v5.model import create_model

    model = create_model(_cfg())
    items = [_passive("a"), _scripted_aggressive("b")]
    baseline = model.predict(items)
    changed = deepcopy(items)
    changed[0]["item_id"] = "totally-different"
    changed[1]["item_id"] = "another-id"
    for item in changed:
        item["window_id"] = "another-window"

    assert model.predict(changed) == baseline


def test_context_marginals_do_not_create_bot_risk():
    from poker44_champion_v5.model import strategic_score

    # Identical response at every decision makes this a pure context-only change;
    # context may condition a response, but must not add marginal bot risk itself.
    base = _item(
        "a",
        ("raise", "raise", "raise", "raise"),
        ("half_pot", "half_pot", "half_pot", "half_pot"),
    )
    shifted = deepcopy(base)
    for index, decision in enumerate(shifted["decisions"]):
        decision["phase"] = ("preflop", "turn", "flop", "river")[index]
        decision["pressure"] = "no_call" if decision["pressure"] == "facing_bet" else "facing_bet"
        decision["position_group"] = ("blinds", "early", "late", "early")[index]

    assert strategic_score(base) == strategic_score(shifted)


def test_unmatched_or_singleton_groups_fall_back_to_safe_global_ranker():
    from poker44_champion_v5.model import create_model

    model = create_model(_cfg())
    passive = _passive("a")
    aggressive = _scripted_aggressive("b")
    # Change one context signature so each group is a singleton. Production
    # red-team windows cannot do this; the model must degrade to independent v4.
    aggressive["decisions"][1]["pressure"] = "no_call"
    scores = model.predict([passive, aggressive])

    assert scores[1] > scores[0]
    assert scores != [0.5, 0.5]


def test_valid_output_and_safe_malformed_fallback():
    from poker44_champion_v5.model import create_model

    model = create_model(_cfg())
    scores = model.predict([{}, {"decisions": "bad"}, 7])  # type: ignore[list-item]

    assert scores == [0.5, 0.5, 0.5]
    assert all(0.0 <= score <= 1.0 for score in scores)
