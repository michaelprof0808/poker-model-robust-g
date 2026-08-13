from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from poker44.validator.settlement.mixin import ValidatorSettlementMixin
from poker44.validator.settlement.weights import (
    emission_scores,
    ranked_score_rows,
    weight_rows,
    winner_uid,
)


def evaluation(uid, score):
    return SimpleNamespace(uid=uid, hotkey=f"hotkey-{uid}", quality_score=score,
                           metrics={}, model_version="test", error=None)


def test_winner_is_highest_positive_score_with_uid_tie_break():
    rows = [evaluation(8, 0.8), evaluation(2, 0.8), evaluation(1, 0.0)]
    assert winner_uid(rows) == 2
    ranked = ranked_score_rows(rows)
    assert [row["uid"] for row in ranked] == [2, 8, 1]
    assert [row["round_reward"] for row in ranked] == [1.0, 0.0, 0.0]


def test_no_positive_finite_score_has_no_winner():
    assert winner_uid([evaluation(1, 0.0), evaluation(2, float("nan"))]) is None


def test_transition_allocation_is_90_burn_5_funding_5_winner():
    scores = emission_scores(
        10,
        winner_uid=2,
        owner_uid=0,
        funding_uid=7,
        burn_fraction=0.90,
        funding_fraction=0.05,
    )
    assert np.isclose(scores[0], 0.90)
    assert np.isclose(scores[7], 0.05)
    assert np.isclose(scores[2], 0.05)
    assert np.isclose(scores.sum(), 1.0)


def test_transition_allocation_combines_funding_and_winner_roles():
    scores = emission_scores(
        10,
        winner_uid=7,
        owner_uid=0,
        funding_uid=7,
        burn_fraction=0.90,
        funding_fraction=0.05,
    )
    assert np.isclose(scores[0], 0.90)
    assert np.isclose(scores[7], 0.10)


@pytest.mark.asyncio
async def test_emission_target_resolves_owner_from_chain_and_funding_from_config():
    settlement = ValidatorSettlementMixin()
    settlement.config = SimpleNamespace(
        netuid=126,
        neuron=SimpleNamespace(
            burn_fraction=0.90,
            funding_fraction=0.05,
            funding_hotkey="funding-hotkey",
        ),
    )
    settlement.metagraph = SimpleNamespace(
        n=4,
        hotkeys=["owner-hotkey", "miner-one", "winner-hotkey", "funding-hotkey"],
    )
    settlement.subtensor = SimpleNamespace(
        get_subnet_owner_hotkey=Mock(return_value="owner-hotkey")
    )

    scores, allocation = await settlement._emission_target(2)

    assert np.allclose(scores, [0.90, 0.0, 0.05, 0.05])
    assert allocation["owner"]["uid"] == 0
    assert allocation["funding"]["uid"] == 3
    assert allocation["winner"]["uid"] == 2
    settlement.subtensor.get_subnet_owner_hotkey.assert_called_once_with(126)


def test_weight_rows_serialize_the_processed_sdk_vector():
    prepared = (np.asarray([2]), np.asarray([1.0]), np.asarray([2]), np.asarray([65535]))
    assert weight_rows(prepared) == [{"uid": 2, "weight": 1.0}]
