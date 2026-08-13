from types import SimpleNamespace
from types import MethodType
from unittest.mock import AsyncMock, Mock

import pytest

from poker44.validator.settlement.mixin import ValidatorSettlementMixin


def _harness(
    tmp_path, *, current_block: int, last_update: int = 0, rate_limit: int = 100
):
    subtensor = SimpleNamespace(
        get_subnet_hyperparameters=Mock(
            return_value=SimpleNamespace(weights_rate_limit=rate_limit)
        ),
        get_current_block=Mock(return_value=current_block),
        commit_reveal_enabled=Mock(return_value=False),
    )
    harness = SimpleNamespace(
        config=SimpleNamespace(
            netuid=44,
            neuron=SimpleNamespace(
                full_path=str(tmp_path),
                wait_for_finalization=False,
            ),
        ),
        uid=0,
        metagraph=SimpleNamespace(last_update=[last_update], hotkeys=["validator", "one", "winner"]),
        subtensor=subtensor,
        subnet_data=SimpleNamespace(advance_evaluation_run=Mock(return_value=True)),
        set_weights=Mock(return_value=True),
        _report_event=AsyncMock(),
        _pending_weight_commits=AsyncMock(return_value=[]),
        _prepared_weights_from_rows=ValidatorSettlementMixin._prepared_weights_from_rows,
    )
    harness._weight_settlement_state_path = lambda: tmp_path / "weight_settlement.json"
    harness._load_weight_settlement = MethodType(
        ValidatorSettlementMixin._load_weight_settlement, harness
    )
    harness._save_weight_settlement = MethodType(
        ValidatorSettlementMixin._save_weight_settlement, harness
    )
    harness._weight_submission_is_due = MethodType(
        ValidatorSettlementMixin._weight_submission_is_due, harness
    )
    return harness


def test_new_round_replaces_target_and_keeps_unsettled_run_evidence(tmp_path):
    harness = _harness(tmp_path, current_block=10)
    harness._record_weight_settlement = MethodType(
        ValidatorSettlementMixin._record_weight_settlement, harness
    )
    first = SimpleNamespace(
        round_id="round-one",
        lease=SimpleNamespace(window_id="window-one"),
    )
    second = SimpleNamespace(
        round_id="round-two",
        lease=SimpleNamespace(window_id="window-two"),
    )

    harness._record_weight_settlement(first, [{"uid": 1, "hotkey": "one", "weight": 1.0}])
    harness._record_weight_settlement(
        second,
        [{"uid": 2, "hotkey": "winner", "weight": 1.0}],
    )

    state = harness._load_weight_settlement()
    assert state["version"] == 3
    assert state["weights"] == [{"uid": 2, "hotkey": "winner", "weight": 1.0}]
    assert state["evaluation_runs"] == [
        {
            "round_id": "round-one",
            "window_id": "window-one",
        },
        {
            "round_id": "round-two",
            "window_id": "window-two",
        },
    ]


@pytest.mark.asyncio
async def test_dirty_weights_remain_pending_until_chain_rate_limit_allows(tmp_path):
    harness = _harness(tmp_path, current_block=100, rate_limit=100)
    harness._save_weight_settlement(
        {
            "version": 1,
            "dirty": True,
            "window_id": "window-1",
            "round_id": "round-1",
            "weights": [{"uid": 2, "hotkey": "winner", "weight": 1.0}],
        }
    )

    submitted = await ValidatorSettlementMixin._attempt_pending_weight_settlement(
        harness
    )

    assert submitted is False
    assert harness._load_weight_settlement()["dirty"] is True
    harness.set_weights.assert_not_called()


@pytest.mark.asyncio
async def test_pending_weights_submit_once_allowed_and_refresh_after_720_blocks(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("POKER44_WEIGHT_REFRESH_BLOCKS", "720")
    harness = _harness(tmp_path, current_block=101, rate_limit=100)
    harness._save_weight_settlement(
        {
            "version": 1,
            "dirty": True,
            "window_id": "window-1",
            "round_id": "round-1",
            "weights": [{"uid": 2, "hotkey": "winner", "weight": 1.0}],
        }
    )

    assert (
        await ValidatorSettlementMixin._attempt_pending_weight_settlement(harness)
        is True
    )
    state = harness._load_weight_settlement()
    assert state["dirty"] is False
    assert state["last_submission_block"] == 101
    assert harness.set_weights.call_count == 1

    harness.subtensor.get_current_block.return_value = 820
    assert (
        await ValidatorSettlementMixin._attempt_pending_weight_settlement(harness)
        is False
    )
    assert harness.set_weights.call_count == 1

    harness.subtensor.get_current_block.return_value = 821
    assert (
        await ValidatorSettlementMixin._attempt_pending_weight_settlement(harness)
        is True
    )
    assert harness.set_weights.call_count == 2
    assert harness._report_event.await_args_list[-1].args[0] == "weights_refreshed"
