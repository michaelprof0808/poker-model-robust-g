from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from neurons.validator import Validator
from poker44.platform.models import LeasedSession, SessionLease, ValidationRound
from poker44.protocol import MicroSessionDetectionSynapse
from poker44.validator.evaluation.mixin import ValidatorEvaluationMixin
from poker44.validator.evaluation.models import MinerEvaluation


@pytest.mark.asyncio
async def test_round_flows_from_platform_lease_through_rewards_and_weights():
    lease = SessionLease(
        lease_id="lease-1",
        window_id="round-1",
        dataset_hash="sha256:dataset",
        expires_at="2099-01-01T00:00:00Z",
        sessions=[
            LeasedSession(
                payload={"item_id": "item-1", "schema_version": "4.1", "window_id": "round-1"},
                is_bot=True,
            )
        ],
    )
    validation_round = ValidationRound(lease=lease)
    evaluation = MinerEvaluation(
        uid=4,
        hotkey="miner-hotkey",
        quality_score=0.9,
        metrics={"accuracy": 1.0},
        response_seconds=0.1,
        model_version="model-v1",
    )
    calls: list[str] = []

    validator = SimpleNamespace(
        poll_interval=0,
        subnet_data=SimpleNamespace(
            complete_lease=Mock(side_effect=lambda *_: calls.append("complete_lease"))
        ),
        round_manager=SimpleNamespace(
            complete=Mock(side_effect=lambda *_: calls.append("complete_round")),
            fail=Mock(),
        ),
        _reconcile_pending_weight_reveals=AsyncMock(),
        _attempt_pending_weight_settlement=AsyncMock(),
        _report_event=AsyncMock(),
    )
    validator._start_validation_round = AsyncMock(
        side_effect=lambda *_: (
            calls.append("load_platform_sessions"),
            validation_round,
        )[1]
    )
    validator._run_evaluation_phase = AsyncMock(
        side_effect=lambda _: (
            calls.append("query_miners_and_compute_rewards"),
            [evaluation],
        )[1]
    )
    validator._run_settlement_phase = AsyncMock(
        side_effect=lambda *_: (
            calls.append("compute_and_set_weights"),
            {"weights": [{"uid": 4, "weight": 1.0}], "submitted": True},
        )[1]
    )

    with patch("neurons.validator.asyncio.sleep", new=AsyncMock()):
        await Validator.forward(validator)

    assert calls == [
        "load_platform_sessions",
        "query_miners_and_compute_rewards",
        "compute_and_set_weights",
        "complete_lease",
        "complete_round",
    ]
    validator.subnet_data.complete_lease.assert_called_once_with("lease-1", "round-1")
    validator.round_manager.complete.assert_called_once_with("round-1")
    validator.round_manager.fail.assert_not_called()


@pytest.mark.asyncio
async def test_failed_round_reports_backoff_and_terminal_state():
    validation_round = ValidationRound(
        lease=SessionLease(
            lease_id="lease-1",
            window_id="round-1",
            dataset_hash="hash",
            expires_at="2099-01-01T00:00:00Z",
            sessions=[LeasedSession(payload={"item_id": "i1", "schema_version": "4.1"}, is_bot=True)],
        )
    )
    validator = SimpleNamespace(
        poll_interval=0,
        round_manager=SimpleNamespace(fail=Mock(return_value=None)),
        _reconcile_pending_weight_reveals=AsyncMock(),
        _attempt_pending_weight_settlement=AsyncMock(),
        _start_validation_round=AsyncMock(return_value=validation_round),
        _run_evaluation_phase=AsyncMock(side_effect=RuntimeError("miner failure")),
        _report_event=AsyncMock(),
    )

    with patch("neurons.validator.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(RuntimeError, match="miner failure"):
            await Validator.forward(validator)

    validator.round_manager.fail.assert_called_once_with("round-1")
    validator._report_event.assert_awaited_once_with(
        "validation_round_failed",
        validation_round,
        {
            "error": "miner failure",
            "retry_in_seconds": None,
            "terminal": True,
        },
    )


@pytest.mark.asyncio
async def test_transient_missing_miner_response_is_retried(monkeypatch):
    monkeypatch.setenv("POKER44_MINER_RETRY_DELAY_SECONDS", "0")
    valid = SimpleNamespace(risk_scores=[0.2])
    missing = SimpleNamespace(risk_scores=None)
    recovered = SimpleNamespace(risk_scores=[0.8])
    validator = SimpleNamespace(
        dendrite=AsyncMock(return_value=[recovered]),
        config=SimpleNamespace(neuron=SimpleNamespace(timeout=10)),
    )
    synapse = MicroSessionDetectionSynapse(
        window_id="window-1",
        dataset_hash="a" * 64,
        query_id="query",
        items=[{"schema_version": "4.1"}],
    )

    merged = await ValidatorEvaluationMixin._retry_transient_responses(
        validator,
        uids=[2, 3],
        axons=["axon-2", "axon-3"],
        synapse=synapse,
        responses=[valid, missing],
    )

    assert merged == [valid, recovered]
    validator.dendrite.assert_awaited_once_with(
        axons=["axon-3"],
        synapse=synapse,
        deserialize=False,
        timeout=10.0,
    )


@pytest.mark.asyncio
async def test_evaluated_round_resumes_without_querying_miners():
    validation_round = ValidationRound(
        lease=SessionLease(
            lease_id="lease-resume",
            window_id="window-resume",
            dataset_hash="hash",
            expires_at="2099-01-01T00:00:00Z",
            sessions=[LeasedSession(payload={"item_id": "i1", "schema_version": "4.1"}, is_bot=True)],
        ),
        resume_state="EVALUATED",
        resume_evidence={
            "evaluations": [
                {
                    "uid": 4,
                    "hotkey": "miner",
                    "quality_score": 0.75,
                    "metrics": {"accuracy": 1.0},
                    "response_seconds": 0.1,
                    "model_version": "v4",
                    "error": None,
                }
            ]
        },
    )
    validator = SimpleNamespace(dendrite=AsyncMock())

    restored = await ValidatorEvaluationMixin._run_evaluation_phase(
        validator, validation_round
    )

    assert restored[0].uid == 4
    validator.dendrite.assert_not_awaited()
