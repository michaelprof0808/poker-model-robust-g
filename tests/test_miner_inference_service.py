from __future__ import annotations

import pytest

from poker44.miner.config import MinerModelConfig
from poker44.miner.model import ReferenceSessionModel
from poker44.miner.service import MinerInferenceService


def make_service(max_sessions=8):
    config = MinerModelConfig(
        factory="poker44.miner.model:create_reference_model",
        model_path=None,
        device="cpu",
        version="test-v1",
        max_sessions_per_request=max_sessions,
    )
    model = ReferenceSessionModel(config)
    model.load()
    return MinerInferenceService(model, config)


def make_micro_session(**overrides):
    session = {
        "schema_version": "4.1",
        "item_id": "item-1",
        "window_id": "window-1",
        "decisions": [
            {
                "decision_number": index + 1,
                "phase": "flop" if index == 0 else "preflop",
                "position_group": "late",
                "pressure": "no_call",
                "action_type": "raise" if index % 2 else "fold",
                "size_bucket": "half_pot" if index % 2 else "not_applicable",
                "is_all_in": False,
            }
            for index in range(4)
        ],
    }
    session.update(overrides)
    return session


@pytest.mark.asyncio
async def test_model_returns_one_bounded_score_per_session():
    scores = await make_service().predict_micro_sessions(
        [make_micro_session(), make_micro_session(item_id="item-2")]
    )

    assert len(scores) == 2
    assert all(0.0 <= score <= 1.0 for score in scores)


@pytest.mark.asyncio
async def test_micro_session_protocol_v41_is_accepted():
    session = make_micro_session()
    session["decisions"] = session["decisions"][:4]
    session["decisions"][1]["phase"] = "preflop"

    scores = await make_service().predict_micro_sessions([session])

    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0


@pytest.mark.asyncio
async def test_micro_session_rejects_label_leak_before_model_execution():
    leaked = make_micro_session(label="bot")
    with pytest.raises(ValueError, match="ground-truth"):
        await make_service().predict_micro_sessions([leaked])


@pytest.mark.asyncio
async def test_contract_rejects_old_schema_payloads():
    with pytest.raises(ValueError, match="micro-session schema 4.1"):
        await make_service().predict_micro_sessions([{"schema_version": "2"}])


@pytest.mark.asyncio
async def test_ground_truth_is_rejected_at_miner_boundary():
    with pytest.raises(ValueError, match="ground-truth"):
        await make_service().predict_micro_sessions([make_micro_session(is_bot=True)])


@pytest.mark.asyncio
async def test_nested_ground_truth_is_rejected_at_miner_boundary():
    session = make_micro_session()
    session["decisions"][0]["bot_family"] = "leaked-family"
    with pytest.raises(ValueError, match="ground-truth"):
        await make_service().predict_micro_sessions([session])


@pytest.mark.asyncio
async def test_non_finite_model_output_is_rejected():
    service = make_service()
    service.model.predict = lambda _sessions: [float("nan")]
    with pytest.raises(ValueError, match="finite"):
        await service.predict_micro_sessions([make_micro_session()])


@pytest.mark.asyncio
async def test_request_limit_is_enforced():
    with pytest.raises(ValueError, match="maximum"):
        await make_service(max_sessions=1).predict_micro_sessions(
            [make_micro_session(), make_micro_session(item_id="item-2")]
        )


@pytest.mark.asyncio
async def test_request_byte_limit_is_enforced():
    service = make_service()
    object.__setattr__(service.config, "max_request_bytes", 10)
    with pytest.raises(ValueError, match="bytes"):
        await service.predict_micro_sessions([make_micro_session()])
