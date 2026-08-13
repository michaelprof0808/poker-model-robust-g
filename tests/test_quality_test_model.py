from __future__ import annotations

from poker44.miner.config import MinerModelConfig
from poker44.miner.model import ReferenceSessionModel
from poker44.miner.test_models import QualityRiskModel


def config() -> MinerModelConfig:
    return MinerModelConfig(
        factory="poker44.miner.test_models:create_quality_model",
        model_path=None,
        device="cpu",
        version="quality-test-v1",
        max_sessions_per_request=8,
    )


def session(session_id: str) -> dict:
    return {
        "schema_version": "4.1",
        "item_id": session_id,
        "window_id": "window-1",
        "decisions": [
            {
                "decision_number": index + 1,
                "phase": "flop" if index == 0 else "preflop",
                "position_group": ("early", "late", "blinds")[index % 3],
                "pressure": "facing_bet" if index % 2 else "no_call",
                "action_type": "raise",
                "size_bucket": "half_pot",
                "is_all_in": False,
            }
            for index in range(4)
        ],
    }


def test_full_quality_matches_reference(monkeypatch):
    monkeypatch.setenv("POKER44_TEST_MODEL_QUALITY", "1")
    model = QualityRiskModel(config())
    payload = session("subject-a")

    assert model.predict([payload]) == [ReferenceSessionModel._session_score(payload)]


def test_zero_quality_inverts_reference(monkeypatch):
    monkeypatch.setenv("POKER44_TEST_MODEL_QUALITY", "0")
    model = QualityRiskModel(config())
    payload = session("subject-b")
    reference = ReferenceSessionModel._session_score(payload)

    assert model.predict([payload]) == [round(1.0 - reference, 6)]


def test_quality_must_be_bounded(monkeypatch):
    monkeypatch.setenv("POKER44_TEST_MODEL_QUALITY", "1.1")

    try:
        QualityRiskModel(config())
    except ValueError as error:
        assert "within [0, 1]" in str(error)
    else:
        raise AssertionError("invalid model quality was accepted")
