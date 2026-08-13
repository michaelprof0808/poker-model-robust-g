from poker44.platform.models import LeasedSession, SessionLease, ValidationRound
from poker44.validator.round_manager import ValidationRoundManager


def make_round(round_id: str = "round-1") -> ValidationRound:
    return ValidationRound(
        lease=SessionLease(
            lease_id="lease-1",
            window_id=round_id,
            dataset_hash="hash",
            expires_at="2099-01-01T00:00:00Z",
            sessions=[LeasedSession(payload={"item_id": "i1", "schema_version": "4.1"}, is_bot=True)],
        )
    )


def test_failed_round_uses_exponential_backoff_then_becomes_terminal():
    manager = ValidationRoundManager(
        max_attempts=3, backoff_base_seconds=10, backoff_max_seconds=60
    )

    manager.start(make_round())
    assert manager.fail("round-1", now=100) == 10
    assert manager.retry_delay("round-1", now=105) == 5
    assert manager.retry_delay("round-1", now=110) == 0

    manager.start(make_round())
    assert manager.fail("round-1", now=110) == 20
    assert manager.retry_delay("round-1", now=120) == 10

    manager.start(make_round())
    assert manager.fail("round-1", now=130) is None
    assert manager.retry_delay("round-1", now=10_000) is None


def test_success_clears_previous_failure_state():
    manager = ValidationRoundManager(backoff_base_seconds=10)
    manager.start(make_round())
    manager.fail("round-1", now=100)
    manager.start(make_round())
    manager.complete("round-1")

    assert manager.retry_delay("round-1", now=100) == 0
