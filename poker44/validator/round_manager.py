"""Local lifecycle state for one validator evaluation round."""

from __future__ import annotations

from dataclasses import dataclass
import time

from poker44.platform.models import ValidationRound


@dataclass
class ValidationRoundManager:
    active: ValidationRound | None = None
    completed_rounds: int = 0
    failure_attempts: dict[str, int] | None = None
    retry_after: dict[str, float] | None = None
    max_attempts: int = 3
    backoff_base_seconds: float = 60.0
    backoff_max_seconds: float = 900.0

    def __post_init__(self) -> None:
        self.failure_attempts = self.failure_attempts or {}
        self.retry_after = self.retry_after or {}

    def start(self, validation_round: ValidationRound) -> None:
        if self.active is not None:
            raise RuntimeError("A validation round is already active")
        self.active = validation_round

    def complete(self, round_id: str) -> None:
        if self.active is None or self.active.round_id != round_id:
            raise RuntimeError("Cannot complete an inactive validation round")
        self.active = None
        self.completed_rounds += 1
        self.failure_attempts.pop(round_id, None)
        self.retry_after.pop(round_id, None)

    def fail(self, round_id: str, *, now: float | None = None) -> float | None:
        if self.active is not None and self.active.round_id == round_id:
            self.active = None
        attempts = self.failure_attempts.get(round_id, 0) + 1
        self.failure_attempts[round_id] = attempts
        if attempts >= self.max_attempts:
            self.retry_after[round_id] = float("inf")
            return None
        delay = min(
            self.backoff_max_seconds,
            self.backoff_base_seconds * (2 ** (attempts - 1)),
        )
        self.retry_after[round_id] = (time.monotonic() if now is None else now) + delay
        return delay

    def retry_delay(self, round_id: str, *, now: float | None = None) -> float | None:
        deadline = self.retry_after.get(round_id)
        if deadline is None:
            return 0.0
        if deadline == float("inf"):
            return None
        return max(0.0, deadline - (time.monotonic() if now is None else now))
