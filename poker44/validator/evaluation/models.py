"""Validator-local miner evaluation results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinerEvaluation:
    uid: int
    hotkey: str
    quality_score: float
    metrics: dict[str, float]
    response_seconds: float | None
    model_version: str | None
    error: str | None = None
