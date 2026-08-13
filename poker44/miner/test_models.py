"""Deterministic test-only miner models used by the Poker44 testnet stack."""

from __future__ import annotations

import hashlib
import os
from typing import Any

from poker44.miner.config import MinerModelConfig
from poker44.miner.model import ReferenceSessionModel


class ConstantRiskModel:
    def __init__(self, config: MinerModelConfig):
        self.version = config.version

    def load(self) -> None:
        return None

    def predict(self, sessions: list[dict[str, Any]]) -> list[float]:
        return [0.5 for _ in sessions]


def create_constant_model(config: MinerModelConfig) -> ConstantRiskModel:
    return ConstantRiskModel(config)


class QualityRiskModel:
    """Deterministic testnet model with a configurable prediction quality.

    This model never reads a label. It starts from the same micro-session strategy
    features as the public reference model, then deterministically inverts a
    subset of predictions based only on ``item_id``. Running several
    qualities gives the E2E stack a stable, progressively ranked miner set.
    """

    def __init__(self, config: MinerModelConfig):
        self.version = config.version
        self.quality = float(os.environ.get("POKER44_TEST_MODEL_QUALITY", "1"))
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("POKER44_TEST_MODEL_QUALITY must be within [0, 1]")

    def load(self) -> None:
        return None

    @staticmethod
    def _unit_interval(session_id: str) -> float:
        digest = hashlib.sha256(session_id.encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64 - 1)

    def predict(self, sessions: list[dict[str, Any]]) -> list[float]:
        scores: list[float] = []
        for session in sessions:
            base = ReferenceSessionModel._session_score(session)
            session_id = str(session.get("item_id") or "")
            score = (
                base if self._unit_interval(session_id) <= self.quality else 1.0 - base
            )
            scores.append(round(score, 6))
        return scores


def create_quality_model(config: MinerModelConfig) -> QualityRiskModel:
    return QualityRiskModel(config)
