"""Shared primitives for isolated validator evaluation tracks."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol

import bittensor as bt

from poker44.platform.models import ValidationRound


class EvaluationTrack(Protocol):
    name: str
    enforce_redteam_gate: bool

    def build_synapse(
        self, validation_round: ValidationRound, validator_hotkey: str
    ) -> bt.Synapse: ...


def query_id(validation_round: ValidationRound, validator_hotkey: str) -> str:
    return sha256(
        (
            f"{validation_round.lease.window_id}:"
            "microsession:"
            f"{validator_hotkey}:"
            f"{validation_round.lease.dataset_hash}"
        ).encode()
    ).hexdigest()
