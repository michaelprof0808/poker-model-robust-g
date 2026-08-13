"""Tournament micro-session validator request builder."""

from __future__ import annotations

from poker44.platform.models import ValidationRound
from poker44.protocol import MicroSessionDetectionSynapse
from poker44.validator.tracks.base import query_id


class MicroSessionEvaluation:
    enforce_redteam_gate = True

    def build_synapse(
        self, validation_round: ValidationRound, validator_hotkey: str
    ) -> MicroSessionDetectionSynapse:
        return MicroSessionDetectionSynapse(
            window_id=validation_round.lease.window_id,
            dataset_hash=validation_round.lease.dataset_hash,
            query_id=query_id(validation_round, validator_hotkey),
            items=validation_round.miner_items,
        )
