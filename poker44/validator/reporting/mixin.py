"""Build ordered, signed validator observability events."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import bittensor as bt

from poker44.platform.models import ValidationRound


class ValidatorReportingMixin:
    async def _report_event(
        self,
        event_type: str,
        validation_round: ValidationRound,
        payload: dict[str, Any],
    ) -> None:
        sequence = int(getattr(self, "_report_sequence", 0))
        self._report_sequence = sequence + 1
        validator_name = (
            os.getenv("POKER44_VALIDATOR_NAME", "").strip()
            or str(self.config.wallet.name).strip()
        )[:128] or None
        event = {
            "schema_version": "3",
            "event_id": uuid4().hex,
            "round_id": validation_round.round_id,
            "sequence": sequence,
            "netuid": int(self.config.netuid),
            "event_type": event_type,
            "validator_hotkey": self.wallet.hotkey.ss58_address,
            "validator_uid": int(self.uid),
            "validator_name": validator_name,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "payload": payload,
        }
        try:
            await asyncio.to_thread(self.dashboard_reporting.send, event)
        except Exception as exc:
            bt.logging.warning(f"Dashboard report failed ({event_type}): {exc}")
