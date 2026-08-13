"""Acquire a sealed session window and validator-specific lease."""

from __future__ import annotations

import asyncio

import bittensor as bt

from poker44.platform.models import ValidationRound


class ValidatorRoundStartMixin:
    async def _start_validation_round(self) -> ValidationRound | None:
        window = await asyncio.to_thread(self.subnet_data.current_window)
        if not window or str(window.get("status") or "").upper() not in {
            "AVAILABLE",
            "LEASING",
        }:
            bt.logging.info("No sealed session window is currently available")
            return None
        window_id = str(window.get("window_id") or "").strip()
        if not window_id:
            raise RuntimeError("Available window response contains no window_id")
        retry_delay = self.round_manager.retry_delay(window_id)
        if retry_delay is None:
            bt.logging.error(
                "Window quarantined after repeated terminal failures | "
                f"window={window_id}"
            )
            return None
        if retry_delay > 0:
            bt.logging.warning(
                "Window retry is in backoff | "
                f"window={window_id} retry_in_seconds={retry_delay:.1f}"
            )
            return None
        lease = await asyncio.to_thread(self.subnet_data.acquire_lease, window_id)
        if lease.completed_at:
            bt.logging.info(
                "Current window was already evaluated by this validator | "
                f"window={lease.window_id} completed_at={lease.completed_at}"
            )
            return None
        # The sealed session window is the network-wide evaluation round.
        # Every validator leasing it must report the same identifier so the
        # dashboard can aggregate participants and miner scores correctly.
        validation_round = ValidationRound(lease=lease, round_id=lease.window_id)
        run = await asyncio.to_thread(
            self.subnet_data.claim_evaluation_run,
            validation_round.lease.window_id,
            validation_round.round_id,
        )
        state = str(run.get("state") or "")
        if state not in {"ACQUIRED", "EVALUATED"}:
            bt.logging.info(
                "Current window already reached a durable settlement phase | "
                f"window={lease.window_id} state={state}"
            )
            return None
        validation_round.resume_state = state
        evidence = run.get("evidence")
        validation_round.resume_evidence = (
            evidence if isinstance(evidence, dict) else {}
        )
        self.round_manager.start(validation_round)
        await self._report_event(
            "validation_round_started",
            validation_round,
            {
                "window_id": lease.window_id,
                "dataset_hash": lease.dataset_hash,
                "fixture_only": lease.fixture_only,
                "single_tournament_production": bool(
                    lease.audit.get("singleTournamentProduction", False)
                ),
            },
        )
        await self._report_event(
            "sessions_acquired",
            validation_round,
            {
                "lease_id": lease.lease_id,
                "session_count": len(lease.sessions),
                "item_count": len(lease.sessions),
                "independent_actor_count": len(
                    {
                        session.actor_group or f"item:{index}"
                        for index, session in enumerate(lease.sessions)
                    }
                ),
                "expires_at": lease.expires_at,
                "fixture_only": lease.fixture_only,
                "single_tournament_production": bool(
                    lease.audit.get("singleTournamentProduction", False)
                ),
                "tournament_count": int(lease.audit.get("tournaments", 0)),
                "bot_family_count": int(lease.audit.get("botFamilies", 0)),
                "rejection_counts": lease.audit.get("rejectionCounts", {}),
            },
        )
        return validation_round
