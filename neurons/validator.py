"""Poker44 v2 validator entrypoint."""

from __future__ import annotations

import asyncio
import os
import time

import bittensor as bt
from dotenv import load_dotenv

from poker44 import __version__
from poker44.base.validator import BaseValidatorNeuron
from poker44.platform.validator_mixin import ValidatorPlatformMixin
from poker44.utils.config import config
from poker44.validator.round_manager import ValidationRoundManager
from poker44.validator.round_start.mixin import ValidatorRoundStartMixin
from poker44.validator.evaluation.mixin import ValidatorEvaluationMixin
from poker44.validator.reporting.client import DashboardReportingClient
from poker44.validator.reporting.mixin import ValidatorReportingMixin
from poker44.validator.settlement.mixin import ValidatorSettlementMixin
from poker44.validator.settlement.weights import winner_fraction

load_dotenv()


class Validator(
    ValidatorRoundStartMixin,
    ValidatorEvaluationMixin,
    ValidatorSettlementMixin,
    ValidatorReportingMixin,
    ValidatorPlatformMixin,
    BaseValidatorNeuron,
):
    """Evaluate session classifiers and calculate weights locally."""

    weights_managed_by_forward = True

    def __init__(self):
        cfg = config(Validator)
        winner_share = winner_fraction(
            float(cfg.neuron.burn_fraction), float(cfg.neuron.funding_fraction)
        )
        if int(cfg.neuron.num_concurrent_forwards) != 1:
            raise ValueError(
                "Poker44 session leases require --neuron.num_concurrent_forwards 1"
            )
        super().__init__(config=cfg)
        funding_hotkey = self._funding_hotkey()
        funding_uid = self._uid_for_hotkey(funding_hotkey, "funding")
        self._initialize_platform_client()
        self.dashboard_reporting = DashboardReportingClient(self.wallet)
        self.round_manager = ValidationRoundManager(
            max_attempts=max(1, int(os.getenv("POKER44_ROUND_MAX_ATTEMPTS", "3"))),
            backoff_base_seconds=max(
                1.0, float(os.getenv("POKER44_ROUND_FAILURE_BACKOFF_SECONDS", "60"))
            ),
            backoff_max_seconds=max(
                1.0,
                float(os.getenv("POKER44_ROUND_FAILURE_BACKOFF_MAX_SECONDS", "900")),
            ),
        )
        # Millisecond epoch makes ordering survive process restarts while the
        # dashboard's (validation_round, validator, sequence) uniqueness remains useful.
        self._report_sequence = int(time.time() * 1000)
        self.poll_interval = max(
            1, int(os.getenv("POKER44_POLL_INTERVAL_SECONDS", "300"))
        )
        bt.logging.info(
            f"Poker44 Validator v{__version__} started | "
            f"session_api={self.subnet_data_config.base_url} "
            f"burn={float(cfg.neuron.burn_fraction):.4f} "
            f"funding={float(cfg.neuron.funding_fraction):.4f}@uid{funding_uid} "
            f"winner={winner_share:.4f}"
        )

    async def forward(self) -> None:
        await self._reconcile_pending_weight_reveals()
        await self._attempt_pending_weight_settlement()
        validation_round = await self._start_validation_round()
        if validation_round is None:
            await asyncio.sleep(self.poll_interval)
            return

        try:
            evaluations = await self._run_evaluation_phase(validation_round)
            if not evaluations:
                raise RuntimeError("Validation round produced no miner evaluations")
            settlement = await self._run_settlement_phase(validation_round, evaluations)
            await asyncio.to_thread(
                self.subnet_data.complete_lease,
                validation_round.lease.lease_id,
                validation_round.round_id,
            )
            self.round_manager.complete(validation_round.round_id)
            bt.logging.info(
                "Validation round completed | "
                f"round={validation_round.round_id} window={validation_round.lease.window_id} "
                f"sessions={len(validation_round.lease.sessions)} miners={len(evaluations)} "
                f"weights_submitted={settlement['submitted']}"
            )
        except Exception as exc:
            retry_delay = self.round_manager.fail(validation_round.round_id)
            await self._report_event(
                "validation_round_failed",
                validation_round,
                {
                    "error": str(exc),
                    "retry_in_seconds": retry_delay,
                    "terminal": retry_delay is None,
                },
            )
            raise
        finally:
            await asyncio.sleep(self.poll_interval)


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            asyncio.run(asyncio.sleep(60))
