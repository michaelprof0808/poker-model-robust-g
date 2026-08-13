"""Query miner axons and calculate local rewards."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from dataclasses import replace
from typing import Any

import bittensor as bt
import numpy as np

from poker44.platform.models import ValidationRound
from poker44.validator.evaluation.models import MinerEvaluation
from poker44.validator.evaluation.redteam_gate import audit_redteam_leakage
from poker44.validator.evaluation.reward import reward
from poker44.utils.encrypted_endpoints import is_masked_axon
from poker44.validator.tracks import MICRO_SESSION_EVALUATION


class ValidatorEvaluationMixin:
    @staticmethod
    def _restore_evaluations(
        validation_round: ValidationRound,
    ) -> list[MinerEvaluation]:
        rows = validation_round.resume_evidence.get("evaluations")
        if validation_round.resume_state != "EVALUATED" or not isinstance(rows, list):
            return []
        restored: list[MinerEvaluation] = []
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("Stored evaluation evidence is malformed")
            restored.append(
                MinerEvaluation(
                    uid=int(row["uid"]),
                    hotkey=str(row["hotkey"]),
                    quality_score=float(row["quality_score"]),
                    metrics={
                        str(key): float(value)
                        for key, value in dict(row.get("metrics") or {}).items()
                    },
                    response_seconds=(
                        float(row["response_seconds"])
                        if row.get("response_seconds") is not None
                        else None
                    ),
                    model_version=(
                        str(row["model_version"])
                        if row.get("model_version") is not None
                        else None
                    ),
                    error=str(row["error"]) if row.get("error") is not None else None,
                )
            )
        if not restored:
            raise RuntimeError("Stored EVALUATED state has no miner evaluations")
        return restored

    def _candidate_miners(self, window_id: str) -> tuple[list[int], list[Any]]:
        try:
            overrides = json.loads(os.getenv("POKER44_AXON_OVERRIDES", "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError("POKER44_AXON_OVERRIDES must be valid JSON") from exc
        resolver = getattr(self, "endpoint_resolver", None)
        if resolver is not None:
            self.refresh_encrypted_endpoints()
        candidates: list[tuple[int, Any, bool]] = []
        for uid, axon in enumerate(self.metagraph.axons):
            if uid == int(self.uid):
                continue
            if bool(self.metagraph.validator_permit[uid]):
                continue
            override = str(overrides.get(str(uid), "")).strip()
            protected = False
            if override:
                host, override_port = override.rsplit(":", 1)
                axon = replace(
                    axon,
                    ip=socket.gethostbyname(host),
                    port=int(override_port),
                )
            elif resolver is not None:
                axon, protected = resolver.resolve(
                    self.metagraph.hotkeys[uid],
                    axon,
                )
            if is_masked_axon(axon) and not protected:
                bt.logging.warning(
                    f"Skipping protected miner UID {uid}: no decryptable endpoint is available."
                )
                continue
            ip = str(getattr(axon, "ip", "") or "")
            port = int(getattr(axon, "port", 0) or 0)
            if ip in {"", "0.0.0.0", "::", "[::]"} or port <= 0:
                continue
            candidates.append((uid, axon, protected))
        selected = sorted(candidates, key=lambda item: item[0])
        bt.logging.info(
            f"Selected all {len(selected)} eligible miner axons | "
            + ", ".join(
                f"uid={uid}@{'protected' if protected else f'{axon.ip}:{axon.port}'}"
                for uid, axon, protected in selected
            )
        )
        return [uid for uid, _, _ in selected], [axon for _, axon, _ in selected]

    @staticmethod
    def _response_seconds(response: Any) -> float | None:
        try:
            value = float(response.dendrite.process_time)
            return value if np.isfinite(value) and value >= 0.0 else None
        except Exception:
            return None

    async def _retry_transient_responses(
        self,
        *,
        uids: list[int],
        axons: list[Any],
        synapse: bt.Synapse,
        responses: list[Any],
    ) -> list[Any]:
        """Retry miners that returned no scores before finalizing the round."""

        missing = [
            index
            for index, response in enumerate(responses)
            if not isinstance(getattr(response, "risk_scores", None), list)
        ]
        if not missing:
            return responses

        delay = max(0.0, float(os.getenv("POKER44_MINER_RETRY_DELAY_SECONDS", "30")))
        bt.logging.warning(
            "Retrying transient miner responses | "
            f"uids={','.join(str(uids[index]) for index in missing)} "
            f"delay_seconds={delay:g}"
        )
        if delay:
            await asyncio.sleep(delay)
        retried = await self.dendrite(
            axons=[axons[index] for index in missing],
            synapse=synapse,
            deserialize=False,
            timeout=float(self.config.neuron.timeout),
        )
        merged = list(responses)
        for index, response in zip(missing, retried):
            merged[index] = response
        return merged

    async def _run_evaluation_phase(
        self, validation_round: ValidationRound
    ) -> list[MinerEvaluation]:
        restored = ValidatorEvaluationMixin._restore_evaluations(validation_round)
        if restored:
            bt.logging.info(
                "Resuming durable evaluated round without querying miners | "
                f"window={validation_round.lease.window_id} miners={len(restored)}"
            )
            return restored
        redteam_threshold = float(os.getenv("POKER44_REDTEAM_MAX_REWARD", "0.15"))
        redteam = audit_redteam_leakage(
            validation_round.miner_items,
            validation_round.labels,
            threshold=redteam_threshold,
        )
        await self._report_event(
            "redteam_gate_checked",
            validation_round,
            redteam.to_dict(),
        )
        enforce_redteam = (
            os.getenv("POKER44_ENFORCE_REDTEAM_GATE", "true").lower() == "true"
        )
        if (
            redteam.skipped
            and os.getenv("POKER44_E2E_ALLOW_SINGLE_CLASS_REWARD", "false").lower()
            != "true"
        ):
            raise RuntimeError("Red-team gate requires a mixed human/bot window")
        if (
            enforce_redteam
            and not redteam.skipped
            and not redteam.passed
        ):
            raise RuntimeError(
                "Red-team leakage gate failed: "
                f"feature={redteam.feature} reward={redteam.reward:.6f} "
                f"threshold={redteam.threshold:.6f}"
            )

        uids, axons = self._candidate_miners(validation_round.lease.window_id)
        if not uids:
            bt.logging.warning("No reachable miner axons were found")
            return []
        synapse = MICRO_SESSION_EVALUATION.build_synapse(
            validation_round,
            self.wallet.hotkey.ss58_address,
        )
        responses = await self.dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=False,
            timeout=float(self.config.neuron.timeout),
        )
        responses = await self._retry_transient_responses(
            uids=uids,
            axons=axons,
            synapse=synapse,
            responses=responses,
        )
        evaluations: list[MinerEvaluation] = []
        for uid, response in zip(uids, responses):
            hotkey = str(self.metagraph.hotkeys[uid])
            try:
                raw_scores = getattr(response, "risk_scores", None)
                if not isinstance(raw_scores, list):
                    raise ValueError("missing risk_scores")
                scores = [float(value) for value in raw_scores]
                if len(scores) != len(validation_round.labels):
                    raise ValueError(
                        f"returned {len(scores)} scores for {len(validation_round.labels)} sessions"
                    )
                metrics = reward(
                    scores,
                    validation_round.labels,
                    sample_weights=validation_round.sample_weights,
                    allow_single_class=os.getenv(
                        "POKER44_E2E_ALLOW_SINGLE_CLASS_REWARD", "false"
                    ).lower()
                    == "true",
                )
                evaluations.append(
                    MinerEvaluation(
                        uid=uid,
                        hotkey=hotkey,
                        quality_score=metrics.reward,
                        metrics=metrics.to_dict(),
                        response_seconds=self._response_seconds(response),
                        model_version=getattr(response, "model_version", None),
                    )
                )
            except Exception as exc:
                evaluations.append(
                    MinerEvaluation(
                        uid=uid,
                        hotkey=hotkey,
                        quality_score=0.0,
                        metrics={},
                        response_seconds=self._response_seconds(response),
                        model_version=getattr(response, "model_version", None),
                        error=str(exc),
                    )
                )
            item = evaluations[-1]
            status_code = getattr(
                getattr(response, "dendrite", None), "status_code", None
            )
            status_message = getattr(
                getattr(response, "dendrite", None), "status_message", None
            )
            bt.logging.info(
                "Miner evaluation | "
                f"uid={uid} quality_score={item.quality_score:.6f} model={item.model_version} "
                f"status={status_code} message={status_message} error={item.error}"
            )
        await self._report_event(
            "miners_queried",
            validation_round,
            {
                "miner_count": len(evaluations),
                "miners": [
                    {
                        "uid": item.uid,
                        "hotkey": item.hotkey,
                        "response_seconds": item.response_seconds,
                        "model_version": item.model_version,
                        "error": item.error,
                    }
                    for item in evaluations
                ],
            },
        )
        advanced = await asyncio.to_thread(
            self.subnet_data.advance_evaluation_run,
            validation_round.lease.window_id,
            validation_round.round_id,
            "EVALUATED",
            {
                "miner_count": len(evaluations),
                "item_count": len(validation_round.labels),
                "query_id": str(getattr(synapse, "query_id", "")),
                "evaluations": [
                    {
                        "uid": item.uid,
                        "hotkey": item.hotkey,
                        "quality_score": item.quality_score,
                        "metrics": item.metrics,
                        "response_seconds": item.response_seconds,
                        "model_version": item.model_version,
                        "error": item.error,
                    }
                    for item in evaluations
                ],
            },
        )
        if not advanced:
            raise RuntimeError("Could not persist EVALUATED state")
        return evaluations
