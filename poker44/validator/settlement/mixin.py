"""Update local score state and publish weights when the chain cadence allows."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import bittensor as bt
import numpy as np

from poker44.base.utils.weight_utils import convert_weights_and_uids_for_emit
from poker44.platform.models import ValidationRound
from poker44.validator.evaluation.models import MinerEvaluation
from poker44.validator.settlement.weights import (
    emission_scores,
    ranked_score_rows,
    weight_rows,
    winner_uid,
)


class ValidatorSettlementMixin:
    def _funding_hotkey(self) -> str:
        hotkey = str(getattr(self.config.neuron, "funding_hotkey", "") or "").strip()
        if not hotkey:
            raise RuntimeError("POKER44_FUNDING_HOTKEY is required")
        return hotkey

    def _uid_for_hotkey(self, hotkey: str, role: str) -> int:
        try:
            return list(self.metagraph.hotkeys).index(hotkey)
        except ValueError as exc:
            raise RuntimeError(
                f"Configured {role} hotkey is not registered on netuid {self.config.netuid}"
            ) from exc

    async def _emission_target(self, winner: int) -> tuple[np.ndarray, dict]:
        owner_hotkey = str(
            await asyncio.to_thread(
                self.subtensor.get_subnet_owner_hotkey, self.config.netuid
            )
        )
        funding_hotkey = self._funding_hotkey()
        owner_uid = self._uid_for_hotkey(owner_hotkey, "owner")
        funding_uid = self._uid_for_hotkey(funding_hotkey, "funding")
        burn_fraction = float(self.config.neuron.burn_fraction)
        funding_fraction = float(self.config.neuron.funding_fraction)
        raw_weights = emission_scores(
            int(self.metagraph.n),
            winner_uid=winner,
            owner_uid=owner_uid,
            funding_uid=funding_uid,
            burn_fraction=burn_fraction,
            funding_fraction=funding_fraction,
        )
        return raw_weights, {
            "owner": {"uid": owner_uid, "hotkey": owner_hotkey, "fraction": burn_fraction},
            "funding": {
                "uid": funding_uid,
                "hotkey": funding_hotkey,
                "fraction": funding_fraction,
            },
            "winner": {
                "uid": winner,
                "hotkey": str(self.metagraph.hotkeys[winner]),
                "fraction": 1.0 - burn_fraction - funding_fraction,
            },
        }

    def _weight_settlement_state_path(self) -> Path:
        configured = os.getenv("POKER44_WEIGHT_SETTLEMENT_PATH", "").strip()
        if configured:
            return Path(configured).expanduser()
        return Path(self.config.neuron.full_path) / "weight_settlement.json"

    def _load_weight_settlement(self) -> dict | None:
        path = self._weight_settlement_state_path()
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("weight settlement state must be an object")
            return value
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            bt.logging.warning(f"Could not read weight settlement state: {exc}")
            return None

    def _save_weight_settlement(self, state: dict) -> None:
        path = self._weight_settlement_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(state, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(path)

    def _record_weight_settlement(
        self, validation_round: ValidationRound, weights: list[dict]
    ) -> None:
        existing = self._load_weight_settlement() or {}
        evaluation_runs = (
            list(existing.get("evaluation_runs") or []) if existing.get("dirty") else []
        )
        if (
            existing.get("dirty")
            and not evaluation_runs
            and existing.get("round_id")
            and existing.get("window_id")
        ):
            evaluation_runs.append(
                {
                    "round_id": str(existing["round_id"]),
                    "window_id": str(existing["window_id"]),
                }
            )
        current_run = {
            "round_id": validation_round.round_id,
            "window_id": str(validation_round.lease.window_id),
        }
        evaluation_runs = [
            run
            for run in evaluation_runs
            if (run.get("window_id"), run.get("round_id"))
            != (current_run["window_id"], current_run["round_id"])
        ]
        evaluation_runs.append(current_run)
        self._save_weight_settlement(
            {
                "version": 3,
                "dirty": True,
                "round_id": validation_round.round_id,
                "window_id": str(validation_round.lease.window_id),
                "evaluation_runs": evaluation_runs,
                "weights": weights,
                **(
                    {"last_submission_block": existing["last_submission_block"]}
                    if "last_submission_block" in existing
                    else {}
                ),
            }
        )

    def _pending_reveal_state_path(self) -> Path:
        configured = os.getenv("POKER44_PENDING_REVEALS_PATH", "").strip()
        if configured:
            return Path(configured).expanduser()
        return Path(self.config.neuron.full_path) / "pending_weight_reveals.json"

    def _load_pending_reveal_reports(self) -> list[dict]:
        path = self._pending_reveal_state_path()
        if not path.exists():
            return []
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise ValueError("pending reveal state must be a list")
            return [item for item in value if isinstance(item, dict)]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            bt.logging.warning(f"Could not read pending weight reveal state: {exc}")
            return []

    def _save_pending_reveal_reports(self, reports: list[dict]) -> None:
        path = self._pending_reveal_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(reports, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(path)

    def _record_pending_reveal(
        self,
        validation_round: ValidationRound,
        weights: list[dict],
        evidence: dict,
        *,
        track_evaluation_run: bool = True,
        evaluation_runs: list[dict] | None = None,
        commit_finalized_reported: bool = False,
    ) -> None:
        reports = self._load_pending_reveal_reports()
        lease = getattr(validation_round, "lease", None)
        record = {
            "round_id": validation_round.round_id,
            "window_id": str(getattr(lease, "window_id", validation_round.round_id)),
            "weights": weights,
            "commit_block": int(evidence["commit_block"]),
            "reveal_round": int(evidence["reveal_round"]),
            "track_evaluation_run": track_evaluation_run,
            "evaluation_runs": evaluation_runs or [],
            "commit_finalized_reported": commit_finalized_reported,
        }
        key = (record["commit_block"], record["reveal_round"])
        reports = [
            item
            for item in reports
            if (item.get("commit_block"), item.get("reveal_round")) != key
        ]
        reports.append(record)
        self._save_pending_reveal_reports(reports)

    async def _pending_weight_commits(self) -> list[dict]:
        commits = await asyncio.to_thread(
            self.subtensor.get_timelocked_weight_commits, self.config.netuid
        )
        return [
            {
                "hotkey": str(hotkey),
                "commit_block": int(commit_block),
                "reveal_round": int(reveal_round),
            }
            for hotkey, commit_block, _ciphertext, reveal_round in commits
            if str(hotkey) == self.wallet.hotkey.ss58_address
        ]

    async def _reported_weights_are_visible(self, report: dict) -> bool:
        chain_weights = await asyncio.to_thread(
            self.subtensor.weights, self.config.netuid
        )
        uid = int(self.uid)
        visible = dict(chain_weights).get(uid)
        if visible is None:
            return False
        rows = report.get("weights") or []
        expected_uids, expected_weights = convert_weights_and_uids_for_emit(
            uids=np.asarray([int(row["uid"]) for row in rows], dtype=np.int64),
            weights=np.asarray(
                [float(row["weight"]) for row in rows], dtype=np.float32
            ),
        )
        expected = {
            int(row_uid): int(row_weight)
            for row_uid, row_weight in zip(expected_uids, expected_weights)
            if int(row_weight) > 0
        }
        actual = {
            int(row_uid): int(row_weight)
            for row_uid, row_weight in visible
            if int(row_weight) > 0
        }
        return bool(expected) and actual == expected

    async def _reconcile_pending_weight_reveals(self) -> None:
        reports = self._load_pending_reveal_reports()
        if not reports:
            return
        pending = {
            (item["commit_block"], item["reveal_round"])
            for item in await self._pending_weight_commits()
        }
        remaining: list[dict] = []
        for report in reports:
            key = (int(report["commit_block"]), int(report["reveal_round"]))
            track_evaluation_run = bool(report.get("track_evaluation_run", True))
            run_refs = list(report.get("evaluation_runs") or [])
            if track_evaluation_run and not run_refs:
                run_refs = [
                    {
                        "window_id": str(report.get("window_id") or report["round_id"]),
                        "round_id": str(report["round_id"]),
                    }
                ]
            if track_evaluation_run:
                for run in run_refs:
                    await asyncio.to_thread(
                        self.subnet_data.advance_evaluation_run,
                        str(run["window_id"]),
                        str(run["round_id"]),
                        "COMMIT_SUBMITTED",
                        {
                            "commit_block": key[0],
                            "reveal_round": key[1],
                            "recovered_from_pending_reveal": True,
                        },
                    )
            if key in pending:
                commit_finalized = False
                for run in run_refs:
                    persisted = await asyncio.to_thread(
                        self.subnet_data.advance_evaluation_run,
                        str(run["window_id"]),
                        str(run["round_id"]),
                        "COMMIT_FINALIZED",
                        {"commit_block": key[0], "recovered_from_pending_reveal": True},
                    )
                    commit_finalized = persisted or commit_finalized
                already_reported = bool(
                    report.get(
                        "commit_finalized_reported",
                        getattr(self.config.neuron, "wait_for_finalization", False),
                    )
                )
                if track_evaluation_run and commit_finalized and not already_reported:
                    await self._report_event(
                        "commit_finalized",
                        SimpleNamespace(round_id=str(report["round_id"])),
                        {
                            "weights": report["weights"],
                            "commit_reveal": True,
                            "commit_block": key[0],
                            "reveal_round": key[1],
                            "chain_state": "committed_pending_reveal",
                            "finalized_scope": "commit_extrinsic",
                        },
                    )
                    report["commit_finalized_reported"] = True
                remaining.append(report)
                continue
            if not await self._reported_weights_are_visible(report):
                remaining.append(report)
                continue
            # If the validator did not wait for finalization when submitting,
            # exact visibility now proves the earlier commit phase as well.
            if track_evaluation_run:
                all_visible = True
                for run in run_refs:
                    run_args = (str(run["window_id"]), str(run["round_id"]))
                    await asyncio.to_thread(
                        self.subnet_data.advance_evaluation_run,
                        *run_args,
                        "COMMIT_FINALIZED",
                        {"commit_block": key[0]},
                    )
                    reveal_persisted = await asyncio.to_thread(
                        self.subnet_data.advance_evaluation_run,
                        *run_args,
                        "REVEALED",
                        {"reveal_round": key[1]},
                    )
                    visible_persisted = (
                        await asyncio.to_thread(
                            self.subnet_data.advance_evaluation_run,
                            *run_args,
                            "VISIBLE_ON_CHAIN",
                            {"commit_block": key[0], "reveal_round": key[1]},
                        )
                        if reveal_persisted
                        else False
                    )
                    all_visible = all_visible and reveal_persisted and visible_persisted
                if not all_visible:
                    remaining.append(report)
                    continue
            event_payload = {
                "weights": report["weights"],
                "commit_reveal": True,
                "commit_block": key[0],
                "reveal_round": key[1],
                "chain_state": "weights_visible",
                "finalized_scope": "weights",
            }
            await self._report_event(
                "weights_revealed",
                SimpleNamespace(round_id=str(report["round_id"])),
                event_payload,
            )
            await self._report_event(
                "weights_finalized",
                SimpleNamespace(round_id=str(report["round_id"])),
                event_payload,
            )
            bt.logging.info(
                "Committed weights are now visible on-chain | "
                f"round={report['round_id']} commit_block={key[0]}"
            )
        self._save_pending_reveal_reports(remaining)

    async def _weight_submission_is_due(
        self, *, dirty: bool, last_submission_block: int = -1
    ) -> tuple[bool, dict]:
        hyperparameters = await asyncio.to_thread(
            self.subtensor.get_subnet_hyperparameters, self.config.netuid
        )
        rate_limit = int(getattr(hyperparameters, "weights_rate_limit", 0) or 0)
        chain_last_update = int(self.metagraph.last_update[self.uid])
        last_update = max(chain_last_update, int(last_submission_block))
        current_block = int(await asyncio.to_thread(self.subtensor.get_current_block))
        refresh_blocks = max(1, int(os.getenv("POKER44_WEIGHT_REFRESH_BLOCKS", "720")))
        required = rate_limit + 1 if dirty else max(refresh_blocks, rate_limit + 1)
        elapsed = current_block - last_update
        evidence = {
            "current_block": current_block,
            "last_update": last_update,
            "chain_last_update": chain_last_update,
            "weights_rate_limit": rate_limit,
            "refresh_blocks": refresh_blocks,
            "remaining_blocks": max(0, required - elapsed),
        }
        return elapsed >= required, evidence

    @staticmethod
    def _prepared_weights_from_rows(weights: list[dict]):
        uids = np.asarray([int(row["uid"]) for row in weights], dtype=np.int64)
        values = np.asarray([float(row["weight"]) for row in weights], dtype=np.float32)
        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=uids, weights=values
        )
        return uids, values, uint_uids, uint_weights

    async def _attempt_pending_weight_settlement(self) -> bool:
        state = self._load_weight_settlement()
        if not state or not state.get("weights"):
            return False
        dirty = bool(state.get("dirty"))
        due, cadence = await self._weight_submission_is_due(
            dirty=dirty,
            last_submission_block=int(state.get("last_submission_block", -1)),
        )
        if not due:
            bt.logging.info(
                "Weight submission deferred by chain cadence | "
                f"dirty={dirty} remaining_blocks={cadence['remaining_blocks']}"
            )
            return False
        commit_reveal = bool(
            await asyncio.to_thread(
                self.subtensor.commit_reveal_enabled, self.config.netuid
            )
        )
        if commit_reveal and await self._pending_weight_commits():
            bt.logging.info(
                "Weight submission deferred while an earlier reveal is pending"
            )
            return False

        weights = list(state["weights"])
        for row in weights:
            uid = int(row["uid"])
            expected_hotkey = str(row.get("hotkey") or "")
            current_hotkey = (
                str(self.metagraph.hotkeys[uid])
                if 0 <= uid < len(self.metagraph.hotkeys)
                else ""
            )
            if not expected_hotkey or current_hotkey != expected_hotkey:
                bt.logging.error(
                    "Weight target is stale because its hotkey is no longer "
                    f"registered at uid={uid}; preserving target without substitution"
                )
                return False
        prepared_weights = self._prepared_weights_from_rows(weights)
        commits_before = await self._pending_weight_commits() if commit_reveal else []
        submitted = bool(await asyncio.to_thread(self.set_weights, prepared_weights))
        if not submitted:
            bt.logging.warning("Pending weight submission failed; it will be retried")
            return False

        validation_round = SimpleNamespace(
            round_id=str(state["round_id"]),
            lease=SimpleNamespace(window_id=str(state["window_id"])),
        )
        evidence: dict = {"commit_reveal": commit_reveal, **cadence}
        if commit_reveal:
            previous_block = max(
                (item["commit_block"] for item in commits_before), default=-1
            )
            commits_after = await self._pending_weight_commits()
            new_commits = [
                item for item in commits_after if item["commit_block"] > previous_block
            ]
            if not new_commits:
                bt.logging.warning(
                    "Weight extrinsic succeeded but its commit is not visible yet; retrying later"
                )
                return False
            evidence.update(new_commits[-1])
            evidence["chain_state"] = "committed_pending_reveal"
            self._record_pending_reveal(
                validation_round,
                weights,
                evidence,
                track_evaluation_run=dirty,
                evaluation_runs=list(state.get("evaluation_runs") or []),
                commit_finalized_reported=bool(
                    self.config.neuron.wait_for_finalization
                ),
            )
        else:
            evidence["chain_state"] = "weights_visible"

        await self._report_event(
            "weights_submitted" if dirty else "weights_refreshed",
            validation_round,
            {"weights": weights, **evidence},
        )
        if dirty:
            evaluation_runs = list(state.get("evaluation_runs") or [])
            if not evaluation_runs:
                evaluation_runs = [
                    {
                        "window_id": state["window_id"],
                        "round_id": state["round_id"],
                    }
                ]
            submitted_results = [
                await asyncio.to_thread(
                    self.subnet_data.advance_evaluation_run,
                    str(run["window_id"]),
                    str(run["round_id"]),
                    "COMMIT_SUBMITTED",
                    evidence,
                )
                for run in evaluation_runs
            ]
            if not all(submitted_results):
                bt.logging.warning(
                    "Chain accepted weights but platform state was not persisted; retrying reconciliation"
                )
                return False
            if bool(self.config.neuron.wait_for_finalization):
                if commit_reveal:
                    for run in evaluation_runs:
                        await asyncio.to_thread(
                            self.subnet_data.advance_evaluation_run,
                            str(run["window_id"]),
                            str(run["round_id"]),
                            "COMMIT_FINALIZED",
                            evidence,
                        )
                    await self._report_event(
                        "commit_finalized",
                        validation_round,
                        {
                            "weights": weights,
                            **evidence,
                            "finalized_scope": "commit_extrinsic",
                        },
                    )
                else:
                    for run in evaluation_runs:
                        for status in (
                            "COMMIT_FINALIZED",
                            "REVEALED",
                            "VISIBLE_ON_CHAIN",
                        ):
                            persisted = await asyncio.to_thread(
                                self.subnet_data.advance_evaluation_run,
                                str(run["window_id"]),
                                str(run["round_id"]),
                                status,
                                evidence,
                            )
                            if not persisted:
                                raise RuntimeError(f"Could not persist {status} state")
                    await self._report_event(
                        "weights_finalized",
                        validation_round,
                        {
                            "weights": weights,
                            **evidence,
                            "finalized_scope": "weights",
                        },
                    )
        state["dirty"] = False
        state["last_submission_block"] = cadence["current_block"]
        self._save_weight_settlement(state)
        return True

    async def _run_settlement_phase(
        self, validation_round: ValidationRound, evaluations: list[MinerEvaluation]
    ) -> dict:
        score_rows = ranked_score_rows(evaluations)
        await self._report_event(
            "scores_computed", validation_round, {"scores": score_rows}
        )
        winner = winner_uid(evaluations)
        if winner is None:
            await self._report_event(
                "validation_round_failed",
                validation_round,
                {"error": "no_positive_finite_quality_score", "terminal": True},
            )
            return {"weights": [], "submitted": False, "scores": score_rows}
        raw_weights, allocation = await self._emission_target(winner)
        prepared_weights = await asyncio.to_thread(self.prepare_weights, raw_weights)
        weights = weight_rows(prepared_weights)
        for row in weights:
            row["hotkey"] = str(self.metagraph.hotkeys[int(row["uid"])])
            row["roles"] = [
                role
                for role, target in allocation.items()
                if int(target["uid"]) == int(row["uid"])
            ]
        await self._report_event(
            "weights_computed",
            validation_round,
            {"weights": weights, "allocation": allocation},
        )
        if weights:
            self._record_weight_settlement(validation_round, weights)
        submitted = await self._attempt_pending_weight_settlement()
        return {
            "weights": weights,
            "allocation": allocation,
            "submitted": submitted,
            "scores": score_rows,
        }
