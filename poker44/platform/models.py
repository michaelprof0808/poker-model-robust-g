"""Typed session lease models kept separate from HTTP transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from poker44.contracts import find_forbidden, validate_v4_micro_session


def canonical_dataset_hash(sessions: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        sessions,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LeasedSession:
    payload: dict[str, Any]
    is_bot: bool
    actor_group: str | None = None


@dataclass(frozen=True)
class SessionLease:
    lease_id: str
    window_id: str
    dataset_hash: str
    expires_at: str
    sessions: list[LeasedSession]
    completed_at: str | None = None
    fixture_only: bool = False
    audit: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, value: Any) -> "SessionLease":
        if not isinstance(value, dict):
            raise ValueError("Session lease response must be an object")
        expected_schema = "4.1"
        id_field = "item_id"
        sessions_raw = value.get("sessions")
        if not isinstance(sessions_raw, list) or not sessions_raw:
            raise ValueError("Session lease contains no sessions")
        sessions: list[LeasedSession] = []
        seen_item_ids: set[str] = set()
        for index, item in enumerate(sessions_raw):
            if not isinstance(item, dict) or not isinstance(item.get("payload"), dict):
                raise ValueError(f"sessions[{index}] has an invalid payload")
            if not isinstance(item.get("is_bot"), bool):
                raise ValueError(f"sessions[{index}] has no boolean ground truth")
            miner_payload = dict(item["payload"])
            leaked = find_forbidden(miner_payload, f"sessions[{index}].payload")
            if leaked:
                raise ValueError(
                    f"sessions[{index}].payload leaks labels: {sorted(leaked)}"
                )
            schema_version = str(miner_payload.get("schema_version") or "")
            if schema_version != expected_schema:
                raise ValueError(
                    f"sessions[{index}] must use micro-session schema {expected_schema}"
                )
            validate_v4_micro_session(miner_payload, index)
            item_id = str(miner_payload.get(id_field) or "").strip()
            if not item_id or item_id in seen_item_ids:
                raise ValueError(
                    f"sessions[{index}] has a missing or duplicate {id_field}"
                )
            seen_item_ids.add(item_id)
            actor_group = str(item.get("actor_group") or "").strip() or None
            if actor_group is None:
                raise ValueError(f"sessions[{index}] has no private actor_group")
            sessions.append(
                LeasedSession(
                    payload=miner_payload,
                    is_bot=item["is_bot"],
                    actor_group=actor_group,
                )
            )
        lease_id = str(value.get("lease_id") or "").strip()
        window_id = str(value.get("window_id") or "").strip()
        if not lease_id or not window_id:
            raise ValueError("Session lease requires lease_id and window_id")
        if any(
            str(session.payload.get("window_id") or "") != window_id
            for session in sessions
        ):
            raise ValueError("Session payload window_id does not match its lease")
        dataset_hash = str(value.get("dataset_hash") or "").strip()
        if not dataset_hash:
            raise ValueError("Session lease requires a dataset_hash")
        computed_hash = canonical_dataset_hash(
            [session.payload for session in sessions]
        )
        if computed_hash != dataset_hash:
            raise ValueError(
                "Session lease dataset_hash does not match miner-visible payloads"
            )
        audit_raw = value.get("audit")
        audit_raw = audit_raw if isinstance(audit_raw, dict) else {}
        rejection_counts_raw = audit_raw.get("rejectionCounts")
        rejection_counts = (
            {
                str(key): int(count)
                for key, count in rejection_counts_raw.items()
                if isinstance(count, int) and count >= 0
            }
            if isinstance(rejection_counts_raw, dict)
            else {}
        )
        return cls(
            lease_id=lease_id,
            window_id=window_id,
            dataset_hash=dataset_hash,
            expires_at=str(value.get("expires_at") or ""),
            sessions=sessions,
            completed_at=(
                str(value["completed_at"]) if value.get("completed_at") else None
            ),
            fixture_only=bool(value.get("fixture_only", False)),
            audit={
                key: raw
                for key in (
                    "itemCount",
                    "humanItems",
                    "botItems",
                    "humanActors",
                    "botActors",
                    "tournaments",
                    "botFamilies",
                    "decisionsPerItem",
                    "postflopDecisionsPerItem",
                    "positionDistributionMaxGap",
                    "singleTournamentProduction",
                )
                if isinstance((raw := audit_raw.get(key)), (int, float, bool))
            }
            | {"rejectionCounts": rejection_counts},
        )


@dataclass
class ValidationRound:
    lease: SessionLease
    round_id: str = ""
    resume_state: str = "ACQUIRED"
    resume_evidence: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.round_id:
            self.round_id = self.lease.window_id

    @property
    def labels(self) -> list[int]:
        return [1 if session.is_bot else 0 for session in self.lease.sessions]

    @property
    def miner_items(self) -> list[dict[str, Any]]:
        return [session.payload for session in self.lease.sessions]

    @property
    def sample_weights(self) -> list[float]:
        """Give every independent actor equal weight within its private class."""

        groups: dict[int, dict[str, list[int]]] = {}
        for index, session in enumerate(self.lease.sessions):
            label = 1 if session.is_bot else 0
            actor = session.actor_group or f"item:{index}"
            groups.setdefault(label, {}).setdefault(actor, []).append(index)
        class_mass = 1.0 / len(groups)
        weights = [0.0] * len(self.lease.sessions)
        for actors in groups.values():
            actor_mass = class_mass / len(actors)
            for indices in actors.values():
                item_mass = actor_mass / len(indices)
                for index in indices:
                    weights[index] = item_mass
        return weights
