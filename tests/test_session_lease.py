import pytest

from poker44.platform.models import (
    SessionLease,
    ValidationRound,
    canonical_dataset_hash,
)


def decisions():
    return [
        {
            "decision_number": index + 1,
            "phase": "flop" if index == 0 else "preflop",
            "position_group": "late",
            "pressure": "no_call",
            "action_type": "check",
            "size_bucket": "not_applicable",
            "is_all_in": False,
        }
        for index in range(4)
    ]


def lease_payload():
    payload = {
        "lease_id": "lease-1",
        "window_id": "window-1",
        "dataset_hash": "",
        "expires_at": "2026-07-15T12:00:00Z",
        "sessions": [
            {
                "is_bot": False,
                "actor_group": "human-a",
                "payload": {
                    "schema_version": "4.1",
                    "item_id": "human-1",
                    "window_id": "window-1",
                    "decisions": decisions(),
                },
            },
            {
                "is_bot": True,
                "actor_group": "bot-a",
                "payload": {
                    "schema_version": "4.1",
                    "item_id": "bot-1",
                    "window_id": "window-1",
                    "decisions": decisions(),
                },
            },
        ],
    }
    payload["dataset_hash"] = canonical_dataset_hash(
        [item["payload"] for item in payload["sessions"]]
    )
    return payload


def test_cycle_keeps_labels_outside_miner_payload():
    validation_round = ValidationRound(SessionLease.from_payload(lease_payload()))

    assert validation_round.round_id == "window-1"
    assert validation_round.labels == [0, 1]
    assert all("is_bot" not in session for session in validation_round.miner_items)


def test_lease_rejects_label_leak_inside_payload():
    payload = lease_payload()
    payload["sessions"][0]["payload"]["label"] = "human"
    with pytest.raises(ValueError, match="leaks labels"):
        SessionLease.from_payload(payload)


def test_lease_rejects_schema_marker_without_telemetry_decisions():
    payload = lease_payload()
    del payload["sessions"][0]["payload"]["decisions"]
    payload["dataset_hash"] = canonical_dataset_hash(
        [item["payload"] for item in payload["sessions"]]
    )
    with pytest.raises(ValueError, match="micro-session contract"):
        SessionLease.from_payload(payload)


def test_lease_preserves_completion_state_for_idempotent_restarts():
    payload = lease_payload()
    payload["completed_at"] = "2026-07-15T12:01:00Z"

    lease = SessionLease.from_payload(payload)

    assert lease.completed_at == "2026-07-15T12:01:00Z"


def test_lease_rejects_payloads_that_do_not_match_the_common_dataset_hash():
    payload = lease_payload()
    payload["sessions"][0]["payload"]["item_id"] = "tampered"

    with pytest.raises(ValueError, match="dataset_hash"):
        SessionLease.from_payload(payload)


def test_microsession_lease_rejects_obsolete_v4_schema():
    decisions = [
        {
            "decision_number": index + 1,
            "phase": "flop" if index < 2 else "preflop",
            "position_group": "late",
            "pressure": "no_call",
            "action_type": "check",
            "size_bucket": "not_applicable",
            "is_all_in": False,
        }
        for index in range(6)
    ]
    sessions = [
        {
            "is_bot": label,
            "actor_group": actor,
            "payload": {
                "schema_version": "4",
                "item_id": f"item-{index}",
                "window_id": "window-v4",
                "decisions": decisions,
            },
        }
        for index, (label, actor) in enumerate(
            [(False, "human-a"), (False, "human-a"), (False, "human-b"), (True, "bot-a")]
        )
    ]
    payload = {
        "lease_id": "lease-v4",
        "window_id": "window-v4",
        "dataset_hash": canonical_dataset_hash(
            [item["payload"] for item in sessions]
        ),
        "expires_at": "2026-07-30T12:00:00Z",
        "fixture_only": True,
        "sessions": sessions,
    }

    with pytest.raises(ValueError, match="micro-session schema 4.1"):
        SessionLease.from_payload(payload)


def test_v41_lease_uses_item_ids_and_protocol_v3():
    decisions = [
        {
            "decision_number": index + 1,
            "phase": "flop" if index == 0 else "preflop",
            "position_group": "late",
            "pressure": "no_call",
            "action_type": "check",
            "size_bucket": "not_applicable",
            "is_all_in": False,
        }
        for index in range(4)
    ]
    sessions = [
        {
            "is_bot": label,
            "actor_group": actor,
            "payload": {
                "schema_version": "4.1",
                "item_id": f"item-v41-{index}",
                "window_id": "window-v41",
                "decisions": decisions,
            },
        }
        for index, (label, actor) in enumerate(
            [(False, "human-a"), (True, "bot-a")]
        )
    ]
    payload = {
        "lease_id": "lease-v41",
        "window_id": "window-v41",
        "dataset_hash": canonical_dataset_hash(
            [item["payload"] for item in sessions]
        ),
        "expires_at": "2026-07-30T12:00:00Z",
        "sessions": sessions,
    }

    validation_round = ValidationRound(SessionLease.from_payload(payload))

    assert validation_round.labels == [0, 1]
    assert all("actor_group" not in item for item in validation_round.miner_items)
