import pytest
from pydantic import ValidationError

from poker44.protocol import MicroSessionDetectionSynapse, validate_micro_session_request


def micro_session():
    return {
        "schema_version": "4.1",
        "item_id": "item-1",
        "window_id": "window",
        "decisions": [
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
        ],
    }


def test_micro_session_is_the_only_signed_contract():
    synapse = MicroSessionDetectionSynapse(
        window_id="window",
        dataset_hash="a" * 64,
        query_id="query",
        items=[micro_session()],
    )
    validate_micro_session_request(synapse)
    assert synapse.contract_version == "microsession-v1"
    assert "items" in synapse.required_hash_fields
    assert "labels" not in type(synapse).model_fields


def test_contract_requires_sha256_dataset_hash():
    synapse = MicroSessionDetectionSynapse(
        window_id="window", dataset_hash="", query_id="query",
        items=[{"schema_version": "4.1"}],
    )
    with pytest.raises(ValueError, match="dataset_hash"):
        validate_micro_session_request(synapse)


def test_contract_rejects_old_versions_and_schemas():
    with pytest.raises(ValidationError):
        MicroSessionDetectionSynapse(contract_version="telemetry-v1")
    with pytest.raises(ValueError, match="schema 4.1"):
        validate_micro_session_request(MicroSessionDetectionSynapse(
            window_id="window", dataset_hash="b" * 64, query_id="query",
            items=[{"schema_version": "2"}],
        ))
