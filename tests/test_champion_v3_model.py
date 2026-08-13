from __future__ import annotations

import asyncio
import hashlib
import json

from poker44.miner.config import MinerModelConfig
from poker44.miner.loader import load_model
from poker44.miner.service import MinerInferenceService
from poker44.protocol import MicroSessionDetectionSynapse


def _item(action="call", size="half_pot", *, item_id="i1"):
    return {
        "schema_version": "4.1",
        "item_id": item_id,
        "window_id": "w1",
        "decisions": [
            {
                "decision_number": 1,
                "phase": "preflop",
                "position_group": "early",
                "pressure": "no_call",
                "action_type": "call",
                "size_bucket": "not_applicable",
                "is_all_in": False,
            },
            {
                "decision_number": 2,
                "phase": "flop",
                "position_group": "late",
                "pressure": "facing_bet",
                "action_type": action,
                "size_bucket": size,
                "is_all_in": action == "all_in" or size == "all_in",
            },
            {
                "decision_number": 3,
                "phase": "turn",
                "position_group": "blinds",
                "pressure": "no_call",
                "action_type": action,
                "size_bucket": size,
                "is_all_in": action == "all_in" or size == "all_in",
            },
            {
                "decision_number": 4,
                "phase": "river",
                "position_group": "late",
                "pressure": "facing_bet",
                "action_type": action,
                "size_bucket": size,
                "is_all_in": action == "all_in" or size == "all_in",
            },
        ],
    }


def test_factory_loads_and_scores_schema_v41_items():
    cfg = MinerModelConfig(
        factory="poker44_champion_v3.model:create_model",
        model_path=None,
        device="cpu",
        version="champion-v3-test",
        max_sessions_per_request=16,
    )
    model = load_model(cfg)
    scores = model.predict([_item("call", "half_pot"), _item("all_in", "all_in")])
    assert model.version == "champion-v3-test"
    assert len(scores) == 2
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[1] > scores[0]


def test_inference_service_matches_v3_synapse_contract():
    item = _item("raise", "pot")
    cfg = MinerModelConfig(
        factory="poker44_champion_v3.model:create_model",
        model_path=None,
        device="cpu",
        version="champion-v3-test",
        max_sessions_per_request=16,
    )
    model = load_model(cfg)
    service = MinerInferenceService(model, cfg)
    scores = asyncio.run(service.predict_micro_sessions([item]))
    synapse = MicroSessionDetectionSynapse(
        window_id="w1",
        dataset_hash=hashlib.sha256(json.dumps([item], sort_keys=True).encode()).hexdigest(),
        query_id="q1",
        items=[item],
    )
    synapse.risk_scores = scores
    synapse.predictions = [score >= 0.5 for score in scores]
    synapse.model_version = model.version
    assert synapse.contract_version == "microsession-v1"
    assert synapse.model_version == "champion-v3-test"
    assert len(synapse.risk_scores) == len(synapse.items) == 1


def test_model_never_raises_on_direct_malformed_input():
    cfg = MinerModelConfig(
        factory="poker44_champion_v3.model:create_model",
        model_path=None,
        device="cpu",
        version="champion-v3-test",
        max_sessions_per_request=16,
    )
    model = load_model(cfg)
    scores = model.predict([{}, {"decisions": "bad"}, 7])  # type: ignore[list-item]
    assert scores == [0.25, 0.25, 0.25]
