"""Probe controlled testnet Axons with a private labelled JSONL stream.

The corpus is read from stdin and never written to disk. Output contains only
aggregate miner metrics, so production labels and actor identities stay private.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import replace
from typing import Any

import bittensor as bt

from poker44.protocol import MicroSessionDetectionSynapse
from poker44.validator.evaluation.reward import reward


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=492)
    parser.add_argument("--network", default="test")
    parser.add_argument("--wallet-path", default="/wallets")
    parser.add_argument("--wallet-name", default="poker44-testnet")
    parser.add_argument("--wallet-hotkey", default="validator")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def load_stream() -> tuple[list[dict[str, Any]], list[int]]:
    items: list[dict[str, Any]] = []
    labels: list[int] = []
    for line_number, line in enumerate(sys.stdin, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        label = int(row.get("label", -1))
        payload = row.get("payload")
        if label not in {0, 1} or not isinstance(payload, dict):
            raise ValueError(f"invalid corpus row at line {line_number}")
        labels.append(label)
        items.append(payload)
    if not items or set(labels) != {0, 1}:
        raise ValueError("a non-empty mixed-class corpus is required")
    return items, labels


async def probe(args: argparse.Namespace) -> dict[str, Any]:
    items, labels = load_stream()
    overrides = json.loads(os.environ.get("POKER44_AXON_OVERRIDES", "{}"))
    if not overrides:
        raise ValueError("POKER44_AXON_OVERRIDES is required")
    wallet = bt.Wallet(
        path=args.wallet_path,
        name=args.wallet_name,
        hotkey=args.wallet_hotkey,
    )
    subtensor = bt.Subtensor(network=args.network)
    metagraph = subtensor.metagraph(args.netuid)
    uids: list[int] = []
    axons: list[Any] = []
    for raw_uid, endpoint in sorted(overrides.items(), key=lambda item: int(item[0])):
        uid = int(raw_uid)
        host, raw_port = str(endpoint).rsplit(":", 1)
        uids.append(uid)
        axons.append(replace(metagraph.axons[uid], ip=host, port=int(raw_port)))
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    synapse = MicroSessionDetectionSynapse(
        window_id="private-testnet-probe-v41",
        dataset_hash=digest,
        query_id=f"private-testnet-probe-{digest[:16]}",
        items=items,
    )
    dendrite = bt.Dendrite(wallet=wallet)
    try:
        responses = await dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=False,
            timeout=args.timeout,
        )
    finally:
        await dendrite.aclose_session()
    miners: list[dict[str, Any]] = []
    for uid, response in zip(uids, responses):
        scores = getattr(response, "risk_scores", None)
        status = getattr(response, "dendrite", None)
        row: dict[str, Any] = {
            "uid": uid,
            "hotkey": str(metagraph.hotkeys[uid]),
            "status_code": getattr(status, "status_code", None),
            "status_message": getattr(status, "status_message", None),
            "model_version": getattr(response, "model_version", None),
        }
        if isinstance(scores, list) and len(scores) == len(labels):
            row["metrics"] = reward([float(value) for value in scores], labels).to_dict()
        else:
            row["error"] = f"expected {len(labels)} scores, received {len(scores or [])}"
        miners.append(row)
    ranked = sorted(
        miners,
        key=lambda item: (-float(item.get("metrics", {}).get("reward", 0.0)), item["uid"]),
    )
    winner_uid = ranked[0]["uid"] if ranked and ranked[0].get("metrics") else None
    return {
        "network": args.network,
        "netuid": args.netuid,
        "items": len(items),
        "class_counts": {"human": labels.count(0), "bot": labels.count(1)},
        "queried_miners": len(miners),
        "successful_miners": sum("metrics" in item for item in miners),
        "winner_uid": winner_uid,
        "miners": miners,
        "raw_payloads_persisted": False,
    }


def main() -> int:
    print(json.dumps(asyncio.run(probe(arguments())), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
