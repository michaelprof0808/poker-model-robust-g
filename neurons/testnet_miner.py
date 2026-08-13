"""Testnet Axon entrypoint for hotkeys whose endpoints are already on-chain."""

import os
import signal
import threading
from types import SimpleNamespace
from typing import Tuple

import bittensor as bt

from poker44.miner.config import MinerModelConfig
from poker44.miner.loader import load_model
from poker44.miner.service import MinerInferenceService
from poker44.protocol import (
    MicroSessionDetectionSynapse,
    validate_micro_session_request,
)
from poker44.utils.encrypted_endpoints import (
    EndpointProtectionError,
    enable_miner_endpoint_protection,
)


class TestnetMiner:
    """Serve a model without coupling availability to public-RPC bootstrap."""

    def __init__(self) -> None:
        port = int(os.environ["POKER44_AXON_PORT"])
        self.allowed_validator = os.environ["POKER44_VALIDATOR_HOTKEY"]
        self.wallet = bt.Wallet(
            name=os.environ.get("POKER44_WALLET_NAME", "poker44-testnet"),
            hotkey=os.environ["POKER44_WALLET_HOTKEY"],
            path=os.environ.get("POKER44_WALLET_PATH", "/wallets"),
        )
        self.model_config = MinerModelConfig.from_env()
        self.model = load_model(self.model_config)
        self.inference = MinerInferenceService(self.model, self.model_config)
        self.axon = bt.Axon(
            wallet=self.wallet,
            ip="0.0.0.0",
            port=port,
            external_ip=os.environ["POKER44_AXON_EXTERNAL_IP"],
            external_port=port,
        )
        self.axon.attach(
            forward_fn=self.forward_micro_sessions,
            blacklist_fn=self.blacklist_micro_sessions,
            priority_fn=self.priority_micro_sessions,
        )
        self.stop_event = threading.Event()

    async def forward_micro_sessions(
        self, synapse: MicroSessionDetectionSynapse
    ) -> MicroSessionDetectionSynapse:
        validate_micro_session_request(synapse)
        scores = await self.inference.predict_micro_sessions(synapse.items)
        synapse.risk_scores = scores
        synapse.predictions = [score >= 0.5 for score in scores]
        synapse.model_version = self.model.version
        bt.logging.info(
            f"Scored {len(scores)} micro-session items for window={synapse.window_id}"
        )
        return synapse

    def _blacklist(
        self, synapse: MicroSessionDetectionSynapse
    ) -> Tuple[bool, str]:
        caller = getattr(synapse.dendrite, "hotkey", None)
        if caller != self.allowed_validator:
            return True, "Hotkey not in validator allowlist"
        return False, "Whitelisted validator hotkey"

    async def blacklist_micro_sessions(
        self, synapse: MicroSessionDetectionSynapse
    ) -> Tuple[bool, str]:
        return self._blacklist(synapse)

    async def priority_micro_sessions(self, synapse: MicroSessionDetectionSynapse) -> float:
        return 1.0

    def run(self) -> None:
        bt.logging.info(
            "Starting pre-registered testnet Axon | "
            f"hotkey={self.wallet.hotkey.ss58_address} "
            f"model={self.model.version} port={self.axon.port}"
        )
        serve_on_chain = (
            os.environ.get("POKER44_SERVE_AXON_ON_CHAIN", "false").lower() == "true"
        )
        protection_enabled = (
            os.environ.get("POKER44_ENCRYPTED_AXON_ENABLED", "false").lower()
            == "true"
        )
        if serve_on_chain or protection_enabled:
            self.subtensor = bt.Subtensor(
                network=os.environ.get(
                    "POKER44_SUBTENSOR_ENDPOINT",
                    "wss://test.finney.opentensor.ai:443",
                )
            )
            netuid = int(os.environ.get("POKER44_NETUID", "492"))
            self.config = SimpleNamespace(netuid=netuid)
            if protection_enabled:
                try:
                    protected = enable_miner_endpoint_protection(self)
                except EndpointProtectionError as exc:
                    protected = False
                    bt.logging.error(
                        "Encrypted Axon endpoint was not enabled; preserving the "
                        f"public endpoint: {exc}"
                    )
                if protected:
                    bt.logging.success(
                        "Encrypted Axon endpoint commitment verified; publishing "
                        "the masked testnet endpoint."
                    )
            response = self.subtensor.serve_axon(
                netuid=netuid,
                axon=self.axon,
            )
            if getattr(response, "success", bool(response)) is False:
                raise RuntimeError(
                    f"Could not publish testnet Axon: {getattr(response, 'message', response)}"
                )
            bt.logging.info("Testnet Axon endpoint published on chain")
        self.axon.start()
        self.stop_event.wait()
        self.axon.stop()


if __name__ == "__main__":
    bt.logging.set_info()
    miner = TestnetMiner()

    def stop(*_args) -> None:
        miner.stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    miner.run()
