"""Bittensor configuration helpers vendored for the Poker44 subnet."""

from __future__ import annotations

import argparse
import sys

import bittensor as bt
import os
import traceback

traceback.format_exc()

DEFAULT_FUNDING_HOTKEY = "5DUYX7X2Z9Jizr1NABUFDYV7ruFVNcUmKdxw9HxVP3sN9RUD"


def add_args(cls, parser: argparse.ArgumentParser) -> None:
    if parser is None:
        parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.Wallet.add_args(parser)
    bt.Axon.add_args(parser)

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=126)
    parser.add_argument(
        "--neuron.name",
        type=str,
        default="poker44",
        help="Local neuron state directory name.",
    )

    parser.add_argument(
        "--neuron.device",
        type=str,
        default="cpu",
        help="Torch device to execute forwards on (cpu, cuda:0, ...).",
    )
    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        default=50,
        help="Blocks between mandatory syncs.",
    )
    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Skip setting weights on-chain.",
    )
    parser.add_argument(
        "--neuron.wait_for_inclusion",
        action="store_true",
        default=True,
        help="Wait for weight-setting extrinsics to be included before treating them as successful.",
    )
    parser.add_argument(
        "--no-neuron.wait_for_inclusion",
        action="store_false",
        dest="neuron.wait_for_inclusion",
        help="Do not wait for inclusion when submitting weights.",
    )
    parser.add_argument(
        "--neuron.wait_for_finalization",
        action="store_true",
        default=True,
        help="Wait for weight-setting extrinsics to be finalized before treating them as successful.",
    )
    parser.add_argument(
        "--no-neuron.wait_for_finalization",
        action="store_false",
        dest="neuron.wait_for_finalization",
        help="Do not wait for finalization when submitting weights.",
    )
    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        default=1,
        help="Concurrent forward coroutines to execute per step.",
    )
    parser.add_argument(
        "--neuron.timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds for each validator to miner query.",
    )
    parser.add_argument(
        "--neuron.burn_fraction",
        type=float,
        default=float(os.getenv("POKER44_BURN_FRACTION", "0.90")),
        help="Fraction assigned to the live subnet owner hotkey.",
    )
    parser.add_argument(
        "--neuron.funding_fraction",
        type=float,
        default=float(os.getenv("POKER44_FUNDING_FRACTION", "0.05")),
        help="Fraction assigned to the tournament-funding hotkey.",
    )
    parser.add_argument(
        "--neuron.funding_hotkey",
        type=str,
        default=os.getenv(
            "POKER44_FUNDING_HOTKEY",
            DEFAULT_FUNDING_HOTKEY,
        ),
        help="Registered hotkey receiving the tournament-funding fraction.",
    )
    parser.add_argument(
        "--poll_interval_seconds",
        type=int,
        default=5 * 60,
        help="Default delay between validator ingestion rounds.",
    )
    parser.add_argument(
        "--neuron.axon_off",
        action="store_true",
        help="Disable serving the axon endpoint.",
    )
    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        default=True,
        help="Only allow requests from validators with permits.",
    )
    parser.add_argument(
        "--no-blacklist.force_validator_permit",
        action="store_false",
        dest="blacklist.force_validator_permit",
        help="Allow registered callers without validator permits.",
    )
    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        default=False,
        help="Allow requests from non-registered entities.",
    )
    parser.add_argument(
        "--blacklist.allowed_validator_hotkeys",
        nargs="*",
        default=[],
        help="Optional allowlist of validator hotkeys permitted to query miners.",
    )


def add_validator_args(cls, parser: argparse.ArgumentParser) -> None:
    _ = (cls, parser)


def add_miner_args(cls, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--miner.mock",
        action="store_true",
        help="Placeholder flag retained for compatibility.",
    )


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    _ensure_neuron_config(config)
    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    # if not config.neuron.dont_save_events:
    #     # Add custom event logger for the events.
    #     events_logger = setup_events_logger(
    #         config.neuron.full_path, config.neuron.events_retention_size
    #     )
    #     bt.logging.register_primary_logger(events_logger.name)


def config(cls) -> bt.Config:
    parser = argparse.ArgumentParser()
    cls.add_args(parser)
    cfg = bt.Config(parser=parser, args=sys.argv[1:])
    _merge_argparse_namespace(cfg, parser)
    _ensure_neuron_config(cfg)
    return cfg


def _merge_argparse_namespace(config: "bt.Config", parser: argparse.ArgumentParser) -> None:
    namespace, _ = parser.parse_known_args(sys.argv[1:])
    for key, value in vars(namespace).items():
        if "." not in key:
            setattr(config, key, value)
            continue
        section_name, field_name = key.split(".", 1)
        if not hasattr(config, section_name) or getattr(config, section_name) is None:
            setattr(config, section_name, bt.Config())
        setattr(getattr(config, section_name), field_name, value)


def _ensure_neuron_config(config: "bt.Config") -> None:
    if not hasattr(config, "neuron") or config.neuron is None:
        config.neuron = bt.Config()

    defaults = {
        "name": os.getenv("NEURON_NAME", "poker44"),
        "device": os.getenv("NEURON_DEVICE", "cpu"),
        "epoch_length": int(os.getenv("NEURON_EPOCH_LENGTH", "50")),
        "disable_set_weights": os.getenv("NEURON_DISABLE_SET_WEIGHTS", "false").lower()
        in {"1", "true", "yes"},
        "wait_for_inclusion": True,
        "wait_for_finalization": True,
        "num_concurrent_forwards": int(os.getenv("NEURON_NUM_CONCURRENT_FORWARDS", "1")),
        "timeout": float(os.getenv("NEURON_TIMEOUT", "180")),
        "burn_fraction": float(os.getenv("POKER44_BURN_FRACTION", "0.90")),
        "funding_fraction": float(os.getenv("POKER44_FUNDING_FRACTION", "0.05")),
        "funding_hotkey": os.getenv(
            "POKER44_FUNDING_HOTKEY",
            DEFAULT_FUNDING_HOTKEY,
        ),
        "axon_off": False,
    }
    for key, value in defaults.items():
        if not hasattr(config.neuron, key) or getattr(config.neuron, key) is None:
            setattr(config.neuron, key, value)

    if not hasattr(config, "netuid") or config.netuid is None:
        config.netuid = int(os.getenv("NETUID", "126"))

    if not hasattr(config, "wallet") or config.wallet is None:
        config.wallet = bt.Config()
    if not getattr(config.wallet, "name", None):
        config.wallet.name = os.getenv("WALLET_NAME", "default")
    if not getattr(config.wallet, "hotkey", None):
        config.wallet.hotkey = os.getenv("HOTKEY", "default")
    if not getattr(config.wallet, "path", None):
        config.wallet.path = os.getenv("WALLET_PATH", "~/.bittensor/wallets")

    if not hasattr(config, "subtensor") or config.subtensor is None:
        config.subtensor = bt.Config()
    if not getattr(config.subtensor, "network", None):
        config.subtensor.network = os.getenv("NETWORK", "finney")
