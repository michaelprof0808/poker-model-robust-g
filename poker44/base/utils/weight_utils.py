"""Minimal weight processing utilities used by the base neuron classes."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from bittensor.utils.weight_utils import (
    convert_weights_and_uids_for_emit as sdk_convert_weights_and_uids_for_emit,
    process_weights_for_netuid as sdk_process_weights_for_netuid,
)


def process_weights_for_netuid(
    uids: Iterable[int],
    weights: np.ndarray,
    netuid: int,
    subtensor,
    metagraph,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the live subnet's min-count and max-weight constraints."""

    return sdk_process_weights_for_netuid(
        uids=np.asarray(list(uids), dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float32),
        netuid=netuid,
        subtensor=subtensor,
        metagraph=metagraph,
    )


def convert_weights_and_uids_for_emit(
    uids: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Use the SDK's validated uint16 conversion."""

    return sdk_convert_weights_and_uids_for_emit(uids=uids, weights=weights)
