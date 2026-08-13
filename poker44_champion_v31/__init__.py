"""Poker44 v3.1 supervised champion training/inference package.

This package is intentionally separate from the untouched reference model in
``poker44.miner.model``. It provides a leakage-safe feature extractor, an
integrity-checked benchmark ingestion pipeline, reward-aligned offline
evaluation, and a CPU inference model exposed through ``POKER44_MODEL_FACTORY``.
"""

from __future__ import annotations

__all__ = ["FEATURE_VERSION"]

FEATURE_VERSION = "champion-v31-f1"
