"""Owner-controlled Poker44 Round-7 robust champion (v6).

A deterministic, artifact-free, context-agnostic per-item scorer selected for
worst-case robustness across plausible policy worlds rather than fit to the small
public preview. See ``CHAMPION_V6_NOTES.md`` for the rationale and deployment.
"""

from __future__ import annotations

from poker44_champion_v6.scoring import VERSION

__all__ = ["VERSION"]
