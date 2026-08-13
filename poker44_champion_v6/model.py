"""Owner-controlled Round-7 champion model exposed through ``POKER44_MODEL_FACTORY``.

Factory: ``poker44_champion_v6.model:create_model``.

The model is a deterministic, artifact-free, per-item robust scorer (see
``poker44_champion_v6.scoring``). There is no trained joblib payload, so there is
nothing to overfit to the 10-item preview and no code-bearing artifact to trust.
Inference is:

* metadata- AND context-invariant -- only the chronological action/size/all-in
  response channel feeds the score;
* item-permutation equivariant     -- each item is scored purely from its own
  content, independently of its siblings or their order (no transductive/batch
  dependence, unlike the research-only v5 path);
* fail-safe                        -- a malformed item scores the neutral 0.5 in
  isolation and can never corrupt a valid sibling; any unexpected error makes the
  whole request fall back to neutral rather than crashing the miner;
* perturbation-free                -- the deployed output is exactly the scoring
  probability, with no tie offset or coarse rounding that could reverse rankings.
"""

from __future__ import annotations

from typing import Any

from poker44.miner.config import MinerModelConfig
from poker44_champion_v6 import scoring

__all__ = ["ChampionV6RobustModel", "create_model"]

# Version sentinels that mean "use the module default" rather than an override.
_DEFAULT_VERSION_SENTINELS = {"", "reference-v2", "unknown"}


class ChampionV6RobustModel:
    """Robust, context-agnostic Poker44 micro-session detector."""

    mode = "robust"

    def __init__(self, config: MinerModelConfig):
        self.config = config
        configured = (getattr(config, "version", "") or "").strip()
        self.version = configured if configured not in _DEFAULT_VERSION_SENTINELS else scoring.VERSION

    def load(self) -> None:
        return None

    def predict(self, sessions: list[dict[str, Any]]) -> list[float]:
        if not sessions:
            return []
        try:
            return [float(scoring.score_session(session)) for session in sessions]
        except Exception:  # noqa: BLE001 -- whole-request neutral fallback, never crash
            return [float(scoring.NEUTRAL) for _ in sessions]


def create_model(config: MinerModelConfig) -> ChampionV6RobustModel:
    return ChampionV6RobustModel(config)
