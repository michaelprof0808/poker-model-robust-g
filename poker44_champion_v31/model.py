"""Runtime champion model exposed through ``POKER44_MODEL_FACTORY``.

Factory: ``poker44_champion_v31.model:create_champion_model``.

The model loads a trained artifact only after its mandatory sibling manifest
passes an artifact hash check, verifies schema/feature compatibility, and
otherwise falls back to an independent, dependency-light safe scorer. Inference
is:

* metadata-invariant   -- only the four strategic decisions feed the features;
* item-permutation equivariant -- each item is scored from its own content;
* fail-safe            -- a malformed estimator output (wrong shape/length,
                          NaN/inf, out-of-range) or any inference exception makes
                          the WHOLE request fall back to the safe scorer;
* perturbation-free    -- the deployed output is exactly the (clipped)
                          probability offline evaluation scores; there is no tie
                          offset or rounding that could reverse rankings.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from poker44.miner.config import MinerModelConfig
from poker44_champion_v31 import FEATURE_VERSION, fallback, features
from poker44_champion_v31 import manifest as manifest_mod

__all__ = [
    "ARTIFACT_SCHEMA",
    "ChampionModel",
    "create_champion_model",
    "default_artifact_path",
    "deserialize_verified_artifact",
]

ARTIFACT_SCHEMA = "poker44-champion-v31-artifact-1"


def deserialize_verified_artifact(
    raw: bytes,
    manifest: dict[str, Any],
    *,
    allow_unpromoted: bool = False,
) -> dict[str, Any] | None:
    """Return an artifact only if the complete production load policy passes.

    Manifest provenance and exact bytes are checked before the code-bearing
    joblib payload is deserialized. This helper is shared by serving and LKG
    publication so their definition of "loadable" cannot drift.
    """
    try:
        if not manifest_mod.validate_artifact_manifest(
            raw, manifest, allow_unpromoted=allow_unpromoted
        ):
            return None

        import io

        import joblib

        artifact = joblib.load(io.BytesIO(raw))
        if not isinstance(artifact, dict):
            return None
        if artifact.get("artifact_schema") != ARTIFACT_SCHEMA:
            return None
        if artifact.get("feature_version") != FEATURE_VERSION:
            return None
        if list(artifact.get("feature_names") or []) != list(features.FEATURE_NAMES):
            return None
        estimator = artifact.get("estimator")
        if estimator is None or not callable(getattr(estimator, "predict_proba", None)):
            return None
        if manifest.get("model_version") != artifact.get("model_version"):
            return None
        return artifact
    except Exception:  # noqa: BLE001 -- untrusted/corrupt artifacts fail closed
        return None


def default_artifact_path() -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "champion.joblib"


class ChampionModel:
    version: str

    def __init__(self, config: MinerModelConfig, artifact_path: str | os.PathLike[str] | None = None):
        self.config = config
        configured_version = (config.version or "").strip()
        self._version_is_default = configured_version in {"", "reference-v2"}
        self.version = "champion-v31" if self._version_is_default else configured_version
        self._artifact_path = (
            Path(artifact_path)
            if artifact_path is not None
            else (Path(config.model_path) if config.model_path else default_artifact_path())
        )
        self._estimator = None
        self.mode = "fallback"
        # Transductive context-matched mode is RESEARCH-ONLY and intentionally NOT
        # wired into the deployed scoring path (see the module notes below). The
        # env flag is recorded for experiments but never alters production scores.
        self._transductive_requested = os.getenv(
            "POKER44_CHAMPION_TRANSDUCTIVE", "0"
        ).strip() in {"1", "true", "yes"}
        # Production default loads only PROMOTED artifacts. A research/candidate
        # smoke may opt in explicitly; an accidental candidate path alone falls
        # back to the safe scorer.
        self._allow_unpromoted = os.getenv(
            "POKER44_CHAMPION_ALLOW_UNPROMOTED", "0"
        ).strip() in {"1", "true", "yes"}

    # -- loading -----------------------------------------------------------
    def load(self) -> None:
        artifact = self._try_load_artifact()
        if artifact is None:
            self.mode = "fallback"
            self._estimator = None
            return
        self._estimator = artifact["estimator"]
        self.mode = "champion"
        if artifact.get("model_version") and self._version_is_default:
            self.version = str(artifact["model_version"])

    def _try_load_artifact(self) -> dict[str, Any] | None:
        path = self._artifact_path
        if not path:
            return None
        # Try the primary pair; if it is missing/interrupted/mismatched, fall back
        # to the verified last-known-good sidecar (crash-safe rollback).
        primary_manifest = Path(str(path) + ".manifest.json")
        artifact = self._load_pair(Path(path), primary_manifest)
        if artifact is not None:
            return artifact
        lkg = Path(str(path) + ".lkg")
        lkg_manifest = Path(str(path) + ".lkg.manifest.json")
        return self._load_pair(lkg, lkg_manifest)

    def _load_pair(self, path: Path, manifest_path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists() or not manifest_path.exists():
                return None
            raw = path.read_bytes()
            manifest = json.loads(manifest_path.read_text())
            return deserialize_verified_artifact(
                raw, manifest, allow_unpromoted=self._allow_unpromoted
            )
        except Exception:  # noqa: BLE001 -- corrupt artifacts must fail closed
            # Any failure (corrupt bytes, incompatible pickle, IO error) is a
            # safe-fallback trigger -- never crash the miner at load time.
            return None

    # -- inference ---------------------------------------------------------
    def _champion_scores(self, matrix, n: int):
        """Return validated 1-D finite [0, 1] champion scores, or ``None`` to
        signal that the whole request must fall back."""
        import numpy as np

        if self.mode != "champion" or self._estimator is None:
            return None
        try:
            raw = self._estimator.predict_proba(matrix)
            # Conversion itself can fail (non-numeric strings / object arrays);
            # keep it inside the guard so it triggers whole-request fallback too.
            raw = np.asarray(raw, dtype=float)
            # Output must be exactly one-dimensional and aligned in length.
            if raw.ndim != 1 or raw.shape[0] != n:
                return None
            # NaN / inf are never acceptable; np.clip would preserve NaN.
            if not np.all(np.isfinite(raw)):
                return None
            # Probabilities must already sit within [0, 1] (tiny float excursions
            # are clipped; anything materially out of range is treated as malformed).
            if float(raw.min()) < -1e-6 or float(raw.max()) > 1.0 + 1e-6:
                return None
            return np.clip(raw, 0.0, 1.0)
        except Exception:  # noqa: BLE001 -- any inference/conversion failure -> fallback
            return None

    def predict(self, sessions: list[dict[str, Any]]) -> list[float]:
        if not sessions:
            return []
        import numpy as np

        n = len(sessions)
        matrix = features.vectorize(sessions)
        scores = self._champion_scores(matrix, n)
        if scores is None:
            # Whole-request fallback: never mix champion and fallback scores, and
            # never fail the request on a malformed model output.
            scores = np.clip(np.asarray(fallback.heuristic_scores(sessions), dtype=float), 0.0, 1.0)
        # No tie offset / rounding: the deployed transformation is the identity on
        # the (clipped) probabilities that offline evaluation scores, so ranking is
        # never perturbed or reversed.
        return [float(s) for s in scores]


# -- transductive context-matched mode (RESEARCH-ONLY, NOT DEPLOYED) ------
# These helpers are retained for offline research only. Even signature-group
# sizes do NOT prove the request satisfies the private class-matching invariant,
# and the naive adjustment below is neither order-independent (no midranks) nor
# exactly group-centered, so it could reorder scores. It is therefore never
# invoked by ``predict``; deployed scoring is always independent per item.
def _signature(payload: dict[str, Any]) -> tuple:
    decisions = payload.get("decisions") or []
    pairs = sorted(
        (str(d.get("phase")), str(d.get("pressure")))
        for d in decisions
        if isinstance(d, dict)
    )
    return tuple(pairs)


def _transductive_applicable(sessions: Sequence[dict[str, Any]]) -> bool:
    """Activate only when every phase|pressure signature group is even and >= 2."""
    from collections import Counter

    counts = Counter(_signature(p) for p in sessions)
    if not counts:
        return False
    return all(n >= 2 and n % 2 == 0 for n in counts.values())


def _transductive_adjust(raw, sessions: Sequence[dict[str, Any]]):
    """Context-matched ranking: sharpen separation within matched signatures."""
    import numpy as np

    groups: dict[tuple, list[int]] = {}
    for idx, payload in enumerate(sessions):
        groups.setdefault(_signature(payload), []).append(idx)
    adjusted = np.array(raw, dtype=float)
    beta = 0.5
    for indices in groups.values():
        vals = adjusted[indices]
        centered = vals + beta * (vals - float(np.mean(vals)))
        adjusted[indices] = np.clip(centered, 0.0, 1.0)
    return adjusted


def create_champion_model(config: MinerModelConfig) -> ChampionModel:
    return ChampionModel(config)
