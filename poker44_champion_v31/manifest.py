"""Machine-readable artifact manifest for the Poker44 v3.1 champion.

The manifest records everything needed to reproduce and trust an artifact:
release ids/versions/hashes and item/class counts, the upstream subnet commit
and model-code identity, the feature version, the seed and hyperparameters, the
calibration strategy, exact dependency versions, offline metrics, the promotion
decision, and the SHA-256 of the serialized artifact bytes.

Trust boundary: joblib artifacts are code-bearing. This manifest binds an
artifact's bytes to its provenance and lets a loader reject a tampered/corrupt
artifact via the SHA-256 check, but it is NOT a cryptographic signature. Only
artifacts produced by this trusted local training pipeline (the trusted
producer) may be loaded; a manifest an attacker can also rewrite is not
authenticity.
"""

from __future__ import annotations

import hashlib
import platform
import re
from datetime import datetime, timedelta
from importlib import metadata as importlib_metadata
from typing import Any

from poker44_champion_v31 import FEATURE_VERSION

__all__ = [
    "MANIFEST_SCHEMA",
    "build_manifest",
    "dependency_versions",
    "validate_artifact_manifest",
    "verify_artifact",
]

MANIFEST_SCHEMA = "poker44-champion-v31-manifest-1"
MODEL_CODE = "poker44_champion_v31"
TRUSTED_PRODUCER = (
    "poker44_champion_v31 local training pipeline; SHA-256 detects "
    "corruption but is not a signature -- load only self-produced artifacts"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_REV_RE = re.compile(r"^src-[0-9a-f]{16}$")

_DEP_PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib")


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for pkg in _DEP_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover
            versions[pkg] = "unknown"
    return versions


def build_manifest(
    *,
    artifact_bytes: bytes,
    source_release_hashes: list[str],
    upstream_commit: str,
    metrics: dict[str, Any],
    promotion: dict[str, Any],
    model_version: str,
    feature_version: str = FEATURE_VERSION,
    created_at: str | None = None,
    release_ids: list[str] | None = None,
    release_versions: list[str] | None = None,
    item_counts: list[int] | None = None,
    class_counts: list[dict[int, int]] | None = None,
    seed: int | None = None,
    hyperparameters: dict[str, Any] | None = None,
    calibration: str | None = None,
    model_code: str = "poker44_champion_v31",
    model_code_revision: str | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "manifest_schema": MANIFEST_SCHEMA,
        "model_version": model_version,
        "model_code": model_code,
        "model_code_revision": model_code_revision,
        "feature_version": feature_version,
        "upstream_commit": upstream_commit,
        "source_release_hashes": list(source_release_hashes),
        "release_ids": list(release_ids or []),
        "release_versions": list(release_versions or []),
        "item_counts": list(item_counts or []),
        "class_counts": [dict(c) for c in (class_counts or [])],
        "seed": seed,
        "hyperparameters": hyperparameters or {},
        "calibration": calibration,
        "dependency_versions": dependency_versions(),
        "metrics": metrics,
        "promotion": promotion,
        "trusted_producer": TRUSTED_PRODUCER,
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    }
    if created_at is not None:
        manifest["created_at"] = created_at
    return manifest


def verify_artifact(artifact_bytes: bytes, manifest: dict[str, Any]) -> bool:
    expected = str(manifest.get("artifact_sha256") or "")
    return bool(expected) and hashlib.sha256(artifact_bytes).hexdigest() == expected


def validate_artifact_manifest(
    artifact_bytes: bytes,
    manifest: dict[str, Any],
    *,
    allow_unpromoted: bool = False,
) -> bool:
    """Validate all trust/deployability fields before joblib deserialization."""
    try:
        if not isinstance(manifest, dict) or manifest.get("manifest_schema") != MANIFEST_SCHEMA:
            return False
        if manifest.get("model_code") != MODEL_CODE:
            return False
        if manifest.get("trusted_producer") != TRUSTED_PRODUCER:
            return False
        if not _SOURCE_REV_RE.fullmatch(manifest.get("model_code_revision", "")):
            return False
        if not _COMMIT_RE.fullmatch(manifest.get("upstream_commit", "")):
            return False
        if manifest.get("feature_version") != FEATURE_VERSION:
            return False
        if not isinstance(manifest.get("model_version"), str) or not manifest["model_version"].strip():
            return False
        if not _SHA256_RE.fullmatch(manifest.get("artifact_sha256", "")):
            return False
        if not verify_artifact(artifact_bytes, manifest):
            return False

        created_at = manifest.get("created_at")
        if not isinstance(created_at, str) or not created_at.endswith("Z"):
            return False
        created = datetime.fromisoformat(created_at[:-1] + "+00:00")
        if created.tzinfo is None or created.utcoffset() != timedelta(0):
            return False

        hashes = manifest.get("source_release_hashes")
        ids = manifest.get("release_ids")
        versions = manifest.get("release_versions")
        item_counts = manifest.get("item_counts")
        class_counts = manifest.get("class_counts")
        sequences = (hashes, ids, versions, item_counts, class_counts)
        if not all(isinstance(value, list) for value in sequences):
            return False
        n = len(hashes)
        if n < 1 or any(len(value) != n for value in sequences):
            return False
        if len(set(hashes)) != n or not all(_SHA256_RE.fullmatch(value or "") for value in hashes):
            return False
        if len(set(ids)) != n or not all(isinstance(value, str) and value.strip() for value in ids):
            return False
        if not all(isinstance(value, str) and value.strip() for value in versions):
            return False
        if not all(type(value) is int and value > 0 for value in item_counts):
            return False
        for total, counts in zip(item_counts, class_counts):
            if not isinstance(counts, dict) or set(map(str, counts)) != {"0", "1"}:
                return False
            human = counts.get(0, counts.get("0"))
            bot = counts.get(1, counts.get("1"))
            if type(human) is not int or type(bot) is not int or human < 1 or bot < 1:
                return False
            if human + bot != total:
                return False

        promotion = manifest.get("promotion")
        if not isinstance(promotion, dict) or type(promotion.get("promote")) is not bool:
            return False
        if not allow_unpromoted and promotion["promote"] is not True:
            return False
        if not isinstance(manifest.get("metrics"), dict):
            return False
        return isinstance(manifest.get("dependency_versions"), dict)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
