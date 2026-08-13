"""Train, evaluate and persist the Poker44 v3.1 champion artifact.

Behaviour by release count:

* 0 releases  -> caller should not train; status is ``unavailable``.
* 1 release   -> a provisional artifact may be fit for experimentation, but
                 champion promotion is blocked (no release-held-out possible).
* >= 2        -> release-held-out (LORO) evaluation drives the promotion
                 decision; the persisted artifact is fit on ALL releases.

The persisted artifact is always written together with a machine-readable
manifest (``<artifact>.manifest.json``) recording full provenance.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from poker44_champion_v31 import FEATURE_VERSION, evaluation, features
from poker44_champion_v31 import manifest as manifest_mod
from poker44_champion_v31.dataset import ParsedRelease
from poker44_champion_v31.estimator import ChampionEstimator
from poker44_champion_v31.model import ARTIFACT_SCHEMA, deserialize_verified_artifact

__all__ = [
    "build_artifact_bytes",
    "candidate_artifact_path",
    "source_revision",
    "train_and_save",
]

UPSTREAM_COMMIT_DEFAULT = "95eb7920cdfcc583da65a000a6ab155c101e1719"


def source_revision() -> str:
    """Stable content revision of the champion package source (not a git commit,
    which would be meaningless for uncommitted/staged code)."""
    pkg_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for py in sorted(pkg_dir.glob("*.py")):
        digest.update(py.name.encode("utf-8"))
        digest.update(py.read_bytes())
    return "src-" + digest.hexdigest()[:16]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def candidate_artifact_path(artifact_path: str | Path) -> Path:
    """Distinct path for a non-promoted (provisional/rejected) candidate so it
    can never masquerade as, or overwrite, the requested champion artifact."""
    p = Path(artifact_path)
    return p.with_name(f"{p.stem}.candidate{p.suffix}")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_pair(artifact_path: Path, artifact_bytes: bytes, manifest: dict[str, Any]) -> Path:
    """Write an artifact + sibling manifest pair (used for CANDIDATE writes).

    Each file is staged in a temp file then swapped in. A crash between the two
    swaps leaves a hash-mismatched pair that the mandatory manifest check
    rejects; candidates never carry last-known-good semantics.
    """
    manifest_path = Path(str(artifact_path) + ".manifest.json")
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write(artifact_path, artifact_bytes)
    _atomic_write(manifest_path, manifest_bytes)
    return manifest_path


def lkg_paths(artifact_path: str | Path) -> tuple[Path, Path]:
    """Return the (artifact, manifest) last-known-good sidecar paths."""
    p = Path(artifact_path)
    return Path(str(p) + ".lkg"), Path(str(p) + ".lkg.manifest.json")


def _publish_pair(
    artifact_path: Path,
    artifact_bytes: bytes,
    manifest: dict[str, Any],
    *,
    _fault_after_artifact=None,
) -> Path:
    """Crash-safe champion publication with last-known-good (LKG) rollback.

    Publishers are serialized by an exclusive fcntl lock. Before overwriting the
    primary pair, the current *verified* primary is copied to ``.lkg`` sidecars.
    The new artifact is swapped in before its manifest, so a crash between the
    two leaves a hash-mismatched primary that the loader detects and skips in
    favour of the verified LKG -- rollback, not a falsely "atomic" pair.
    """
    import fcntl

    artifact_path = Path(artifact_path)
    manifest_path = Path(str(artifact_path) + ".manifest.json")
    lkg_artifact, lkg_manifest = lkg_paths(artifact_path)
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(artifact_path) + ".publish.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if artifact_path.exists() and manifest_path.exists():
                current_artifact = artifact_path.read_bytes()
                current_manifest_bytes = manifest_path.read_bytes()
                try:
                    current_manifest = json.loads(current_manifest_bytes)
                except ValueError:
                    current_manifest = {}
                # Preserve only a pair that the production loader would accept.
                if deserialize_verified_artifact(
                    current_artifact, current_manifest, allow_unpromoted=False
                ) is not None:
                    _atomic_write(lkg_artifact, current_artifact)
                    _atomic_write(lkg_manifest, current_manifest_bytes)
            _atomic_write(artifact_path, artifact_bytes)
            if _fault_after_artifact is not None:
                _fault_after_artifact()
            _atomic_write(manifest_path, manifest_bytes)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return manifest_path


def _fit_all(releases: Sequence[ParsedRelease]) -> ChampionEstimator:
    X = np.vstack([features.vectorize(r.payloads) for r in releases])
    y = np.concatenate([np.asarray(r.labels, dtype=int) for r in releases])
    # One group per release so calibration is release-grouped out-of-fold when
    # two or more releases are available.
    groups = np.concatenate(
        [np.full(r.item_count, i, dtype=int) for i, r in enumerate(releases)]
    )
    return ChampionEstimator().fit(X, y, groups=groups)


def build_artifact_bytes(estimator: ChampionEstimator, model_version: str) -> bytes:
    import joblib

    buffer = io.BytesIO()
    joblib.dump(
        {
            "artifact_schema": ARTIFACT_SCHEMA,
            "model_version": model_version,
            "feature_version": FEATURE_VERSION,
            "feature_names": list(features.FEATURE_NAMES),
            "estimator": estimator,
        },
        buffer,
    )
    return buffer.getvalue()


def train_and_save(
    releases: Sequence[ParsedRelease],
    artifact_path: str | Path,
    *,
    upstream_commit: str = UPSTREAM_COMMIT_DEFAULT,
    model_version: str = "champion-v31",
    created_at: str | None = None,
    model_code_revision: str | None = None,
) -> dict[str, Any]:
    if not releases:
        return {
            "status": "unavailable",
            "promote": False,
            "reason": "no releases available; nothing trained",
            "artifact_written": False,
        }

    n = len(releases)
    status_override = None
    if n >= 2:
        try:
            eval_result = evaluation.release_held_out_eval(releases)
            promotion = evaluation.promotion_decision(n, eval_result)
            metrics = {
                "held_out_macro": eval_result["macro"],
                "held_out_pooled_diagnostic": eval_result["pooled"],
                "n_releases": n,
            }
        except evaluation.LeakageError as exc:
            eval_result = {"leakage": str(exc)}
            promotion = {
                "promote": False,
                "status": "blocked",
                "reason": f"blocked: {exc}",
            }
            metrics = {"n_releases": n, "blocked": str(exc)}
            status_override = "blocked"
    else:
        eval_result = None
        promotion = evaluation.promotion_decision(1, None)
        metrics = {"n_releases": 1, "note": "provisional; no release-held-out metric"}

    estimator = _fit_all(releases)
    artifact_bytes = build_artifact_bytes(estimator, model_version)

    manifest = manifest_mod.build_manifest(
        artifact_bytes=artifact_bytes,
        source_release_hashes=[r.dataset_hash for r in releases],
        upstream_commit=upstream_commit,
        metrics=metrics,
        promotion=promotion,
        model_version=model_version,
        feature_version=FEATURE_VERSION,
        created_at=created_at if created_at is not None else _utc_now_iso(),
        model_code_revision=model_code_revision or source_revision(),
        release_ids=[r.release_id for r in releases],
        release_versions=[r.release_version for r in releases],
        item_counts=[r.item_count for r in releases],
        class_counts=[r.class_counts for r in releases],
        seed=estimator.seed,
        hyperparameters=estimator.hyperparameters,
        calibration=estimator.calibration_,
    )

    artifact_path = Path(artifact_path)
    promote = bool(promotion["promote"])
    # A rejected/provisional/blocked candidate must NEVER overwrite the requested
    # champion path or its manifest -- it goes to a distinct candidate path so the
    # prior champion is preserved.
    if promote:
        # Crash-safe champion publication preserving the prior last-known-good.
        target = artifact_path
        manifest_path = _publish_pair(target, artifact_bytes, manifest)
    else:
        target = candidate_artifact_path(artifact_path)
        manifest_path = _write_pair(target, artifact_bytes, manifest)

    return {
        "status": status_override or promotion["status"],
        "promote": promote,
        "is_champion": promote,
        "reason": promotion["reason"],
        "artifact_written": True,
        "artifact_path": str(target),
        "champion_path": str(artifact_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "evaluation": eval_result,
    }
