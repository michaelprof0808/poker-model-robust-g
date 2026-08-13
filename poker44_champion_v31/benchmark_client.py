"""Client for the public Poker44 staging benchmark API with an immutable cache.

The staging base is ``https://staging.platform.poker44.net/api/v1/benchmark``.
All endpoints are public and unauthenticated. Downloads are verified by SHA-256
over the exact returned bytes against the release ``datasetHash`` BEFORE any
parsing, and verified bytes are cached immutably keyed by their hash.

Network access is isolated behind an injectable ``fetcher`` callable so the
ingestion logic is fully unit-testable offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

__all__ = [
    "STAGING_BASE_URL",
    "BenchmarkClient",
    "BenchmarkError",
    "default_cache_dir",
    "sha256_hex",
]

STAGING_BASE_URL = "https://staging.platform.poker44.net/api/v1/benchmark"

Fetcher = Callable[[str], bytes]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BenchmarkError(RuntimeError):
    """Raised on transport failures or integrity violations."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Concurrency-safe write: unique temp file in the same dir, then os.replace.

    A per-writer temp name (mkstemp) means concurrent writers never collide on a
    shared ``.tmp`` path, and ``os.replace`` is atomic on the same filesystem so
    readers only ever see a complete file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def default_cache_dir() -> Path:
    root = os.getenv("POKER44_BENCHMARK_CACHE_DIR")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "poker44_champion_v31" / "releases"


_MAX_REDIRECTS = 5


def _origin(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc


def _validate_trusted_url(url: str, base_url: str) -> str:
    """Return ``url`` iff it is an https, credential-free, fragment-free URL on
    the exact same origin as the trusted benchmark base; else raise."""
    parsed = urlparse(url)
    base = urlparse(base_url)
    if parsed.scheme != "https":
        raise BenchmarkError(f"download URL must use https, got {parsed.scheme!r}")
    if parsed.username or parsed.password or "@" in (parsed.netloc or ""):
        raise BenchmarkError("download URL must not embed credentials")
    if parsed.fragment:
        raise BenchmarkError("download URL must not contain a fragment")
    if parsed.scheme != base.scheme or parsed.netloc != base.netloc:
        raise BenchmarkError("download URL origin does not match benchmark API origin")
    return url


def _requests_fetcher(timeout: float, base_url: str) -> Fetcher:
    def fetch(url: str) -> bytes:
        import requests

        current = url
        for _ in range(_MAX_REDIRECTS + 1):
            try:
                # Never auto-follow: a redirect can escape the trusted origin.
                response = requests.get(
                    current, timeout=timeout, allow_redirects=False
                )
            except Exception as exc:  # pragma: no cover - network dependent
                raise BenchmarkError(f"request failed for {current}: {exc}") from exc
            status = response.status_code
            if status in (301, 302, 303, 307, 308):
                location = (response.headers or {}).get("Location")
                if not location:
                    raise BenchmarkError(f"redirect without Location from {current}")
                nxt = urljoin(current, location)
                # Validate every redirect hop against the trusted origin.
                current = _validate_trusted_url(nxt, base_url)
                continue
            if status != 200:
                raise BenchmarkError(f"http {status} for {current}")
            return response.content
        raise BenchmarkError("too many redirects while downloading release")

    return fetch


def _expected_int(release: dict[str, Any], *keys: str):
    """Return a metadata count as an exact non-negative, non-boolean int.

    No ``int()`` coercion: strings, floats and booleans are rejected so that
    malformed or wrong-typed metadata fails closed instead of being normalized.
    """
    for key in keys:
        if key in release and release[key] is not None:
            value = release[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BenchmarkError(
                    f"metadata {key} must be a non-negative integer, got {value!r}"
                )
            return value
    return None


def _reconcile_counts(release: dict[str, Any], parsed) -> None:
    """Cross-check body-derived counts against index metadata when supplied."""
    item_count = _expected_int(release, "itemCount", "itemsCount", "n_items")
    if item_count is not None and item_count != parsed.item_count:
        raise BenchmarkError(
            f"itemCount metadata {item_count} disagrees with body {parsed.item_count}"
        )

    class_counts = parsed.class_counts
    human_count = _expected_int(release, "humanItems", "humanCount", "humans")
    if human_count is not None and human_count != class_counts.get(0, 0):
        raise BenchmarkError(
            f"humanCount metadata {human_count} disagrees with body {class_counts.get(0, 0)}"
        )
    bot_count = _expected_int(release, "botItems", "botCount", "bots")
    if bot_count is not None and bot_count != class_counts.get(1, 0):
        raise BenchmarkError(
            f"botCount metadata {bot_count} disagrees with body {class_counts.get(1, 0)}"
        )

    decision_count = _expected_int(release, "decisionCount", "decisionsCount")
    if decision_count is not None:
        actual = sum(len(p.get("decisions") or []) for p in parsed.payloads)
        if decision_count != actual:
            raise BenchmarkError(
                f"decisionCount metadata {decision_count} disagrees with body {actual}"
            )


class BenchmarkClient:
    def __init__(
        self,
        base_url: str = STAGING_BASE_URL,
        cache_dir: str | os.PathLike[str] | None = None,
        fetcher: Fetcher | None = None,
        timeout: float = 30.0,
    ):
        if not isinstance(base_url, str):
            raise BenchmarkError("benchmark base URL must be a string")
        self.base_url = base_url.rstrip("/")
        if self.base_url != STAGING_BASE_URL:
            raise BenchmarkError(
                "benchmark base URL must be the official HTTPS staging origin/path"
            )
        _validate_trusted_url(self.base_url, STAGING_BASE_URL)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.fetcher: Fetcher = fetcher or _requests_fetcher(timeout, self.base_url)

    # -- JSON envelope endpoints -------------------------------------------
    def _get_json(self, path: str) -> Any:
        raw = self.fetcher(f"{self.base_url}{path}")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise BenchmarkError(f"invalid JSON from {path}: {exc}") from exc
        if isinstance(decoded, dict) and "success" in decoded:
            if not decoded.get("success", False):
                error = decoded.get("error") or {}
                raise BenchmarkError(f"api error on {path}: {error}")
            return decoded.get("data")
        return decoded

    def index(self) -> dict[str, Any]:
        """Return the ``/benchmark`` schedule + latest-release metadata."""
        data = self._get_json("")
        return data if isinstance(data, dict) else {}

    def list_releases(self) -> list[dict[str, Any]]:
        data = self._get_json("/releases")
        releases = data.get("releases") if isinstance(data, dict) else None
        if not isinstance(releases, list):
            return []
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()
        validated: list[dict[str, Any]] = []
        for release in releases:
            if not isinstance(release, dict):
                raise BenchmarkError("release index entry must be an object")
            release_id = release.get("releaseId")
            dataset_hash = release.get("datasetHash")
            if not isinstance(release_id, str) or not release_id.strip():
                raise BenchmarkError("release index releaseId must be a non-empty string")
            if not isinstance(dataset_hash, str) or not _SHA256_RE.fullmatch(dataset_hash):
                raise BenchmarkError("release index datasetHash must be lowercase SHA-256")
            if release_id in seen_ids:
                raise BenchmarkError(f"duplicate releaseId in index: {release_id!r}")
            if dataset_hash in seen_hashes:
                raise BenchmarkError(f"duplicate datasetHash in index: {dataset_hash}")
            seen_ids.add(release_id)
            seen_hashes.add(dataset_hash)
            validated.append(release)
        return validated

    def has_releases(self) -> bool:
        return len(self.list_releases()) > 0

    def latest_release(self) -> dict[str, Any]:
        """Return the latest release by explicit index identity, never by list
        ordering. Requires the index to name a latest releaseId."""
        index = self.index()
        latest = index.get("latest")
        if isinstance(latest, dict):
            latest_id = str(latest.get("releaseId") or "").strip()
        elif isinstance(latest, str):
            latest_id = latest.strip()
        else:
            latest_id = ""
        if not latest_id:
            raise BenchmarkError(
                "benchmark index does not name a latest release; refusing to "
                "assume list ordering (pass an explicit release id instead)"
            )
        for release in self.list_releases():
            if str(release.get("releaseId") or "") == latest_id:
                return release
        raise BenchmarkError(f"latest release {latest_id!r} not present in /releases")

    def find_release(self, release_id: str) -> dict[str, Any]:
        for release in self.list_releases():
            if str(release.get("releaseId") or "") == release_id:
                return release
        raise BenchmarkError(f"release {release_id!r} not found")

    # -- Download + immutable cache ----------------------------------------
    def _download_url(self, release: dict[str, Any]) -> str:
        url = release.get("downloadUrl")
        if isinstance(url, str) and url:
            candidate = urljoin(self.base_url + "/", url)
        else:
            release_id = str(release.get("releaseId") or "").strip()
            if not release_id:
                raise BenchmarkError("release has neither downloadUrl nor releaseId")
            candidate = f"{self.base_url}/releases/{release_id}/download"
        return _validate_trusted_url(candidate, self.base_url)

    def _cache_path(self, expected_hash: str) -> Path:
        return self.cache_dir / f"{expected_hash}.json"

    # -- release-id -> hash immutability ledger ----------------------------
    def _ledger_path(self) -> Path:
        return self.cache_dir / "_release_ledger.json"

    def _provenance_path(self) -> Path:
        return self.cache_dir / "_release_provenance.json"

    def _read_json_ledger(self, path: Path):
        """Read a JSON-object ledger, failing CLOSED on any corruption.

        A missing ledger is an empty history; anything present-but-unreadable or
        of the wrong shape raises rather than silently erasing immutability."""
        if not path.exists():
            return {}
        try:
            raw = path.read_text()
        except OSError as exc:
            raise BenchmarkError(f"cannot read ledger {path.name}: {exc}") from exc
        try:
            loaded = json.loads(raw)
        except ValueError as exc:
            raise BenchmarkError(f"malformed ledger {path.name}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise BenchmarkError(f"malformed ledger {path.name}: not a JSON object")
        return loaded

    def _read_ledger(self) -> dict[str, str]:
        loaded = self._read_json_ledger(self._ledger_path())
        ledger: dict[str, str] = {}
        for key, value in loaded.items():
            if not isinstance(key, str) or not key.strip():
                raise BenchmarkError(
                    f"malformed ledger {self._ledger_path().name}: invalid release id key"
                )
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise BenchmarkError(
                    f"malformed ledger {self._ledger_path().name}: value for {key!r} "
                    "is not a lowercase SHA-256"
                )
            ledger[key] = value
        return ledger

    def _check_release_immutability(self, release_id: str, expected_hash: str) -> None:
        """Reject a known release id reappearing with a different datasetHash."""
        if not release_id:
            return
        prior = self._read_ledger().get(release_id)
        if prior is not None and prior != expected_hash:
            raise BenchmarkError(
                f"release {release_id!r} datasetHash changed from {prior} to "
                f"{expected_hash}; releases are immutable"
            )

    def _locked(self, fn):
        """Run ``fn()`` under the exclusive cross-process ledger lock."""
        import fcntl

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_dir / "_release_ledger.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return fn()
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _record_release_hash(self, release_id: str, expected_hash: str) -> None:
        if not release_id:
            return

        # The preliminary check in download_bytes avoids needless network work,
        # but only this locked read-check-write is authoritative: two processes
        # may pass the preliminary check concurrently with different hashes.
        def _op():
            ledger = self._read_ledger()
            prior = ledger.get(release_id)
            if prior is not None and prior != expected_hash:
                raise BenchmarkError(
                    f"release {release_id!r} datasetHash changed from {prior} to "
                    f"{expected_hash}; releases are immutable"
                )
            if prior == expected_hash:
                return
            ledger[release_id] = expected_hash
            _atomic_write_bytes(
                self._ledger_path(), json.dumps(ledger, sort_keys=True).encode("utf-8")
            )

        self._locked(_op)

    def _record_release_provenance(self, release_id: str, provenance: dict[str, Any]) -> None:
        """Bind a release id to its immutable version + body-derived counts.

        Recorded under the same lock as the hash ledger. A repeated release id
        whose provenance differs from the stored record is rejected."""
        if not release_id:
            return

        def _op():
            loaded = self._read_json_ledger(self._provenance_path())
            expected_keys = {
                "hash", "version", "item_count", "human_count", "bot_count"
            }
            for key, value in loaded.items():
                if (
                    not isinstance(key, str)
                    or not key.strip()
                    or not isinstance(value, dict)
                    or set(value) != expected_keys
                    or not _SHA256_RE.fullmatch(str(value.get("hash") or ""))
                    or not isinstance(value.get("version"), str)
                    or not value.get("version", "").strip()
                    or any(
                        isinstance(value.get(name), bool)
                        or not isinstance(value.get(name), int)
                        or value[name] < 0
                        for name in ("item_count", "human_count", "bot_count")
                    )
                ):
                    raise BenchmarkError(
                        f"malformed provenance ledger entry for {key!r}"
                    )
            prior = loaded.get(release_id)
            if prior is not None and prior != provenance:
                raise BenchmarkError(
                    f"release {release_id!r} provenance changed from {prior} to "
                    f"{provenance}; release provenance is immutable"
                )
            if prior == provenance:
                return
            loaded[release_id] = provenance
            _atomic_write_bytes(
                self._provenance_path(),
                json.dumps(loaded, sort_keys=True).encode("utf-8"),
            )

        self._locked(_op)

    def download_bytes(self, release: dict[str, Any]) -> bytes:
        """Return verified release bytes, using the immutable cache when valid."""
        expected_raw = release.get("datasetHash")
        if not isinstance(expected_raw, str) or not _SHA256_RE.fullmatch(expected_raw):
            raise BenchmarkError("release datasetHash must be 64 lowercase hex characters")
        expected = expected_raw

        release_id_raw = release.get("releaseId")
        if not isinstance(release_id_raw, str) or not release_id_raw.strip():
            raise BenchmarkError("releaseId must be a non-empty string")
        release_id = release_id_raw
        self._check_release_immutability(release_id, expected)

        cached = self._cache_path(expected)
        if cached.exists():
            data = cached.read_bytes()
            if sha256_hex(data) == expected:
                self._record_release_hash(release_id, expected)
                return data
            # Corrupt cache entry: drop it and re-fetch.
            cached.unlink()

        data = self.fetcher(self._download_url(release))
        actual = sha256_hex(data)
        if actual != expected:
            raise BenchmarkError(
                f"dataset hash mismatch: expected {expected}, got {actual}"
            )
        _atomic_write_bytes(cached, data)
        self._record_release_hash(release_id, expected)
        return data

    def fetch_release(self, release: dict[str, Any]):
        """Download (hash-verified), then parse a release into a ParsedRelease."""
        import dataclasses

        from poker44_champion_v31.dataset import parse_release

        expected = release["datasetHash"]
        raw = self.download_bytes(release)
        body = json.loads(raw.decode("utf-8"))
        parsed = parse_release(body)

        # The index is authoritative for release identity. The live canonical
        # dataset wrapper omits releaseId/releaseVersion; when either field is
        # present in a body, require exact agreement without coercion.
        index_release_id = release.get("releaseId")
        if index_release_id not in (None, ""):
            body_id = body.get("releaseId")
            if body_id is not None and (
                not isinstance(body_id, str)
                or not body_id.strip()
                or body_id != str(index_release_id)
            ):
                raise BenchmarkError("download body releaseId disagrees with index metadata")
        index_version = release.get("releaseVersion")
        if index_version not in (None, ""):
            if not isinstance(index_version, str) or not index_version.strip():
                raise BenchmarkError("index releaseVersion must be a non-empty string")
            body_version = body.get("releaseVersion")
            if body_version is not None and (
                not isinstance(body_version, str)
                or not body_version.strip()
                or body_version != str(index_version)
            ):
                raise BenchmarkError("download body releaseVersion disagrees with index metadata")
        # datasetHash inside the body is optional (a body cannot contain its own
        # hash), but must be exact if present. The index hash is authoritative.
        body_hash = body.get("datasetHash")
        if body_hash is not None and (
            not isinstance(body_hash, str) or body_hash != expected
        ):
            raise BenchmarkError("download body datasetHash disagrees with verified bytes")
        _reconcile_counts(release, parsed)

        resolved_id = str(index_release_id or parsed.release_id or "")
        resolved_version = str(index_version or parsed.release_version or "")
        # Bind immutable version + body-derived provenance under the ledger lock.
        self._record_release_provenance(
            resolved_id,
            {
                "hash": expected or parsed.dataset_hash,
                "version": resolved_version,
                "item_count": parsed.item_count,
                "human_count": parsed.class_counts.get(0, 0),
                "bot_count": parsed.class_counts.get(1, 0),
            },
        )
        # Stamp the verified hash / release id from the authoritative index entry.
        return dataclasses.replace(
            parsed,
            dataset_hash=expected or parsed.dataset_hash,
            release_id=resolved_id or parsed.release_id,
            release_version=resolved_version or parsed.release_version,
        )
