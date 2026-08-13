"""Runtime self-checks for a loaded Poker44 model.

Verifies the three contract invariants the champion must satisfy:
finite bounded output, metadata invariance, and item-permutation equivariance.
Used by the smoke script and safe to run against any object with ``predict``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

__all__ = ["runtime_selfcheck"]

_META_PROBES = {
    "item_id": "diagnostic-probe-item-id-XXXXXXXXXXXXXXXX",
    "window_id": "diagnostic-probe-window-id",
}
_NUMERIC_EQUIVARIANCE_ATOL = 1e-12


def _finite_bounded(scores: Sequence[float]) -> bool:
    return all(isinstance(s, (int, float)) and math.isfinite(s) and 0.0 <= s <= 1.0 for s in scores)


def _scores_close(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=_NUMERIC_EQUIVARIANCE_ATOL)
        for a, b in zip(left, right)
    )


def runtime_selfcheck(model: Any, payloads: Sequence[dict]) -> dict[str, Any]:
    payloads = list(payloads)
    base = list(model.predict(payloads))

    finite_bounded = len(base) == len(payloads) and _finite_bounded(base)

    # Metadata invariance: rewriting only transport metadata must not move scores.
    relabeled = []
    for p in payloads:
        clone = dict(p)
        clone.update(_META_PROBES)
        relabeled.append(clone)
    meta_scores = list(model.predict(relabeled))
    metadata_invariant = _scores_close(meta_scores, base)

    # Item-permutation equivariance: reversing input reverses output.
    permutation_equivariant = True
    if len(payloads) >= 2:
        reversed_scores = list(model.predict(list(reversed(payloads))))
        permutation_equivariant = _scores_close(reversed_scores, list(reversed(base)))

    ok = bool(finite_bounded and metadata_invariant and permutation_equivariant)
    return {
        "ok": ok,
        "mode": getattr(model, "mode", "unknown"),
        "n_items": len(payloads),
        "finite_bounded": bool(finite_bounded),
        "metadata_invariant": bool(metadata_invariant),
        "permutation_equivariant": bool(permutation_equivariant),
        "scores": [round(float(s), 6) for s in base] if finite_bounded else None,
    }
