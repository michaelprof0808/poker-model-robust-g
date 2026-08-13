"""Reward-aligned offline evaluation with release-based splits.

Metrics use the official ``poker44.validator.evaluation.reward.reward`` with
UNIFORM sample weights. The public corpus has no actor IDs, so these are honest
uniform-weight offline estimates and are explicitly NOT the actor-balanced
validator metric -- they are labeled as such and must never be presented as
reproducing the validator's weighting.

Generalization is measured leave-one-release-out (LORO): the model is trained on
all-but-one release and scored on the held-out release. There is deliberately no
random item-split headline metric. Promotion is gated on release count.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from poker44.validator.evaluation.reward import reward
from poker44_champion_v31 import FEATURE_VERSION, fallback, features
from poker44_champion_v31.dataset import ParsedRelease, behavior_key
from poker44_champion_v31.estimator import ChampionEstimator

__all__ = [
    "ACTOR_CLUSTER_CONFIDENCE_NOTE",
    "BOOTSTRAP_ALPHA",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "MAX_CROSS_CLASS_TIE_FRACTION",
    "MIN_ITEMS_PER_CLASS",
    "MIN_ITEMS_PER_RELEASE",
    "PROMOTION_MARGIN",
    "LeakageError",
    "cross_release_leakage",
    "paired_stratified_bootstrap_reward_diff",
    "promotion_decision",
    "release_held_out_eval",
    "score_tie_diagnostics",
    "uniform_offline_metrics",
]


class LeakageError(ValueError):
    """Raised when behavioral payloads leak across release boundaries."""


def cross_release_leakage(releases: Sequence[ParsedRelease]) -> dict[str, Any]:
    """Detect identical behavioral payloads appearing in more than one release.

    Only the canonical behavioral decision payload is compared (transport ids,
    windows and decision_number *values* are excluded), so the same session
    reappearing under different ids in a train and a held-out release is caught.
    Within-release repeats are reported separately and do not, alone, invalidate
    leave-one-release-out evidence.
    """
    release_ids = [release.release_id for release in releases]
    release_hashes = [release.dataset_hash for release in releases]
    if any(not value for value in release_ids) or len(set(release_ids)) != len(release_ids):
        raise LeakageError("release identities must have unique non-empty release IDs")
    if any(not value for value in release_hashes) or len(set(release_hashes)) != len(release_hashes):
        raise LeakageError("release identities must have unique non-empty dataset hashes")

    releases_by_key: dict[Any, set[str]] = {}
    within_release_repeats = 0
    for release_index, rel in enumerate(releases):
        seen_here: set[Any] = set()
        rel_tag = f"{release_index}:{rel.release_id}"
        for record in rel.records:
            key = behavior_key(record)
            if key in seen_here:
                within_release_repeats += 1
            seen_here.add(key)
            releases_by_key.setdefault(key, set()).add(str(rel_tag))
    crossing = {k: sorted(v) for k, v in releases_by_key.items() if len(v) >= 2}
    return {
        "n_cross_release_duplicates": len(crossing),
        "n_within_release_repeats": within_release_repeats,
        "colliding_release_sets": sorted({tuple(v) for v in crossing.values()}),
    }

# The champion must beat the safe fallback baseline by at least this reward
# margin, out-of-release (macro over held releases), before it can be promoted.
PROMOTION_MARGIN = 0.01

# On any single usable held release the champion may trail the baseline by at
# most this much (noise tolerance) and still count as non-inferior. Kept small
# so that materially losing an entire release blocks promotion.
NONINFERIORITY_TOL = 0.005

# A held release must carry enough labeled evidence for its fold reward to mean
# anything: at least this many items overall and this many of EACH class.
MIN_ITEMS_PER_RELEASE = 40
MIN_ITEMS_PER_CLASS = 10

# Ranking skill built on degenerate score ties is illusory: if this fraction of a
# fold's items share their exact score with an opposite-class item, the champion
# is not actually separating the classes and promotion is blocked.
MAX_CROSS_CLASS_TIE_FRACTION = 0.05
MAX_LOW_FPR_BLOCKED_BOT_FRACTION = 0.05

# Deterministic paired stratified item-bootstrap of the champion-minus-baseline
# reward difference. The one-sided lower bound (BOOTSTRAP_ALPHA quantile) must
# clear zero for a fold to count as statistically ahead of the baseline.
BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 44

# The public benchmark exposes no actor IDs, so we cannot cluster the bootstrap
# by actor. This is stated on every eval result so the interval is never mistaken
# for an actor-clustered (validator-grade) confidence statement.
ACTOR_CLUSTER_CONFIDENCE_NOTE = (
    "actor-cluster confidence unavailable: the public corpus exposes no actor "
    "IDs, so the paired reward-difference interval is an item-level bootstrap, "
    "NOT actor-clustered; it may understate uncertainty if items share an actor"
)

# Reward components that must be finite for a fold to count as valid.
_FINITE_KEYS = (
    "reward",
    "average_precision_skill",
    "recall_at_fpr_05",
    "brier_skill",
)

_UNIFORM_NOTE = (
    "uniform-weight offline estimate; the public corpus has no actor IDs so this "
    "is NOT the actor-balanced validator metric and is not reproducible as one"
)

FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def uniform_offline_metrics(
    predictions: Sequence[float], labels: Sequence[int]
) -> dict[str, Any]:
    """Return official-reward metrics under uniform weighting, clearly labeled."""
    result = reward(list(map(float, predictions)), list(map(int, labels)))
    metrics = result.to_dict()
    metrics["weighting"] = "uniform"
    metrics["note"] = _UNIFORM_NOTE
    return metrics


def score_tie_diagnostics(
    scores: Sequence[float], labels: Sequence[int]
) -> dict[str, Any]:
    """Measure real score ties from the champion output (not a proxy metric).

    ``cross_class_tie_fraction`` is the fraction of items whose EXACT champion
    score is shared by at least one opposite-class item: those items are
    ranking-ambiguous, so a large fraction means the reported ranking skill is an
    artefact of tie-breaking rather than genuine separation.
    """
    from collections import defaultdict

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n = int(s.shape[0])
    if n == 0:
        return {
            "n": 0,
            "n_distinct_scores": 0,
            "tie_fraction": 0.0,
            "cross_class_tie_fraction": 0.0,
        }
    classes_by_score: dict[float, set[int]] = defaultdict(set)
    count_by_score: dict[float, int] = defaultdict(int)
    for score, label in zip(s.tolist(), y.tolist()):
        classes_by_score[score].add(int(label))
        count_by_score[score] += 1
    tied_items = sum(c for c in count_by_score.values() if c >= 2)
    cross_items = sum(
        count_by_score[score]
        for score, classes in classes_by_score.items()
        if len(classes) >= 2
    )
    negatives = max(1, int(np.sum(y == 0)))
    positives = max(1, int(np.sum(y == 1)))
    fp = 0
    blocked_bots = 0
    for score in sorted(count_by_score, reverse=True):
        at_score = s == score
        group_fp = int(np.sum((y == 0) & at_score))
        group_tp = int(np.sum((y == 1) & at_score))
        if fp / negatives <= 0.05 < (fp + group_fp) / negatives:
            blocked_bots = group_tp
            break
        fp += group_fp
    return {
        "n": n,
        "n_distinct_scores": len(count_by_score),
        "tie_fraction": tied_items / n,
        "cross_class_tie_fraction": cross_items / n,
        "low_fpr_blocked_bot_fraction": blocked_bots / positives,
    }


def paired_stratified_bootstrap_reward_diff(
    champion: Sequence[float],
    baseline: Sequence[float],
    labels: Sequence[int],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Deterministic paired, class-stratified item-bootstrap of the champion-minus-
    baseline reward difference.

    * paired      -- each resample draws ONE set of item indices scored under both
                     the champion and the baseline, so the difference cancels the
                     shared sampling noise;
    * stratified  -- indices are resampled within each class at its original size,
                     preserving prevalence and guaranteeing both classes are
                     present in every resample;
    * deterministic -- a fixed-seed ``default_rng`` makes the lower bound exactly
                     reproducible across runs.

    Returns the mean difference and the one-sided ``alpha`` lower bound.
    """
    champ = np.clip(np.asarray(champion, dtype=float), 0.0, 1.0)
    base = np.clip(np.asarray(baseline, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=int)
    class_idx = [np.where(y == c)[0] for c in (0, 1)]
    if any(idx.size == 0 for idx in class_idx):
        return {
            "resamples": 0,
            "alpha": alpha,
            "seed": seed,
            "mean_reward_diff": 0.0,
            "reward_diff_lower_bound": 0.0,
            "note": "single-class fold; paired reward-difference bootstrap undefined",
            "actor_cluster_confidence": ACTOR_CLUSTER_CONFIDENCE_NOTE,
        }
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples, dtype=float)
    for b in range(resamples):
        picks = np.concatenate(
            [rng.choice(idx, size=idx.size, replace=True) for idx in class_idx]
        )
        yb = y[picks]
        champ_reward = reward(champ[picks].tolist(), yb.tolist()).reward
        base_reward = reward(base[picks].tolist(), yb.tolist()).reward
        diffs[b] = champ_reward - base_reward
    diffs.sort()
    return {
        "resamples": int(resamples),
        "alpha": alpha,
        "seed": seed,
        "mean_reward_diff": float(np.mean(diffs)),
        "reward_diff_lower_bound": float(np.quantile(diffs, alpha)),
        "actor_cluster_confidence": ACTOR_CLUSTER_CONFIDENCE_NOTE,
    }


def _champion_fit_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, groups_train: np.ndarray
) -> np.ndarray:
    return ChampionEstimator().fit(X_train, y_train, groups=groups_train).predict_proba(X_test)


def _matrix(release: ParsedRelease) -> tuple[np.ndarray, np.ndarray]:
    return features.vectorize(release.payloads), np.asarray(release.labels, dtype=int)


def _finite_1d(arr: np.ndarray, n: int) -> bool:
    return arr.ndim == 1 and arr.shape[0] == n and bool(np.all(np.isfinite(arr)))


def release_held_out_eval(
    releases: Sequence[ParsedRelease],
    fit_predict: FitPredict | None = None,
) -> dict[str, Any]:
    """Leave-one-release-out evaluation of champion vs fallback baseline.

    Requires at least two releases. Returns pooled and per-release uniform
    metrics for both the supervised champion and the heuristic baseline.
    """
    if len(releases) < 2:
        raise ValueError("release-held-out evaluation requires >= 2 releases")
    leakage = cross_release_leakage(releases)
    if leakage["n_cross_release_duplicates"] > 0:
        raise LeakageError(
            "cross-release behavioral leakage detected "
            f"({leakage['n_cross_release_duplicates']} payloads appear in >1 release); "
            "leave-one-release-out evidence would be invalid"
        )
    fit_predict = fit_predict or _champion_fit_predict

    per_release: list[dict[str, Any]] = []
    champ_pred: list[float] = []
    base_pred: list[float] = []
    pooled_labels: list[int] = []

    for held in range(len(releases)):
        test = releases[held]
        train_idx = [i for i in range(len(releases)) if i != held]
        train = [releases[i] for i in train_idx]
        X_train = np.vstack([_matrix(r)[0] for r in train])
        y_train = np.concatenate([_matrix(r)[1] for r in train])
        # Release-grouped calibration relies on the train group labels; use the
        # release ordinal so each training release is a distinct group.
        groups_train = np.concatenate(
            [np.full(train[k].item_count, train_idx[k], dtype=int) for k in range(len(train))]
        )
        X_test, y_test = _matrix(test)

        champ = np.asarray(fit_predict(X_train, y_train, X_test, groups_train), dtype=float)
        base = np.asarray(fallback.heuristic_scores(test.payloads), dtype=float)

        entry: dict[str, Any] = {
            "release_id": test.release_id,
            "n_items": test.item_count,
            "class_counts": test.class_counts,
        }
        n = int(test.item_count)
        finite = _finite_1d(champ, n) and _finite_1d(base, n)
        mixed = len(set(y_test.tolist())) >= 2
        if finite:
            champ_pred.extend(champ.tolist())
            base_pred.extend(base.tolist())
            pooled_labels.extend(y_test.tolist())
        if not finite:
            entry["champion"] = None
            entry["baseline"] = None
            entry["invalid"] = True
            entry["note"] = "champion produced non-finite or misshaped predictions"
        elif mixed:
            entry["champion"] = uniform_offline_metrics(champ, y_test)
            entry["baseline"] = uniform_offline_metrics(base, y_test)
            # Real score-tie audit + paired reward-difference confidence interval,
            # both computed from the actual fold scores (not proxy metrics).
            entry["tie_diagnostics"] = score_tie_diagnostics(champ, y_test)
            entry["reward_diff_bootstrap"] = paired_stratified_bootstrap_reward_diff(
                champ, base, y_test
            )
            counts = test.class_counts
            entry["meets_min_items"] = bool(
                n >= MIN_ITEMS_PER_RELEASE
                and counts.get(0, 0) >= MIN_ITEMS_PER_CLASS
                and counts.get(1, 0) >= MIN_ITEMS_PER_CLASS
            )
        else:
            entry["champion"] = None
            entry["baseline"] = None
            entry["note"] = "single-class held-out release; ranking skill undefined"
        per_release.append(entry)

    if pooled_labels and len(set(pooled_labels)) >= 2:
        champion = uniform_offline_metrics(champ_pred, pooled_labels)
        baseline = uniform_offline_metrics(base_pred, pooled_labels)
    else:
        champion = None
        baseline = None
    valid_releases = [row for row in per_release if row["champion"] is not None]
    macro = {
        "champion_reward": float(
            np.mean([row["champion"]["reward"] for row in valid_releases])
        )
        if valid_releases
        else 0.0,
        "baseline_reward": float(
            np.mean([row["baseline"]["reward"] for row in valid_releases])
        )
        if valid_releases
        else 0.0,
        "n_valid_releases": len(valid_releases),
    }
    return {
        "n_releases": len(releases),
        "feature_version": FEATURE_VERSION,
        "pooled": champion,
        "champion": champion,
        "baseline": baseline,
        "macro": macro,
        "promotion_metric": "macro_mean_release_reward",
        "per_release": per_release,
        "actor_cluster_confidence": ACTOR_CLUSTER_CONFIDENCE_NOTE,
    }


def _fold_finite(metrics: dict[str, Any] | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    return all(
        isinstance(metrics.get(k), (int, float)) and math.isfinite(float(metrics[k]))
        for k in _FINITE_KEYS
    )


def promotion_decision(
    n_releases: int, result: dict[str, Any] | None
) -> dict[str, Any]:
    """Gate champion promotion on conservative, release-aware evidence.

    Beyond the release-count gate this requires, over the leave-one-release-out
    folds: at least two valid mixed-class folds, each with enough items
    (>= MIN_ITEMS_PER_RELEASE and >= MIN_ITEMS_PER_CLASS of each class); all
    component metrics finite; per-release non-inferiority to the baseline; a macro
    reward that beats the baseline by the promotion margin; no pathological
    score-tie block measured from the real fold scores; and, on every fold, a
    paired stratified item-bootstrap reward-difference lower bound above zero. All
    gates must pass; small datasets simply fail conservatively rather than
    inventing confidence.
    """
    if n_releases <= 0:
        return {
            "promote": False,
            "status": "unavailable",
            "reason": "no benchmark releases published; nothing to train or promote",
        }
    if n_releases == 1:
        return {
            "promote": False,
            "status": "provisional",
            "reason": (
                "single release cannot be release-held-out; provisional fit only, "
                "champion promotion blocked"
            ),
        }

    result = result or {}
    per_release = result.get("per_release") or []
    valid = [
        row for row in per_release
        if isinstance(row.get("champion"), dict) and isinstance(row.get("baseline"), dict)
    ]

    all_finite = bool(valid) and all(
        _fold_finite(row["champion"]) and _fold_finite(row["baseline"]) for row in valid
    ) and not any(row.get("invalid") for row in per_release)

    champ_rewards = [float(r["champion"]["reward"]) for r in valid] if all_finite else []
    base_rewards = [float(r["baseline"]["reward"]) for r in valid] if all_finite else []
    champion_reward = float(np.mean(champ_rewards)) if champ_rewards else 0.0
    baseline_reward = float(np.mean(base_rewards)) if base_rewards else 0.0

    per_release_non_inferior = all_finite and all(
        float(r["champion"]["reward"]) >= float(r["baseline"]["reward"]) - NONINFERIORITY_TOL
        for r in valid
    )

    def _fold_size_ok(row: dict[str, Any]) -> bool:
        counts = row.get("class_counts") or {}
        humans = int(counts.get(0, counts.get("0", 0)))
        bots = int(counts.get(1, counts.get("1", 0)))
        return (
            int(row.get("n_items", 0)) >= MIN_ITEMS_PER_RELEASE
            and humans >= MIN_ITEMS_PER_CLASS
            and bots >= MIN_ITEMS_PER_CLASS
        )

    enough_items = bool(valid) and all(_fold_size_ok(r) for r in valid)

    # Real score-tie block: both global cross-class ambiguity and the bot mass
    # stranded at the official <=5% FPR boundary are measured from exact scores.
    no_pathological_score_ties = all_finite and all(
        float((r.get("tie_diagnostics") or {}).get("cross_class_tie_fraction", 1.0))
        <= MAX_CROSS_CLASS_TIE_FRACTION
        and float(
            (r.get("tie_diagnostics") or {}).get(
                "low_fpr_blocked_bot_fraction", 1.0
            )
        )
        <= MAX_LOW_FPR_BLOCKED_BOT_FRACTION
        for r in valid
    )

    # Paired stratified item-bootstrap reward-difference lower bound must clear 0
    # on every fold (missing bootstrap fails closed).
    bootstrap_reward_diff_positive = all_finite and all(
        float(
            (r.get("reward_diff_bootstrap") or {}).get(
                "reward_diff_lower_bound", -1.0
            )
        )
        > 0.0
        for r in valid
    )

    gates = {
        "min_valid_folds": len(valid) >= 2,
        "min_items_per_release": enough_items,
        "all_finite": all_finite,
        "per_release_non_inferior": per_release_non_inferior,
        "macro_beats_baseline": (
            champion_reward > 0.0
            and champion_reward >= baseline_reward + PROMOTION_MARGIN
        ),
        "no_pathological_score_ties": no_pathological_score_ties,
        "bootstrap_reward_diff_positive": bootstrap_reward_diff_positive,
    }
    promote = all(gates.values())
    failed = [name for name, ok in gates.items() if not ok]
    reason = (
        f"macro champion {champion_reward:.4f} vs baseline {baseline_reward:.4f} "
        f"over {len(valid)} valid folds; "
        + ("all promotion gates passed" if promote else f"failed gates: {failed}")
    )
    return {
        "promote": bool(promote),
        "status": "evaluated",
        "reason": reason,
        "champion_reward": champion_reward,
        "baseline_reward": baseline_reward,
        "metric": "macro_mean_release_reward",
        "n_valid_folds": len(valid),
        "gates": gates,
        "actor_cluster_confidence": ACTOR_CLUSTER_CONFIDENCE_NOTE,
    }
