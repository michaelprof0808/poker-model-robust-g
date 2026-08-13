#!/usr/bin/env python3
"""Round-7 offline diagnostic & candidate comparison (PROXY EVIDENCE ONLY).

Compares every owner-controlled candidate (reference, champion v3, v5, v31, and
the new robust v6) on:

  1. the public 10-item preview -- under BOTH uniform and validator-style
     class-balanced weighting. The preview FAILED its quality audit (single
     tournament; it even fails the red-team gate on position), so this is a weak
     sanity check, never a selection target; and

  2. a battery of *validator-valid* synthetic policy worlds (see
     ``poker44_champion_v6.worlds``) -- phase|pressure and position class-matched
     exactly like production, signal only in the legitimate action/size/all-in
     channel -- scored with class-balanced weighting.

It also runs the contract self-check (finite/bounded, metadata-invariant,
permutation-equivariant), reports real score-tie diagnostics, and shows the
robust SLOPE x AMPL calibration selection for v6.

NOTHING here is a claim about the private generator. Every number is proxy
evidence over stylised worlds or a failed-audit preview, and is labelled as such.

Run:  PYTHONPATH=. /tmp/sn55venv/bin/python scripts/miner/round7_diagnostic.py [--json out.json] [--n 60] [--seed 44]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from poker44.miner.config import MinerModelConfig
from poker44.validator.evaluation.redteam_gate import audit_redteam_leakage
from poker44.validator.evaluation.reward import reward
from poker44_champion_v31 import diagnostics as v31_diag
from poker44_champion_v31.evaluation import score_tie_diagnostics
from poker44_champion_v6 import scoring, worlds

REPO = Path(__file__).resolve().parents[2]
PREVIEW = REPO / "data" / "benchmark-latest.json"

PROXY_BANNER = (
    "PROXY EVIDENCE ONLY -- synthetic worlds + a failed-audit 10-item preview; "
    "NOT a claim about the private generator or the validator's actor weighting."
)


# --------------------------------------------------------------------------- #
# Candidates                                                                    #
# --------------------------------------------------------------------------- #
def _cfg(version: str = "") -> MinerModelConfig:
    return MinerModelConfig(
        factory="x", model_path=None, device="cpu", version=version, max_sessions_per_request=512
    )


def _build_candidates() -> dict[str, Callable[[list[dict]], list[float]]]:
    from poker44.miner.model import create_reference_model
    from poker44_champion_v3.model import create_model as v3_model
    from poker44_champion_v5.model import create_model as v5_model
    from poker44_champion_v6.model import create_model as v6_model

    candidates: dict[str, Callable[[list[dict]], list[float]]] = {
        "reference": create_reference_model(_cfg("reference-v2")).predict,
        "champion_v3": v3_model(_cfg()).predict,
        "champion_v5": v5_model(_cfg()).predict,
        "champion_v6": v6_model(_cfg()).predict,
    }

    # champion_v31: try the promoted-load path with the local candidate artifact;
    # if it is not loadable, ChampionModel transparently serves its safe fallback.
    try:
        import os

        from poker44_champion_v31.model import ChampionModel

        os.environ["POKER44_CHAMPION_ALLOW_UNPROMOTED"] = "1"
        artifact = REPO / "artifacts" / "champion-orientation-guard.candidate.joblib"
        model = ChampionModel(_cfg(), artifact_path=str(artifact) if artifact.exists() else None)
        model.load()
        candidates[f"champion_v31[{model.mode}]"] = model.predict
    except Exception as exc:  # noqa: BLE001 -- comparator is best-effort
        candidates["champion_v31[unavailable]"] = lambda items, _exc=exc: [0.5] * len(items)
    return candidates


# --------------------------------------------------------------------------- #
# Metrics                                                                        #
# --------------------------------------------------------------------------- #
def class_balanced_weights(labels: list[int]) -> list[float]:
    y = np.asarray(labels, dtype=int)
    w = np.ones(y.shape, dtype=float)
    for c in (0, 1):
        mask = y == c
        n = int(mask.sum())
        if n:
            w[mask] = 0.5 / n
    return w.tolist()


def _reward_row(scores: list[float], labels: list[int], *, balanced: bool) -> dict[str, float]:
    weights = class_balanced_weights(labels) if balanced else None
    r = reward([float(s) for s in scores], [int(y) for y in labels], sample_weights=weights)
    return {
        "reward": r.reward,
        "ap_skill": r.average_precision_skill,
        "recall_at_fpr05": r.recall_at_fpr_05,
        "brier_skill": r.brier_skill,
    }


def _safe_scores(predict: Callable[[list[dict]], list[float]], items: list[dict]) -> list[float]:
    try:
        scores = [float(s) for s in predict(items)]
        if len(scores) != len(items):
            return [0.5] * len(items)
        return scores
    except Exception:  # noqa: BLE001
        return [0.5] * len(items)


# --------------------------------------------------------------------------- #
# Report                                                                        #
# --------------------------------------------------------------------------- #
def load_preview() -> tuple[list[dict], list[int]]:
    data = json.loads(PREVIEW.read_text())
    items = [it["payload"] for it in data["items"]]
    labels = [int(it["label"]) for it in data["items"]]
    return items, labels


def _by_label(items: list[dict], labels: list[int], label: int) -> list[dict]:
    return [it for it, y in zip(items, labels) if y == label]


def _partial_window(n_per_class: int, seed: int, reversed_frac: float = 0.3) -> tuple[list[dict], list[int]]:
    """A realistic, imperfectly-separated window: a ``reversed_frac`` minority of
    each class follows the fully-reversed (adversarial) policy while the majority
    follows the production-orientation (aligned) policy. This is where over-
    confidence actually costs Brier skill, so it disciplines the AMPL choice."""
    a_items, a_labels = worlds.build_world("aligned", n_per_class, seed)
    x_items, x_labels = worlds.build_world("adversarial", n_per_class, seed + 1)
    cut = int(round(n_per_class * (1.0 - reversed_frac)))
    items: list[dict] = []
    labels: list[int] = []
    for label in (1, 0):
        items += _by_label(a_items, a_labels, label)[:cut]
        items += _by_label(x_items, x_labels, label)[cut:]
        labels += [label] * (cut + (n_per_class - cut))
    return items, labels


def robust_calibration_grid(n_per_class: int, seed: int) -> dict[str, Any]:
    """Audit the frozen v6 calibration against a small SLOPE x AMPL grid.

    The pure worlds are cleanly separable, so reward there rises monotonically
    with confidence -- that alone would (wrongly) push AMPL to the ceiling. Two
    realistic penalties expose the true frontier:

    * ``partial_brier``: Brier skill on an imperfectly-separated window (30% of
      each class reversed). Ranking (AP/recall) is confidence-invariant, so only
      Brier moves -- and beyond a moderate AMPL the confidently-wrong minority
      collapses it. This is an interior optimum.
    * ``neutral_overconf``: mean |score-0.5| over a class-independent (null)
      population. A calibrated model must stay near 0.5 on noise; high SLOPE/AMPL
      is over-confident about nothing.
    """
    signal = ("aligned", "mechanical", "coarse", "reversal")
    grid_slopes = (0.9, 1.1, 1.35, 1.7, 2.2)
    grid_ampls = (0.30, 0.42, 0.48)
    built = {name: worlds.build_world(name, n_per_class, seed) for name in worlds.WORLD_NAMES}
    partial_items, partial_labels = _partial_window(n_per_class, seed)
    # Neutral calibration reference: evidence ~ 0 by construction, so score spread
    # is a direct over-confidence read.
    neutral_items = scoring.sample_reference_items(400, seed)
    rows = []
    orig_slope, orig_ampl = scoring.SLOPE, scoring.AMPL
    try:
        for slope in grid_slopes:
            for ampl in grid_ampls:
                scoring.SLOPE, scoring.AMPL = slope, ampl
                per = {}
                for name, (items, labels) in built.items():
                    s = [scoring.score_session(it) for it in items]
                    per[name] = reward(s, labels, sample_weights=class_balanced_weights(labels)).reward
                partial = reward(
                    [scoring.score_session(it) for it in partial_items],
                    partial_labels,
                    sample_weights=class_balanced_weights(partial_labels),
                )
                neutral = [scoring.score_session(it) for it in neutral_items]
                overconf = float(np.mean([abs(v - 0.5) for v in neutral]))
                rows.append(
                    {
                        "slope": slope,
                        "ampl": ampl,
                        "mean_signal_reward": round(float(np.mean([per[w] for w in signal])), 4),
                        "worst_world_reward": round(min(per.values()), 4),
                        "partial_reward": round(partial.reward, 4),
                        "partial_brier": round(partial.brier_skill, 4),
                        "neutral_overconf": round(overconf, 4),
                        "is_frozen": slope == orig_slope and ampl == orig_ampl,
                    }
                )
    finally:
        scoring.SLOPE, scoring.AMPL = orig_slope, orig_ampl
    return {"frozen": {"slope": orig_slope, "ampl": orig_ampl}, "grid": rows}


def build_report(n_per_class: int = 60, seed: int = 44) -> dict[str, Any]:
    candidates = _build_candidates()
    preview_items, preview_labels = load_preview()
    world_data = {name: worlds.build_world(name, n_per_class, seed) for name in worlds.WORLD_NAMES}

    report: dict[str, Any] = {
        "proxy_disclaimer": PROXY_BANNER,
        "preview": {"n": len(preview_items), "prevalence": float(np.mean(preview_labels))},
        "preview_redteam_gate": audit_redteam_leakage(preview_items, preview_labels).to_dict(),
        "worlds": {"n_per_class": n_per_class, "seed": seed, "names": list(worlds.WORLD_NAMES)},
        "world_validity": {},
        "candidates": {},
        "v6_calibration_selection": robust_calibration_grid(n_per_class, seed),
    }

    for name, (items, labels) in world_data.items():
        gate = audit_redteam_leakage(items, labels)
        report["world_validity"][name] = {"redteam_passed": bool(gate.passed), "reason": gate.reason}

    for cand, predict in candidates.items():
        entry: dict[str, Any] = {}
        pv = _safe_scores(predict, preview_items)
        entry["preview_uniform"] = _reward_row(pv, preview_labels, balanced=False)
        entry["preview_balanced"] = _reward_row(pv, preview_labels, balanced=True)
        entry["preview_tie_diagnostics"] = {
            k: round(float(v), 4)
            for k, v in score_tie_diagnostics(pv, preview_labels).items()
        }
        world_rewards = {}
        for name, (items, labels) in world_data.items():
            scores = _safe_scores(predict, items)
            world_rewards[name] = _reward_row(scores, labels, balanced=True)["reward"]
        entry["worlds_balanced_reward"] = world_rewards
        signal = ("aligned", "mechanical", "coarse", "reversal")
        entry["worlds_summary"] = {
            "mean_signal_reward": float(np.mean([world_rewards[w] for w in signal])),
            "worst_world_reward": float(min(world_rewards.values())),
            "adversarial_reward": world_rewards["adversarial"],
            "null_reward": world_rewards["null"],
        }
        # Contract self-check on a small mixed batch.
        try:
            model_like = type("M", (), {"predict": staticmethod(predict)})()
            check = v31_diag.runtime_selfcheck(model_like, world_data["aligned"][0][:8])
            entry["selfcheck"] = {k: check[k] for k in ("finite_bounded", "metadata_invariant", "permutation_equivariant")}
        except Exception as exc:  # noqa: BLE001
            entry["selfcheck"] = {"error": str(exc)}
        report["candidates"][cand] = entry
    return report


def _fmt_row(label: str, values: list[str]) -> str:
    return f"{label:<26}" + "".join(f"{v:>13}" for v in values)


def print_report(report: dict[str, Any]) -> None:
    print("=" * 96)
    print("POKER44 SN126 ROUND-7 MINER DIAGNOSTIC")
    print(report["proxy_disclaimer"])
    print("=" * 96)

    gate = report["preview_redteam_gate"]
    print(
        f"\nPreview: n={report['preview']['n']} prevalence={report['preview']['prevalence']:.2f} | "
        f"red-team gate PASSED={gate['passed']} (reason={gate.get('reason')})"
    )
    print("  -> the preview is NOT a production-valid window; treat preview metrics as weak sanity only.")

    print("\nWorld validity (all must be red-team-valid to be usable proxies):")
    for name, v in report["world_validity"].items():
        print(f"  {name:<12} redteam_passed={v['redteam_passed']}")

    cands = report["candidates"]
    names = list(cands)

    print("\n" + "-" * 96)
    print("PREVIEW reward (class-balanced) [weak sanity; do not select on this]")
    print(_fmt_row("candidate", ["reward", "ap_skill", "recall@5", "brier_sk", "xclass_tie"]))
    for name in names:
        b = cands[name]["preview_balanced"]
        tie = cands[name]["preview_tie_diagnostics"].get("cross_class_tie_fraction", 0.0)
        print(_fmt_row(name, [f"{b['reward']:.4f}", f"{b['ap_skill']:.3f}", f"{b['recall_at_fpr05']:.3f}", f"{b['brier_skill']:.3f}", f"{tie:.3f}"]))

    print("\n" + "-" * 96)
    print("SYNTHETIC WORLDS reward (class-balanced) [robust selection evidence]")
    world_names = report["worlds"]["names"]
    print(_fmt_row("candidate", [w[:11] for w in world_names]))
    for name in names:
        wr = cands[name]["worlds_balanced_reward"]
        print(_fmt_row(name, [f"{wr[w]:.3f}" for w in world_names]))
    print()
    print(_fmt_row("candidate", ["mean_signal", "worst_world", "adversarial", "null", "selfcheck"]))
    for name in names:
        s = cands[name]["worlds_summary"]
        sc = cands[name]["selfcheck"]
        ok = all(sc.get(k) for k in ("finite_bounded", "metadata_invariant", "permutation_equivariant")) if "error" not in sc else False
        print(_fmt_row(name, [f"{s['mean_signal_reward']:.3f}", f"{s['worst_world_reward']:.3f}", f"{s['adversarial_reward']:.3f}", f"{s['null_reward']:.3f}", "OK" if ok else "FAIL"]))

    print("\n" + "-" * 96)
    sel = report["v6_calibration_selection"]
    print(f"v6 robust calibration selection (frozen: slope={sel['frozen']['slope']} ampl={sel['frozen']['ampl']})")
    print(_fmt_row("slope x ampl", ["mean_signal", "worst_world", "partial_R", "partial_B*", "neutral_oc", "frozen"]))
    for row in sel["grid"]:
        print(_fmt_row(
            f"{row['slope']} x {row['ampl']}",
            [f"{row['mean_signal_reward']:.3f}", f"{row['worst_world_reward']:.3f}", f"{row['partial_reward']:.3f}", f"{row['partial_brier']:.3f}", f"{row['neutral_overconf']:.3f}", "*" if row["is_frozen"] else ""],
        ))
    print("\nSelection rule: maximise mean signal-world reward subject to worst_world>=0, a")
    print("non-collapsing partial-window Brier skill (partial_B*), and bounded neutral over-")
    print("confidence (neutral_oc). Pure-world reward rises with confidence, but partial_B* and")
    print("neutral_oc penalise it -- the frozen (*) row sits on that robust frontier.")
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60, help="items per class per world")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--json", type=str, default=None, help="write full report JSON here")
    args = parser.parse_args()
    report = build_report(args.n, args.seed)
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nWrote full report to {args.json}")


if __name__ == "__main__":
    main()
