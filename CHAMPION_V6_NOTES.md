# Champion v6 — Round-7 robust micro-session miner

Owner-controlled, neutral Round-7 candidate for the registered poker-wallet
production UID on Poker44 SN126 (schema v4.1). This document is the rationale,
evidence, risks and deployment factory for `poker44_champion_v6`.

> **Evidence posture.** Every offline number below is **proxy evidence** over
> stylised synthetic worlds and/or the failed-audit 10-item public preview. None
> of it is a claim about the private generator or the validator's actor-balanced
> weighting, and nothing here is fit to the 10 preview labels. We do not claim
> certainty.

## What the validator rewards

`reward = 0.50·AP-skill + 0.30·recall@FPR≤5% + 0.20·Brier-skill`, each component
floored at 0, class-balanced by weight on mixed windows
(`poker44/validator/evaluation/reward.py`). Two structural consequences drive the
design:

1. **AP-skill and recall are ranking-only.** A monotone re-scaling of scores
   cannot change them. So the *ranking* must carry the signal, and calibration
   (score magnitude) only trades within the 20% Brier term.
2. **Each component floors at 0.** A wrong-orientation or uninformative model
   scores ~0, never negative. So the robust move is to *commit* to signal where
   we have evidence and stay bounded elsewhere — hedging only cancels signal.

## The legitimate-signal invariant (why v6 is context-agnostic)

The validator red-team gate (`redteam_gate.py`) only passes a window when the
`phase|pressure` signature multiset **and** the `position_group` marginal are
class-matched. Those channels therefore carry **no** legitimate marginal signal;
the surviving signal is the **chronological action / size / all-in** channel.

The public preview even **fails** that gate on position (max gap 0.30 > 0.15):
its one real separator is position — exactly the illegitimate, non-generalizing
channel. Models that look good on the preview (`reference` 0.399, the supervised
`champion_v31` artifact 0.934) do so largely by reading position/preview-specific
structure that a production window removes.

**v6 reads only `action_type`, `size_bucket`, `is_all_in` and their chronological
order.** It is invariant to ids, window, schema version **and the entire
phase/pressure/position context**. It therefore *cannot* exploit the preview's
position leak — and indeed scores 0.0 on the preview. Here, preview score is
anti-correlated with production robustness; we accept the 0.0 preview and buy
robustness on the legitimate channel.

## Model (deterministic, artifact-free)

`poker44_champion_v6.scoring` builds a per-item **bot-evidence** scalar from
sign-fixed, non-negative-weighted features grouped by the policy-world channel
each covers:

- **backbone / levels** — action & size risk, all-in fraction (production-validated
  orientation; action & size marginals are *not* gate-matched, so legitimate);
- **consistency / mechanicalness** — action repetition, response repetition, low
  action entropy (orientation-agnostic: catches a repetitive bot whatever its
  aggression level);
- **chronological dynamics** — longest identical-action run, net escalation;
- **coarseness / extremity** — big-size fraction, extreme (all-in/overbet) fraction.

Evidence is centred on a broad, label-free "typical play" prior (8 000 i.i.d.
reference lines) — centres only place where 0.5 falls (ranking is invariant to
them) — then mapped `p = 0.5 + AMPL·tanh(SLOPE·evidence)`.

Guarantees (all covered by `tests/test_champion_v6_model.py`):

- **finite, bounded** in `[0.02, 0.98]`; malformed/empty → neutral **0.5**;
- **metadata- and full-context-invariant**; **item-permutation equivariant**
  (pure per-item function — no transductive/batch dependence, unlike the
  research-only v5 path);
- **orientation-guarded** — monotone non-decreasing in every bot-ward feature; no
  input can invert the orientation;
- **fine-grained, tie-free** ranking (no coarse rounding) — protects AP,
  recall@FPR and the validator tie-diagnostics gate; genuinely identical lines
  tie (correctly);
- **fail-safe** — a malformed item scores 0.5 in isolation and cannot corrupt a
  valid sibling; any error falls the whole request back to neutral;
- **latency-safe** — pure Python, ~microseconds/item, no numpy/joblib at inference
  and no code-bearing artifact to trust or overfit.

## Robust / adversarial selection (proxy)

`poker44_champion_v6.worlds` generates six **validator-valid** windows (matched
context + position, signal only in the response channel; all pass the red-team
gate). Class-balanced reward
(`scripts/miner/round7_diagnostic.py`, n=80/class, seed=44):

| candidate | aligned | mechanical | coarse | reversal | adversarial | null | **mean signal** |
|---|---|---|---|---|---|---|---|
| reference | 0.904 | 0.340 | 0.830 | 0.000 | 0.000 | 0.000 | 0.519 |
| champion_v3 | 0.894 | 0.316 | 0.893 | 0.000 | 0.000 | 0.023 | 0.526 |
| champion_v5 | 0.885 | 0.316 | 0.874 | 0.000 | 0.000 | 0.001 | 0.519 |
| **champion_v6** | **0.953** | **0.944** | 0.810 | **0.931** | 0.000 | 0.033 | **0.910** |
| champion_v31[champion] | 0.000 | 0.000 | 0.265 | 0.360 | **0.418** | 0.057 | 0.156 |

Reading:

- **v6 wins every legitimate-signal world**, including the two a single action-level
  aggression bet is blind to (`mechanical`, `coarse`) and the one it is reversed in
  (`reversal`, rescued by the orientation-agnostic consistency channel).
- **v6 is correctly floored (0.000) in `adversarial`** — every channel reversed —
  and ~0 on `null`. It is never harmful.
- **The supervised `champion_v31` artifact is anti-robust**: 0.934 on the preview
  but ~0 on the production-orientation worlds and **0.418 on `adversarial`**. It
  learned the preview's reversed / position-leaked pattern and generalizes
  backwards — the "tiny-preview overfit" failure mode, demonstrated.

### Calibration is a Brier-only, conservative choice

Because the score map is monotone, SLOPE/AMPL do not move AP or recall — they only
shape Brier. The grid in the diagnostic shows pure-world reward rising with
confidence, but on a realistic **partially-reversed** window (30% of each class
reversed) a bold AMPL collapses Brier skill (0.135 → 0.00) and over-confidence on
class-independent noise climbs. Frozen **SLOPE=1.1, AMPL=0.30** sits on that robust
frontier (partial Brier-skill 0.135, neutral over-confidence 0.117, within 0.04 of
the unrealistic easy-world max). Scores live in ~[0.20, 0.80]: confident only when
evidence is strong, bounded so a mis-orientation cannot cause a catastrophic Brier.

## Deployment factory (SN126)

`poker44_champion_v6` is a drop-in factory; it does **not** touch wallet
credentials, the VPS, remotes, or any existing candidate.

```bash
export NETUID=126
export WALLET_NAME=poker
export HOTKEY=sn126_1
export AXON_PORT=8195
export POKER44_MODEL_FACTORY=poker44_champion_v6.model:create_model
export POKER44_MODEL_VERSION=champion-v6-robust-1   # or a UID-specific label
export POKER44_MAX_SESSIONS_PER_REQUEST=256
export POKER44_MAX_REQUEST_BYTES=16777216
```

Launched with `PYTHONPATH` including the repo root (as the existing run scripts
do). No artifact/manifest is required.

## Risks & limitations

- **Orientation risk.** The backbone assumes the production-validated orientation
  (mechanical/aggressive/coarse → bot). If the private orientation is reversed on
  aggression, the backbone floors — but the consistency channel still separates
  mechanical bots (see `reversal`), and per-component flooring bounds the loss at 0.
- **Proxy worlds are self-authored.** They are stylised and cleanly separable; real
  windows are far weaker. Treat absolute world rewards as directional, not
  predictive. This is why calibration is conservative, not tuned to world reward.
- **Preview score is 0.0** by design (v6 refuses the position leak). If a future
  *audited* preview passes the red-team gate, re-run the diagnostic before reading
  anything into preview numbers.
- **No supervised lift.** v6 deliberately fits nothing. If/when multiple audited,
  red-team-valid labelled releases exist, a release-held-out supervised model could
  add lift on top of this prior — but not from a single failed-audit tournament.

## Verification

```bash
PY=/tmp/sn55venv/bin/python   # any env with numpy+scikit-learn+pytest
python -m py_compile poker44_champion_v6/*.py scripts/miner/round7_diagnostic.py
PYTHONPATH=. $PY -m pytest -q tests/test_champion_v6_model.py tests/test_champion_v6_worlds.py
PYTHONPATH=. $PY scripts/miner/round7_diagnostic.py --n 80 --seed 44 --json /tmp/round7_report.json
```
