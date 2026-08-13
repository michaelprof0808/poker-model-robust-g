# Tournament evaluation workflow

Poker44 is moving from scheduled competition rounds to a continuous,
tournament-sourced evaluation pipeline. This document explains what changes,
what remains stable, and exactly what a miner model receives.

## The short version

- Competition rounds disappear. A miner does not join round 1, round 2, or any
  later tournament stage.
- Poker tournaments run independently and continuously generate completed
  hands plus consented interaction telemetry.
- The platform waits until it has enough comparable, quality-checked sessions
  to seal an evaluation window.
- Validators query miners only when a sealed window is available. There is no
  guaranteed daily evaluation schedule; sufficient data may be available after
  one day or after several days.
- A miner remains online as usual. It receives an ordered list of sanitized
  subject sessions and returns one calibrated bot-risk score per session.
- Ground truth remains inside the platform-to-validator lease and is removed
  before the Bittensor synapse reaches a miner.

The word `round` still appears in some internal class names, environment
variables and lifecycle events for backward compatibility. In those places it
means one validator evaluation cycle over a sealed window. It does **not** mean
a competition round or a tournament stage.

## End-to-end lifecycle

### 1. Tournament registration

Players register before the scheduled start. An individual tournament may use
private, single-use invitations and may require a verified payout coldkey, but
those access rules are separate from the miner protocol.

Registration closes when the tournament starts:

- a player who is absent is not awaited;
- late entry is not allowed after play begins;
- a disconnected player cannot block the tournament;
- when an action timer expires, play continues under the table timeout rules.

### 2. Tables and tournament progression

The platform creates the required tables from the registered field, seats
players, and starts all active tables. As players are eliminated, it
automatically rebalances tables and eventually consolidates the surviving
players at one final table. The tournament ends when the winner is determined.

Table creation, rebalancing, blind progression, eliminations and final standings
are platform concerns. A miner does not need to reproduce tournament state or
connect to a tournament table.

### 3. Session assembly

The platform records durable poker actions and consented telemetry throughout
play. It assembles a subject session only from completed hands that pass data
quality checks. In the current tournament collector, a session contains three
consecutive completed hands for one subject. Miners should not hardcode that
number: session size is part of the versioned payload and may evolve.

The assembler retains:

- poker decisions and the game state visible at each decision;
- relative decision and action timing;
- sanitized, bucketed interaction events;
- aggregate timing and activity statistics.

Incomplete or low-quality chunks are not promoted into the evaluation pool.
Internal engine fixtures, synthetic data and historical private actors are also
excluded from subnet evaluation windows.

Before sealing, the platform builds the exact miner-visible representation and
runs a shallow provenance-leak canary over payload size and telemetry-event
cardinality. A window is withheld if one of those capture-shape features can
separate humans from agents above the configured balanced-accuracy limit.
Unscoped page-surface clicks are removed at this boundary; semantic control
interactions remain available to models.

### 4. Evaluation-window readiness

Completed sessions accumulate across recurring tournaments. The platform seals
an immutable window only when its configured quality, comparability and
diversity requirements are met. Comparable sessions use the same collector
version and hand count. Window construction also prevents one subject from
supplying multiple samples to the same window. The standard policy additionally
prevents the same subject from being reused in later windows, reducing identity
memorization and repeated-player leakage.

This makes evaluation data-driven rather than calendar-driven:

```text
recurring tournaments
        |
        v
completed hands + telemetry
        |
        v
quality-checked subject sessions
        |
        v
enough comparable sessions?
   no --------> keep collecting
   yes
        |
        v
sealed evaluation window
        |
        v
validator leases -> miner requests -> local rewards
```

Validators poll for an available window. If no window is ready, they send no
evaluation request and continue polling. Depending on tournament volume, the
next window may become available in one day, three days, four days, or another
data-dependent interval.

The standard deployment profile seals a 20-session window: 10 eligible human
sessions and 10 eligible bot sessions from at least one tournament, with bot
family diversity enforced. With the current three-hand session collector, one
full 20-seat tournament can therefore produce a complete evaluation window.
Sessions still have to pass quality and telemetry checks, so registrations or
raw dealt-hand counts alone do not guarantee readiness.

### 5. Validator lease and miner request

Each validator acquires an idempotent lease for the current window. The lease
contains the exact same persisted payload list plus labels for local scoring.
The validator recomputes `dataset_hash` over the ordered miner-visible payloads
and rejects any mismatch. Before
constructing the track-specific Synapse, the validator separates the labels
and sends only the ordered feature payloads.

For strategic v3, the validator verifies that decision counts and the complete
`phase × position_group × pressure` context multiset are identical across
subjects. The gate deliberately ignores the chosen action and size bucket,
which are the intended strategic signal. Any context mismatch stops evaluation
before a miner receives the snapshot.

For micro-session v4.1, every item has four decisions and at least one
postflop decision. Human and bot items are matched on `phase × pressure`;
position is balanced globally with a maximum 0.15 class-distribution gap.
Source decisions are single-use within a window.

The tournament telemetry request is:

```python
MicroSessionDetectionSynapse(
    window_id="window_...",
    dataset_hash="<sha256 of ordered items>",
    query_id="<validator-bound query id>",
    items=[item_0, item_1, ...],
)
```

This is the only evaluation endpoint. There is no schema negotiation or fallback.

### 6. Miner inference

The configured miner model receives the `sessions` list:

```python
def predict(self, sessions: list[dict]) -> list[float]:
    ...
```

It must return one finite probability-like value for each input session, in the
same order:

```python
[0.08, 0.71, 0.43, ...]
```

- `0.0` means strongest confidence that the session is human.
- `1.0` means strongest confidence that the session is bot-generated.
- Output length must equal input length.
- Every value must be finite and within `[0, 1]`.

The miner returns the values in `risk_scores`. A boolean classification alone
is insufficient because calibration is part of validator scoring.

### 7. Validator scoring and settlement

The validator keeps labels locally, validates the response, computes quality,
selects the deterministic winner and submits the transition allocation when
cadence permits: 90% to the live subnet owner, 5% to the configured tournament
funding hotkey and 5% to the winner by default. The dashboard receives signed
observability events only; it does not calculate or provide weights.

An invalid response, missing response, wrong output length, non-finite value or
out-of-range score receives zero reward for that evaluation cycle.

## Miner-visible telemetry micro-session v4.1

The normative tournament contract is
[`contracts/subject-session.v4.1.schema.json`](../contracts/subject-session.v4.1.schema.json).
Each payload contains exactly four coarse poker decisions with at least one
postflop decision. Human and bot items are context-matched without exposing
labels, actors, cards, exact chips, timing provenance or tournament identity.

The contract intentionally contains no hands, raw telemetry, clocks, cards,
chip amounts, tournament results or source identifiers. Each decision exposes
only `phase`, `position_group`, `pressure`, `action_type`, `size_bucket` and
`is_all_in`, plus its one-based decision number.

## Privacy and label separation

The miner-visible boundary creates an opaque `item_id` per evaluation window
and removes cross-window and platform identifiers.

The following internal fields are forbidden recursively from miner payloads:

- `is_bot`, `is_human`, `ground_truth`, `label`
- `bot_family`, `capture_source`, `collector_version`
- `simulation`, `session_index`
- `tournament_id`, `user_id`

Miners should also treat `item_id`, `window_id`, array order, hashes and
pagination or request timing as metadata, not model features. These values are
not stable behavioral signals and using them encourages leakage and
overfitting.

## Migration from competition rounds

| Previous competition format | Tournament evaluation format |
| --- | --- |
| A miner could join a later competition round | There are no joinable competition rounds |
| Evaluations followed an announced round schedule | Requests occur only when a sealed data window exists |
| Inputs focused primarily on poker hands | Inputs are four coarse strategic decisions |
| A round was a participant-facing competition phase | `round_id` means only an internal validator evaluation cycle |
| Models could assume the old chunk structure | Models must consume the versioned subject-session contract |

The miner's operational responsibility remains the same: keep the axon
reachable and return model predictions when queried. Miners do not register for
data-generation tournaments unless their operators independently choose to
participate as players in a community test.

## Model adaptation checklist

Before running the tournament-based release:

1. Pull the latest `dev` branch.
2. Configure `POKER44_MODEL_FACTORY=your_package.module:create_model`.
3. Ensure the factory returns an object with `version`, `load()` and
   `predict(sessions)`.
4. Accept only `schema_version: "4.1"` through `MicroSessionDetectionSynapse`.
5. Parse a list of sessions containing `decisions`.
6. Produce exactly one score per session and preserve input order.
7. Return finite values in `[0, 1]`; do not return only class labels.
8. Treat nullable fields, empty arrays and future additive fields defensively.
9. Do not hardcode the current three-hands-per-session collector setting.
10. Remove features derived from IDs, window metadata, request order or wall
    clock.
11. Confirm the scored model does not require telemetry, timestamps, cards,
    exact chips or tournament outcomes.
12. Size inference for the configured session and byte limits.
13. Avoid classifiers based only on payload byte length or raw event volume;
    these are monitored as provenance shortcuts and are not stable behavioral
    features.
14. Run the repository tests before deployment:

    ```bash
    PYTHONPATH=. pytest -q
    ```

## Defensive feature extraction example

This minimal example demonstrates shape handling, not a competitive detection
strategy:

```python
from typing import Any


def extract_features(session: dict[str, Any]) -> dict[str, float]:
    decisions = [
        decision
        for decision in (session.get("decisions") or [])
        if isinstance(decision, dict)
    ]
    aggressive = sum(
        decision.get("action_type") in {"bet", "raise", "all_in"}
        for decision in decisions
    )
    facing_bet = sum(
        decision.get("pressure") == "facing_bet" for decision in decisions
    )
    return {
        "decision_count": float(len(decisions)),
        "aggression_rate": aggressive / max(1, len(decisions)),
        "facing_bet_rate": facing_bet / max(1, len(decisions)),
    }
```

Production models should validate their own feature assumptions, pin a model
version, and monitor distributions across multiple evaluation windows rather
than fitting one tournament or one window.
