# Validator

Poker44 v0.2.1 has one evaluation path: manually published schema-v4.1
tournament micro-sessions derived from consented telemetry. There is no legacy
hand JSON track, EMA, GitHub/model-repository check, W&B integration or
coldkey-level hotkey restriction.

## Window acquisition

Validators poll `POKER44_SUBNET_DATA_URL` (default `https://api.poker44.net`)
for the current `AVAILABLE` or `LEASING` window. Platform requests are signed
with the validator hotkey, timestamp and one-use nonce. The platform performs
the chain-side validator authorization.

Every validator leases the same immutable item ordering. Before querying a
miner, the validator:

1. requires schema `4.1`, exactly four strategic decisions and at least one
   postflop decision per item;
2. rejects labels or other forbidden private fields in miner-visible payloads;
3. recomputes and verifies the SHA-256 dataset hash;
4. keeps ground-truth labels and actor groups outside the Synapse;
5. runs the capture-shape red-team gate.

The red-team gate fails closed when context signatures differ by class,
position distributions differ materially, or a shallow visible-feature rule
exceeds `POKER44_REDTEAM_MAX_REWARD` (default `0.15`). Single-class windows are
rejected outside an explicit E2E-only override.

## Miner selection and scoring

The validator queries every registered, reachable miner Axon that is not its
own UID and does not have a validator permit. Public and successfully resolved
encrypted Axons are supported together. There is no arbitrary 10/32-miner cap
and no one-hotkey-per-coldkey rule. Missing responses are retried once after
`POKER44_MINER_RETRY_DELAY_SECONDS` (default 30 seconds).

Each miner returns one probability per item. Actor-balanced sample weights give
equal mass to each private actor within each class, preventing one prolific
actor from dominating the round. Quality is:

```text
0.50 * average-precision skill
+ 0.30 * recall at <=5% false-positive rate
+ 0.20 * Brier skill
```

Invalid, missing, non-finite, out-of-range or wrong-length responses score
zero. The highest positive finite quality wins; exact ties use the lower UID.
All miner scores, ranks and errors are reported to the dashboard. The dashboard
is observability only and never supplies consensus weights.

## Transition emission policy

The default on-chain target is:

| Role | Default | Resolution |
| --- | ---: | --- |
| Burn | 90% | Current subnet owner hotkey read from chain |
| Tournament funding | 5% | `POKER44_FUNDING_HOTKEY` |
| Round winner | 5% | Remainder after burn and funding |

Configuration:

```bash
export POKER44_BURN_FRACTION=0.90
export POKER44_FUNDING_FRACTION=0.05
export POKER44_FUNDING_HOTKEY=5DUYX7X2Z9Jizr1NABUFDYV7ruFVNcUmKdxw9HxVP3sN9RUD
```

The funding hotkey must be registered and different from the owner. Fractions
must be finite, non-negative and leave a positive winner remainder. If the
funding hotkey itself wins, its funding and winner shares combine. The
validator rejects the target if live subnet constraints alter the configured
fractions.

Weights are support signals consumed by Yuma Consensus; assigning 5% weight
does not guarantee exactly 5% realized alpha.

## Settlement and refresh cadence

Only a valid newly evaluated window can change the target. The latest target
is persisted locally with its UID-to-hotkey mapping. If any target hotkey moves
or unregisters before submission, settlement stops rather than substituting a
different UID.

A new dirty target submits as soon as the chain `weights_rate_limit` permits.
Afterward, the exact unchanged target is refreshed every
`POKER44_WEIGHT_REFRESH_BLOCKS` (default 720), or later if required by the live
rate limit. A refresh does not query miners and is reported separately as
`weights_refreshed`.

When commit-reveal is enabled, the validator waits for an earlier pending
commit before submitting another. It records commit block and reveal round,
then reports `weights_finalized` only after the exact emitted vector is visible
on chain. Process restarts resume durable evaluated rounds and pending reveals
without querying miners twice.

## Reporting and runtime configuration

Dashboard events use schema v3, are signed with the validator hotkey and are
queued durably in a local SQLite outbox before delivery. A dashboard outage
does not make it the source of weights.

Important variables:

| Variable | Default |
| --- | --- |
| `POKER44_SUBNET_DATA_URL` | `https://api.poker44.net` |
| `POKER44_DASHBOARD_REPORT_URL` | platform validator-events endpoint |
| `POKER44_POLL_INTERVAL_SECONDS` | `300` |
| `POKER44_WEIGHT_REFRESH_BLOCKS` | `720` |
| `POKER44_ROUND_MAX_ATTEMPTS` | `3` |
| `POKER44_ENFORCE_REDTEAM_GATE` | `true` |

Run the validator with:

```bash
bash scripts/validator/run/run_vali.sh
```

The runner starts two persisted PM2 processes by default: the validator and
`poker44-validator-auto-update`. Every ten minutes the watcher fetches
`origin/main` and compares `VALIDATOR_DEPLOY_VERSION`. A strictly newer deploy
version is fast-forwarded, dependencies are checked, the existing validator is
restarted with its current environment, and the applied version is persisted
in a mode-600 state file. Failed deployments are not marked applied and are
retried by the PM2-supervised watcher. Set `AUTO_UPDATE_ENABLED=false` only when
an operator intentionally owns updates by another process supervisor.

Existing installations created before this watcher was enabled need one
manual bootstrap; remote code cannot start a process on a third-party machine:

```bash
git pull --ff-only origin main
AUTO_UPDATE_ENABLED=true bash scripts/validator/run/run_vali.sh
pm2 describe poker44-validator-auto-update
pm2 save
```

Tracked local modifications or a checkout outside `main` stop auto-update
instead of being stashed or overwritten. Validator secrets remain in the local
environment and `.env`; the watcher never runs with shell xtrace.

The script defaults to Finney netuid 126, one concurrent forward and deploy
version `0.2.1`. Before deployment run:

```bash
ruff check poker44 neurons tests
PYTHONPATH=. pytest -q
```

After deployment, verify the fetched commit, PM2 process, platform lease,
responses from eligible miners, computed scores, commit-reveal evidence, exact
on-chain vector and dashboard events. None of those layers alone proves the
complete round succeeded.
