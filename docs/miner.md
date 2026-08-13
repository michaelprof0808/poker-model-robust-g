# Miner

Poker44 miners classify tournament micro-sessions as human or bot generated.
The Axon exposes only `MicroSessionDetectionSynapse` with
`contract_version = "microsession-v1"`. There is no legacy JSON, hand-based
request, version negotiation, GitHub check, manifest check or W&B integration.

## Request contract

`items` is an ordered list of schema-v4.1 micro-sessions. Every item contains
exactly four coarse strategic decisions and at least one postflop decision. The
only decision fields are:

- `decision_number`
- `phase`
- `position_group`
- `pressure`
- `action_type`
- `size_bucket`
- `is_all_in`

These features are derived from consented tournament telemetry, but miners do
not receive raw mouse, keyboard or timing events. Labels, cards, exact chip
amounts, actor IDs, bot families, tournament IDs and capture provenance remain
private. Old schemas, extra fields and label leakage are rejected before model
inference. The normative schema is
[`contracts/subject-session.v4.1.schema.json`](../contracts/subject-session.v4.1.schema.json).

The response must contain one finite `risk_scores` value in `[0, 1]` per item,
in the original order. `0` means strongest human confidence and `1` means
strongest bot confidence. The Axon also derives `predictions` at threshold
`0.5`; validator scoring uses the continuous values, not those booleans.

## Model interface

Set `POKER44_MODEL_FACTORY=module:create_model`. The factory receives a
`MinerModelConfig` and returns an object with:

```python
class Model:
    version: str

    def load(self) -> None: ...
    def predict(self, sessions: list[dict]) -> list[float]: ...
```

The included reference model is only a runnable baseline. A competitive miner
should provide its own factory. Relevant environment variables are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `POKER44_MODEL_FACTORY` | reference factory | Import path `module:create_model` |
| `POKER44_MODEL_PATH` | empty | Optional model artifact path |
| `POKER44_MODEL_DEVICE` | `cpu` | Model device hint |
| `POKER44_MODEL_VERSION` | `reference-v2` | Reported model version |
| `POKER44_MAX_SESSIONS_PER_REQUEST` | `256` | Hard item limit |
| `POKER44_MAX_REQUEST_BYTES` | `16777216` | Hard serialized request limit |

Model loading happens once during startup. Inference is serialized with a
process-local lock, and synchronous models run outside the event loop.

## Validator authentication

By default, callers must be registered on the subnet and have a validator
permit. `ALLOWED_VALIDATOR_HOTKEYS` passed to the run script enables a stricter
explicit allowlist and requires the normal signed Axon verification for those
hotkeys. Do not enable non-registered callers in production.

The validator queries every registered, reachable non-validator miner Axon. It
does not cap the request at 10 or 32 miners and does not deduplicate hotkeys
that share a coldkey. Running several hotkeys therefore creates several
independent candidates, but only the highest-scoring hotkey wins the miner
share for a round.

## Endpoint protection

Encrypted Axon commitments are optional. If enabled and finalized successfully,
the miner publishes a masked metagraph endpoint and authorized validators
resolve the encrypted commitment. If commitment publication fails, the miner
keeps its public endpoint rather than becoming unreachable. See
[`encrypted-axon-endpoints.md`](encrypted-axon-endpoints.md) before enabling it.

## Run and verify

Configure `WALLET_NAME`, `HOTKEY`, `NETUID` and `AXON_PORT`, then run:

```bash
bash scripts/miner/run/run_miner.sh
```

The script defaults to Finney netuid 126, uses PM2 and enforces validator
permits unless an explicit allowlist is provided. Before deployment run:

```bash
ruff check poker44 neurons tests
PYTHONPATH=. pytest -q
```

After startup, verify registration, Axon reachability, model version, recent
request logs and incentive on chain. A PM2 `online` state alone is not proof
that the miner answered a validator request.
