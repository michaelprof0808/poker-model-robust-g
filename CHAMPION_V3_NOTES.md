# Champion v3 micro-session miner notes

This repo is a Poker44 v3-compatible miner build based on the official `Poker44/Poker44-subnet` micro-session contract.

## Contract

- Synapse: `MicroSessionDetectionSynapse`
- Contract version: `microsession-v1`
- Input: schema-v4.1 `items` with four coarse strategic decisions
- Output: one finite `risk_scores` value in `[0, 1]` per item plus `model_version`

This retired compatibility model does not load any legacy hand-chunk artifact because the live v3 validator sends schema-v4.1 micro-session items (`MicroSessionDetectionSynapse(items=...)`). The v3 model factory is:

```bash
export POKER44_MODEL_FACTORY=poker44_champion_v3.model:create_model
export POKER44_MODEL_VERSION=champion-v3-ms-1
```

## Deployment env for SN126

```bash
export NETUID=126
export WALLET_NAME=poker
export HOTKEY=sn126_1
export AXON_PORT=8195
export POKER44_MODEL_FACTORY=poker44_champion_v3.model:create_model
export POKER44_MODEL_VERSION=champion-v3-ms-1
export POKER44_MAX_SESSIONS_PER_REQUEST=256
export POKER44_MAX_REQUEST_BYTES=16777216
```

## Encrypted Axon

Official v3 code supports encrypted endpoint commitments. Enable only when validators can resolve protected endpoints:

```bash
export POKER44_ENCRYPTED_AXON_ENABLED=true
export POKER44_AXON_EXTERNAL_IP=<public-origin-ip>
export POKER44_AXON_EXTERNAL_PORT=8195
```

If commitment publication/readback fails, the base miner logs the failure and continues with the public endpoint to preserve validator connectivity.

## Verification

```bash
python -m py_compile neurons/miner.py poker44_champion_v3/model.py
PYTHONPATH=. pytest -q tests/test_champion_v3_model.py
```
