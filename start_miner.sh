#!/usr/bin/env bash
set -euo pipefail
APP=${POKER44_APP_DIR:-/opt/poker-model-robust-a}
cd "$APP"
export PYTHONPATH="$APP"
export NETUID=${NETUID:-126}
export NETWORK=${NETWORK:-finney}
export WALLET_NAME=${WALLET_NAME:-poker}
export HOTKEY=${HOTKEY:-sn126_1}
export AXON_PORT=${AXON_PORT:-8195}
export POKER44_MODEL_FACTORY=${POKER44_MODEL_FACTORY:-poker44_champion_v31.model:create_champion_model}
export POKER44_MODEL_VERSION=${POKER44_MODEL_VERSION:-round7-primary-v31-1}
export POKER44_MODEL_PATH=${POKER44_MODEL_PATH:-$APP/artifacts/champion-orientation-guard.candidate.joblib}
export POKER44_CHAMPION_ALLOW_UNPROMOTED=${POKER44_CHAMPION_ALLOW_UNPROMOTED:-1}
export POKER44_MAX_SESSIONS_PER_REQUEST=${POKER44_MAX_SESSIONS_PER_REQUEST:-256}
export POKER44_MAX_REQUEST_BYTES=${POKER44_MAX_REQUEST_BYTES:-16777216}
export POKER44_ENCRYPTED_AXON_ENABLED=${POKER44_ENCRYPTED_AXON_ENABLED:-true}
export POKER44_AXON_EXTERNAL_IP=${POKER44_AXON_EXTERNAL_IP:-161.35.119.64}
export POKER44_AXON_EXTERNAL_PORT=${POKER44_AXON_EXTERNAL_PORT:-$AXON_PORT}
exec "$APP/.venv/bin/python" neurons/miner.py \
  --netuid "$NETUID" --wallet.name "$WALLET_NAME" --wallet.hotkey "$HOTKEY" \
  --subtensor.network "$NETWORK" --axon.port "$AXON_PORT" \
  --blacklist.force_validator_permit --logging.info
