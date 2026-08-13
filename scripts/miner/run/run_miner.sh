#!/usr/bin/env bash
set -euo pipefail

NETUID="${NETUID:-126}"
NETWORK="${NETWORK:-finney}"
WALLET_NAME="${WALLET_NAME:-poker44-miner}"
HOTKEY="${HOTKEY:-miner}"
WALLET_PATH="${WALLET_PATH:-}"
AXON_PORT="${AXON_PORT:-8091}"
PM2_NAME="${PM2_NAME:-poker44-miner}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MINER_SCRIPT="${MINER_SCRIPT:-./neurons/miner.py}"
MINER_EXTRA_ARGS="${MINER_EXTRA_ARGS:-}"

: "${POKER44_ENCRYPTED_AXON_ENABLED:=false}"
: "${POKER44_ENDPOINT_PUBLIC_KEY:=}"
: "${POKER44_AXON_EXTERNAL_IP:=}"
: "${POKER44_AXON_EXTERNAL_PORT:=}"
export POKER44_ENCRYPTED_AXON_ENABLED POKER44_ENDPOINT_PUBLIC_KEY
export POKER44_AXON_EXTERNAL_IP POKER44_AXON_EXTERNAL_PORT

command -v pm2 >/dev/null || { echo "pm2 is required" >&2; exit 1; }
test -f "$MINER_SCRIPT" || { echo "Missing $MINER_SCRIPT" >&2; exit 1; }
"$PYTHON_BIN" -c 'import bittensor, dotenv, nacl, poker44' || {
  echo "Install the Poker44 runtime dependencies first" >&2; exit 1;
}

args=(
  "$MINER_SCRIPT"
  --netuid "$NETUID"
  --subtensor.network "$NETWORK"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --axon.port "$AXON_PORT"
  --logging.info
)
if [[ -n "$WALLET_PATH" ]]; then args+=(--wallet.path "$WALLET_PATH"); fi
if [[ -n "${ALLOWED_VALIDATOR_HOTKEYS:-}" ]]; then
  read -r -a validators <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  args+=(--blacklist.allowed_validator_hotkeys "${validators[@]}")
else
  args+=(--blacklist.force_validator_permit)
fi
if [[ -n "$MINER_EXTRA_ARGS" ]]; then
  read -r -a extra <<< "$MINER_EXTRA_ARGS"
  args+=("${extra[@]}")
fi

pm2 delete "$PM2_NAME" >/dev/null 2>&1 || true
pm2 start "$PYTHON_BIN" --name "$PM2_NAME" -- "${args[@]}"
pm2 save
echo "Started $PM2_NAME on netuid=$NETUID with $WALLET_NAME/$HOTKEY port=$AXON_PORT"
echo "Encrypted Axon endpoint protection: $POKER44_ENCRYPTED_AXON_ENABLED"
