#!/usr/bin/env bash
set -euo pipefail

NETUID="${NETUID:-126}"
NETWORK="${NETWORK:-finney}"
WALLET_NAME="${WALLET_NAME:-poker44-validator}"
HOTKEY="${HOTKEY:-validator}"
WALLET_PATH="${WALLET_PATH:-}"
PM2_NAME="${PM2_NAME:-poker44-validator}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VALIDATOR_SCRIPT="${VALIDATOR_SCRIPT:-./neurons/validator.py}"
NEURON_TIMEOUT="${NEURON_TIMEOUT:-180}"
VALIDATOR_EXTRA_ARGS="${VALIDATOR_EXTRA_ARGS:-}"
VALIDATOR_ENV_DIR="${VALIDATOR_ENV_DIR:-validator_env}"
AUTO_UPDATE_ENABLED="${AUTO_UPDATE_ENABLED:-true}"
AUTO_UPDATE_PM2_NAME="${AUTO_UPDATE_PM2_NAME:-poker44-validator-auto-update}"
AUTO_UPDATE_INTERVAL_SECONDS="${AUTO_UPDATE_INTERVAL_SECONDS:-600}"

: "${POKER44_SUBNET_DATA_URL:=https://api.poker44.net}"
: "${POKER44_DASHBOARD_REPORT_URL:=https://api.poker44.net/api/v1/validator-events}"
: "${POKER44_POLL_INTERVAL_SECONDS:=300}"
: "${POKER44_BURN_FRACTION:=0.90}"
: "${POKER44_FUNDING_FRACTION:=0.05}"
: "${POKER44_FUNDING_HOTKEY:=5DUYX7X2Z9Jizr1NABUFDYV7ruFVNcUmKdxw9HxVP3sN9RUD}"
: "${POKER44_ENDPOINT_PRIVATE_KEY:=}"
: "${POKER44_ENDPOINT_PRIVATE_KEY_FILE:=}"
: "${POKER44_ENDPOINT_REFRESH_SECONDS:=300}"
: "${POKER44_ENDPOINT_AUTO_PROVISION:=true}"
: "${POKER44_ENDPOINT_PROVISIONING_URL:=https://api.poker44.net/internal/validators/runtime/endpoint-key}"
: "${POKER44_ENDPOINT_CACHE_FILE:=}"
export POKER44_SUBNET_DATA_URL POKER44_DASHBOARD_REPORT_URL
export POKER44_POLL_INTERVAL_SECONDS
export POKER44_BURN_FRACTION POKER44_FUNDING_FRACTION POKER44_FUNDING_HOTKEY
export POKER44_ENDPOINT_PRIVATE_KEY POKER44_ENDPOINT_PRIVATE_KEY_FILE
export POKER44_ENDPOINT_REFRESH_SECONDS POKER44_ENDPOINT_AUTO_PROVISION
export POKER44_ENDPOINT_PROVISIONING_URL POKER44_ENDPOINT_CACHE_FILE

command -v pm2 >/dev/null || { echo "pm2 is required" >&2; exit 1; }
test -f "$VALIDATOR_SCRIPT" || { echo "Missing $VALIDATOR_SCRIPT" >&2; exit 1; }
"$PYTHON_BIN" -c 'import bittensor, dotenv, nacl, numpy, sklearn, poker44' || {
  echo "Install the Poker44 runtime dependencies first" >&2; exit 1;
}

args=(
  "$VALIDATOR_SCRIPT"
  --netuid "$NETUID"
  --subtensor.network "$NETWORK"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --neuron.timeout "$NEURON_TIMEOUT"
  --neuron.num_concurrent_forwards 1
  --logging.info
)
if [[ -n "$WALLET_PATH" ]]; then args+=(--wallet.path "$WALLET_PATH"); fi
if [[ -n "$VALIDATOR_EXTRA_ARGS" ]]; then
  read -r -a extra <<< "$VALIDATOR_EXTRA_ARGS"
  args+=("${extra[@]}")
fi

pm2 delete "$PM2_NAME" >/dev/null 2>&1 || true
pm2 start "$PYTHON_BIN" --name "$PM2_NAME" -- "${args[@]}"

if [[ "$AUTO_UPDATE_ENABLED" == "true" ]]; then
  AUTO_UPDATE_SCRIPT="$(git rev-parse --show-toplevel)/scripts/validator/update/auto_update_validator.sh"
  test -f "$AUTO_UPDATE_SCRIPT" || { echo "Missing $AUTO_UPDATE_SCRIPT" >&2; exit 1; }
  pm2 delete "$AUTO_UPDATE_PM2_NAME" >/dev/null 2>&1 || true
  PROCESS_NAME="$PM2_NAME" \
  WALLET_NAME="$WALLET_NAME" \
  WALLET_HOTKEY="$HOTKEY" \
  SUBTENSOR_PARAM="--subtensor.network $NETWORK" \
  VALIDATOR_ENV_DIR="$VALIDATOR_ENV_DIR" \
  VALIDATOR_EXTRA_ARGS="$VALIDATOR_EXTRA_ARGS" \
  SLEEP_INTERVAL="$AUTO_UPDATE_INTERVAL_SECONDS" \
  pm2 start bash --name "$AUTO_UPDATE_PM2_NAME" -- "$AUTO_UPDATE_SCRIPT"
fi
pm2 save
echo "Started $PM2_NAME on netuid=$NETUID with $WALLET_NAME/$HOTKEY"
if [[ "$AUTO_UPDATE_ENABLED" == "true" ]]; then
  echo "Auto-update watcher: $AUTO_UPDATE_PM2_NAME every ${AUTO_UPDATE_INTERVAL_SECONDS}s"
fi
if [[ -n "$POKER44_ENDPOINT_PRIVATE_KEY" || -n "$POKER44_ENDPOINT_PRIVATE_KEY_FILE" ]]; then
  echo "Encrypted Axon endpoint resolver: enabled"
elif [[ "$POKER44_ENDPOINT_AUTO_PROVISION" == "true" ]]; then
  echo "Encrypted Axon endpoint resolver: automatic signed provisioning enabled"
else
  echo "Encrypted Axon endpoint resolver: disabled"
fi
