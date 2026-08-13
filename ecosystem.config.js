module.exports = { apps: [{
  name: 'poker-model-robust-a',
  script: '/opt/poker-model-robust-a/start_miner.sh',
  cwd: '/opt/poker-model-robust-a',
  env: {
    PATH: '/opt/poker-model-robust-a/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
    VIRTUAL_ENV: '/opt/poker-model-robust-a/.venv',
    PYTHONPATH: '/opt/poker-model-robust-a',
    WALLET_NAME: 'poker', HOTKEY: 'sn126_1', AXON_PORT: '8195',
    POKER44_MODEL_FACTORY: 'poker44_champion_v6.model:create_model',
    POKER44_MODEL_VERSION: 'round7-primary-v6-robust-1',
    POKER44_MAX_SESSIONS_PER_REQUEST: '256', POKER44_MAX_REQUEST_BYTES: '16777216',
    POKER44_ENCRYPTED_AXON_ENABLED: 'true', POKER44_AXON_EXTERNAL_IP: '161.35.119.64',
    POKER44_AXON_EXTERNAL_PORT: '8195'
  },
  autorestart: true, max_restarts: 5, restart_delay: 30000,
  max_memory_restart: '2G', kill_timeout: 10000,
  log_date_format: 'YYYY-MM-DDTHH:mm:ss'
}]};
