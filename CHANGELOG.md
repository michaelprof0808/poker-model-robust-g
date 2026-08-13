# Changelog

## 0.2.1 - 2026-07-31

- Trigger validator deployment of the production micro-session, 90/5/5
  settlement, dashboard reporting and reveal-reconciliation fixes.
- Start a PM2-supervised validator auto-update watcher by default.
- Verify the applied version and Git commit, retry failed deployments, support
  both documented PM2 validator names and keep updater state private.

## 0.2.0 - 2026-07-18

### Breaking

- Validator protocol spec increased to 2.
- Removed the `cycle` validator flow and old subnet-backend configuration
  aliases. Only signed platform round leases are supported.
- Validator reports require schema v2; schema-v1 reports are rejected.

### Added

- Local miner evaluation, reward calculation and on-chain weight settlement.
- Exponential failed-round backoff and terminal in-process quarantine.
- Explicit, disabled-by-default single-class scoring for bot-only release E2E.

### Fixed

- Failed settlement no longer hot-loops against the same lease.
- Weight submission evidence distinguishes commit acceptance and finalization.
