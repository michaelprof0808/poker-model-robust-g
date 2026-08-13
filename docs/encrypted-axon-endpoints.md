# Encrypted Axon Endpoints

Poker44 supports opt-in encrypted endpoint commitments to reduce public exposure
of miner origin IP addresses.

## Security Model

When protection is enabled:

1. the miner encrypts a versioned, hotkey-bound `IPv4:port` payload with the
   Poker44 endpoint public key;
2. the miner publishes the ciphertext through Bittensor commitment metadata;
3. the miner reads the finalized commitment back and requires an exact match;
4. only after verification succeeds, the miner advertises a non-routable
   placeholder endpoint in the metagraph;
5. updated validators decrypt the commitment and query a local copy of the Axon
   using the recovered endpoint.

The encrypted payload is bound to the miner hotkey. A commitment copied from a
different hotkey is rejected. Validators also reject private, loopback,
documentation-only, malformed, and out-of-range endpoints.

The mechanism hides the origin from public metagraph readers. It does not hide
the origin from authorized validators and does not replace upstream DDoS
scrubbing.

## Compatibility

Protection is disabled for miners unless explicitly enabled. Public miners keep
using their existing metagraph endpoints.

Validators without a trusted decryption key keep their existing behavior for
public miners and skip unresolved masked endpoints. Miners must not opt in until
Poker44 confirms that the active validator set has upgraded.

If commitment publication fails, the miner keeps its public endpoint. It never
switches to the placeholder endpoint on an unconfirmed publication.

## Miner Activation

Update dependencies and set:

```bash
export POKER44_ENCRYPTED_AXON_ENABLED=true
export POKER44_AXON_EXTERNAL_IP=<new_public_ipv4>
export POKER44_AXON_EXTERNAL_PORT=<axon_port>
```

The production public key for netuid `126` is included in the subnet package.
`POKER44_ENDPOINT_PUBLIC_KEY` may override it only for controlled test
deployments.

After Poker44 confirms validator readiness, a miner whose previous IP has
already been exposed should move to a new origin IP before the protected
restart.

## Validator Activation

Validators automatically request the shared key through a signed endpoint. The
provisioning response is encrypted to an ephemeral transport key, bound to the
requesting validator hotkey and checked against the fingerprint embedded in the
release. The resulting cache is written with owner-only permissions.

```bash
export POKER44_ENDPOINT_AUTO_PROVISION=true
export POKER44_ENDPOINT_PROVISIONING_URL=https://api.poker44.net/internal/validators/runtime/endpoint-key
export POKER44_ENDPOINT_CACHE_FILE=<owner_only_state_path>
export POKER44_ENDPOINT_REFRESH_SECONDS=300
```

Operators may instead configure exactly one of
`POKER44_ENDPOINT_PRIVATE_KEY` or `POKER44_ENDPOINT_PRIVATE_KEY_FILE`. Key
material must never be committed, logged or placed in process arguments.

The resolver supports mixed public and protected miners. Commitment RPC
failures retain the last valid in-memory endpoint set. Endpoint resolution
happens before reachability filtering. Every eligible miner hotkey remains an
independent candidate; there is no coldkey deduplication.
