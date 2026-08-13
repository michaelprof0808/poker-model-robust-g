# Poker44 contracts

`subject-session.v4.1.schema.json` is the only miner-visible data contract: four
strategic decisions with at least one postflop decision. Private labels, actor
groups and source provenance remain exclusively in the signed validator lease.

The Bittensor transport is `MicroSessionDetectionSynapse` with
`contract_version = "microsession-v1"`. There is no version negotiation or
fallback. Validator observability uses dashboard event schema v3 and never
controls consensus.
