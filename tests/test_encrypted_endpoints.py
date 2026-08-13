import hashlib
import json
import stat
from types import SimpleNamespace
import urllib.error

import pytest
from nacl.public import Box, PrivateKey, PublicKey

from poker44.utils import encrypted_endpoints as endpoints


def _keypair():
    private_key = PrivateKey.generate()
    return bytes(private_key), bytes(private_key.public_key)


def _commitment_row(hotkey, ciphertext, block=1):
    return (
        hotkey,
        {
            "block": block,
            "info": {
                "fields": [
                    {
                        f"Raw{len(ciphertext)}": f"0x{ciphertext.hex()}",
                    }
                ]
            },
        },
    )


def _provisioning_response(
    request,
    *,
    endpoint_private_key,
    validator_hotkey,
    key_id="test-key-v1",
    fingerprint=None,
):
    request_payload = json.loads(request.data.decode("utf-8"))
    transport_public_key = PublicKey(
        bytes.fromhex(request_payload["transport_public_key"])
    )
    server_private_key = PrivateKey.generate()
    plaintext = json.dumps(
        {
            "version": 1,
            "key_id": key_id,
            "validator_hotkey": validator_hotkey,
            "private_key": bytes(endpoint_private_key).hex(),
        }
    ).encode("utf-8")
    encrypted = Box(server_private_key, transport_public_key).encrypt(plaintext)
    endpoint_fingerprint = hashlib.sha256(
        bytes(endpoint_private_key.public_key)
    ).hexdigest()[:16]
    envelope = {
        "version": 1,
        "algorithm": endpoints.PROVISIONING_ALGORITHM,
        "key_id": key_id,
        "key_fingerprint": fingerprint or endpoint_fingerprint,
        "server_public_key": bytes(server_private_key.public_key).hex(),
        "nonce": bytes(encrypted.nonce).hex(),
        "ciphertext": bytes(encrypted.ciphertext).hex(),
    }

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return json.dumps({"success": True, "data": envelope}).encode("utf-8")

    return Response()


class FakeSubtensor:
    def __init__(self, rows):
        self.rows = rows
        self.fail = False

    def query_map(self, **kwargs):
        assert kwargs == {
            "module": "Commitments",
            "name": "CommitmentOf",
            "params": [126],
        }
        if self.fail:
            raise RuntimeError("temporary RPC failure")
        return list(self.rows)


def test_resolver_reads_private_key_from_owner_only_file(monkeypatch, tmp_path):
    private_key, _ = _keypair()
    key_file = tmp_path / "endpoint.key"
    key_file.write_text(private_key.hex(), encoding="ascii")
    key_file.chmod(0o600)
    monkeypatch.delenv(endpoints.PRIVATE_KEY_ENV, raising=False)
    monkeypatch.setenv(endpoints.PRIVATE_KEY_FILE_ENV, str(key_file))

    resolver = endpoints.ValidatorEndpointResolver.from_env(
        subtensor=FakeSubtensor([]),
        netuid=126,
    )

    assert resolver.enabled is True
    assert resolver.private_key == private_key


def test_resolver_rejects_insecure_private_key_file(monkeypatch, tmp_path):
    private_key, _ = _keypair()
    key_file = tmp_path / "endpoint.key"
    key_file.write_text(private_key.hex(), encoding="ascii")
    key_file.chmod(0o644)
    monkeypatch.delenv(endpoints.PRIVATE_KEY_ENV, raising=False)
    monkeypatch.setenv(endpoints.PRIVATE_KEY_FILE_ENV, str(key_file))

    with pytest.raises(endpoints.EndpointProtectionError, match="group or others"):
        endpoints.ValidatorEndpointResolver.from_env(
            subtensor=FakeSubtensor([]),
            netuid=126,
        )


def test_resolver_rejects_ambiguous_private_key_configuration(
    monkeypatch,
    tmp_path,
):
    private_key, _ = _keypair()
    key_file = tmp_path / "endpoint.key"
    key_file.write_text(private_key.hex(), encoding="ascii")
    key_file.chmod(0o600)
    monkeypatch.setenv(endpoints.PRIVATE_KEY_ENV, private_key.hex())
    monkeypatch.setenv(endpoints.PRIVATE_KEY_FILE_ENV, str(key_file))

    with pytest.raises(endpoints.EndpointProtectionError, match="only one"):
        endpoints.ValidatorEndpointResolver.from_env(
            subtensor=FakeSubtensor([]),
            netuid=126,
        )


def test_resolver_auto_provisions_and_caches_key(monkeypatch, tmp_path):
    endpoint_private_key = PrivateKey.generate()
    fingerprint = hashlib.sha256(
        bytes(endpoint_private_key.public_key)
    ).hexdigest()[:16]
    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(
            ss58_address="validator-hotkey",
            sign=lambda _message: b"\x01\x02",
        )
    )
    cache_path = tmp_path / "endpoint-resolver.key"
    requests = []

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return _provisioning_response(
            request,
            endpoint_private_key=endpoint_private_key,
            validator_hotkey="validator-hotkey",
        )

    monkeypatch.delenv(endpoints.PRIVATE_KEY_ENV, raising=False)
    monkeypatch.delenv(endpoints.PRIVATE_KEY_FILE_ENV, raising=False)
    monkeypatch.setenv(endpoints.AUTO_PROVISION_ENV, "true")
    monkeypatch.setenv(
        endpoints.PROVISIONING_URL_ENV,
        "https://api.example.test/internal/validators/runtime/endpoint-key",
    )
    monkeypatch.setitem(
        endpoints.EXPECTED_PROVISIONED_KEY_FINGERPRINTS,
        126,
        fingerprint,
    )
    monkeypatch.setattr(endpoints.urllib.request, "urlopen", urlopen)

    resolver = endpoints.ValidatorEndpointResolver.from_env(
        subtensor=FakeSubtensor([]),
        netuid=126,
        wallet=wallet,
        cache_path=cache_path,
    )

    assert resolver.enabled is True
    assert resolver.private_key == bytes(endpoint_private_key)
    assert resolver.key_source == "provisioned:test-key-v1"
    assert resolver.provisioning_error == ""
    assert cache_path.read_text(encoding="ascii").strip() == bytes(
        endpoint_private_key
    ).hex()
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600
    assert len(requests) == 1
    request, timeout = requests[0]
    assert timeout == 5.0
    assert request.get_header("X-validator-hotkey") == "validator-hotkey"
    assert request.get_header("X-validator-signature") == "0102"


def test_resolver_uses_secure_cache_when_provisioning_is_unavailable(
    monkeypatch,
    tmp_path,
):
    endpoint_private_key = PrivateKey.generate()
    fingerprint = hashlib.sha256(
        bytes(endpoint_private_key.public_key)
    ).hexdigest()[:16]
    cache_path = tmp_path / "endpoint-resolver.key"
    cache_path.write_text(bytes(endpoint_private_key).hex(), encoding="ascii")
    cache_path.chmod(0o600)
    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(
            ss58_address="validator-hotkey",
            sign=lambda _message: b"\x01",
        )
    )
    monkeypatch.delenv(endpoints.PRIVATE_KEY_ENV, raising=False)
    monkeypatch.delenv(endpoints.PRIVATE_KEY_FILE_ENV, raising=False)
    monkeypatch.setenv(endpoints.AUTO_PROVISION_ENV, "true")
    monkeypatch.setitem(
        endpoints.EXPECTED_PROVISIONED_KEY_FINGERPRINTS,
        126,
        fingerprint,
    )
    monkeypatch.setattr(
        endpoints.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("offline")
        ),
    )

    resolver = endpoints.ValidatorEndpointResolver.from_env(
        subtensor=FakeSubtensor([]),
        netuid=126,
        wallet=wallet,
        cache_path=cache_path,
    )

    assert resolver.enabled is True
    assert resolver.private_key == bytes(endpoint_private_key)
    assert resolver.key_source == "cache"
    assert resolver.provisioning_error == "Provisioning server could not be reached"


def test_resolver_rejects_insecure_provisioning_cache(monkeypatch, tmp_path):
    endpoint_private_key = PrivateKey.generate()
    cache_path = tmp_path / "endpoint-resolver.key"
    cache_path.write_text(bytes(endpoint_private_key).hex(), encoding="ascii")
    cache_path.chmod(0o644)
    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(
            ss58_address="validator-hotkey",
            sign=lambda _message: b"\x01",
        )
    )
    monkeypatch.delenv(endpoints.PRIVATE_KEY_ENV, raising=False)
    monkeypatch.delenv(endpoints.PRIVATE_KEY_FILE_ENV, raising=False)
    monkeypatch.setenv(endpoints.AUTO_PROVISION_ENV, "true")
    monkeypatch.setattr(
        endpoints.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("offline")
        ),
    )

    resolver = endpoints.ValidatorEndpointResolver.from_env(
        subtensor=FakeSubtensor([]),
        netuid=126,
        wallet=wallet,
        cache_path=cache_path,
    )

    assert resolver.enabled is False
    assert resolver.key_source == "disabled"


def test_resolver_rejects_untrusted_provisioned_fingerprint(monkeypatch, tmp_path):
    endpoint_private_key = PrivateKey.generate()
    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(
            ss58_address="validator-hotkey",
            sign=lambda _message: b"\x01",
        )
    )
    monkeypatch.delenv(endpoints.PRIVATE_KEY_ENV, raising=False)
    monkeypatch.delenv(endpoints.PRIVATE_KEY_FILE_ENV, raising=False)
    monkeypatch.setenv(endpoints.AUTO_PROVISION_ENV, "true")
    monkeypatch.setitem(
        endpoints.EXPECTED_PROVISIONED_KEY_FINGERPRINTS,
        126,
        "0000000000000000",
    )
    monkeypatch.setattr(
        endpoints.urllib.request,
        "urlopen",
        lambda request, timeout: _provisioning_response(
            request,
            endpoint_private_key=endpoint_private_key,
            validator_hotkey="validator-hotkey",
        ),
    )

    resolver = endpoints.ValidatorEndpointResolver.from_env(
        subtensor=FakeSubtensor([]),
        netuid=126,
        wallet=wallet,
        cache_path=tmp_path / "endpoint-resolver.key",
    )

    assert resolver.enabled is False
    assert resolver.key_source == "disabled"
    assert "fingerprint" in resolver.provisioning_error


def test_configured_private_key_skips_auto_provisioning(monkeypatch, tmp_path):
    private_key, _ = _keypair()
    monkeypatch.setenv(endpoints.PRIVATE_KEY_ENV, private_key.hex())
    monkeypatch.delenv(endpoints.PRIVATE_KEY_FILE_ENV, raising=False)
    monkeypatch.setattr(
        endpoints.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("manual configuration must take precedence")
        ),
    )

    resolver = endpoints.ValidatorEndpointResolver.from_env(
        subtensor=FakeSubtensor([]),
        netuid=126,
        wallet=SimpleNamespace(),
        cache_path=tmp_path / "unused.key",
    )

    assert resolver.enabled is True
    assert resolver.key_source == "configured"


def test_endpoint_round_trip_is_bound_to_hotkey():
    private_key, public_key = _keypair()
    ciphertext = endpoints.encrypt_endpoint(
        "1.1.1.1",
        8091,
        public_key,
        hotkey="miner-hotkey",
    )

    assert endpoints.decrypt_endpoint(
        ciphertext,
        private_key,
        expected_hotkey="miner-hotkey",
    ) == ("1.1.1.1", 8091)

    with pytest.raises(endpoints.EndpointProtectionError, match="hotkey"):
        endpoints.decrypt_endpoint(
            ciphertext,
            private_key,
            expected_hotkey="different-hotkey",
        )


def test_endpoint_rejects_unversioned_plaintext():
    private_key, public_key = _keypair()
    _, _, PublicKey, SealedBox = endpoints._load_nacl()
    ciphertext = SealedBox(PublicKey(public_key)).encrypt(
        b"miner-hotkey|1.1.1.1:8091"
    )

    with pytest.raises(endpoints.EndpointProtectionError, match="format"):
        endpoints.decrypt_endpoint(
            ciphertext,
            private_key,
            expected_hotkey="miner-hotkey",
        )


def test_maximum_ipv4_payload_fits_chain_commitment_limit():
    _, public_key = _keypair()
    ciphertext = endpoints.encrypt_endpoint(
        "223.255.255.254",
        65535,
        public_key,
        hotkey="5" + ("A" * 47),
    )

    assert len(ciphertext) <= endpoints.MAX_COMMITMENT_BYTES


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "192.0.2.1", "not-an-ip"])
def test_endpoint_rejects_non_public_addresses(ip):
    _, public_key = _keypair()
    with pytest.raises(endpoints.EndpointProtectionError):
        endpoints.encrypt_endpoint(ip, 8091, public_key, hotkey="miner-hotkey")


def test_resolver_supports_mixed_public_and_protected_miners():
    private_key, public_key = _keypair()
    ciphertext = endpoints.encrypt_endpoint(
        "8.8.8.8",
        9001,
        public_key,
        hotkey="protected-hotkey",
    )
    subtensor = FakeSubtensor([_commitment_row("protected-hotkey", ciphertext, 42)])
    resolver = endpoints.ValidatorEndpointResolver(
        subtensor=subtensor,
        netuid=126,
        private_key_hex=private_key.hex(),
    )

    assert resolver.refresh(["public-hotkey", "protected-hotkey"], force=True) == 1

    masked_axon = SimpleNamespace(
        ip=endpoints.MASKED_AXON_IP,
        port=endpoints.MASKED_AXON_PORT,
    )
    resolved_axon, resolved = resolver.resolve("protected-hotkey", masked_axon)
    assert resolved is True
    assert (resolved_axon.ip, resolved_axon.port) == ("8.8.8.8", 9001)
    assert (masked_axon.ip, masked_axon.port) == (
        endpoints.MASKED_AXON_IP,
        endpoints.MASKED_AXON_PORT,
    )

    public_axon = SimpleNamespace(ip="9.9.9.9", port=9002)
    unchanged_axon, resolved = resolver.resolve("public-hotkey", public_axon)
    assert resolved is False
    assert unchanged_axon is public_axon


def test_resolver_does_not_override_a_public_axon_with_a_stale_commitment():
    private_key, public_key = _keypair()
    ciphertext = endpoints.encrypt_endpoint(
        "8.8.8.8",
        9001,
        public_key,
        hotkey="miner-a",
    )
    resolver = endpoints.ValidatorEndpointResolver(
        subtensor=FakeSubtensor([_commitment_row("miner-a", ciphertext)]),
        netuid=126,
        private_key_hex=private_key.hex(),
    )
    assert resolver.refresh(["miner-a"], force=True) == 1

    public_axon = SimpleNamespace(ip="9.9.9.9", port=9002)
    resolved_axon, resolved = resolver.resolve("miner-a", public_axon)

    assert resolved is False
    assert resolved_axon is public_axon


def test_resolver_rejects_commitment_copied_to_another_hotkey():
    private_key, public_key = _keypair()
    ciphertext = endpoints.encrypt_endpoint(
        "8.8.4.4",
        9001,
        public_key,
        hotkey="miner-a",
    )
    subtensor = FakeSubtensor(
        [
            _commitment_row("miner-a", ciphertext),
            _commitment_row("miner-b", ciphertext),
        ]
    )
    resolver = endpoints.ValidatorEndpointResolver(
        subtensor=subtensor,
        netuid=126,
        private_key_hex=private_key.hex(),
    )

    assert resolver.refresh(["miner-a", "miner-b"], force=True) == 1
    assert resolver.protected_hotkeys == frozenset({"miner-a"})


def test_resolver_keeps_cache_when_commitment_iteration_fails():
    private_key, public_key = _keypair()
    ciphertext = endpoints.encrypt_endpoint(
        "1.0.0.1",
        9001,
        public_key,
        hotkey="miner-a",
    )
    subtensor = FakeSubtensor([_commitment_row("miner-a", ciphertext)])
    resolver = endpoints.ValidatorEndpointResolver(
        subtensor=subtensor,
        netuid=126,
        private_key_hex=private_key.hex(),
    )
    assert resolver.refresh(["miner-a"], force=True) == 1

    class BrokenRows:
        def query_map(self, **_kwargs):
            def rows():
                yield _commitment_row("miner-a", ciphertext)
                raise RuntimeError("incomplete RPC stream")

            return rows()

    resolver.subtensor = BrokenRows()
    assert resolver.refresh(["miner-a"], force=True) == 1
    assert resolver.protected_hotkeys == frozenset({"miner-a"})
    assert resolver.public_status() == {
        "enabled": True,
        "key_fingerprint": resolver.key_fingerprint,
        "protected_miners": 1,
        "last_refresh_succeeded": False,
        "last_refresh_error": "RuntimeError",
        "key_source": "configured",
        "provisioning_error": "",
    }


def test_commitment_account_bytes_are_normalized_to_ss58():
    from bittensor_wallet import Keypair

    raw_hotkey = bytes(range(32))
    expected = Keypair(public_key=raw_hotkey.hex(), ss58_format=42).ss58_address
    assert endpoints._key_to_ss58((tuple(raw_hotkey),)) == expected


def test_resolver_keeps_last_valid_state_during_rpc_failure():
    private_key, public_key = _keypair()
    ciphertext = endpoints.encrypt_endpoint(
        "1.0.0.1",
        9001,
        public_key,
        hotkey="miner-a",
    )
    subtensor = FakeSubtensor([_commitment_row("miner-a", ciphertext)])
    resolver = endpoints.ValidatorEndpointResolver(
        subtensor=subtensor,
        netuid=126,
        private_key_hex=private_key.hex(),
    )
    assert resolver.refresh(["miner-a"], force=True) == 1

    subtensor.fail = True
    assert resolver.refresh(["miner-a"], force=True) == 1
    assert resolver.protected_hotkeys == frozenset({"miner-a"})


def test_miner_masks_only_after_confirmed_publication(monkeypatch):
    private_key, public_key = _keypair()
    _ = private_key
    miner = SimpleNamespace(
        config=SimpleNamespace(netuid=126),
        axon=SimpleNamespace(
            ip="8.8.8.8",
            port=8091,
            external_ip="8.8.8.8",
            external_port=8091,
        ),
        wallet=SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address="miner-hotkey"),
        ),
        subtensor=object(),
    )
    monkeypatch.setenv(endpoints.PROTECTION_ENABLED_ENV, "true")
    monkeypatch.setenv(endpoints.PUBLIC_KEY_ENV, public_key.hex())

    monkeypatch.setattr(endpoints, "publish_endpoint_commitment", lambda **_: False)
    assert endpoints.enable_miner_endpoint_protection(miner) is False
    assert (miner.axon.external_ip, miner.axon.external_port) == ("8.8.8.8", 8091)

    monkeypatch.setattr(endpoints, "publish_endpoint_commitment", lambda **_: True)
    monkeypatch.setattr(endpoints, "verify_endpoint_commitment", lambda **_: True)
    assert endpoints.enable_miner_endpoint_protection(miner) is True
    assert (miner.axon.external_ip, miner.axon.external_port) == (
        endpoints.MASKED_AXON_IP,
        endpoints.MASKED_AXON_PORT,
    )


def test_miner_does_not_mask_when_commitment_readback_fails(monkeypatch):
    _, public_key = _keypair()
    miner = SimpleNamespace(
        config=SimpleNamespace(netuid=126),
        axon=SimpleNamespace(
            ip="8.8.8.8",
            port=8091,
            external_ip="8.8.8.8",
            external_port=8091,
        ),
        wallet=SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address="miner-hotkey"),
        ),
        subtensor=object(),
    )
    monkeypatch.setenv(endpoints.PROTECTION_ENABLED_ENV, "true")
    monkeypatch.setenv(endpoints.PUBLIC_KEY_ENV, public_key.hex())
    monkeypatch.setattr(endpoints, "publish_endpoint_commitment", lambda **_: True)
    monkeypatch.setattr(endpoints, "verify_endpoint_commitment", lambda **_: False)

    assert endpoints.enable_miner_endpoint_protection(miner) is False
    assert (miner.axon.external_ip, miner.axon.external_port) == (
        "8.8.8.8",
        8091,
    )


def test_disabled_miner_does_not_publish(monkeypatch):
    monkeypatch.delenv(endpoints.PROTECTION_ENABLED_ENV, raising=False)
    assert endpoints.enable_miner_endpoint_protection(SimpleNamespace()) is False


def test_miner_uses_published_subnet_key_when_enabled(monkeypatch):
    monkeypatch.setenv(endpoints.PROTECTION_ENABLED_ENV, "true")
    monkeypatch.delenv(endpoints.PUBLIC_KEY_ENV, raising=False)
    miner = SimpleNamespace(
        config=SimpleNamespace(netuid=126),
        axon=SimpleNamespace(
            ip="8.8.8.8",
            port=8091,
            external_ip="8.8.8.8",
            external_port=8091,
        ),
        wallet=SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address="miner-hotkey"),
        ),
        subtensor=object(),
    )
    published = {}

    def publish(**kwargs):
        published.update(kwargs)
        return True

    monkeypatch.setattr(endpoints, "publish_endpoint_commitment", publish)
    monkeypatch.setattr(endpoints, "verify_endpoint_commitment", lambda **_: True)

    assert endpoints.enable_miner_endpoint_protection(miner) is True
    assert published["netuid"] == 126
    assert len(published["ciphertext"]) <= endpoints.MAX_COMMITMENT_BYTES
    assert hashlib.sha256(
        bytes.fromhex(endpoints.DEFAULT_PUBLIC_KEYS[126])
    ).hexdigest()[:16] == "d4a77a56d68268df"


def test_publish_uses_extrinsic_success_flag(monkeypatch):
    from bittensor.core.extrinsics import serving

    response = SimpleNamespace(success=False)
    monkeypatch.setattr(
        serving,
        "publish_metadata_extrinsic",
        lambda **_: response,
    )
    assert endpoints.publish_endpoint_commitment(
        subtensor=object(),
        wallet=object(),
        netuid=126,
        ciphertext=b"ciphertext",
    ) is False

    response.success = True
    assert endpoints.publish_endpoint_commitment(
        subtensor=object(),
        wallet=object(),
        netuid=126,
        ciphertext=b"ciphertext",
    ) is True


def test_commitment_readback_requires_exact_ciphertext():
    _, public_key = _keypair()
    ciphertext = endpoints.encrypt_endpoint(
        "8.8.8.8",
        8091,
        public_key,
        hotkey="miner-hotkey",
    )
    subtensor = FakeSubtensor(
        [_commitment_row("miner-hotkey", ciphertext, block=42)]
    )

    assert endpoints.read_endpoint_commitment(
        subtensor,
        126,
        "miner-hotkey",
    ) == ciphertext
    assert endpoints.verify_endpoint_commitment(
        subtensor,
        126,
        "miner-hotkey",
        ciphertext,
        attempts=1,
        delay_seconds=0,
    )
    assert not endpoints.verify_endpoint_commitment(
        subtensor,
        126,
        "miner-hotkey",
        b"different",
        attempts=1,
        delay_seconds=0,
    )


def test_end_to_end_publish_verify_mask_and_resolve(monkeypatch):
    private_key, public_key = _keypair()

    class InMemoryChain:
        def __init__(self):
            self.rows = []

        def query_map(self, **kwargs):
            assert kwargs == {
                "module": "Commitments",
                "name": "CommitmentOf",
                "params": [126],
            }
            return list(self.rows)

    chain = InMemoryChain()
    miner = SimpleNamespace(
        config=SimpleNamespace(netuid=126),
        axon=SimpleNamespace(
            ip="8.8.8.8",
            port=8091,
            external_ip="8.8.8.8",
            external_port=8091,
        ),
        wallet=SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address="miner-hotkey"),
        ),
        subtensor=chain,
    )
    monkeypatch.setenv(endpoints.PROTECTION_ENABLED_ENV, "true")
    monkeypatch.setenv(endpoints.PUBLIC_KEY_ENV, public_key.hex())

    def publish(**kwargs):
        chain.rows = [
            _commitment_row(
                "miner-hotkey",
                kwargs["ciphertext"],
                block=42,
            )
        ]
        return True

    monkeypatch.setattr(endpoints, "publish_endpoint_commitment", publish)

    assert endpoints.enable_miner_endpoint_protection(miner) is True
    assert (miner.axon.external_ip, miner.axon.external_port) == (
        endpoints.MASKED_AXON_IP,
        endpoints.MASKED_AXON_PORT,
    )

    resolver = endpoints.ValidatorEndpointResolver(
        subtensor=chain,
        netuid=126,
        private_key_hex=private_key.hex(),
    )
    assert resolver.refresh(["miner-hotkey"], force=True) == 1

    masked_axon = SimpleNamespace(
        ip=endpoints.MASKED_AXON_IP,
        port=endpoints.MASKED_AXON_PORT,
    )
    resolved_axon, resolved = resolver.resolve("miner-hotkey", masked_axon)
    assert resolved is True
    assert (resolved_axon.ip, resolved_axon.port) == ("8.8.8.8", 8091)


def test_candidate_selection_uses_resolved_endpoint(monkeypatch):
    from poker44.validator.evaluation.mixin import ValidatorEvaluationMixin

    public_axon = SimpleNamespace(ip="9.9.9.9", port=9002)
    masked_axon = SimpleNamespace(
        ip=endpoints.MASKED_AXON_IP,
        port=endpoints.MASKED_AXON_PORT,
    )

    class FakeResolver:
        def resolve(self, hotkey, axon):
            if hotkey == "protected-hotkey":
                return SimpleNamespace(ip="8.8.8.8", port=9001), True
            return axon, False

    validator = SimpleNamespace(
        metagraph=SimpleNamespace(
            axons=[
                SimpleNamespace(ip="0.0.0.0", port=0),
                masked_axon,
                public_axon,
            ],
            hotkeys=["uid-zero", "protected-hotkey", "public-hotkey"],
            coldkeys=["validator-cold", "protected-cold", "public-cold"],
            validator_permit=[False, False, False],
        ),
        uid=0,
        wallet=SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address="validator-hotkey")
        ),
        endpoint_resolver=FakeResolver(),
        refresh_encrypted_endpoints=lambda: 1,
    )
    uids, axons = ValidatorEvaluationMixin._candidate_miners(
        validator, "window-protected"
    )

    assert set(uids) == {1, 2}
    assert {
        uid: (axon.ip, axon.port) for uid, axon in zip(uids, axons)
    } == {1: ("8.8.8.8", 9001), 2: ("9.9.9.9", 9002)}


def test_disabled_resolver_preserves_public_candidate_behavior(monkeypatch):
    from poker44.validator.evaluation.mixin import ValidatorEvaluationMixin

    class QueryMustNotRun:
        def query_map(self, **_kwargs):
            raise AssertionError("disabled resolver must not query commitments")

    public_axon = SimpleNamespace(ip="9.9.9.9", port=9002)
    resolver = endpoints.ValidatorEndpointResolver(
        subtensor=QueryMustNotRun(),
        netuid=126,
        private_key_hex="",
    )
    validator = SimpleNamespace(
        metagraph=SimpleNamespace(
            axons=[
                SimpleNamespace(ip="0.0.0.0", port=0),
                public_axon,
            ],
            hotkeys=["uid-zero", "public-hotkey"],
            coldkeys=["validator-cold", "public-cold"],
            validator_permit=[False, False],
        ),
        uid=0,
        wallet=SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address="validator-hotkey")
        ),
        endpoint_resolver=resolver,
        refresh_encrypted_endpoints=lambda: resolver.refresh(
            ["uid-zero", "public-hotkey"]
        ),
    )
    uids, axons = ValidatorEvaluationMixin._candidate_miners(
        validator, "window-public"
    )

    assert uids == [1]
    assert axons == [public_axon]
    assert axons[0] is public_axon


def test_candidate_selection_skips_unresolved_masked_endpoint(monkeypatch):
    from poker44.validator.evaluation.mixin import ValidatorEvaluationMixin

    masked_axon = SimpleNamespace(
        ip=endpoints.MASKED_AXON_IP,
        port=endpoints.MASKED_AXON_PORT,
    )

    class EmptyResolver:
        def resolve(self, _hotkey, axon):
            return axon, False

    validator = SimpleNamespace(
        metagraph=SimpleNamespace(
            axons=[
                SimpleNamespace(ip="0.0.0.0", port=0),
                masked_axon,
            ],
            hotkeys=["uid-zero", "protected-hotkey"],
            coldkeys=["validator-cold", "protected-cold"],
            validator_permit=[False, False],
        ),
        uid=0,
        wallet=SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address="validator-hotkey")
        ),
        endpoint_resolver=EmptyResolver(),
        refresh_encrypted_endpoints=lambda: 0,
    )
    uids, axons = ValidatorEvaluationMixin._candidate_miners(
        validator, "window-unresolved"
    )

    assert uids == []
    assert axons == []


def test_validator_recovers_endpoint_provisioning_without_restart(
    monkeypatch,
    tmp_path,
):
    from poker44.base.validator import BaseValidatorNeuron

    class TestValidator(BaseValidatorNeuron):
        async def forward(self):
            return None

    private_key, _ = _keypair()
    subtensor = FakeSubtensor([])
    disabled = endpoints.ValidatorEndpointResolver(
        subtensor=subtensor,
        netuid=126,
        private_key_hex="",
    )
    recovered = endpoints.ValidatorEndpointResolver(
        subtensor=subtensor,
        netuid=126,
        private_key_hex=private_key.hex(),
    )
    validator = object.__new__(TestValidator)
    validator.subtensor = subtensor
    validator.config = SimpleNamespace(netuid=126)
    validator.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address="validator-hotkey")
    )
    validator.metagraph = SimpleNamespace(hotkeys=["validator-hotkey"])
    validator.endpoint_resolver = disabled
    validator._endpoint_cache_path = tmp_path / "endpoint.key"
    validator._endpoint_provision_retry_at = 0.0
    provision_calls = []

    def provision(cls, **kwargs):
        provision_calls.append(kwargs)
        return recovered

    monkeypatch.setattr(
        endpoints.ValidatorEndpointResolver,
        "from_env",
        classmethod(provision),
    )

    assert validator.refresh_encrypted_endpoints(force=True) == 0
    assert validator.endpoint_resolver is recovered
    assert validator.endpoint_resolver.enabled
    assert len(provision_calls) == 1
