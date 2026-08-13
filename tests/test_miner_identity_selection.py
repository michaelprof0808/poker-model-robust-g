from types import SimpleNamespace

from poker44.validator.evaluation.mixin import ValidatorEvaluationMixin


class Harness(ValidatorEvaluationMixin):
    def __init__(self):
        self.uid = 0
        self.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
        self.metagraph = SimpleNamespace(
            axons=[
                SimpleNamespace(ip="127.0.0.1", port=9000),
                SimpleNamespace(ip="127.0.0.1", port=9001),
                SimpleNamespace(ip="127.0.0.1", port=9002),
                SimpleNamespace(ip="127.0.0.1", port=9003),
            ],
            hotkeys=["validator", "miner-a", "miner-b", "miner-c"],
            coldkeys=["validator-cold", "same-cold", "same-cold", "other-cold"],
            validator_permit=[True, False, False, False],
        )


def test_candidate_selection_queries_every_hotkey_even_when_coldkey_is_shared():
    uids, _ = Harness()._candidate_miners("window-1")

    assert uids == [1, 2, 3]


def test_candidate_selection_has_no_arbitrary_miner_cap():
    harness = Harness()
    miner_count = 40
    harness.metagraph = SimpleNamespace(
        axons=[SimpleNamespace(ip="127.0.0.1", port=9000 + uid) for uid in range(miner_count + 1)],
        hotkeys=["validator"] + [f"miner-{uid}" for uid in range(1, miner_count + 1)],
        coldkeys=["validator-cold"] + [f"cold-{uid}" for uid in range(1, miner_count + 1)],
        validator_permit=[True] + [False] * miner_count,
    )

    uids, _ = harness._candidate_miners("window-all")

    assert uids == list(range(1, miner_count + 1))
