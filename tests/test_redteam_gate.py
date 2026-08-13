from poker44.validator.evaluation.redteam_gate import audit_redteam_leakage


def strategic_session(
    action: str,
    *,
    context_shift: bool = False,
    schema_version: str = "4.1",
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "session_id": f"session_{action}_{context_shift}",
        "window_id": "window_example",
        "decisions": [
            {
                "decision_number": index + 1,
                "phase": "turn" if context_shift and index == 0 else (
                    "preflop" if index < 9 else "flop"
                ),
                "position_group": ("early", "late", "blinds")[index % 3],
                "pressure": "facing_bet" if index % 2 else "no_call",
                "action_type": action,
                "size_bucket": (
                    "half_pot" if action in {"bet", "raise"} else "not_applicable"
                ),
                "is_all_in": False,
            }
            for index in range(12)
        ],
    }


def test_gate_accepts_matched_contexts_when_only_strategy_differs() -> None:
    sessions = [
        strategic_session("check"),
        strategic_session("call"),
        strategic_session("raise"),
        strategic_session("fold"),
    ]
    result = audit_redteam_leakage(sessions, [0, 0, 1, 1], threshold=0.15)

    assert result.passed is True
    assert result.reward == 0.0


def test_gate_rejects_v3_context_distribution_shortcuts() -> None:
    sessions = [
        strategic_session("check"),
        strategic_session("call"),
        strategic_session("raise", context_shift=True),
        strategic_session("fold", context_shift=True),
    ]
    result = audit_redteam_leakage(sessions, [0, 0, 1, 1], threshold=0.15)

    assert result.passed is False
    assert result.feature == "strategic_context_signature"


def test_gate_accepts_multiple_contexts_when_balanced_per_class() -> None:
    sessions = [
        strategic_session("check"),
        strategic_session("call", context_shift=True),
        strategic_session("raise"),
        strategic_session("fold", context_shift=True),
    ]

    result = audit_redteam_leakage(sessions, [0, 0, 1, 1], threshold=0.15)

    assert result.passed is True
    assert result.reward == 0.0


def test_v41_matches_phase_pressure_but_audits_global_position_balance() -> None:
    sessions = [
        strategic_session("check", schema_version="4.1"),
        strategic_session("call", schema_version="4.1"),
        strategic_session("raise", schema_version="4.1"),
        strategic_session("fold", schema_version="4.1"),
    ]
    for item in sessions:
        item["decisions"] = item["decisions"][:4]
    for decision in sessions[2]["decisions"] + sessions[3]["decisions"]:
        decision["position_group"] = "early"

    result = audit_redteam_leakage(sessions, [0, 0, 1, 1], threshold=0.15)

    assert result.passed is False
    assert result.feature == "strategic_position_distribution"
