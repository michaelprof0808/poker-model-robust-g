import pytest

from poker44.validator.evaluation.reward import reward


def test_perfect_session_classifier_receives_full_reward():
    result = reward([0.01, 0.05, 0.95, 0.99], [0, 0, 1, 1])

    assert result.reward == pytest.approx(1.0, abs=0.01)
    assert result.average_precision == pytest.approx(1.0)
    assert result.recall_at_fpr_05 == pytest.approx(1.0)


def test_single_class_window_cannot_generate_reward():
    result = reward([0.1, 0.2], [0, 0])
    assert result.reward == 0.0


def test_single_class_e2e_reward_is_explicit_and_uses_calibration():
    result = reward([0.8, 1.0], [1, 1], allow_single_class=True)

    assert result.reward == pytest.approx(0.98)
    assert result.brier_skill == pytest.approx(0.98)
    assert result.accuracy == pytest.approx(1.0)


def test_constant_prevalence_prediction_has_zero_skill_reward():
    result = reward([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1])

    assert result.reward == pytest.approx(0.0)
    assert result.average_precision_skill == pytest.approx(0.0)
    assert result.recall_at_fpr_05 == pytest.approx(0.0)
    assert result.brier_skill == pytest.approx(0.0)


def test_tied_scores_are_evaluated_as_one_threshold():
    result = reward([0.9, 0.9, 0.1, 0.1], [1, 0, 1, 0])

    assert result.recall_at_fpr_05 == pytest.approx(0.0)


def test_invalid_score_is_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        reward([1.1, 0.2], [1, 0])


def test_actor_balanced_weights_prevent_one_actor_from_dominating():
    # The duplicated positive actor has four items, but the two actors in each
    # class retain equal total influence.
    result = reward(
        [0.1, 0.2, 0.9, 0.9, 0.9, 0.9],
        [0, 0, 1, 1, 1, 1],
        sample_weights=[0.25, 0.25, 0.125, 0.125, 0.125, 0.125],
    )
    assert result.average_precision == pytest.approx(1.0)
    assert result.reward == pytest.approx(1.0, abs=0.03)
