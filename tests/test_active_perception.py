from core.driver.active_perception import score_internal


class DummyAff:
    setpoint = 0.0


def test_score_internal_epistemic_zero_invariant():
    s = {
        "valence": 0.4,
        "arousal": 0.2,
        "Ns": 0.1,
        "bb_model": 0.05,
    }
    aff = DummyAff()
    score_low = score_internal(s, aff, current_I=0.0, predicted_I=0.0, w_epistemic=0.0)
    score_high = score_internal(s, aff, current_I=0.0, predicted_I=1.0, w_epistemic=0.0)
    assert abs(score_low - score_high) < 1e-9
