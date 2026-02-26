import pytest

from core.ipsundrum_model import Builder, LoopParams, AffectParams


def test_stage_b_efference_tracks_motor_command():
    params = LoopParams(efference_decay=0.5)
    b = Builder(params=params, affect=AffectParams(enabled=False))
    net, _ = b.stage_B()

    # set reflexive motor command via Ns activation
    net.get("Ns").activation = 0.8
    nm = net.get("Nm")
    nm.actuator_effect(0.8)

    ne = net.get("Ne")
    assert ne.value == pytest.approx(0.4)
