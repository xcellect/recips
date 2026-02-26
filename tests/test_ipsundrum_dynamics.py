import copy

import numpy as np

from core.driver.ipsundrum_dynamics import ipsundrum_step
from core.ipsundrum_model import Builder, LoopParams, AffectParams


def _random_state(rng: np.random.Generator) -> dict:
    return {
        "reafferent": float(rng.random()),
        "internal": float(rng.random()),
        "motor": float(rng.random()),
        "efference": float(rng.random()),
        "g": float(0.5 + 0.5 * rng.random()),
        "bb_true": float(rng.normal(0.0, 0.1)),
        "bb_model": float(rng.normal(0.0, 0.1)),
        "bb_pred": float(rng.normal(0.0, 0.1)),
        "pe": float(rng.normal(0.0, 0.1)),
        "valence": float(rng.random()),
        "arousal": float(rng.random()),
        "drive": 0.0,
        "drive_base": 0.0,
        "precision_eff": 1.0,
        "g_eff": 1.0,
        "demand": 0.0,
        "alpha_eff": 0.0,
        "lesion_integrator": False,
        "lesion_affect": False,
        "lesion_feedback": False,
    }


def test_ipsundrum_step_deterministic():
    rng = np.random.default_rng(123)
    state = _random_state(rng)
    state_copy = copy.deepcopy(state)

    loop = LoopParams(
        g=1.1,
        h=0.9,
        internal_decay=0.6,
        fatigue=0.02,
        nonlinearity="linear",
        saturation=True,
        sensor_bias=0.5,
        divisive_norm=0.8,
        efference_decay=0.7,
    )
    aff = AffectParams(
        enabled=True,
        valence_scale=3.0,
        k_homeo=0.10,
        k_pe=0.50,
        demand_motor=0.20,
        demand_stim=0.30,
        modulate_g=True,
        k_g_arousal=0.8,
        k_g_unpleasant=0.8,
        modulate_precision=True,
        precision_base=1.0,
        k_precision_arousal=0.5,
        bb_noise_std=0.0,
    )

    I_ext = 0.25
    out1 = ipsundrum_step(state, I_ext, loop, aff, rng=np.random.default_rng(0))
    out2 = ipsundrum_step(state, I_ext, loop, aff, rng=np.random.default_rng(0))

    for k in out1:
        assert np.isclose(float(out1[k]), float(out2[k]), atol=1e-9)

    for k, v in state_copy.items():
        assert np.isclose(float(state[k]), float(v), atol=1e-9)


def test_ipsundrum_step_matches_online_update():
    rng = np.random.default_rng(321)
    base_state = _random_state(rng)

    loop = LoopParams(
        g=1.0,
        h=1.0,
        internal_decay=0.6,
        fatigue=0.02,
        nonlinearity="linear",
        saturation=True,
        sensor_bias=0.5,
        divisive_norm=0.8,
        efference_decay=0.7,
    )
    aff = AffectParams(
        enabled=True,
        valence_scale=3.0,
        k_homeo=0.10,
        k_pe=0.50,
        demand_motor=0.20,
        demand_stim=0.30,
        modulate_g=True,
        k_g_arousal=0.8,
        k_g_unpleasant=0.8,
        modulate_precision=True,
        precision_base=1.0,
        k_precision_arousal=0.5,
        bb_noise_std=0.0,
    )

    b = Builder(params=loop, affect=aff)
    net, _ = b.stage_D(efference_threshold=0.05)
    net.start_root(True)

    net_state = net._ipsundrum_state  # type: ignore[attr-defined]
    net_state.clear()
    net_state.update(copy.deepcopy(base_state))

    I_ext = -0.4
    pred = ipsundrum_step(base_state, I_ext, loop, aff, rng=np.random.default_rng(0))
    net._update_ipsundrum_sensor(I_ext, rng=np.random.default_rng(0))  # type: ignore[attr-defined]
    st = dict(net._ipsundrum_state)  # type: ignore[attr-defined]

    keys = [
        "Ns",
        "internal",
        "motor",
        "efference",
        "reafferent",
        "g",
        "drive",
        "drive_base",
        "precision_eff",
        "g_eff",
        "demand",
        "bb_true",
        "bb_model",
        "bb_pred",
        "pe",
        "valence",
        "arousal",
        "alpha_eff",
    ]
    for key in keys:
        assert np.isclose(float(st.get(key, 0.0)), float(pred.get(key, 0.0)), atol=1e-9)
