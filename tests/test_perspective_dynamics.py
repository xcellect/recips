import copy

import numpy as np

from core.driver.perspective_dynamics import perspective_step
from core.driver.perspective_forward import predict_one_step_perspective
from core.ipsundrum_model import AffectParams
from core.model_factory import build_attached_stage_d_network
from core.perspective_model import initial_perspective_state, make_perspective_params


def _aff() -> AffectParams:
    return AffectParams(
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


def test_perspective_step_deterministic_and_non_mutating():
    params = make_perspective_params(arch_seed=3)
    state = initial_perspective_state(params, plastic=True)
    state["z"] = np.linspace(-0.2, 0.3, params.z_dim)
    state["p"] = np.linspace(-0.1, 0.2, params.p_dim)
    state["bb_model"] = 0.1
    state["bb_true"] = -0.2
    state["valence"] = 0.7
    state["arousal"] = 0.3
    state_copy = copy.deepcopy(state)
    obs = (0.25, 0.8, -0.1, 0.05)

    out1 = perspective_step(state, obs[0], params, _aff(), obs_components=obs, rng=np.random.default_rng(0))
    out2 = perspective_step(state, obs[0], params, _aff(), obs_components=obs, rng=np.random.default_rng(0))

    assert np.allclose(out1["z"], out2["z"])
    assert np.allclose(out1["p"], out2["p"])
    assert np.allclose(out1["delta_w_in"], out2["delta_w_in"])
    assert np.isclose(out1["Ns"], out2["Ns"])
    assert np.allclose(state["z"], state_copy["z"])
    assert np.allclose(state["p"], state_copy["p"])
    assert np.allclose(state["delta_w_in"], state_copy["delta_w_in"])


def test_perspective_forward_matches_online_update():
    params = make_perspective_params(arch_seed=4)
    base_state = initial_perspective_state(params, plastic=True)
    base_state["z"] = np.linspace(-0.1, 0.2, params.z_dim)
    base_state["p"] = np.linspace(0.05, -0.1, params.p_dim)
    aff = _aff()
    net = build_attached_stage_d_network(
        params=params,
        affect=aff,
        initial_state=copy.deepcopy(base_state),
        step_fn=perspective_step,
        efference_threshold=0.05,
    )
    net.start_root(True)
    net._ipsundrum_state.clear()  # type: ignore[attr-defined]
    net._ipsundrum_state.update(copy.deepcopy(base_state))  # type: ignore[attr-defined]
    obs = (-0.35, 0.0, -0.6, 0.1)

    pred = predict_one_step_perspective(base_state, params, aff, obs[0], np.random.default_rng(0), obs_components=obs)
    net._update_ipsundrum_sensor(obs[0], rng=np.random.default_rng(0), obs_components=obs)  # type: ignore[attr-defined]
    st = dict(net._ipsundrum_state)  # type: ignore[attr-defined]

    assert np.allclose(st["z"], pred["z"])
    assert np.allclose(st["p"], pred["p"])
    assert np.allclose(st["delta_w_in"], pred["delta_w_in"])
    for key in ("Ns", "motor", "efference", "bb_model", "valence", "arousal"):
        assert np.isclose(float(st[key]), float(pred[key]))


def test_perspective_slow_latent_changes_more_slowly_than_fast_latent():
    params = make_perspective_params(arch_seed=7)
    state = initial_perspective_state(params, plastic=False)
    aff = _aff()
    obs = (0.45, 0.9, 0.2, -0.1)
    dz = []
    dp = []
    for _ in range(12):
        prev = copy.deepcopy(state)
        state = perspective_step(state, obs[0], params, aff, obs_components=obs, rng=np.random.default_rng(0))
        dz.append(float(np.linalg.norm(state["z"] - prev["z"])))
        dp.append(float(np.linalg.norm(state["p"] - prev["p"])))
    assert np.mean(dp) < np.mean(dz) * 0.5


def test_plasticity_lesion_freezes_delta_w():
    params = make_perspective_params(arch_seed=9)
    aff = _aff()
    state = initial_perspective_state(params, plastic=True)
    obs = (0.3, 0.7, 0.2, 0.1)
    updated = perspective_step(state, obs[0], params, aff, obs_components=obs, rng=np.random.default_rng(0))
    assert float(np.linalg.norm(updated["delta_w_in"])) > 0.0
    updated["lesion_plasticity"] = True
    frozen = perspective_step(updated, obs[0], params, aff, obs_components=obs, rng=np.random.default_rng(0))
    assert np.allclose(frozen["delta_w_in"], updated["delta_w_in"])
