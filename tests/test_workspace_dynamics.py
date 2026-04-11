import copy

import numpy as np

from core.driver.workspace_dynamics import workspace_step
from core.driver.workspace_forward import predict_one_step_workspace
from core.ipsundrum_model import AffectParams
from core.model_factory import build_attached_stage_d_network
from core.workspace_model import initial_workspace_state, make_workspace_params


def _aff() -> AffectParams:
    return AffectParams(enabled=True, valence_scale=3.0, k_homeo=0.10, k_pe=0.50, demand_motor=0.20, demand_stim=0.30, bb_noise_std=0.0)


def test_workspace_step_deterministic_and_non_mutating():
    params = make_workspace_params(arch_seed=5)
    state = initial_workspace_state(params)
    state["workspace"] = np.linspace(-0.2, 0.2, params.w_dim)
    state_copy = copy.deepcopy(state)
    obs = (0.2, -0.5, 0.7, 0.1)
    out1 = workspace_step(state, obs[0], params, _aff(), obs_components=obs, rng=np.random.default_rng(0))
    out2 = workspace_step(state, obs[0], params, _aff(), obs_components=obs, rng=np.random.default_rng(0))
    assert np.allclose(out1["workspace"], out2["workspace"])
    assert np.allclose(out1["selector_weights"], out2["selector_weights"])
    assert np.allclose(state["workspace"], state_copy["workspace"])


def test_workspace_forward_matches_online_update():
    params = make_workspace_params(arch_seed=8)
    base_state = initial_workspace_state(params)
    base_state["workspace"] = np.linspace(0.1, -0.2, params.w_dim)
    aff = _aff()
    net = build_attached_stage_d_network(
        params=params,
        affect=aff,
        initial_state=copy.deepcopy(base_state),
        step_fn=workspace_step,
        efference_threshold=0.05,
    )
    net.start_root(True)
    net._ipsundrum_state.clear()  # type: ignore[attr-defined]
    net._ipsundrum_state.update(copy.deepcopy(base_state))  # type: ignore[attr-defined]
    obs = (0.4, -0.2, 0.0, 0.8)
    pred = predict_one_step_workspace(base_state, params, aff, obs[0], np.random.default_rng(0), obs_components=obs)
    net._update_ipsundrum_sensor(obs[0], rng=np.random.default_rng(0), obs_components=obs)  # type: ignore[attr-defined]
    st = dict(net._ipsundrum_state)  # type: ignore[attr-defined]
    assert np.allclose(st["workspace"], pred["workspace"])
    assert np.allclose(st["selector_weights"], pred["selector_weights"])
    for key in ("Ns", "motor", "efference", "bb_model"):
        assert np.isclose(float(st[key]), float(pred[key]))


def test_selector_lesion_changes_selector_weights_and_internal_state():
    params = make_workspace_params(arch_seed=10)
    aff = _aff()
    state = initial_workspace_state(params)
    obs = (0.6, 0.1, -0.7, 0.2)
    intact = workspace_step(state, obs[0], params, aff, obs_components=obs, rng=np.random.default_rng(0))
    state["lesion_selector"] = True
    lesioned = workspace_step(state, obs[0], params, aff, obs_components=obs, rng=np.random.default_rng(0))
    assert not np.allclose(intact["selector_weights"], lesioned["selector_weights"])
    assert np.allclose(lesioned["selector_weights"], np.full(params.n_modalities, 0.25))
