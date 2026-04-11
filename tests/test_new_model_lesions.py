import numpy as np

from core.driver.perspective_dynamics import perspective_step
from core.driver.workspace_dynamics import workspace_step
from core.ipsundrum_model import AffectParams
from core.perspective_model import initial_perspective_state, make_perspective_params
from core.workspace_model import initial_workspace_state, make_workspace_params


def _aff() -> AffectParams:
    return AffectParams(enabled=True, valence_scale=3.0, k_homeo=0.10, k_pe=0.50, demand_motor=0.20, demand_stim=0.30, bb_noise_std=0.0)


def _hysteresis_score(params, aff, lesion_perspective: bool) -> float:
    state = initial_perspective_state(params, plastic=False)
    state["lesion_perspective"] = lesion_perspective
    up = []
    down = []
    lambdas = np.linspace(0.0, 1.0, 9)
    for lam in lambdas:
        obs = (lam, 0.8 * lam, 0.2 - lam, lam - 0.1)
        state = perspective_step(state, obs[0], params, aff, obs_components=obs, rng=np.random.default_rng(0))
        up.append(state["p"].copy())
    for lam in lambdas[::-1]:
        obs = (lam, 0.8 * lam, 0.2 - lam, lam - 0.1)
        state = perspective_step(state, obs[0], params, aff, obs_components=obs, rng=np.random.default_rng(0))
        down.append(state["p"].copy())
    return float(np.mean([np.linalg.norm(a - b) for a, b in zip(up, down[::-1])]))


def _robustness_score(params, aff, lesion_perspective: bool) -> float:
    state = initial_perspective_state(params, plastic=False)
    state["lesion_perspective"] = lesion_perspective
    clean = perspective_step(state, 0.4, params, aff, obs_components=(0.4, 0.0, 0.3, 0.5), rng=np.random.default_rng(0))
    corrupt = perspective_step(state, 0.4, params, aff, obs_components=(0.4, 0.0, 0.3, -0.5), rng=np.random.default_rng(0))
    return abs(float(clean["Ns"]) - float(corrupt["Ns"]))


def test_perspective_lesion_hits_hysteresis_more_than_corruption_robustness_proxy():
    params = make_perspective_params(arch_seed=2)
    aff = _aff()
    intact_h = _hysteresis_score(params, aff, False)
    lesion_h = _hysteresis_score(params, aff, True)
    intact_r = _robustness_score(params, aff, False)
    lesion_r = _robustness_score(params, aff, True)
    assert (intact_h - lesion_h) > abs(intact_r - lesion_r)


def test_selector_lesion_reduces_conflict_resolution_proxy_in_gw_lite():
    params = make_workspace_params(arch_seed=4)
    aff = _aff()
    state = initial_workspace_state(params)
    conflict_obs = (0.2, 0.0, 0.9, -0.9)
    intact = workspace_step(state, conflict_obs[0], params, aff, obs_components=conflict_obs, rng=np.random.default_rng(0))
    state["lesion_selector"] = True
    lesioned = workspace_step(state, conflict_obs[0], params, aff, obs_components=conflict_obs, rng=np.random.default_rng(0))
    assert float(np.max(intact["selector_weights"])) > float(np.max(lesioned["selector_weights"]))
