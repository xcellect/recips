import copy

import numpy as np

from core.driver.ipsundrum_dynamics import ipsundrum_step
from core.driver.perspective_dynamics import perspective_step
from core.ipsundrum_model import AffectParams, LoopParams
from core.perspective_model import initial_perspective_state, make_perspective_params


def test_same_observation_after_different_histories_yields_larger_perspective_separation_than_scalar_baseline():
    aff = AffectParams(enabled=True, valence_scale=3.0, k_homeo=0.10, k_pe=0.50, demand_motor=0.20, demand_stim=0.30, bb_noise_std=0.0)
    p_params = make_perspective_params(arch_seed=12)
    p_a = initial_perspective_state(p_params, plastic=True)
    p_b = initial_perspective_state(p_params, plastic=True)
    hist_a = [(0.8, 0.9, 0.4, -0.2)] * 5
    hist_b = [(-0.8, -0.2, -0.7, 0.1)] * 5
    for obs in hist_a:
        p_a = perspective_step(p_a, obs[0], p_params, aff, obs_components=obs, rng=np.random.default_rng(0))
    for obs in hist_b:
        p_b = perspective_step(p_b, obs[0], p_params, aff, obs_components=obs, rng=np.random.default_rng(0))
    same_obs = (0.15, 0.0, 0.2, 0.25)
    p_a2 = perspective_step(p_a, same_obs[0], p_params, aff, obs_components=same_obs, rng=np.random.default_rng(0))
    p_b2 = perspective_step(p_b, same_obs[0], p_params, aff, obs_components=same_obs, rng=np.random.default_rng(0))
    perspective_sep = float(
        np.linalg.norm(
            np.concatenate([p_a2["z"], p_a2["p"]])
            - np.concatenate([p_b2["z"], p_b2["p"]])
        )
    )

    loop = LoopParams(g=1.0, h=1.0, internal_decay=0.6, fatigue=0.02, nonlinearity="linear", saturation=True, sensor_bias=0.5, divisive_norm=0.8, efference_decay=0.7)
    s_a = {"reafferent": 0.0, "internal": 0.0, "motor": 0.0, "efference": 0.0, "g": 1.0, "bb_true": 0.0, "bb_model": 0.0, "bb_pred": 0.0, "pe": 0.0, "valence": 1.0, "arousal": 0.0}
    s_b = copy.deepcopy(s_a)
    for obs in hist_a:
        s_a = ipsundrum_step(s_a, obs[0], loop, aff, rng=np.random.default_rng(0))
    for obs in hist_b:
        s_b = ipsundrum_step(s_b, obs[0], loop, aff, rng=np.random.default_rng(0))
    s_a2 = ipsundrum_step(s_a, same_obs[0], loop, aff, rng=np.random.default_rng(0))
    s_b2 = ipsundrum_step(s_b, same_obs[0], loop, aff, rng=np.random.default_rng(0))
    scalar_sep = abs(float(s_a2["internal"]) - float(s_b2["internal"]))
    assert perspective_sep > scalar_sep
