import numpy as np

from core.envs.social_foodshare import FoodShareToy
from core.ipsundrum_model import AffectParams, LoopParams
from core.social_forward import SocialForwardContext, predict_one_step_social
from core.social_homeostat import HomeostatParams, SocialCouplingParams, initial_homeostat, social_state_to_net_dict


def _base_state(self_energy=0.55, partner_energy=0.2):
    params = HomeostatParams()
    state = social_state_to_net_dict(initial_homeostat(self_energy, params), params)
    state["partner_state"] = social_state_to_net_dict(initial_homeostat(partner_energy, params), params)
    state["x"] = 0.0
    state["has_food"] = 0.0
    return state


def test_pass_decreases_predicted_partner_distress():
    env = FoodShareToy(horizon=1)
    ctx = SocialForwardContext(
        env_model=env,
        homeostat_params=HomeostatParams(),
        social_params=SocialCouplingParams(lambda_affective=0.75),
    )
    loop = LoopParams(sensor_bias=0.5)
    aff = AffectParams(enabled=True, setpoint=0.7)
    stay = predict_one_step_social(_base_state(), loop, aff, 0.0, rng=np.random.default_rng(0), social_ctx=ctx, action="STAY")
    pas = predict_one_step_social(_base_state(), loop, aff, 0.0, rng=np.random.default_rng(0), social_ctx=ctx, action="PASS")
    assert pas["partner_state"]["distress_self"] < stay["partner_state"]["distress_self"]


def test_sham_lesion_leaves_trajectory_unchanged():
    env = FoodShareToy(horizon=1)
    loop = LoopParams(sensor_bias=0.5)
    aff = AffectParams(enabled=True, setpoint=0.7)
    sham_ctx = SocialForwardContext(
        env_model=env,
        homeostat_params=HomeostatParams(),
        social_params=SocialCouplingParams(lambda_affective=0.75, lesion_mode="sham"),
    )
    base_ctx = SocialForwardContext(
        env_model=env,
        homeostat_params=HomeostatParams(),
        social_params=SocialCouplingParams(lambda_affective=0.75, lesion_mode="none"),
    )
    sham = predict_one_step_social(_base_state(), loop, aff, 0.0, rng=np.random.default_rng(0), social_ctx=sham_ctx, action="PASS")
    base = predict_one_step_social(_base_state(), loop, aff, 0.0, rng=np.random.default_rng(0), social_ctx=base_ctx, action="PASS")
    assert sham["distress_coupled"] == base["distress_coupled"]
