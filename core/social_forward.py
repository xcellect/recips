from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from core.driver.ipsundrum_dynamics import ipsundrum_step
from core.social_homeostat import (
    HomeostatParams,
    SocialCouplingParams,
    SocialObservation,
    homeostat_from_net_dict,
    initial_homeostat,
    social_state_to_net_dict,
    step_homeostat,
)


@dataclass
class SocialForwardContext:
    env_model: Any
    homeostat_params: HomeostatParams
    social_params: SocialCouplingParams
    partner_homeostat_params: Optional[HomeostatParams] = None
    scripted_partner: Any = None


@dataclass
class SocialRollout:
    self_state: Dict[str, float]
    partner_state: Dict[str, float]
    observation: Dict[str, Any]


def ensure_social_state_dict(net_state: Dict[str, Any], params: HomeostatParams, *, default_energy: float = 0.7) -> Dict[str, Any]:
    state = dict(net_state)
    if "energy_true" not in state:
        init = social_state_to_net_dict(initial_homeostat(default_energy, params), params)
        for key, value in init.items():
            state.setdefault(key, value)
    state.setdefault("alive", True)
    return state


def social_predict_one_step(
    model_state: Dict[str, Any],
    action_self: str,
    *,
    social_ctx: SocialForwardContext,
    loop: Any,
    aff: Any,
    I_ext: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    eval_info: Any = None,
) -> SocialRollout:
    rg = rng or np.random.default_rng(0)
    self_state = ensure_social_state_dict(model_state, social_ctx.homeostat_params)
    partner_state = ensure_social_state_dict(
        dict(model_state.get("partner_state", {})),
        social_ctx.partner_homeostat_params or social_ctx.homeostat_params,
    )

    sensor_state = ipsundrum_step(self_state, I_ext, loop, aff, rng=rg)
    transition = social_ctx.env_model.predict_transition(
        self_state=self_state,
        partner_state=partner_state,
        action=action_self,
        eval_info=eval_info,
    )

    partner_social = step_homeostat(
        homeostat_from_net_dict(partner_state, social_ctx.partner_homeostat_params or social_ctx.homeostat_params),
        social_ctx.partner_homeostat_params or social_ctx.homeostat_params,
        SocialCouplingParams(lambda_affective=0.0),
        move_amount=float(transition.get("move_amount_partner", 0.0)),
        hazard_contact=float(transition.get("hazard_contact_partner", 0.0)),
        self_ate=float(transition.get("partner_ate", 0.0)),
        received_food=float(transition.get("received_food_partner", 0.0)),
        rng=rg,
    )
    observed_partner_energy = transition.get("other_energy_observed", partner_social.state.energy_true)
    observed_partner_expression = transition.get("other_expression", partner_social.state.distress_self)
    self_social = step_homeostat(
        homeostat_from_net_dict(self_state, social_ctx.homeostat_params),
        social_ctx.homeostat_params,
        social_ctx.social_params,
        move_amount=float(transition.get("move_amount_self", 0.0)),
        hazard_contact=float(transition.get("hazard_contact_self", 0.0)),
        self_ate=float(transition.get("self_ate", 0.0)),
        received_food=float(transition.get("received_food_self", 0.0)),
        partner_observation=SocialObservation(
            other_energy_est=observed_partner_energy,
            other_expression=observed_partner_expression,
            shuffled_energy_est=transition.get("shuffled_other_energy", partner_state.get("energy_true")),
        ),
        rng=rg,
    )

    sensor_state.update(social_state_to_net_dict(self_social.state, social_ctx.homeostat_params))
    sensor_state["x"] = float(transition.get("next_x", self_state.get("x", 0.0)))
    sensor_state["has_food"] = float(transition.get("has_food_next", self_state.get("has_food", 0.0)))
    sensor_state["partner_state"] = social_state_to_net_dict(
        partner_social.state,
        social_ctx.partner_homeostat_params or social_ctx.homeostat_params,
    )
    return SocialRollout(self_state=sensor_state, partner_state=sensor_state["partner_state"], observation=transition)


def predict_one_step_social(
    state: Dict[str, Any],
    loop: Any,
    aff: Any,
    I_ext: float,
    rng: np.random.Generator,
    **kwargs: Any,
) -> Dict[str, Any]:
    social_ctx = kwargs.get("social_ctx")
    action = kwargs.get("action", "stay")
    eval_info = kwargs.get("eval_info")
    if social_ctx is None:
        return ipsundrum_step(state, I_ext, loop, aff, rng=rng)
    rollout = social_predict_one_step(
        state,
        action,
        social_ctx=social_ctx,
        loop=loop,
        aff=aff,
        I_ext=I_ext,
        rng=rng,
        eval_info=eval_info,
    )
    return copy.deepcopy(rollout.self_state)
