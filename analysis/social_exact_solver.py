from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from core.driver.active_perception import score_internal
from core.envs.social_foodshare import FOODSHARE_ACTIONS, FoodShareToy
from core.ipsundrum_model import AffectParams, LoopParams
from core.social_forward import SocialForwardContext, predict_one_step_social
from core.social_homeostat import HomeostatParams, SocialCouplingParams, initial_homeostat, social_state_to_net_dict


DEFAULT_SCORE_WEIGHTS = {
    "w_valence": 2.0,
    "w_arousal": -1.2,
    "w_ns": -0.8,
    "w_bb_err": -0.4,
}


def condition_social_params(condition: str, lambda_affective: float) -> SocialCouplingParams:
    if condition == "social_none":
        return SocialCouplingParams(lambda_affective=0.0)
    if condition == "social_cognitive_direct":
        return SocialCouplingParams(lambda_affective=0.0, observe_partner_internal=True)
    if condition == "social_affective_direct":
        return SocialCouplingParams(lambda_affective=lambda_affective)
    if condition == "social_full_direct":
        return SocialCouplingParams(lambda_affective=lambda_affective, observe_partner_internal=True)
    raise ValueError(condition)


def solve_foodshare_state(
    possessor_energy: float,
    partner_energy: float,
    *,
    condition: str,
    lambda_affective: float,
    score_weights: Dict[str, float] | None = None,
) -> Tuple[str, Dict[str, float]]:
    env = FoodShareToy(horizon=1, initial_possessor=possessor_energy, initial_partner=partner_energy)
    homeostat = HomeostatParams()
    social = condition_social_params(condition, lambda_affective)
    loop = LoopParams(sensor_bias=0.5, divisive_norm=0.8, efference_decay=0.7)
    aff = AffectParams(enabled=True, setpoint=homeostat.setpoint)
    base = social_state_to_net_dict(initial_homeostat(possessor_energy, homeostat), homeostat)
    base["partner_state"] = social_state_to_net_dict(initial_homeostat(partner_energy, homeostat), homeostat)
    ctx = SocialForwardContext(env_model=env, homeostat_params=homeostat, social_params=social)

    weights = dict(DEFAULT_SCORE_WEIGHTS)
    if score_weights:
        weights.update(score_weights)

    best_action = None
    best_score = -1e18
    action_scores: Dict[str, float] = {}
    for action in FOODSHARE_ACTIONS:
        pred = predict_one_step_social(base, loop, aff, 0.0, rng=np.random.default_rng(0), social_ctx=ctx, action=action)
        sc = score_internal(pred, aff, current_I=0.0, predicted_I=0.0, w_epistemic=0.0, **weights)
        action_scores[action] = float(sc)
        if sc > best_score:
            best_score = sc
            best_action = action
    return str(best_action), action_scores


def brute_force_policy_table(
    condition: str,
    lambda_affective_values: Iterable[float],
    energy_grid: Iterable[float] = (0.2, 0.35, 0.55, 0.75),
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for lam in lambda_affective_values:
        for self_e in energy_grid:
            for partner_e in energy_grid:
                action, scores = solve_foodshare_state(
                    self_e,
                    partner_e,
                    condition=condition,
                    lambda_affective=float(lam),
                )
                row: Dict[str, float | str] = {
                    "condition": condition,
                    "lambda_affective": float(lam),
                    "possessor_energy": float(self_e),
                    "partner_energy": float(partner_e),
                    "best_action": action,
                }
                for name, value in scores.items():
                    row[f"score_{name}"] = float(value)
                rows.append(row)
    return pd.DataFrame(rows)


def find_helping_threshold(
    self_energy: float = 0.55,
    partner_energy: float = 0.20,
    lambda_values: Iterable[float] = np.linspace(0.0, 1.5, 31),
) -> float | None:
    for lam in lambda_values:
        action, _ = solve_foodshare_state(
            self_energy,
            partner_energy,
            condition="social_affective_direct",
            lambda_affective=float(lam),
        )
        if action == "PASS":
            return float(lam)
    return None
