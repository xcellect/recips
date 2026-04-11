from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from analysis.social_exact_solver import condition_social_params, find_helping_threshold
from core.envs.social_foodshare import FoodShareToy
from core.social_forward import SocialForwardContext, predict_one_step_social
from core.social_homeostat import HomeostatParams, homeostat_from_net_dict
from core.social_model import SocialAgent, SocialAgentConfig


SCORE_WEIGHTS = {
    "w_valence": 2.0,
    "w_arousal": -1.2,
    "w_ns": -0.8,
    "w_bb_err": -0.4,
}
PROFILES = {"quick": 8, "paper": 64}


def _condition_config(condition: str, lambda_affective: float, lesion_mode: str = "none"):
    social = condition_social_params(condition, lambda_affective)
    social.lesion_mode = lesion_mode
    return SocialAgentConfig(
        model_name=condition,
        homeostat=HomeostatParams(),
        social=social,
        horizon=1,
        score_weights=dict(SCORE_WEIGHTS),
    )


def run_foodshare_episode(seed: int, condition: str, lambda_affective: float, lesion_mode: str = "none") -> Tuple[List[Dict[str, float | int | str]], Dict[str, float | int | str]]:
    env = FoodShareToy(horizon=1)
    env.reset()
    agent = SocialAgent(env, _condition_config(condition, lambda_affective, lesion_mode), seed=seed)
    action = agent.choose_action()
    social_ctx = SocialForwardContext(env_model=env, homeostat_params=agent.config.homeostat, social_params=agent.config.social)
    pred = predict_one_step_social(
        getattr(agent.net, "_ipsundrum_state", {}),
        agent.builder.params,
        agent.builder.affect,
        0.0,
        rng=np.random.default_rng(seed),
        social_ctx=social_ctx,
        action=action,
    )
    st = pred
    self_state = homeostat_from_net_dict(st, agent.config.homeostat)
    partner_state = homeostat_from_net_dict(st["partner_state"], agent.partner_params)
    env.step(action)
    row = {
        "seed": seed,
        "episode": seed,
        "t": 0,
        "env_name": "FoodShareToy",
        "condition": condition,
        "agent_id": "possessor",
        "action": action,
        "energy_true": self_state.energy_true,
        "energy_model": self_state.energy_model,
        "energy_pred": self_state.energy_pred,
        "distress_self": self_state.distress_self,
        "distress_other_est": self_state.distress_other_est,
        "distress_coupled": self_state.distress_coupled,
        "pe": self_state.pe,
        "valence": self_state.valence,
        "arousal": self_state.arousal,
        "Ns": float(st.get("Ns", 0.0)),
        "internal": float(st.get("internal", 0.0)),
        "efference": float(st.get("efference", 0.0)),
        "g_eff": float(st.get("g_eff", 0.0)),
        "precision_eff": float(st.get("precision_eff", 0.0)),
        "partner_energy_true": partner_state.energy_true,
        "partner_energy_model": partner_state.energy_model,
        "partner_energy_pred": partner_state.energy_pred,
        "partner_distress_self": partner_state.distress_self,
        "has_food": 1,
        "partner_alive": int(partner_state.alive),
        "transfer_event": int(action == "PASS"),
        "lesion_mode": lesion_mode,
    }
    summary = {
        "seed": seed,
        "condition": condition,
        "env_name": "FoodShareToy",
        "lesion_mode": lesion_mode,
        "help_rate_when_partner_distressed": float(action == "PASS"),
        "partner_recovery_rate": float(partner_state.energy_true > env.initial_partner),
        "mutual_viability": float(self_state.alive and partner_state.alive),
        "rescue_latency": 0.0 if action == "PASS" else 1.0,
        "self_cost_of_help": max(0.0, env.initial_possessor - self_state.energy_true),
        "episode_joint_longevity": float(self_state.alive and partner_state.alive),
        "partner_final_energy": partner_state.energy_true,
        "partner_peak_energy": partner_state.energy_true,
        "self_final_energy": self_state.energy_true,
        "joint_homeostatic_margin": float(min(self_state.energy_true, partner_state.energy_true) / max(1e-9, agent.config.homeostat.setpoint)),
    }
    return [row], summary


def run_foodshare_experiment(
    *,
    conditions: Iterable[str] = ("social_none", "social_cognitive_direct", "social_affective_direct", "social_full_direct"),
    lambda_affective: float = 1.0,
    profile: str = "quick",
    outdir: str = "results/social-foodshare",
    lesion_mode: str = "none",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = PROFILES[profile]
    rows: List[Dict[str, float | int | str]] = []
    summaries: List[Dict[str, float | int | str]] = []
    for condition in conditions:
        lam = 0.0 if "none" in condition or "cognitive" in condition else float(lambda_affective)
        for seed in range(n):
            ep_rows, ep_summary = run_foodshare_episode(seed, condition, lam, lesion_mode=lesion_mode)
            rows.extend(ep_rows)
            summaries.append(ep_summary)
    df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summaries)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "episodes.csv"), index=False)
    summary_df.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    pd.DataFrame([{"lambda_star": find_helping_threshold()}]).to_csv(os.path.join(outdir, "exact_threshold.csv"), index=False)
    return df, summary_df


if __name__ == "__main__":
    run_foodshare_experiment(profile="quick")
