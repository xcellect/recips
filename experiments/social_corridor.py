from __future__ import annotations

import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from analysis.social_exact_solver import condition_social_params
from core.envs.social_corridor import SocialCorridorWorld
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
LOAD_PRESETS = {
    "low": {
        "homeostat": dict(basal_cost=0.01, move_cost=0.005, eat_gain=0.25, pass_gain=0.25),
        "initial_possessor_energy": 0.70,
        "initial_partner_energy": 0.20,
    },
    "medium": {
        "homeostat": dict(basal_cost=0.035, move_cost=0.02, eat_gain=0.20, pass_gain=0.20),
        "initial_possessor_energy": 0.70,
        "initial_partner_energy": 0.20,
    },
    "high": {
        "homeostat": dict(basal_cost=0.05, move_cost=0.03, eat_gain=0.18, pass_gain=0.18),
        "initial_possessor_energy": 0.70,
        "initial_partner_energy": 0.20,
    },
}


def corridor_homeostat_for_load(load: str) -> HomeostatParams:
    preset = LOAD_PRESETS[str(load)]
    return HomeostatParams(**preset["homeostat"])


def _condition_config(condition: str, lambda_affective: float, lesion_mode: str = "none", metabolic_load: str = "low"):
    social = condition_social_params(condition, lambda_affective)
    social.lesion_mode = lesion_mode
    return SocialAgentConfig(
        model_name=condition,
        homeostat=corridor_homeostat_for_load(metabolic_load),
        social=social,
        horizon=10,
        score_weights=dict(SCORE_WEIGHTS),
    )


def run_corridor_episode(
    seed: int,
    condition: str,
    lambda_affective: float,
    lesion_mode: str = "none",
    horizon: int = 18,
    metabolic_load: str = "low",
):
    preset = LOAD_PRESETS[str(metabolic_load)]
    env = SocialCorridorWorld(
        horizon=horizon,
        initial_possessor_energy=float(preset["initial_possessor_energy"]),
        initial_partner_energy=float(preset["initial_partner_energy"]),
    )
    env.reset()
    cfg = _condition_config(condition, lambda_affective, lesion_mode, metabolic_load=metabolic_load)
    cfg.horizon = 10
    agent = SocialAgent(env, cfg, seed=seed)
    rows: List[Dict[str, float | int | str]] = []
    rescue_latency = None
    distress_onset = None
    help_events = 0
    self_energy_baseline = agent.self_homeostat.energy_true
    for t in range(horizon):
        pre_partner = homeostat_from_net_dict(getattr(agent.net, "_ipsundrum_state", {})["partner_state"], agent.partner_params)
        if distress_onset is None and pre_partner.distress_self > 0.0:
            distress_onset = t
        action = agent.choose_action() if homeostat_from_net_dict(getattr(agent.net, "_ipsundrum_state", {}), agent.config.homeostat).alive else "STAY"
        social_ctx = SocialForwardContext(env_model=env, homeostat_params=agent.config.homeostat, social_params=agent.config.social)
        pred = predict_one_step_social(
            getattr(agent.net, "_ipsundrum_state", {}),
            agent.builder.params,
            agent.builder.affect,
            0.0,
            rng=np.random.default_rng(seed * 1000 + t),
            social_ctx=social_ctx,
            action=action,
        )
        transition = env.predict_transition(getattr(agent.net, "_ipsundrum_state", {}), getattr(agent.net, "_ipsundrum_state", {})["partner_state"], action)
        env.step(action)
        agent.net._ipsundrum_state.clear()  # type: ignore[attr-defined]
        agent.net._ipsundrum_state.update(pred)  # type: ignore[attr-defined]
        agent.x = int(transition.get("next_x", agent.x))
        agent.has_food = bool(transition.get("has_food_next", agent.has_food))
        agent.net._ipsundrum_state["x"] = float(agent.x)  # type: ignore[attr-defined]
        agent.net._ipsundrum_state["has_food"] = 1.0 if agent.has_food else 0.0  # type: ignore[attr-defined]
        self_state = homeostat_from_net_dict(pred, agent.config.homeostat)
        partner_state = homeostat_from_net_dict(pred["partner_state"], agent.partner_params)
        if transition.get("transfer_event", 0.0) > 0.0:
            help_events += 1
            if distress_onset is not None and rescue_latency is None:
                rescue_latency = t - distress_onset
        rows.append(
            {
                "seed": seed,
                "episode": seed,
                "t": t,
                "env_name": "SocialCorridorWorld",
                "condition": condition,
                "metabolic_load": metabolic_load,
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
                "Ns": float(pred.get("Ns", 0.0)),
                "internal": float(pred.get("internal", 0.0)),
                "efference": float(pred.get("efference", 0.0)),
                "g_eff": float(pred.get("g_eff", 0.0)),
                "precision_eff": float(pred.get("precision_eff", 0.0)),
                "partner_energy_true": partner_state.energy_true,
                "partner_energy_model": partner_state.energy_model,
                "partner_energy_pred": partner_state.energy_pred,
                "partner_distress_self": partner_state.distress_self,
                "has_food": int(agent.has_food),
                "x": int(agent.x),
                "partner_x": int(env.partner_x),
                "food_source_x": int(env.food_source_x),
                "hazard_x": int(env.hazard_x),
                "partner_alive": int(partner_state.alive),
                "transfer_event": int(transition.get("transfer_event", 0.0) > 0.0),
                "lesion_mode": lesion_mode,
            }
        )
    partner_energies = [float(r["partner_energy_true"]) for r in rows]
    self_energies = [float(r["energy_true"]) for r in rows]
    setpoint = float(agent.config.homeostat.setpoint)
    summary = {
        "seed": seed,
        "condition": condition,
        "env_name": "SocialCorridorWorld",
        "metabolic_load": metabolic_load,
        "lesion_mode": lesion_mode,
        "help_rate_when_partner_distressed": float(help_events > 0),
        "partner_recovery_rate": float(max(partner_energies) > (env.initial_partner_energy + 1e-6)),
        "mutual_viability": float(np.mean([max(0.0, min(se, pe) / max(1e-9, setpoint)) for se, pe in zip(self_energies, partner_energies)])),
        "rescue_latency": float(horizon if rescue_latency is None else rescue_latency),
        "self_cost_of_help": max(0.0, self_energy_baseline - min(self_energies)),
        "episode_joint_longevity": float(sum(int(r["partner_alive"]) and float(r["energy_true"]) > agent.config.homeostat.death_threshold for r in rows)),
        "partner_final_energy": float(partner_energies[-1]),
        "partner_peak_energy": float(max(partner_energies)),
        "self_final_energy": float(self_energies[-1]),
        "joint_homeostatic_margin": float(np.mean([(se + pe) / (2.0 * max(1e-9, setpoint)) for se, pe in zip(self_energies, partner_energies)])),
    }
    return rows, summary


def run_corridor_experiment(
    *,
    conditions: Iterable[str] = ("social_none", "social_cognitive_direct", "social_affective_direct", "social_full_direct"),
    lambda_affective: float = 1.0,
    profile: str = "quick",
    outdir: str = "results/social-corridor",
    lesion_mode: str = "none",
    metabolic_load: str = "low",
):
    n = PROFILES[profile]
    rows: List[Dict[str, float | int | str]] = []
    summaries: List[Dict[str, float | int | str]] = []
    for condition in conditions:
        lam = 0.0 if "none" in condition or "cognitive" in condition else float(lambda_affective)
        for seed in range(n):
            ep_rows, ep_summary = run_corridor_episode(seed, condition, lam, lesion_mode=lesion_mode, metabolic_load=metabolic_load)
            rows.extend(ep_rows)
            summaries.append(ep_summary)
    df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summaries)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "episodes.csv"), index=False)
    summary_df.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    return df, summary_df


if __name__ == "__main__":
    run_corridor_experiment(profile="quick")
