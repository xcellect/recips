from __future__ import annotations

import copy
from typing import Any, Optional, Tuple

import numpy as np

from core.perspective_model import make_obs_vector
from core.recon_core import clamp01, sigmoid


def _update_affect(s: dict, aff: Any, I_ext: float, motor: float, rng: np.random.Generator) -> None:
    if not bool(getattr(aff, "enabled", False)) or bool(s.get("lesion_affect", False)):
        return

    def stim_cost(i: float) -> float:
        if i >= 0.0:
            return float(getattr(aff, "stim_cost_pos", 1.0)) * abs(i)
        return -float(getattr(aff, "stim_gain_neg", 0.5)) * abs(i)

    demand = float(aff.demand_motor) * abs(motor) + float(aff.demand_stim) * stim_cost(float(I_ext))
    s["demand"] = float(demand)

    u = -float(aff.k_homeo) * (float(s.get("bb_model", 0.0)) - float(aff.setpoint))
    bb_pred = float(s.get("bb_model", 0.0)) + u
    bb_true = float(s.get("bb_true", 0.0)) + u - float(demand)
    bb_noise_std = float(getattr(aff, "bb_noise_std", 0.0))
    y = bb_true + (float(rng.normal(0.0, bb_noise_std)) if bb_noise_std > 0.0 else 0.0)

    pe = y - bb_pred
    bb_model = float(s.get("bb_model", 0.0)) + float(aff.k_pe) * pe
    dist = abs(bb_model - float(aff.setpoint))
    valence = 1.0 - dist / max(1e-9, float(aff.valence_scale))
    arousal = float(aff.arousal_scale) * (abs(pe) + abs(demand))

    s["bb_true"] = float(bb_true)
    s["bb_pred"] = float(bb_pred)
    s["pe"] = float(pe)
    s["bb_model"] = float(bb_model)
    s["valence"] = float(clamp01(valence))
    s["arousal"] = float(clamp01(arousal))


def perspective_step(
    state: dict,
    I_ext: float,
    loop_params: Any,
    affect_params: Any,
    obs_components: Optional[Tuple[float, float, float, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    s = copy.deepcopy(state)
    p = loop_params
    rg = rng or np.random.default_rng(0)

    obs_vec = make_obs_vector(I_ext, s, obs_components)
    s["drive_base"] = float(obs_vec[0])
    s["drive"] = float(obs_vec[0])

    pi = np.clip(
        float(p.pi0) + np.asarray(p.B_p) @ np.asarray(s.get("p", np.zeros(p.p_dim))) + float(p.b_ar) * float(s.get("arousal", 0.0)),
        float(p.pi_min),
        float(p.pi_max),
    )
    obs_gated = obs_vec.copy()
    obs_gated[:4] *= pi

    z_prev = np.asarray(s.get("z", np.zeros(p.z_dim)), dtype=float)
    pers_prev = np.asarray(s.get("p", np.zeros(p.p_dim)), dtype=float)
    delta_w = np.asarray(s.get("delta_w_in", np.zeros((p.z_dim, p.obs_dim))), dtype=float)

    recurrent = np.zeros_like(z_prev) if bool(s.get("lesion_feedback", False)) else np.asarray(p.W_rec) @ z_prev
    topdown = np.zeros(p.z_dim, dtype=float) if bool(s.get("lesion_feedback", False)) else np.asarray(p.W_pz) @ pers_prev
    z_prop = np.tanh((np.asarray(p.W_in0) + delta_w) @ obs_gated + recurrent + topdown)
    if bool(s.get("lesion_integrator", False)):
        z_t = z_prop
    else:
        z_t = (1.0 - float(p.alpha_z)) * z_prev + float(p.alpha_z) * z_prop

    p_prop = np.tanh(np.asarray(p.W_zp) @ z_t + np.asarray(p.W_pp) @ pers_prev)
    if bool(s.get("lesion_perspective", False)):
        p_t = np.zeros(p.p_dim, dtype=float)
    else:
        p_t = (1.0 - float(p.alpha_p)) * pers_prev + float(p.alpha_p) * p_prop

    lesion_affect = bool(s.get("lesion_affect", False))
    if bool(getattr(affect_params, "enabled", False)) and not lesion_affect:
        precision = float(getattr(affect_params, "precision_base", 1.0))
        if bool(getattr(affect_params, "modulate_precision", False)):
            precision *= 1.0 + float(getattr(affect_params, "k_precision_arousal", 0.0)) * float(s.get("arousal", 0.0))
        g_eff = float(s.get("g", getattr(p, "g", 1.0)))
        if bool(getattr(affect_params, "modulate_g", False)):
            unpleasant = 1.0 - float(s.get("valence", 1.0))
            g_eff *= 1.0 + float(getattr(affect_params, "k_g_arousal", 0.0)) * float(s.get("arousal", 0.0)) + float(getattr(affect_params, "k_g_unpleasant", 0.0)) * unpleasant
        g_eff = max(0.0, g_eff)
    else:
        precision = 1.0
        g_eff = float(s.get("g", getattr(p, "g", 1.0)))

    Ns = clamp01(sigmoid(float(np.asarray(p.w_ns) @ z_t + float(p.b_ns))))
    motor = clamp01(sigmoid(float(np.asarray(p.w_m) @ z_t + float(p.b_m))))
    d_e = float(getattr(p, "efference_decay", 0.7))
    eff_prev = float(s.get("efference", 0.0))
    eff = d_e * eff_prev + (1.0 - d_e) * abs(motor)
    reaff = clamp01(float(g_eff) * abs(motor))

    s["z"] = z_t
    s["p"] = p_t
    s["Ns"] = float(Ns)
    s["internal"] = float(np.mean(np.abs(z_t)))
    s["motor"] = float(motor)
    s["efference"] = float(eff)
    s["reafferent"] = float(reaff)
    s["precision_vec"] = np.asarray(pi, dtype=float)
    s["precision_eff"] = float(np.mean(pi))
    s["g_eff"] = float(g_eff)
    s["alpha_eff"] = float(p.alpha_z)

    _update_affect(s, affect_params, I_ext, motor, rg)

    mu = np.clip(
        float(p.c_pe) * abs(float(s.get("pe", 0.0)))
        + float(p.c_ar) * float(s.get("arousal", 0.0))
        + float(p.c_bb) * abs(float(s.get("bb_model", 0.0)) - float(getattr(affect_params, "setpoint", 0.0))),
        0.0,
        float(p.mu_max),
    )
    if bool(s.get("lesion_perspective", False)):
        open_t = float(sigmoid(float(p.b_open)))
    else:
        open_t = float(sigmoid(float(p.b_open) + float(np.asarray(p.v_open) @ p_t)))
    if bool(s.get("lesion_plasticity", False)):
        next_delta_w = delta_w
    else:
        o_tilde = obs_gated.copy()
        o_tilde[-1] = 0.0
        next_delta_w = np.clip(
            (1.0 - float(p.lambda_w)) * delta_w + float(p.eta_w) * mu * open_t * np.outer(z_t, o_tilde),
            -float(p.delta_max),
            float(p.delta_max),
        )
    s["delta_w_in"] = next_delta_w
    s["plasticity_mod"] = float(mu)
    s["plasticity_open"] = float(open_t)
    s["delta_w_norm"] = float(np.linalg.norm(next_delta_w))
    return s
