from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np

from core.recon_core import clamp01, sigmoid


def ipsundrum_step(
    state: dict,
    I_ext: float,
    loop_params: Any,
    affect_params: Any,
    modulation_params: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Pure one-step ipsundrum update aligned with ipsundrum_model._update_ipsundrum_sensor.

    Returns a NEW state dict and does not mutate the input state.
    """
    _ = modulation_params  # reserved for future use
    s = copy.deepcopy(state)
    loop = loop_params
    aff = affect_params
    rg = rng or np.random.default_rng(0)

    # LESION HANDLING
    lesion_integrator = bool(s.get("lesion_integrator", False))
    lesion_affect = bool(s.get("lesion_affect", False))
    lesion_feedback = bool(s.get("lesion_feedback", False))

    # Injections (override state values)
    if "inject_internal" in s:
        s["internal"] = float(s["inject_internal"])
    if "inject_reafferent" in s:
        s["reafferent"] = float(s["inject_reafferent"])
    if "inject_bb_model" in s:
        s["bb_model"] = float(s["inject_bb_model"])

    affect_enabled = bool(getattr(aff, "enabled", False)) and not lesion_affect
    I_drive = float(I_ext)
    if not affect_enabled:
        # No affect: treat negative I as neutral (no "pleasantness" benefit).
        I_drive = max(0.0, I_drive)

    # precision baseline
    precision = 1.0
    if affect_enabled and getattr(aff, "modulate_precision", False):
        precision = float(aff.precision_base) * (
            1.0 + float(aff.k_precision_arousal) * float(s.get("arousal", 0.0))
        )

    # g_eff
    g_base = float(s.get("g", getattr(loop, "g", 1.0)))
    g_eff = g_base
    if affect_enabled and getattr(aff, "modulate_g", False):
        unpleasant = 1.0 - float(s.get("valence", 1.0))
        g_eff = g_eff * (
            1.0
            + float(aff.k_g_arousal) * float(s.get("arousal", 0.0))
            + float(aff.k_g_unpleasant) * unpleasant
        )
        g_eff = max(0.0, g_eff)

    # drive
    reaff_prev = float(s.get("reafferent", 0.0))
    if lesion_feedback:
        reaff_prev = 0.0
        precision = 0.0

    noise_std = float(getattr(loop, "noise_std", 0.0))
    noise = float(rg.normal(0.0, noise_std)) if noise_std > 0.0 else 0.0
    drive_base = float(I_drive) + precision * reaff_prev + float(getattr(loop, "sensor_bias", 0.0)) + noise

    divn = float(getattr(loop, "divisive_norm", 0.0))
    if divn > 0.0:
        denom = 1.0 + divn * abs(precision * reaff_prev)
        drive = drive_base / denom
    else:
        drive = drive_base

    s["drive_base"] = float(drive_base)
    s["drive"] = float(drive)
    s["precision_eff"] = float(precision)
    s["g_eff"] = float(g_eff)

    # sensory nonlinearity
    if getattr(loop, "nonlinearity", "linear") == "linear":
        ns_val = float(drive)
    elif getattr(loop, "nonlinearity", "linear") == "sigmoid":
        ns_val = float(sigmoid(float(drive)))
    else:
        raise ValueError(f"Unknown nonlinearity: {getattr(loop, 'nonlinearity', None)}")

    if getattr(loop, "saturation", True):
        ns_val = float(clamp01(ns_val))

    s["Ns"] = float(ns_val)

    # integrator
    if lesion_integrator:
        d_eff = 0.0
        x = float(ns_val)
    else:
        d_eff = float(getattr(loop, "internal_decay", 0.6))
        x = d_eff * float(s.get("internal", 0.0)) + (1.0 - d_eff) * float(ns_val)
    s["internal"] = float(x)

    # motor
    m = float(getattr(loop, "h", 1.0)) * float(x)
    if getattr(loop, "saturation", True):
        m = float(clamp01(m))
    s["motor"] = float(m)

    # efference filtered
    d_e = float(getattr(loop, "efference_decay", 0.0))
    ne_prev = float(s.get("efference", 0.0))
    ne = d_e * ne_prev + (1.0 - d_e) * float(m)
    s["efference"] = float(ne)

    # reafferent
    e = float(g_eff) * float(m)
    if getattr(loop, "saturation", True):
        e = float(clamp01(e))
    s["reafferent"] = float(e)

    # Barrett interoception update
    def stim_cost(i: float) -> float:
        if i >= 0.0:
            return float(getattr(aff, "stim_cost_pos", 1.0)) * abs(i)
        return -float(getattr(aff, "stim_gain_neg", 0.5)) * abs(i)

    if affect_enabled:
        demand = float(aff.demand_motor) * abs(m) + float(aff.demand_stim) * stim_cost(float(I_ext))
        s["demand"] = float(demand)

        u = -float(aff.k_homeo) * (float(s.get("bb_model", 0.0)) - float(aff.setpoint))
        bb_pred = float(s.get("bb_model", 0.0)) + u

        bb_true = float(s.get("bb_true", 0.0)) + u - float(demand)
        bb_noise_std = float(getattr(aff, "bb_noise_std", 0.0))
        y = bb_true + (float(rg.normal(0.0, bb_noise_std)) if bb_noise_std > 0.0 else 0.0)

        pe = y - bb_pred
        bb_model = float(s.get("bb_model", 0.0)) + float(aff.k_pe) * pe

        dist = abs(bb_model - float(aff.setpoint))
        valence = 1.0 - dist / max(1e-9, float(aff.valence_scale))
        valence = float(clamp01(valence))

        arousal = float(aff.arousal_scale) * (abs(pe) + abs(demand))
        arousal = float(clamp01(arousal))

        s["bb_true"] = float(bb_true)
        s["bb_pred"] = float(bb_pred)
        s["pe"] = float(pe)
        s["bb_model"] = float(bb_model)
        s["valence"] = float(valence)
        s["arousal"] = float(arousal)
    else:
        for key in ("demand", "bb_true", "bb_model", "bb_pred", "pe", "valence", "arousal"):
            s.pop(key, None)

    # fatigue on g
    fatigue = float(getattr(loop, "fatigue", 0.0))
    if fatigue > 0.0 and (abs(m) > 1e-12 or abs(e) > 1e-12):
        g_base = max(0.0, float(g_base) * (1.0 - fatigue * abs(m)))
    s["g"] = float(g_base)

    # alpha_eff diagnostic
    alpha = d_eff + (1.0 - d_eff) * (float(g_eff) * float(getattr(loop, "h", 1.0)) * float(precision))
    s["alpha_eff"] = float(alpha)

    return s
