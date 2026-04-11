from __future__ import annotations

import copy
from typing import Any, Optional, Tuple

import numpy as np

from core.recon_core import clamp01, sigmoid
from core.workspace_model import make_workspace_obs
from .perspective_dynamics import _update_affect


def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
    y = np.asarray(x, dtype=float) / max(1e-9, float(tau))
    y = y - np.max(y)
    e = np.exp(y)
    return e / np.sum(e)


def workspace_step(
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
    x_touch, x_smell, x_vision, x_eff, sal = make_workspace_obs(I_ext, s, obs_components)
    workspace_prev = np.asarray(s.get("workspace", np.zeros(p.w_dim)), dtype=float)
    embeddings = np.stack(
        [
            np.tanh(np.asarray(p.E_touch) @ x_touch),
            np.tanh(np.asarray(p.E_smell) @ x_smell),
            np.tanh(np.asarray(p.E_vision) @ x_vision),
            np.tanh(np.asarray(p.E_eff) @ x_eff),
        ],
        axis=0,
    )
    logits = np.asarray(p.b_sel) + np.asarray(p.U_sal) @ sal + np.asarray(p.U_ctx) @ workspace_prev
    if bool(s.get("lesion_selector", False)):
        weights = np.full(p.n_modalities, 1.0 / p.n_modalities, dtype=float)
    else:
        weights = _softmax(logits, p.tau_sel)
    broadcast = np.sum(weights[:, None] * embeddings, axis=0)
    w_prop = np.tanh(np.asarray(p.W_ww) @ workspace_prev + broadcast)
    if bool(s.get("lesion_workspace", False)):
        workspace_t = broadcast
    else:
        workspace_t = (1.0 - float(p.alpha_w)) * workspace_prev + float(p.alpha_w) * w_prop

    Ns = clamp01(sigmoid(float(np.asarray(p.r_ns) @ workspace_t + float(p.b_ns))))
    motor = clamp01(sigmoid(float(np.asarray(p.r_m) @ workspace_t + float(p.b_m))))
    d_e = float(getattr(p, "efference_decay", 0.7))
    eff_prev = float(s.get("efference", 0.0))
    eff = d_e * eff_prev + (1.0 - d_e) * abs(motor)
    g_eff = float(s.get("g", getattr(p, "g", 1.0)))
    reaff = clamp01(float(g_eff) * abs(motor))

    s["workspace"] = workspace_t
    s["selector_logits"] = logits
    s["selector_weights"] = weights
    s["Ns"] = float(Ns)
    s["internal"] = float(np.mean(np.abs(workspace_t)))
    s["motor"] = float(motor)
    s["efference"] = float(eff)
    s["reafferent"] = float(reaff)
    s["precision_eff"] = float(np.max(weights))
    s["g_eff"] = float(g_eff)
    s["drive_base"] = float(I_ext)
    s["drive"] = float(I_ext)
    s["alpha_eff"] = float(p.alpha_w)
    _update_affect(s, affect_params, I_ext, motor, rg)
    return s
