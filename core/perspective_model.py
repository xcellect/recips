from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class PerspectiveParams:
    obs_dim: int = 6
    z_dim: int = 8
    p_dim: int = 4
    alpha_z: float = 0.48
    alpha_p: float = 0.05
    pi0: float = 1.0
    pi_min: float = 0.25
    pi_max: float = 1.75
    b_ar: float = 0.25
    lambda_w: float = 0.02
    eta_w: float = 0.03
    delta_max: float = 0.35
    mu_max: float = 1.5
    c_pe: float = 0.8
    c_ar: float = 0.6
    c_bb: float = 0.4
    b_open: float = -0.25
    efference_decay: float = 0.7
    g: float = 1.0

    W_in0: np.ndarray | None = None
    W_rec: np.ndarray | None = None
    W_pz: np.ndarray | None = None
    W_zp: np.ndarray | None = None
    W_pp: np.ndarray | None = None
    B_p: np.ndarray | None = None
    v_open: np.ndarray | None = None
    w_ns: np.ndarray | None = None
    w_m: np.ndarray | None = None
    b_ns: float = 0.0
    b_m: float = 0.0

    def parameter_count(self) -> int:
        total = 0
        for key in (
            "W_in0",
            "W_rec",
            "W_pz",
            "W_zp",
            "W_pp",
            "B_p",
            "v_open",
            "w_ns",
            "w_m",
        ):
            val = getattr(self, key)
            if val is not None:
                total += int(np.asarray(val).size)
        total += 2
        return total


def make_perspective_params(arch_seed: int = 0) -> PerspectiveParams:
    rng = np.random.default_rng(int(arch_seed))

    def scaled(shape: Tuple[int, ...], scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, size=shape)

    params = PerspectiveParams()
    params.W_in0 = scaled((params.z_dim, params.obs_dim), 0.28)
    params.W_rec = scaled((params.z_dim, params.z_dim), 0.16)
    params.W_pz = scaled((params.z_dim, params.p_dim), 0.22)
    params.W_zp = scaled((params.p_dim, params.z_dim), 0.18)
    params.W_pp = scaled((params.p_dim, params.p_dim), 0.08)
    params.B_p = scaled((4, params.p_dim), 0.12)
    params.v_open = scaled((params.p_dim,), 0.18)
    params.w_ns = scaled((params.z_dim,), 0.35)
    params.w_m = scaled((params.z_dim,), 0.35)
    params.b_ns = float(rng.normal(0.0, 0.1))
    params.b_m = float(rng.normal(0.0, 0.1))
    return params


def initial_perspective_state(params: PerspectiveParams, *, plastic: bool) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "model_family": "perspective",
        "z": np.zeros(params.z_dim, dtype=float),
        "p": np.zeros(params.p_dim, dtype=float),
        "delta_w_in": np.zeros((params.z_dim, params.obs_dim), dtype=float),
        "Ns": 0.0,
        "internal": 0.0,
        "motor": 0.0,
        "efference": 0.0,
        "reafferent": 0.0,
        "precision_eff": 1.0,
        "precision_vec": np.ones(4, dtype=float),
        "g": float(params.g),
        "g_eff": float(params.g),
        "bb_true": 0.0,
        "bb_model": 0.0,
        "bb_pred": 0.0,
        "pe": 0.0,
        "valence": 1.0,
        "arousal": 0.0,
        "plasticity_mod": 0.0,
        "plasticity_open": 0.0,
        "delta_w_norm": 0.0,
        "drive": 0.0,
        "drive_base": 0.0,
        "alpha_eff": float(params.alpha_z),
        "lesion_integrator": False,
        "lesion_feedback": False,
        "lesion_affect": False,
        "lesion_perspective": False,
        "lesion_plasticity": not plastic,
    }
    return state


def perspective_builder(params: PerspectiveParams, affect: Any, *, plastic: bool) -> Any:
    score_weights = {
        "w_valence": 2.0,
        "w_arousal": -1.2,
        "w_ns": -0.8,
        "w_bb_err": -0.4,
        "w_epistemic": 0.35,
    }
    return SimpleNamespace(params=params, affect=affect, plastic=plastic, score_weights=score_weights)


def make_obs_vector(
    I_ext: float,
    state: Dict[str, Any],
    obs_components: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    if obs_components is None:
        total = float(I_ext)
        touch = max(0.0, total)
        smell = 0.5 * float(total)
        vision = float(total) - smell
    else:
        total, touch, smell, vision = [float(v) for v in obs_components]
    eff_prev = float(state.get("efference", 0.0))
    return np.array([total, touch, smell, vision, eff_prev, 1.0], dtype=float)
