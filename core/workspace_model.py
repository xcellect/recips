from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class WorkspaceParams:
    w_dim: int = 8
    modality_dim: int = 2
    n_modalities: int = 4
    alpha_w: float = 0.42
    tau_sel: float = 0.8
    efference_decay: float = 0.7
    g: float = 1.0

    E_touch: np.ndarray | None = None
    E_smell: np.ndarray | None = None
    E_vision: np.ndarray | None = None
    E_eff: np.ndarray | None = None
    U_sal: np.ndarray | None = None
    U_ctx: np.ndarray | None = None
    W_ww: np.ndarray | None = None
    r_ns: np.ndarray | None = None
    r_m: np.ndarray | None = None
    b_sel: np.ndarray | None = None
    b_ns: float = 0.0
    b_m: float = 0.0

    def parameter_count(self) -> int:
        total = 0
        for key in (
            "E_touch",
            "E_smell",
            "E_vision",
            "E_eff",
            "U_sal",
            "U_ctx",
            "W_ww",
            "r_ns",
            "r_m",
            "b_sel",
        ):
            val = getattr(self, key)
            if val is not None:
                total += int(np.asarray(val).size)
        total += 2
        return total


def make_workspace_params(arch_seed: int = 0) -> WorkspaceParams:
    rng = np.random.default_rng(int(arch_seed) + 7919)

    def scaled(shape: Tuple[int, ...], scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, size=shape)

    params = WorkspaceParams()
    params.E_touch = scaled((params.w_dim, params.modality_dim), 0.22)
    params.E_smell = scaled((params.w_dim, params.modality_dim), 0.22)
    params.E_vision = scaled((params.w_dim, params.modality_dim), 0.22)
    params.E_eff = scaled((params.w_dim, params.modality_dim), 0.22)
    params.U_sal = scaled((params.n_modalities, 5), 0.25)
    params.U_ctx = scaled((params.n_modalities, params.w_dim), 0.12)
    params.W_ww = scaled((params.w_dim, params.w_dim), 0.18)
    params.r_ns = scaled((params.w_dim,), 0.32)
    params.r_m = scaled((params.w_dim,), 0.32)
    params.b_sel = scaled((params.n_modalities,), 0.08)
    params.b_ns = float(rng.normal(0.0, 0.1))
    params.b_m = float(rng.normal(0.0, 0.1))
    return params


def initial_workspace_state(params: WorkspaceParams) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "model_family": "gw_lite",
        "workspace": np.zeros(params.w_dim, dtype=float),
        "selector_logits": np.zeros(params.n_modalities, dtype=float),
        "selector_weights": np.full(params.n_modalities, 1.0 / params.n_modalities, dtype=float),
        "Ns": 0.0,
        "internal": 0.0,
        "motor": 0.0,
        "efference": 0.0,
        "reafferent": 0.0,
        "precision_eff": 1.0,
        "g": float(params.g),
        "g_eff": float(params.g),
        "bb_true": 0.0,
        "bb_model": 0.0,
        "bb_pred": 0.0,
        "pe": 0.0,
        "valence": 1.0,
        "arousal": 0.0,
        "drive": 0.0,
        "drive_base": 0.0,
        "alpha_eff": float(params.alpha_w),
        "lesion_selector": False,
        "lesion_workspace": False,
        "lesion_affect": False,
    }
    return state


def workspace_builder(params: WorkspaceParams, affect: Any) -> Any:
    score_weights = {
        "w_valence": 2.0,
        "w_arousal": -1.2,
        "w_ns": -0.8,
        "w_bb_err": -0.4,
        "w_epistemic": 0.35,
    }
    return SimpleNamespace(params=params, affect=affect, score_weights=score_weights)


def make_workspace_obs(
    I_ext: float,
    state: Dict[str, Any],
    obs_components: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if obs_components is None:
        total = float(I_ext)
        touch = max(0.0, total)
        smell = 0.5 * float(total)
        vision = float(total) - smell
    else:
        total, touch, smell, vision = [float(v) for v in obs_components]
    eff = float(state.get("efference", 0.0))
    x_touch = np.array([touch, np.sign(touch)], dtype=float)
    x_smell = np.array([smell, np.sign(smell)], dtype=float)
    x_vision = np.array([vision, np.sign(vision)], dtype=float)
    x_eff = np.array([eff, 1.0], dtype=float)
    sal = np.array([abs(touch), abs(smell), abs(vision), eff, float(state.get("arousal", 0.0))], dtype=float)
    _ = total
    return x_touch, x_smell, x_vision, x_eff, sal
