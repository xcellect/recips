from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from core.ipsundrum_model import AffectParams, Builder, LoopParams
from core.recon_core import clamp01


def build_attached_stage_d_network(
    *,
    params: Any,
    affect: AffectParams,
    initial_state: Dict[str, Any],
    step_fn: Callable[..., Dict[str, Any]],
    efference_threshold: float = 0.05,
) -> Any:
    shell_loop = LoopParams(g=float(getattr(params, "g", 1.0)), efference_decay=float(getattr(params, "efference_decay", 0.7)))
    shell_builder = Builder(params=shell_loop, affect=affect)
    net, _ = shell_builder.stage_D(efference_threshold=efference_threshold)
    state = initial_state

    def nr_effect(_a: float) -> None:
        net.set_sensor_value("Ne", clamp01(abs(float(state.get("efference", 0.0)))))

    net.get("Nr").actuator_effect = nr_effect

    def update_sensor(
        I_ext: float,
        rng: Optional[np.random.Generator] = None,
        obs_components: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        rg = rng or np.random.default_rng(0)
        next_state = step_fn(state, float(I_ext), params, affect, obs_components=obs_components, rng=rg)
        state.clear()
        state.update(next_state)

        if "Ns" in net.nodes:
            net.set_sensor_value("Ns", float(state.get("Ns", 0.0)))
        if "Ne" in net.nodes:
            net.set_sensor_value("Ne", clamp01(abs(float(state.get("efference", 0.0)))))
        if "Ni" in net.nodes:
            net.set_sensor_value("Ni", clamp01(0.5 + 0.5 * float(state.get("bb_model", 0.0))))
        if "Nv" in net.nodes:
            net.set_sensor_value("Nv", clamp01(float(state.get("valence", 0.0))))
        if "Na" in net.nodes:
            net.set_sensor_value("Na", clamp01(float(state.get("arousal", 0.0))))

    net._update_ipsundrum_sensor = update_sensor  # type: ignore[attr-defined]
    net._ipsundrum_state = state  # type: ignore[attr-defined]
    return net


def flatten_latent_state(state: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, val in state.items():
        arr = None
        if isinstance(val, np.ndarray):
            arr = np.asarray(val)
        elif isinstance(val, (list, tuple)) and val and not isinstance(val[0], (str, bytes, dict)):
            arr = np.asarray(val)
        if arr is not None and arr.ndim >= 1:
            if arr.ndim == 1:
                for i, item in enumerate(arr.tolist()):
                    flat[f"{key}_{i}"] = float(item)
            elif arr.ndim == 2:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        flat[f"{key}_{i}_{j}"] = float(arr[i, j])
    if "selector_weights_0" in flat:
        labels = ["touch", "smell", "vision", "efference"]
        for idx, label in enumerate(labels):
            key = f"selector_weights_{idx}"
            if key in flat:
                flat[f"sel_{label}"] = flat[key]
    return flat
