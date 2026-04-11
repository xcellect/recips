from __future__ import annotations

from typing import Any
from typing import Optional, Tuple

import numpy as np

from .ipsundrum_dynamics import ipsundrum_step


def predict_one_step(
    state: dict,
    loop: Any,
    aff: Any,
    I_ext: float,
    rng: np.random.Generator,
    obs_components: Optional[Tuple[float, float, float, float]] = None,
) -> dict:
    """
    One-step internal forward model aligned with ipsundrum_model update_sensor.

    NOTE: Used by gridworld and corridor active-perception policy.
    """
    _ = obs_components
    return ipsundrum_step(state, I_ext, loop, aff, rng=rng)
