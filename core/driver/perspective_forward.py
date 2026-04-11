from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .perspective_dynamics import perspective_step


def predict_one_step_perspective(
    state: dict,
    loop: Any,
    aff: Any,
    I_ext: float,
    rng: np.random.Generator,
    obs_components: Optional[Tuple[float, float, float, float]] = None,
) -> dict:
    return perspective_step(state, I_ext, loop, aff, obs_components=obs_components, rng=rng)
