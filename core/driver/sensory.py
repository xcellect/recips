from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def compute_I_affect(env: Any, y: int, x: int, heading: int) -> Tuple[float, float, float, float]:
    """
    Sensor fusion into a signed affect-relevant scalar in [-1, 1].

    Requires the environment to implement touch, smell, and vision_cone_features.
    """
    touch = env.touch(y, x)
    smell = env.smell(y, x)
    hz_v, bt_v = env.vision_cone_features(y, x, heading, radius=5, fov_deg=70)
    vision = float(hz_v - 0.6 * bt_v)

    I = float(np.clip(1.2 * touch + 0.7 * smell + 0.6 * vision, -1.0, 1.0))
    return I, touch, smell, vision
