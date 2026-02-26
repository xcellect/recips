from __future__ import annotations

import copy
from typing import Any

import numpy as np


def predict_one_step_recon(
    state: dict,
    loop: Any,
    aff: Any,
    I_ext: float,
    rng: np.random.Generator,
) -> dict:
    """
    One-step forward model for Stage-B (ReCoN) planning.

    Matches stage_B online semantics:
      Ns := clip(0.5 + 0.5 * I_ext, 0, 1)
    No integrator, reafferent, efference, or affect updates are performed.
    """
    s = copy.deepcopy(state)
    I_drive = float(I_ext)
    if not bool(getattr(aff, "enabled", False)):
        # Stage-B / non-affect: negative I does not confer positive value.
        I_drive = max(0.0, I_drive)
    s["Ns"] = float(np.clip(0.5 + 0.5 * I_drive, 0.0, 1.0))
    return s
