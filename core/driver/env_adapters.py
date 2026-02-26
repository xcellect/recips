from __future__ import annotations

from typing import Any, Callable

from .active_perception import ActionEval, EnvAdapter
from .sensory import compute_I_affect as _compute_I_affect


ACTIONS = ("forward", "turn_left", "turn_right", "stay")


def gridworld_adapter(
    compute_I_affect: Callable[[Any, int, int, int], tuple] = _compute_I_affect,
) -> EnvAdapter:
    def eval_action(env: Any, y: int, x: int, heading: int, action: str) -> ActionEval:
        y2, x2, h2 = env.step(y, x, action, heading)
        bumped = (y2, x2) == (y, x) and action == "forward"
        return ActionEval(y=y2, x=x2, heading=h2, pred_y=y2, pred_x=x2, pred_heading=h2, bumped=bumped)

    return EnvAdapter(
        actions=ACTIONS,
        compute_I_affect=compute_I_affect,
        eval_action=eval_action,
        bump_penalty_fn=lambda env: float(getattr(env, "bump_penalty", 0.20)),
        hazard_penalty_fn=lambda env: float(getattr(env, "hazard_penalty", 0.10)),
    )


def corridor_adapter(
    compute_I_affect: Callable[[Any, int, int, int], tuple] = _compute_I_affect,
) -> EnvAdapter:
    def eval_action(env: Any, y: int, x: int, heading: int, action: str) -> ActionEval:
        y2, x2, h2 = env.step(y, x, action, heading)
        bumped = (y2, x2) == (y, x) and action == "forward"
        pred_y, pred_x, pred_h = y2, x2, h2
        if hasattr(env, "split_pose") and action in ("turn_left", "turn_right"):
            split_y, split_x, _ = env.split_pose
            if (y, x) == (split_y, split_x):
                pred_y, pred_x, pred_h = env.step(y2, x2, "forward", h2)
        return ActionEval(
            y=y2,
            x=x2,
            heading=h2,
            pred_y=pred_y,
            pred_x=pred_x,
            pred_heading=pred_h,
            bumped=bumped,
        )

    def progress(_env: Any, y: int, _x: int, y2: int, _x2: int, _action: str) -> float:
        return float(max(0, y2 - y))

    return EnvAdapter(
        actions=ACTIONS,
        compute_I_affect=compute_I_affect,
        eval_action=eval_action,
        forward_prior_fn=lambda env: float(getattr(env, "forward_prior", 0.10)),
        turn_cost_fn=lambda env: float(getattr(env, "turn_cost", 0.20)),
        stay_penalty_fn=lambda env: float(getattr(env, "stay_penalty", 0.30)),
        bump_penalty_fn=lambda env: float(getattr(env, "bump_penalty", 0.0)),
        hazard_penalty_fn=lambda env: float(getattr(env, "hazard_penalty", 0.10)),
        progress_fn=progress,
    )
