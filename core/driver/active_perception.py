from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .ipsundrum_forward import predict_one_step as _predict_one_step


@dataclass
class PolicyContext:
    env: Any
    y: int
    x: int
    heading: int
    rng: np.random.Generator
    loop: Any
    aff: Any
    net_state: Dict[str, Any]
    efference_threshold: Optional[float] = None


@dataclass
class PolicyState:
    last_action: Optional[str] = None
    viewpoint_counts: Optional[Dict[Tuple[int, int, int], int]] = None


class ActivePerceptionPolicy:
    def __init__(self, adapter: "EnvAdapter", forward_model: Optional[Callable[..., dict]] = None) -> None:
        self.adapter = adapter
        self.forward_model = forward_model
        self.state = PolicyState()

    def reset(self) -> None:
        self.state = PolicyState()

    def choose_action(self, ctx: PolicyContext, **kwargs) -> str:
        return choose_action_feelings(
            ctx,
            self.adapter,
            policy_state=self.state,
            forward_model=self.forward_model,
            **kwargs,
        )


@dataclass
class ActionEval:
    y: int
    x: int
    heading: int
    pred_y: int
    pred_x: int
    pred_heading: int
    bumped: bool = False


def _default_forward_prior(_: Any) -> float:
    return 0.10


def _default_turn_cost(_: Any) -> float:
    return 0.20


def _default_stay_penalty(_: Any) -> float:
    return 0.30


def _default_bump_penalty(_: Any) -> float:
    return 0.0

def _default_hazard_penalty(_: Any) -> float:
    return 0.0


@dataclass
class EnvAdapter:
    actions: Sequence[str]
    compute_I_affect: Callable[[Any, int, int, int], Tuple[float, float, float, float]]
    eval_action: Callable[[Any, int, int, int, str], ActionEval]
    forward_prior_fn: Callable[[Any], float] = _default_forward_prior
    turn_cost_fn: Callable[[Any], float] = _default_turn_cost
    stay_penalty_fn: Callable[[Any], float] = _default_stay_penalty
    bump_penalty_fn: Callable[[Any], float] = _default_bump_penalty
    hazard_penalty_fn: Callable[[Any], float] = _default_hazard_penalty
    progress_fn: Optional[Callable[[Any, int, int, int, int, str], float]] = None


def score_internal_components(
    s,
    aff,
    current_I,
    predicted_I,
    w_epistemic=0.35,
    last_action=None,
    w_valence=2.0,
    w_arousal=-1.2,
    w_ns=-0.8,
    w_bb_err=-0.4,
):
    Nv = float(s.get("valence", 0.5))
    Na = float(s.get("arousal", 0.0))
    Ns = float(s.get("Ns", 0.0))
    bb = float(s.get("bb_model", 0.0))
    sp = float(getattr(aff, "setpoint", 0.0))
    bb_err = abs(bb - sp)

    base = (
        float(w_valence) * Nv
        + float(w_arousal) * Na
        + float(w_ns) * Ns
        + float(w_bb_err) * bb_err
    )

    sensory_change = abs(predicted_I - current_I)
    if float(w_epistemic) <= 0.0:
        epistemic = 0.0
    elif sensory_change < 0.01:
        epistemic = -0.25
    else:
        epistemic = float(w_epistemic) * sensory_change

    commit_pen = 0.0
    if last_action is not None:
        a = str(s.get("action", ""))
        if last_action == "turn_left" and a == "turn_right":
            commit_pen += 0.15
        if last_action == "turn_right" and a == "turn_left":
            commit_pen += 0.15

    return base, epistemic, commit_pen


def score_internal(
    s,
    aff,
    current_I,
    predicted_I,
    w_epistemic=0.35,
    last_action=None,
    w_valence=2.0,
    w_arousal=-1.2,
    w_ns=-0.8,
    w_bb_err=-0.4,
):
    base, epistemic, commit_pen = score_internal_components(
        s,
        aff,
        current_I,
        predicted_I,
        w_epistemic=w_epistemic,
        last_action=last_action,
        w_valence=w_valence,
        w_arousal=w_arousal,
        w_ns=w_ns,
        w_bb_err=w_bb_err,
    )
    return base + epistemic - commit_pen


def choose_action_feelings(
    ctx: PolicyContext,
    adapter: EnvAdapter,
    policy_state: Optional[PolicyState] = None,
    horizon: int = 2,
    curiosity: bool = False,
    w_progress: float = 0.0,
    w_epistemic: float = 0.35,
    beauty_weight: float = 1.0,
    use_beauty_term: Optional[bool] = None,
    forward_model: Optional[Callable[..., dict]] = None,
    w_valence: float = 2.0,
    w_arousal: float = -1.2,
    w_ns: float = -0.8,
    w_bb_err: float = -0.4,
    novelty_scale: float = 0.50,
) -> str:
    """
    Active-inference style action selection (purely internal).

    policy_state is mutated in-place to preserve last_action and novelty counts.
    """
    if policy_state is None:
        policy_state = PolicyState()
    if curiosity and policy_state.viewpoint_counts is None:
        policy_state.viewpoint_counts = {}

    forward_model = forward_model or _predict_one_step

    loop = ctx.loop
    aff = ctx.aff
    current_I, *_ = adapter.compute_I_affect(ctx.env, ctx.y, ctx.x, ctx.heading)

    base = dict(ctx.net_state)
    base["g"] = float(base.get("g", getattr(loop, "g", 1.0)))
    valence = float(base.get("valence", 0.5))
    arousal = float(base.get("arousal", 0.0))
    lesion_affect = bool(base.get("lesion_affect", False))
    affect_enabled = bool(getattr(aff, "enabled", False) and not lesion_affect)

    internal_mode = True
    if ctx.efference_threshold is not None:
        internal_mode = float(base.get("efference", 0.0)) >= float(ctx.efference_threshold)

    w_epistemic_eff = float(w_epistemic) if internal_mode else 0.0
    if (not affect_enabled) and (not curiosity) and (ctx.efference_threshold is not None):
        w_epistemic_eff = 0.0

    curiosity_gain = 1.0
    if not internal_mode:
        curiosity_gain = 0.0
    elif not affect_enabled:
        curiosity_gain *= 1.0
    else:
        curiosity_gain *= max(0.0, 1.0 - valence)

    caution_floor = 0.05
    caution_gain = 1.50

    last_action = policy_state.last_action

    best_a = None
    best_s = -1e18

    action_order = list(adapter.actions)
    ctx.rng.shuffle(action_order)
    for a in action_order:
        eval_info = adapter.eval_action(ctx.env, ctx.y, ctx.x, ctx.heading, a)
        predicted_I, touch_pred, *_ = adapter.compute_I_affect(
            ctx.env, eval_info.pred_y, eval_info.pred_x, eval_info.pred_heading
        )
        sensory_change = abs(predicted_I - current_I)

        s_pred = dict(base)
        for _ in range(horizon):
            s_pred = forward_model(
                s_pred,
                loop,
                aff,
                predicted_I,
                rng=ctx.rng,
            )

        s_pred["action"] = a
        internal_pred = float(s_pred.get("internal", base.get("internal", 0.0)))
        arousal_pred = float(s_pred.get("arousal", arousal))
        sc = score_internal(
            s_pred,
            aff,
            current_I,
            predicted_I,
            w_epistemic=w_epistemic_eff,
            last_action=last_action,
            w_valence=w_valence,
            w_arousal=w_arousal,
            w_ns=w_ns,
            w_bb_err=w_bb_err,
        )

        if eval_info.bumped:
            sc -= float(adapter.bump_penalty_fn(ctx.env))
        hazard_penalty = float(adapter.hazard_penalty_fn(ctx.env))
        if hazard_penalty and touch_pred > 0.0:
            sc -= hazard_penalty * float(touch_pred)

        use_beauty = use_beauty_term
        if use_beauty is None:
            use_beauty = bool(getattr(ctx.env, "use_beauty_term", True))
        if use_beauty and beauty_weight != 0.0 and hasattr(ctx.env, "beauty"):
            scenic_dest = float(ctx.env.beauty[eval_info.y, eval_info.x])
            sc += float(beauty_weight) * scenic_dest

        if w_progress and adapter.progress_fn is not None:
            progress = float(adapter.progress_fn(ctx.env, ctx.y, ctx.x, eval_info.y, eval_info.x, a))
            sc += float(w_progress) * progress

        if a == "forward" and internal_mode and affect_enabled:
            arousal_drive = max(0.0, float(arousal_pred) - caution_floor)
            sc -= float(caution_gain) * arousal_drive * max(0.0, float(internal_pred))

        if a in ("turn_left", "turn_right"):
            sc -= float(adapter.turn_cost_fn(ctx.env))
            if internal_mode and affect_enabled:
                sc += 1.0 * float(arousal) * float(sensory_change)

        if curiosity and policy_state.viewpoint_counts is not None and curiosity_gain > 0.0:
            pred_y, pred_x, pred_h = eval_info.pred_y, eval_info.pred_x, eval_info.pred_heading
            loc_visits = 0
            for h in range(4):
                loc_visits += policy_state.viewpoint_counts.get((pred_y, pred_x, h), 0)
            if loc_visits <= 0:
                loc_visits = policy_state.viewpoint_counts.get((pred_y, pred_x), 0)
            heading_visits = policy_state.viewpoint_counts.get((pred_y, pred_x, pred_h), 0)
            loc_novelty = 1.0 / np.sqrt(1.0 + loc_visits)
            heading_novelty = 1.0 / np.sqrt(1.0 + heading_visits)
            move_pred = (pred_y, pred_x) != (ctx.y, ctx.x)
            scale = float(novelty_scale)
            if move_pred and not affect_enabled:
                scale *= 1.5
            loc_w, head_w = 0.5, 0.5
            novelty_bonus = scale * float(curiosity_gain) * (loc_w * loc_novelty + head_w * heading_novelty)
            sc += novelty_bonus

        if a == "stay":
            sc -= float(adapter.stay_penalty_fn(ctx.env))

        if a == "forward":
            sc += float(adapter.forward_prior_fn(ctx.env))

        if sc > best_s:
            best_s = sc
            best_a = a

    chosen = best_a if best_a is not None else "stay"
    policy_state.last_action = chosen
    if curiosity and policy_state.viewpoint_counts is not None:
        current_viewpoint = (ctx.y, ctx.x, ctx.heading)
        policy_state.viewpoint_counts[current_viewpoint] = (
            policy_state.viewpoint_counts.get(current_viewpoint, 0) + 1
        )

    return chosen
