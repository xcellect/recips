"""
evaluation_harness.py  (Jupyter-friendly)

Goal:
- Mechanism-level sanity checks (Humphrey / Barrett / divisive norm / alpha_eff / forward-model alignment).
- Behavioral sweeps on BOTH GridWorld + CorridorWorld using the SAME “active-inference feelings” policy.
- Produces per-episode logs + aggregated tables + plots.

IMPORTANT:
- gridworld_exp.Agent hardcodes choose_action_feelings(..., horizon=10).
  This harness does NOT rely on Agent.step(). It runs the loop explicitly so horizons in sweeps are real.
- corridor_exp.Agent already supports horizons, but we still run explicitly for consistency.

Usage (in notebook):
    %run experiments/evaluation_harness.py
or:
    import experiments.evaluation_harness as eh
    checks = eh.run_mechanism_checks()
    df_g, summ_g = eh.sweep_gridworld(...)
    df_c, summ_c = eh.sweep_corridor(...)
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global figure styling (Times-like serif)
from utils.plot_style import apply_times_style

apply_times_style()

# Your experiment modules (contain compute_I_affect / choose_action_feelings / predict_one_step)
import experiments.gridworld_exp as gw
import experiments.corridor_exp as cw

from core.ipsundrum_model import Builder, LoopParams, AffectParams
from core.driver.perspective_dynamics import perspective_step
from core.driver.workspace_dynamics import workspace_step
from core.model_factory import build_attached_stage_d_network, flatten_latent_state
from core.perspective_model import initial_perspective_state, make_perspective_params, perspective_builder
from core.workspace_model import initial_workspace_state, make_workspace_params, workspace_builder
from utils.model_naming import (
    MODEL_ID_ORDER,
    MODEL_DISPLAY_ORDER,
    canonical_model_display,
    canonical_model_id,
)

# Internal model ids (used for building/logic). Display names are standardized
# separately for all saved results.
MODEL_ORDER = list(MODEL_ID_ORDER)
MODEL_ORDER_DISPLAY = list(MODEL_DISPLAY_ORDER)


def order_models(df: pd.DataFrame, model_col: str = "model", order: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    if model_col not in df.columns:
        return df, list(order or MODEL_ORDER)
    base_order = list(order or MODEL_ORDER)
    extras = [m for m in pd.unique(df[model_col]) if m not in base_order]
    ordered = base_order + sorted(extras)
    out = df.copy()
    out[model_col] = pd.Categorical(out[model_col], categories=ordered, ordered=True)
    return out, ordered


# =============================================================================
# Small eval-only agent wrapper (keeps us faithful to gw/cw choose_action_feelings)
# =============================================================================

class EvalAgent:
    """
    Minimal wrapper with the fields expected by:
      - gw.choose_action_feelings(agent, horizon=...)
      - cw.choose_action_feelings(agent, horizon=...)
    """
    def __init__(
        self,
        env: Any,
        model: str,
        seed: int,
        *,
        start: Tuple[int, int],
        heading: int,
        eps: float,
        efference_threshold: float = 0.05,
    ):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.model = canonical_model_id(model)

        self.y, self.x = start
        self.heading = int(heading)
        self.eps = float(eps)

        # Build the ReCoN / Ipsundrum network using the SAME parameterization as gridworld/corridor code.
        self.b, self.net = build_model_network(self.model, efference_threshold=efference_threshold)
        self.score_weights = getattr(self.b, "score_weights", None)

        # Start Root
        self.net.start_root(True)

        # for convenience / metrics
        self.hazard_contacts = 0
        self.actions: List[str] = []
        self.positions: List[Tuple[int, int]] = []

    def read_state(self) -> Dict[str, float]:
        st = getattr(self.net, "_ipsundrum_state", {})
        out = dict(st) if isinstance(st, dict) else {}
        if isinstance(st, dict):
            out.update(flatten_latent_state(st))
        # visible sensors (if present)
        for k in ("Ns", "Ne", "Nv", "Na", "Ni"):
            if k in getattr(self.net, "nodes", {}):
                out[f"node_{k}"] = float(self.net.get(k).activation)
        if "Nv" not in getattr(self.net, "nodes", {}):
            out.pop("valence", None)
        if "Na" not in getattr(self.net, "nodes", {}):
            out.pop("arousal", None)
        return out


# =============================================================================
# Model construction (variants / ablations)
# =============================================================================

def default_loop_params() -> LoopParams:
    # Must match your experiment defaults (bias + divisive_norm to avoid Ns saturating at 1)
    return LoopParams(
        g=1.0,
        h=1.0,
        internal_decay=0.6,
        fatigue=0.02,
        nonlinearity="linear",
        saturation=True,
        sensor_bias=0.5,
        divisive_norm=0.8,
    )

def default_affect_params(enabled: bool) -> AffectParams:
    if not enabled:
        return AffectParams(enabled=False)
    return AffectParams(
        enabled=True,
        valence_scale=3.0,
        k_homeo=0.10,
        k_pe=0.50,
        demand_motor=0.20,
        demand_stim=0.30,
        modulate_g=True,
        k_g_arousal=0.8,
        k_g_unpleasant=0.8,
        modulate_precision=True,
        precision_base=1.0,
        k_precision_arousal=0.5,
    )

def build_model_network(model: str, efference_threshold: float = 0.05, arch_seed: int = 0) -> Tuple[Any, Any]:
    """
    Supported model strings:
      - "recon"              : Stage B (no ipsundrum loop)
      - "humphrey"           : Stage D, affect OFF
      - "humphrey_barrett"   : Stage D, affect ON
      - "humphrey_barrett_readout_only"    : affect ON, no gain/precision modulation
      - "humphrey_barrett_modulation_only" : gain/precision modulation ON, but scoring ignores Nv/Na/bb_err
      - "full"               : Deprecated alias for "humphrey_barrett"
    """
    loop = default_loop_params()
    model = canonical_model_id(model)

    if model == "recon":
        b = Builder(params=loop, affect=default_affect_params(False))
        net, _ = b.stage_B()
        b.score_weights = {
            "w_valence": 0.0,
            "w_arousal": 0.0,
            "w_ns": 0.0,
            "w_bb_err": 0.0,
            "w_epistemic": 0.0,
        }

    elif model == "humphrey":
        b = Builder(params=loop, affect=default_affect_params(False))
        net, _ = b.stage_D(efference_threshold=efference_threshold)

    elif model in ("humphrey_barrett", "full"):
        b = Builder(params=loop, affect=default_affect_params(True))
        net, _ = b.stage_D(efference_threshold=efference_threshold)

    elif model == "humphrey_barrett_readout_only":
        aff = default_affect_params(True)
        aff.modulate_g = False
        aff.modulate_precision = False
        b = Builder(params=loop, affect=aff)
        net, _ = b.stage_D(efference_threshold=efference_threshold)

    elif model == "humphrey_barrett_modulation_only":
        aff = default_affect_params(True)
        aff.modulate_g = True
        aff.modulate_precision = True
        b = Builder(params=loop, affect=aff)
        b.score_weights = {
            "w_valence": 0.0,
            "w_arousal": 0.0,
            "w_bb_err": 0.0,
        }
        net, _ = b.stage_D(efference_threshold=efference_threshold)

    elif model == "perspective":
        aff = default_affect_params(True)
        params = make_perspective_params(arch_seed=arch_seed)
        b = perspective_builder(params, aff, plastic=False)
        net = build_attached_stage_d_network(
            params=params,
            affect=aff,
            initial_state=initial_perspective_state(params, plastic=False),
            step_fn=perspective_step,
            efference_threshold=efference_threshold,
        )

    elif model == "perspective_plastic":
        aff = default_affect_params(True)
        params = make_perspective_params(arch_seed=arch_seed)
        b = perspective_builder(params, aff, plastic=True)
        net = build_attached_stage_d_network(
            params=params,
            affect=aff,
            initial_state=initial_perspective_state(params, plastic=True),
            step_fn=perspective_step,
            efference_threshold=efference_threshold,
        )

    elif model == "gw_lite":
        aff = default_affect_params(True)
        params = make_workspace_params(arch_seed=arch_seed)
        b = workspace_builder(params, aff)
        net = build_attached_stage_d_network(
            params=params,
            affect=aff,
            initial_state=initial_workspace_state(params),
            step_fn=workspace_step,
            efference_threshold=efference_threshold,
        )

    else:
        raise ValueError(f"Unknown model variant: {model}")

    return b, net


def canonicalize_model_column_display(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    """Return a copy with standardized display names in `model_col`."""
    if model_col not in df.columns:
        return df
    out = df.copy()
    out[model_col] = out[model_col].map(canonical_model_display)
    return out


# =============================================================================
# Goal functions + distances
# =============================================================================

def make_goal_fn_gridworld(env: gw.GridWorld, q: float = 0.9) -> Callable[[int, int], bool]:
    # define goal as top-q quantile of beauty among non-hazard cells
    mask = (env.hazard == 0.0)
    vals = env.beauty[mask]
    thr = float(np.quantile(vals, q)) if vals.size else float(np.max(env.beauty))
    goal_mask = (env.beauty >= thr) & mask

    def is_goal(y: int, x: int) -> bool:
        return bool(goal_mask[y, x])

    return is_goal

def min_dist_to_grid_goal(env: gw.GridWorld, y: int, x: int, q: float = 0.9) -> float:
    mask = (env.hazard == 0.0)
    vals = env.beauty[mask]
    thr = float(np.quantile(vals, q)) if vals.size else float(np.max(env.beauty))
    goal_pos = np.argwhere((env.beauty >= thr) & mask)
    if goal_pos.size == 0:
        gy, gx = np.unravel_index(int(np.argmax(env.beauty)), env.beauty.shape)
        return float(math.hypot(y - gy, x - gx))
    d2 = np.min((goal_pos[:, 0] - y) ** 2 + (goal_pos[:, 1] - x) ** 2)
    return float(math.sqrt(float(d2)))

def make_goal_fn_corridor_y(env: cw.CorridorWorld) -> Callable[[int, int], bool]:
    # robust “reach end of corridor” criterion
    gy = int(env.goal_y)
    def is_goal(y: int, x: int) -> bool:
        return bool(y >= gy) and bool(env.is_free(y, x))
    return is_goal

def min_dist_to_corridor_goal(env: cw.CorridorWorld, y: int, x: int) -> float:
    return float(math.hypot(y - int(env.goal_y), x - int(env.goal_x)))


# =============================================================================
# One episode rollout (task-agnostic)
# =============================================================================

@dataclass
class EpisodeResult:
    task: str
    model: str
    horizon: int
    seed: int
    T: int
    success: bool
    time_to_goal: int
    hazard_contacts: int
    min_dist_to_goal: float
    final_dist_to_goal: float
    forward_frac: float
    turn_frac: float
    stay_frac: float
    unique_pos_frac: float
    mean_Ns: float
    mean_alpha_eff: float
    mean_valence: float
    mean_arousal: float

def rollout_episode(
    *,
    task: str,
    env: Any,
    module: Any,              # gw or cw
    goal_fn: Callable[[int, int], bool],
    dist_fn: Callable[[int, int], float],
    model: str,
    horizon: int,
    seed: int,
    T: int,
    start: Tuple[int, int],
    heading: int,
    eps: float,
) -> EpisodeResult:
    agent = EvalAgent(env, model=model, seed=seed, start=start, heading=heading, eps=eps)
    if task == "corridor":
        if agent.score_weights is None:
            agent.score_weights = {}
        if "w_progress" not in agent.score_weights:
            agent.score_weights["w_progress"] = 0.20

    Ns_hist: List[float] = []
    alpha_hist: List[float] = []
    Nv_hist: List[float] = []
    Na_hist: List[float] = []

    t_goal: Optional[int] = None
    min_d = float("inf")

    for t in range(T):
        agent.positions.append((agent.y, agent.x))

        # --- sensory evidence stream (NOT reward) ---
        I_total, I_touch, I_smell, I_vision = module.compute_I_affect(env, agent.y, agent.x, agent.heading)

        # hazard contacts measured at time of sensation ("touch now")
        if I_touch > 0.5:
            agent.hazard_contacts += 1

        # --- physiology update ---
        if hasattr(agent.net, "_update_ipsundrum_sensor"):
            agent.net._update_ipsundrum_sensor(float(I_total), rng=agent.rng, obs_components=(I_total, I_touch, I_smell, I_vision))  # type: ignore[attr-defined]
        else:
            # Stage B: map signed evidence to [0,1]
            agent.net.set_sensor_value("Ns", float(np.clip(0.5 + 0.5 * I_total, 0.0, 1.0)))

        agent.net.step()

        # record internals
        st = agent.read_state()
        Ns_hist.append(float(st.get("node_Ns", np.nan)))
        alpha_hist.append(float(st.get("alpha_eff", np.nan)))
        Nv_hist.append(float(st.get("node_Nv", np.nan)))
        Na_hist.append(float(st.get("node_Na", np.nan)))

        # goal + distance tracking (after perception update, before action)
        d = dist_fn(agent.y, agent.x)
        min_d = min(min_d, d)
        if t_goal is None and goal_fn(agent.y, agent.x):
            t_goal = t

        # --- choose action (internal “feelings” planning; no RL reward) ---
        if agent.rng.random() < agent.eps:
            action = agent.rng.choice(["forward", "turn_left", "turn_right"])
        else:
            action = module.choose_action_feelings(agent, horizon=int(horizon))

        agent.actions.append(str(action))

        # --- apply action to env ---
        agent.y, agent.x, agent.heading = env.step(agent.y, agent.x, str(action), agent.heading)

    # final goal check
    if t_goal is None and goal_fn(agent.y, agent.x):
        t_goal = T - 1

    success = (t_goal is not None)
    time_to_goal = int(t_goal) if success else int(T)

    # action stats
    n = max(1, len(agent.actions))
    fwd = sum(1 for a in agent.actions if a == "forward") / n
    turn = sum(1 for a in agent.actions if a in ("turn_left", "turn_right")) / n
    stay = sum(1 for a in agent.actions if a == "stay") / n

    uniq = len(set(agent.positions)) / max(1, len(agent.positions))

    # summary signals
    def nanmean(xs: List[float]) -> float:
        arr = np.asarray(xs, dtype=float)
        return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

    return EpisodeResult(
        task=task,
        model=model,
        horizon=int(horizon),
        seed=int(seed),
        T=int(T),
        success=bool(success),
        time_to_goal=int(time_to_goal),
        hazard_contacts=int(agent.hazard_contacts),
        min_dist_to_goal=float(min_d),
        final_dist_to_goal=float(dist_fn(agent.y, agent.x)),
        forward_frac=float(fwd),
        turn_frac=float(turn),
        stay_frac=float(stay),
        unique_pos_frac=float(uniq),
        mean_Ns=nanmean(Ns_hist),
        mean_alpha_eff=nanmean(alpha_hist),
        mean_valence=nanmean(Nv_hist),
        mean_arousal=nanmean(Na_hist),
    )


# =============================================================================
# Sweeps + summarization + plotting
# =============================================================================

def summarize(rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    df, ordered = order_models(df)
    df = df.sort_values(["task", "model", "horizon", "seed"])
    grp = df.groupby(["task", "model", "horizon"], as_index=False, sort=False, observed=False).agg(
        mean_hazards=("hazard_contacts", "mean"),
        mean_time=("time_to_goal", "mean"),
        success_rate=("success", "mean"),
        mean_min_dist=("min_dist_to_goal", "mean"),
        mean_final_dist=("final_dist_to_goal", "mean"),
        mean_forward=("forward_frac", "mean"),
        mean_unique=("unique_pos_frac", "mean"),
        mean_Ns=("mean_Ns", "mean"),
        mean_alpha=("mean_alpha_eff", "mean"),
        mean_valence=("mean_valence", "mean"),
        mean_arousal=("mean_arousal", "mean"),
        n=("seed", "count"),
    )
    grp, _ = order_models(grp, order=ordered)
    grp = grp.sort_values(["task", "model", "horizon"])
    return df, grp

def plot_metric(summary: pd.DataFrame, task: str, metric: str, title: str, *, show: bool = True):
    def _pretty_axis_label(raw: str) -> str:
        overrides = {
            "success_rate": "Success Rate",
            "mean_hazards": "Hazard Contacts",
            "mean_time": "Time to Goal",
            "mean_min_dist": "Min Distance to Goal",
            "mean_final_dist": "Final Distance to Goal",
            "mean_forward": "Forward Action Share",
            "mean_unique": "Unique Position Share",
            "mean_Ns": "Mean Ns",
            "mean_alpha": "Mean Alpha",
            "mean_valence": "Mean Valence",
            "mean_arousal": "Mean Arousal",
        }
        if raw in overrides:
            return overrides[raw]
        return str(raw).replace("_", " ").strip().title()

    sub = summary[summary["task"] == task].copy()
    if "model" in sub.columns:
        sub["model"] = sub["model"].map(canonical_model_display)
    sub, ordered = order_models(sub, order=MODEL_DISPLAY_ORDER)
    fig, ax = plt.subplots()
    horizons = sorted(pd.unique(sub["horizon"]))
    present = [m for m in ordered if m in sub["model"].values]
    n_models = max(1, len(present))

    if len(horizons) > 1:
        min_spacing = float(np.min(np.diff(horizons)))
        offset_step = 0.08 * min_spacing
    else:
        offset_step = 0.05

    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    linestyles = ["-", "--", ":", "-."]

    for idx, model in enumerate(present):
        s2 = sub[sub["model"] == model].sort_values("horizon")
        offset = (idx - (n_models - 1) / 2.0) * offset_step if n_models > 1 else 0.0
        x = s2["horizon"].to_numpy(dtype=float) + offset
        ax.plot(
            x,
            s2[metric],
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=str(model),
        )
    ax.set_title(title)
    ax.set_xlabel("Planning Horizon")
    ax.set_ylabel(_pretty_axis_label(metric))
    if horizons:
        ax.set_xticks(horizons)
        ax.set_xlim(min(horizons) - 0.5, max(horizons) + 0.5)
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend()
    if show:
        plt.show()
    return fig

def sweep_task(
    *,
    task: str,
    env_factory: Callable[[int], Any],
    module: Any,
    start_fn: Callable[[Any], Tuple[int, int]],
    heading_fn: Callable[[Any], int],
    goal_fn_factory: Callable[[Any], Callable[[int, int], bool]],
    dist_fn_factory: Callable[[Any], Callable[[int, int], float]],
    models: List[str],
    horizons: List[int],
    seeds: List[int],
    T: int,
    eps: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in seeds:
        env = env_factory(s)
        goal_fn = goal_fn_factory(env)
        dist_fn = dist_fn_factory(env)
        start = start_fn(env)
        heading = heading_fn(env)

        for model in models:
            for h in horizons:
                ep = rollout_episode(
                    task=task,
                    env=env_factory(s),   # fresh env per run for fairness
                    module=module,
                    goal_fn=goal_fn_factory(env_factory(s)),
                    dist_fn=dist_fn_factory(env_factory(s)),
                    model=model,
                    horizon=h,
                    seed=s,               # agent RNG seed
                    T=T,
                    start=start_fn(env_factory(s)),
                    heading=heading_fn(env_factory(s)),
                    eps=eps,
                )
                rows.append(asdict(ep))
    return rows


def sweep_gridworld(
    *,
    H=18, W=18,
    horizons=(1, 2, 3, 5, 10, 20),
    seeds=tuple(range(20)),
    models=tuple(MODEL_ORDER),
    T=250,
    q_goal=0.9,
    eps=0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = sweep_task(
        task="gridworld",
        env_factory=lambda s: gw.GridWorld(H=H, W=W, seed=s),
        module=gw,
        start_fn=lambda env: (env.H // 2, env.W // 2),
        heading_fn=lambda env: 1,
        goal_fn_factory=lambda env: make_goal_fn_gridworld(env, q=q_goal),
        dist_fn_factory=lambda env: (lambda y, x: min_dist_to_grid_goal(env, y, x, q=q_goal)),
        models=list(models),
        horizons=list(horizons),
        seeds=list(seeds),
        T=int(T),
        eps=float(eps),
    )
    return summarize(rows)

def sweep_corridor(
    *,
    H=18, W=18,
    horizons=(1, 2, 3, 5, 10, 20),
    seeds=tuple(range(20)),
    models=tuple(MODEL_ORDER),
    T=250,
    eps=0.08,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = sweep_task(
        task="corridor",
        env_factory=lambda s: cw.CorridorWorld(H=H, W=W, seed=s),
        module=cw,
        start_fn=lambda env: (1, int(env.goal_x)),       # match corridor_exp animation default:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
        heading_fn=lambda env: 2,                        # face down corridor:contentReference[oaicite:2]{index=2}
        goal_fn_factory=lambda env: make_goal_fn_corridor_y(env),
        dist_fn_factory=lambda env: (lambda y, x: min_dist_to_corridor_goal(env, y, x)),
        models=list(models),
        horizons=list(horizons),
        seeds=list(seeds),
        T=int(T),
        eps=float(eps),
    )
    return summarize(rows)


# =============================================================================
# Mechanism-level checks (the “checklist”)
# =============================================================================

def run_mechanism_checks(verbose: bool = True) -> Dict[str, Any]:
    """
    Returns dict of check_name -> {ok:bool, ...details}
    """

    checks: Dict[str, Any] = {}

    # -------------------------
    # 1) Barrett: signed demand (deposit vs cost)
    # -------------------------
    def _barrett_signed_demand():
        loop = LoopParams(
            g=1.0, h=1.0, internal_decay=0.6, fatigue=0.0,
            nonlinearity="linear", saturation=True, sensor_bias=0.5, divisive_norm=0.0
        )
        # isolate the stimulus term to make the sign obvious
        aff = AffectParams(
            enabled=True,
            setpoint=0.0,
            k_homeo=0.0,
            k_pe=0.0,
            demand_motor=0.0,
            demand_stim=1.0,
            stim_cost_pos=1.0,
            stim_gain_neg=1.0,
            valence_scale=3.0,
            arousal_scale=1.0,
            modulate_g=False,
            modulate_precision=False,
        )
        b = Builder(params=loop, affect=aff)
        net, _ = b.stage_D(efference_threshold=0.05)
        net.start_root(True)
        rng = np.random.default_rng(0)

        net._update_ipsundrum_sensor(0.5, rng=rng)  # type: ignore[attr-defined]
        dpos = float(net._ipsundrum_state.get("demand", np.nan))  # type: ignore[attr-defined]
        bbpos = float(net._ipsundrum_state.get("bb_true", np.nan))  # type: ignore[attr-defined]

        b = Builder(params=loop, affect=aff)
        net2, _ = b.stage_D(efference_threshold=0.05)
        net2.start_root(True)
        rng2 = np.random.default_rng(0)

        net2._update_ipsundrum_sensor(-0.5, rng=rng2)  # type: ignore[attr-defined]
        dneg = float(net2._ipsundrum_state.get("demand", np.nan))  # type: ignore[attr-defined]
        bbneg = float(net2._ipsundrum_state.get("bb_true", np.nan))  # type: ignore[attr-defined]

        ok = (dpos > 0.0) and (dneg < 0.0) and (bbpos < 0.0) and (bbneg > 0.0)
        return {"ok": ok, "demand_pos": dpos, "bb_true_pos": bbpos, "demand_neg": dneg, "bb_true_neg": bbneg}

    checks["barrett_signed_demand"] = _barrett_signed_demand()

    # -------------------------
    # 2) alpha_eff exists and matches formula
    # -------------------------
    def _alpha_eff_present():
        loop = default_loop_params()
        b = Builder(params=loop, affect=default_affect_params(True))
        net, _ = b.stage_D(efference_threshold=0.05)
        net.start_root(True)
        rng = np.random.default_rng(0)

        net._update_ipsundrum_sensor(1.0, rng=rng)  # type: ignore[attr-defined]
        net._update_ipsundrum_sensor(1.0, rng=rng)  # type: ignore[attr-defined]
        st = net._ipsundrum_state  # type: ignore[attr-defined]

        d = float(loop.internal_decay)
        g_eff = float(st.get("g_eff", np.nan))
        precision = float(st.get("precision_eff", np.nan))
        alpha_expected = d + (1.0 - d) * (g_eff * float(loop.h) * precision)

        alpha = float(st.get("alpha_eff", np.nan))
        ok = np.isfinite(alpha) and np.isfinite(alpha_expected) and abs(alpha - alpha_expected) < 1e-9
        return {"ok": ok, "alpha_eff": alpha, "alpha_expected": alpha_expected}

    checks["alpha_eff_present"] = _alpha_eff_present()

    # -------------------------
    # 3) divisive normalization reduces drive under strong feedback
    # -------------------------
    def _divisive_norm_effect_observed():
        loop = LoopParams(
            g=1.2, h=1.0, internal_decay=0.6, fatigue=0.0,
            nonlinearity="linear", saturation=True, sensor_bias=0.5, divisive_norm=0.8
        )
        b = Builder(params=loop, affect=default_affect_params(False))
        net, _ = b.stage_D(efference_threshold=0.05)
        net.start_root(True)
        rng = np.random.default_rng(0)

        # step 1 creates reafferent, step 2 has large precision*E_prev
        net._update_ipsundrum_sensor(1.0, rng=rng)  # type: ignore[attr-defined]
        net._update_ipsundrum_sensor(1.0, rng=rng)  # type: ignore[attr-defined]
        st = net._ipsundrum_state  # type: ignore[attr-defined]
        drive_base = float(st.get("drive_base", np.nan))
        drive = float(st.get("drive", np.nan))
        ok = np.isfinite(drive_base) and np.isfinite(drive) and (drive < drive_base)
        return {"ok": ok, "drive_base": drive_base, "drive": drive}

    checks["divisive_norm_effect_observed"] = _divisive_norm_effect_observed()

    # -------------------------
    # 4) forward-model alignment: predict_one_step vs ipsundrum_model update_sensor
    #    (uses gw.predict_one_step because it includes all debug fields)
    # -------------------------
    def _forward_model_alignment(n_trials: int = 20):
        loop = default_loop_params()
        aff = default_affect_params(True)

        b = Builder(params=loop, affect=aff)
        net, _ = b.stage_D(efference_threshold=0.05)
        net.start_root(True)

        abs_diffs: List[float] = []
        for k in range(n_trials):
            I = float(np.sin(0.1 * k))  # deterministic stimulus sequence

            # copy pre-state
            st0 = copy.deepcopy(net._ipsundrum_state)  # type: ignore[attr-defined]

            # predict (no noise in defaults => deterministic)
            pred = gw.predict_one_step(st0, loop, aff, I, rng=np.random.default_rng(0))

            # actual update
            net._update_ipsundrum_sensor(I, rng=np.random.default_rng(0))  # type: ignore[attr-defined]
            st1 = net._ipsundrum_state  # type: ignore[attr-defined]

            # compare a small set of core scalars
            keys = ["drive", "drive_base", "precision_eff", "g_eff", "reafferent",
                    "internal", "motor", "efference", "demand", "bb_true", "bb_model", "pe",
                    "valence", "arousal", "alpha_eff"]
            diffs = []
            for kk in keys:
                a = float(st1.get(kk, 0.0))
                p = float(pred.get(kk, 0.0))
                diffs.append(abs(a - p))
            abs_diffs.append(float(np.mean(diffs)))

        mean_abs_diff = float(np.mean(abs_diffs))
        ok = mean_abs_diff < 1e-9
        return {"ok": ok, "mean_abs_diff": mean_abs_diff, "n": n_trials}

    checks["forward_model_alignment"] = _forward_model_alignment()

    if verbose:
        for k, v in checks.items():
            status = "OK" if v.get("ok") else "FAIL"
            print(f"[{status}] {k}: {v}")

    return checks


# =============================================================================
# Convenience “main” (safe to run in notebook)
# =============================================================================

if __name__ == "__main__":
    # 1) Mechanism-level checks
    _ = run_mechanism_checks(verbose=True)

    # 2) Sweeps (gridworld + corridor)
    df_g, summ_g = sweep_gridworld()
    df_c, summ_c = sweep_corridor()

    # 3) Plots: Gridworld
    plot_metric(summ_g, "gridworld", "mean_hazards", "Gridworld: hazard contacts vs horizon")
    plot_metric(summ_g, "gridworld", "mean_time", "Gridworld: time-to-goal vs horizon")
    plot_metric(summ_g, "gridworld", "success_rate", "Gridworld: success rate vs horizon")
    plot_metric(summ_g, "gridworld", "mean_min_dist", "Gridworld: mean min dist-to-goal vs horizon")

    # 4) Plots: Corridor
    plot_metric(summ_c, "corridor", "mean_hazards", "Corridor: hazard contacts vs horizon")
    plot_metric(summ_c, "corridor", "mean_time", "Corridor: time-to-goal vs horizon")
    plot_metric(summ_c, "corridor", "success_rate", "Corridor: success rate vs horizon")
    plot_metric(summ_c, "corridor", "mean_min_dist", "Corridor: mean min dist-to-goal vs horizon")

    # show tables
    print("\n=== GRIDWORLD SUMMARY ===")
    print(summ_g.sort_values(["model", "horizon"]).to_string(index=False))
    print("\n=== CORRIDOR SUMMARY ===")
    print(summ_c.sort_values(["model", "horizon"]).to_string(index=False))
