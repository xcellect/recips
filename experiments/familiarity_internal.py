"""Internal familiarity control with cross-episode curiosity memory."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.plot_style import apply_times_style

apply_times_style()

import experiments.corridor_exp as cw
from experiments.qualiaphilia_assay import QualiaphiliaCorridorWorld, update_choice_entry
from experiments.evaluation_harness import EvalAgent, MODEL_ORDER, order_models
from core.driver.active_perception import score_internal_components
from core.driver.env_adapters import corridor_adapter
from utils.model_naming import MODEL_DISPLAY_ORDER, canonical_model_display


_ADAPTER = corridor_adapter(cw.compute_I_affect)

N_FAM = 10
FAM_STEPS_PER_EP = 80
N_TEST = 5
NOVELTY_SCALE = 0.50

CONDITIONS = (
    "scenic_familiar",
    "dull_familiar",
    "both_familiar",
    "none_familiar",
)


@dataclass
class FamiliarityInternalResult:
    model: str
    seed: int
    scenic_side: str
    condition: str
    phase: str
    episode_idx: int
    choice_entry: Optional[str]
    decided: bool
    scenic_choice: bool
    split_novelty_left: float
    split_novelty_right: float
    split_novelty_delta: float
    split_novelty_scenic: float
    split_novelty_dull: float
    split_valence: float
    split_arousal: float
    split_body_budget: float
    score_base: float
    score_epistemic: float
    score_progress: float
    score_curiosity: float
    score_total: float
    score_other: float
    action: str
    action_random: bool


def new_agent_with_memory(env, model, seed, start, heading, eps, memory=None):
    agent = EvalAgent(env, model=model, seed=seed, start=start, heading=heading, eps=eps)
    if memory is not None:
        policy = getattr(agent, "policy", None)
        if not isinstance(policy, cw.ActivePerceptionPolicy) or policy.adapter is not cw._CORRIDOR_ADAPTER:
            policy = cw.ActivePerceptionPolicy(
                cw._CORRIDOR_ADAPTER,
                forward_model=cw.select_forward_model(model=model),
            )
            try:
                agent.policy = policy
            except Exception:
                pass
        policy.state.viewpoint_counts = memory
    return agent


def heading_toward(split_x: int, target_x: int, default_heading: int) -> int:
    if target_x < split_x:
        return 3
    if target_x > split_x:
        return 1
    return default_heading


def novelty_bonus(memory: Dict[Tuple[int, int, int], int], y: int, x: int, heading: int) -> float:
    visits = memory.get((y, x, heading), 0)
    return NOVELTY_SCALE / np.sqrt(1.0 + visits)


def split_novelty_bonus(memory: Dict[Tuple[int, int, int], int], env: QualiaphiliaCorridorWorld):
    split_y, split_x, split_heading = env.split_pose
    left_x = min(env.scenic_x, env.dull_x)
    right_x = max(env.scenic_x, env.dull_x)
    left_heading = heading_toward(split_x, left_x, split_heading)
    right_heading = heading_toward(split_x, right_x, split_heading)
    left_bonus = novelty_bonus(memory, split_y, split_x, left_heading)
    right_bonus = novelty_bonus(memory, split_y, split_x, right_heading)
    scenic_heading = heading_toward(split_x, env.scenic_x, split_heading)
    dull_heading = heading_toward(split_x, env.dull_x, split_heading)
    scenic_bonus = novelty_bonus(memory, split_y, split_x, scenic_heading)
    dull_bonus = novelty_bonus(memory, split_y, split_x, dull_heading)
    return left_bonus, right_bonus, right_bonus - left_bonus, scenic_bonus, dull_bonus


def step_with_memory(env, memory, y, x, heading, action):
    memory[(y, x, heading)] = memory.get((y, x, heading), 0) + 1
    return env.step(y, x, action, heading)


def scripted_familiarize_lane(env, memory, side, steps):
    split_y, split_x, heading = env.split_pose
    y, x, heading = split_y, split_x, heading
    lane_x = env.scenic_x if side == "scenic" else env.dull_x
    steps_remaining = int(steps)

    if lane_x < split_x:
        y, x, heading = step_with_memory(env, memory, y, x, heading, "turn_right")
        steps_remaining -= 1
    elif lane_x > split_x:
        y, x, heading = step_with_memory(env, memory, y, x, heading, "turn_left")
        steps_remaining -= 1

    while steps_remaining > 0 and x != lane_x:
        y2, x2, h2 = step_with_memory(env, memory, y, x, heading, "forward")
        steps_remaining -= 1
        if (y2, x2) == (y, x):
            break
        y, x, heading = y2, x2, h2

    if steps_remaining > 0 and heading != 2:
        action = "turn_left" if heading == 3 else "turn_right"
        y, x, heading = step_with_memory(env, memory, y, x, heading, action)
        steps_remaining -= 1

    while steps_remaining > 0:
        if y >= env.goal_y:
            break
        y2, x2, h2 = step_with_memory(env, memory, y, x, heading, "forward")
        steps_remaining -= 1
        if (y2, x2) == (y, x):
            break
        y, x, heading = y2, x2, h2


def read_affect_state(agent: EvalAgent):
    st = agent.read_state()
    valence = st.get("node_Nv", np.nan)
    arousal = st.get("node_Na", np.nan)
    body_budget = st.get("bb_model", np.nan)
    return float(valence), float(arousal), float(body_budget)


def score_action_components(
    agent: EvalAgent,
    env: QualiaphiliaCorridorWorld,
    action: str,
    memory_snapshot: Dict[Tuple[int, int, int], int],
    last_action: Optional[str],
    horizon: int,
    w_progress: float,
    w_epistemic: float,
    beauty_weight: float,
    use_beauty_term: Optional[bool],
    curiosity: bool,
    rng_state: Optional[dict] = None,
) -> Dict[str, float]:
    current_I, *_ = _ADAPTER.compute_I_affect(env, agent.y, agent.x, agent.heading)
    eval_info = _ADAPTER.eval_action(env, agent.y, agent.x, agent.heading, action)
    predicted_I, *_ = _ADAPTER.compute_I_affect(
        env, eval_info.pred_y, eval_info.pred_x, eval_info.pred_heading
    )

    base = dict(getattr(agent.net, "_ipsundrum_state", {}))
    base["g"] = float(base.get("g", getattr(agent.b.params, "g", 1.0)))

    if rng_state is None:
        rng_state = copy.deepcopy(agent.rng.bit_generator.state)
    rng_copy = np.random.default_rng()
    rng_copy.bit_generator.state = rng_state

    s_pred = dict(base)
    forward_model = cw.select_forward_model(agent=agent)
    for _ in range(horizon):
        s_pred = forward_model(
            s_pred,
            agent.b.params,
            agent.b.affect,
            predicted_I,
            rng=rng_copy,
        )

    s_pred["action"] = action
    base_score, epistemic, commit_pen = score_internal_components(
        s_pred,
        agent.b.affect,
        current_I,
        predicted_I,
        w_epistemic=w_epistemic,
        last_action=last_action,
    )

    score = base_score + epistemic - commit_pen

    progress_term = 0.0
    if w_progress and _ADAPTER.progress_fn is not None:
        progress = float(_ADAPTER.progress_fn(env, agent.y, agent.x, eval_info.y, eval_info.x, action))
        progress_term = float(w_progress) * progress
        score += progress_term

    curiosity_bonus = 0.0
    if curiosity and memory_snapshot is not None:
        viewpoint_next = (eval_info.y, eval_info.x, eval_info.heading)
        visit_count = memory_snapshot.get(viewpoint_next, 0)
        curiosity_bonus = NOVELTY_SCALE / np.sqrt(1.0 + visit_count)
        score += curiosity_bonus

    use_beauty = use_beauty_term
    if use_beauty is None:
        use_beauty = bool(getattr(env, "use_beauty_term", True))
    beauty_term = 0.0
    if use_beauty and beauty_weight != 0.0 and hasattr(env, "beauty"):
        scenic_dest = float(env.beauty[eval_info.y, eval_info.x])
        beauty_term = float(beauty_weight) * scenic_dest
        score += beauty_term

    bump_penalty = 0.0
    if eval_info.bumped:
        bump_penalty = float(_ADAPTER.bump_penalty_fn(env))
        score -= bump_penalty

    turn_cost = 0.0
    if action in ("turn_left", "turn_right"):
        turn_cost = float(_ADAPTER.turn_cost_fn(env))
        score -= turn_cost

    stay_penalty = 0.0
    if action == "stay":
        stay_penalty = float(_ADAPTER.stay_penalty_fn(env))
        score -= stay_penalty

    forward_prior = 0.0
    if action == "forward":
        forward_prior = float(_ADAPTER.forward_prior_fn(env))
        score += forward_prior

    return {
        "score_base": float(base_score),
        "score_epistemic": float(epistemic),
        "score_progress": float(progress_term),
        "score_curiosity": float(curiosity_bonus),
        "score_total": float(score),
        "score_other": float(score - (base_score + epistemic + progress_term + curiosity_bonus)),
        "score_commit_penalty": float(commit_pen),
        "score_bump_penalty": float(bump_penalty),
        "score_turn_cost": float(turn_cost),
        "score_stay_penalty": float(stay_penalty),
        "score_forward_prior": float(forward_prior),
        "score_beauty": float(beauty_term),
    }


def run_test_episode(
    env: QualiaphiliaCorridorWorld,
    model: str,
    seed: int,
    scenic_side: str,
    condition: str,
    memory: Dict[Tuple[int, int, int], int],
    episode_idx: int,
    T: int = 80,
    w_progress: float = 0.20,
):
    split_y, split_x, heading = env.split_pose
    agent = new_agent_with_memory(
        env, model, seed, start=(split_y, split_x), heading=heading, eps=0.15, memory=memory
    )

    memory_snapshot = dict(memory)
    (
        split_left,
        split_right,
        split_delta,
        split_scenic,
        split_dull,
    ) = split_novelty_bonus(memory_snapshot, env)

    choice_entry = None
    decided_entry = False
    valence0 = np.nan
    arousal0 = np.nan
    body_budget0 = np.nan

    score_base = np.nan
    score_epistemic = np.nan
    score_progress = np.nan
    score_curiosity = np.nan
    score_total = np.nan
    score_other = np.nan
    action = "stay"
    action_random = False

    for t in range(T):
        I_total, *_ = cw.compute_I_affect(env, agent.y, agent.x, agent.heading)
        if hasattr(agent.net, "_update_ipsundrum_sensor"):
            agent.net._update_ipsundrum_sensor(float(I_total), rng=agent.rng)
        else:
            agent.net.set_sensor_value("Ns", float(np.clip(0.5 + 0.5 * I_total, 0.0, 1.0)))
        agent.net.step()

        if t == 0:
            valence0, arousal0, body_budget0 = read_affect_state(agent)

        choice_entry, decided_entry = update_choice_entry(
            choice_entry, decided_entry, env, agent.y, agent.x
        )

        if agent.y >= env.goal_y:
            break

        explore_eps = 0.50 if choice_entry is None else agent.eps
        last_action = getattr(getattr(agent, "policy", None), "state", None)
        last_action = getattr(last_action, "last_action", None)
        rng_state_before = copy.deepcopy(agent.rng.bit_generator.state)

        policy_action = cw.choose_action_feelings(
            agent,
            horizon=5,
            curiosity=True,
            w_progress=w_progress,
            w_epistemic=0.0,
            beauty_weight=0.0,
            use_beauty_term=False,
        )
        if agent.rng.random() < explore_eps:
            action = agent.rng.choice(["forward", "turn_left", "turn_right"])
            action_random = True
            if hasattr(agent, "policy"):
                agent.policy.state.last_action = action
        else:
            action = policy_action
            action_random = False

        if t == 0:
            comps = score_action_components(
                agent,
                env,
                action,
                memory_snapshot,
                last_action,
                horizon=5,
                w_progress=w_progress,
                w_epistemic=0.0,
                beauty_weight=0.0,
                use_beauty_term=False,
                curiosity=True,
                rng_state=rng_state_before,
            )
            score_base = comps["score_base"]
            score_epistemic = comps["score_epistemic"]
            score_progress = comps["score_progress"]
            score_curiosity = comps["score_curiosity"]
            score_total = comps["score_total"]
            score_other = comps["score_other"]

        agent.y, agent.x, agent.heading = env.step(agent.y, agent.x, action, agent.heading)

    scenic_choice = (choice_entry == "scenic")

    return FamiliarityInternalResult(
        model=model,
        seed=seed,
        scenic_side=scenic_side,
        condition=condition,
        phase="test",
        episode_idx=episode_idx,
        choice_entry=choice_entry,
        decided=bool(decided_entry),
        scenic_choice=bool(scenic_choice),
        split_novelty_left=float(split_left),
        split_novelty_right=float(split_right),
        split_novelty_delta=float(split_delta),
        split_novelty_scenic=float(split_scenic),
        split_novelty_dull=float(split_dull),
        split_valence=float(valence0),
        split_arousal=float(arousal0),
        split_body_budget=float(body_budget0),
        score_base=float(score_base),
        score_epistemic=float(score_epistemic),
        score_progress=float(score_progress),
        score_curiosity=float(score_curiosity),
        score_total=float(score_total),
        score_other=float(score_other),
        action=action,
        action_random=bool(action_random),
    )


def run_familiarization(env, memory, condition):
    if condition == "none_familiar":
        return
    for i in range(N_FAM):
        if condition == "both_familiar":
            side = "scenic" if i % 2 == 0 else "dull"
        elif condition == "scenic_familiar":
            side = "scenic"
        elif condition == "dull_familiar":
            side = "dull"
        else:
            side = "scenic"
        scripted_familiarize_lane(env, memory, side=side, steps=FAM_STEPS_PER_EP)


def plot_scenic_choice(summary, delta_summary, condition_order, out_path):
    n_models = len(MODEL_ORDER)
    ncols = min(3, max(1, n_models))
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7), sharey="row")
    axes = np.atleast_2d(axes)
    label_map = {
        "scenic_familiar": "pre: scenic",
        "dull_familiar": "pre: dull",
        "both_familiar": "pre: both",
        "none_familiar": "pre: none",
    }
    tick_labels = [label_map.get(c, c) for c in condition_order]
    for idx, model in enumerate(MODEL_ORDER):
        top = axes[0, idx]
        sub = summary[summary["model"] == model].copy()
        sub = sub.set_index("condition").reindex(condition_order).reset_index()
        means = sub["mean"].to_numpy()
        sems = sub["sem"].to_numpy()
        xs = np.arange(len(condition_order))
        top.bar(xs, means, yerr=sems, capsize=3, color="#5a6d7b")
        top.set_title(canonical_model_display(model))
        top.set_xticks(xs)
        top.set_xticklabels(tick_labels, rotation=20, ha="right")
        top.set_ylim(0.0, 1.0)
        if idx == 0:
            top.set_ylabel("P(enter scenic)")

        bottom = axes[1, idx]
        dsub = delta_summary[delta_summary["model"] == model].copy()
        dsub = dsub.set_index("condition").reindex(condition_order).reset_index()
        dmeans = dsub["mean"].to_numpy()
        dsems = dsub["sem"].to_numpy()
        bottom.bar(xs, dmeans, yerr=dsems, capsize=3, color="#7b6d5a")
        bottom.axhline(0.0, color="#333333", lw=1.0, alpha=0.6)
        bottom.set_xticks(xs)
        bottom.set_xticklabels(tick_labels, rotation=20, ha="right")
        if idx == 0:
            bottom.set_ylabel("Δ P(enter scenic)\nvs pre: none")
    for ax in axes[0, n_models:]:
        ax.set_visible(False)
    for ax in axes[1, n_models:]:
        ax.set_visible(False)
    fig.suptitle(
        "Internal familiarity control: scenic entry by pre-exposure (top) and Δ vs none (bottom)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_experiment():
    all_results = []
    for model in MODEL_ORDER:
        for seed in range(5):
            scenic_side = "left" if seed % 2 == 0 else "right"
            for condition in CONDITIONS:
                print(f"Running {model}, seed={seed}, condition={condition}...")
                env = QualiaphiliaCorridorWorld(H=18, W=18, seed=seed, scenic_side=scenic_side)
                env.use_beauty_term = False
                memory: Dict[Tuple[int, int, int], int] = {}

                run_familiarization(env, memory, condition)

                for ep_idx in range(1, N_TEST + 1):
                    res = run_test_episode(
                        env,
                        model,
                        seed,
                        scenic_side,
                        condition,
                        memory,
                        episode_idx=ep_idx,
                    )
                    all_results.append(asdict(res))

    os.makedirs("results/familiarity", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/familiarity/familiarity_internal_{timestamp}.csv"
    fig_path = f"results/familiarity/familiarity_internal_{timestamp}.png"

    df = pd.DataFrame(all_results)
    df, _ = order_models(df)
    df_out = df.copy()
    if "model" in df_out.columns:
        df_out["model"] = df_out["model"].map(canonical_model_display)
        df_out, _ = order_models(df_out, order=MODEL_DISPLAY_ORDER)
    df_out.to_csv(csv_path, index=False)

    decided = df[df["decided"] == True].copy()
    seed_rates = (
        decided.groupby(["model", "condition", "seed"], sort=False, observed=False)["scenic_choice"]
        .mean()
        .reset_index()
    )
    summary = seed_rates.groupby(["model", "condition"], sort=False, observed=False).agg(
        mean=("scenic_choice", "mean"),
        std=("scenic_choice", "std"),
        n=("scenic_choice", "count"),
    ).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    summary["sem"] = summary["sem"].fillna(0.0)
    summary, _ = order_models(summary)

    baseline = seed_rates[seed_rates["condition"] == "none_familiar"].copy()
    baseline = baseline.rename(columns={"scenic_choice": "baseline"})[["model", "seed", "baseline"]]
    delta = seed_rates.merge(baseline, on=["model", "seed"], how="left")
    delta["delta"] = delta["scenic_choice"] - delta["baseline"]
    delta_summary = delta.groupby(["model", "condition"], sort=False, observed=False).agg(
        mean=("delta", "mean"),
        std=("delta", "std"),
        n=("delta", "count"),
    ).reset_index()
    delta_summary["sem"] = delta_summary["std"] / np.sqrt(delta_summary["n"])
    delta_summary["sem"] = delta_summary["sem"].fillna(0.0)
    delta_summary, _ = order_models(delta_summary)

    plot_scenic_choice(summary, delta_summary, CONDITIONS, fig_path)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    run_experiment()
