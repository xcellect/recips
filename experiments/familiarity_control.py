"""Qualiaphilia familiarity control with designed exposure and novelty metrics."""
import copy
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from typing import Optional

import experiments.corridor_exp as cw
from experiments.qualiaphilia_assay import QualiaphiliaCorridorWorld, update_choice_entry
from experiments.evaluation_harness import EvalAgent, MODEL_ORDER, order_models
from utils.model_naming import MODEL_DISPLAY_ORDER, canonical_model_display, canonical_model_id


@dataclass
class FamiliarityResult:
    model: str
    seed: int
    scenic_side: str
    phase: str
    familiarize_side: str
    morning_idx: int
    valid: bool
    decided: bool
    choice_entry: Optional[str]
    choice_commit: str
    decided_commit: bool
    scenic_choice: bool
    scenic_time: int
    dull_time: int
    scenic_time_share: float
    scenic_time_barrier: int
    dull_time_barrier: int
    scenic_time_share_barrier: float
    split_delta_novelty: float
    split_scenic_novelty: float
    split_dull_novelty: float
    split_pred_I_scenic: float
    split_pred_I_dull: float
    split_pred_valence_scenic: float
    split_pred_valence_dull: float
    split_pred_arousal_scenic: float
    split_pred_arousal_dull: float
    mean_valence_scenic: float
    mean_valence_dull: float
    mean_arousal_scenic: float
    mean_arousal_dull: float


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


def safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def read_affect_state(agent: EvalAgent):
    st = agent.read_state()
    valence = st.get("node_Nv", np.nan)
    arousal = st.get("node_Na", np.nan)
    return float(valence), float(arousal)


def loc_visits(memory, y, x):
    return sum(memory.get((y, x, h), 0) for h in range(4))


def novelty(memory, y, x):
    visits = loc_visits(memory, y, x)
    return 0.50 / np.sqrt(1.0 + visits)


def split_novelty(memory, env):
    split_y = env.split_pose[0]
    s_nov = novelty(memory, split_y, env.scenic_x)
    d_nov = novelty(memory, split_y, env.dull_x)
    return s_nov, d_nov, (s_nov - d_nov)


def validate_delta_novelty(delta, familiarize_side, tol):
    if familiarize_side == "scenic":
        return delta < 0
    if familiarize_side == "dull":
        return delta > 0
    if familiarize_side == "both":
        return abs(delta) <= tol
    return True


def scripted_familiarize(env, memory, start_pose, steps, action_sequence=None):
    y, x, heading = start_pose
    if action_sequence is None:
        turn_remaining = 0
        for _ in range(int(steps)):
            memory[(y, x, heading)] = memory.get((y, x, heading), 0) + 1
            if turn_remaining > 0:
                action = "turn_left"
                turn_remaining -= 1
            else:
                action = "forward"
            y2, x2, h2 = env.step(y, x, action, heading)
            if action == "forward" and (y2, x2) == (y, x):
                turn_remaining = 2
            y, x, heading = y2, x2, h2
        return

    seq = list(action_sequence)
    for i in range(int(steps)):
        memory[(y, x, heading)] = memory.get((y, x, heading), 0) + 1
        action = seq[i % len(seq)]
        y, x, heading = env.step(y, x, action, heading)


N_FAM = 10
FAM_STEPS_PER_EP = 80
N_POST_REPEAT = 5
NOVELTY_TOL = 0.05


def run_choice_episode(
    env,
    model,
    seed,
    scenic_side,
    familiarize_side,
    memory,
    phase,
    morning_idx,
    T=80,
    w_progress=0.20,
    update_memory=True,
    score_kwargs=None,
    novelty_scale=None,
):
    split_y, split_x, heading = env.split_pose
    agent = new_agent_with_memory(
        env, model, seed, start=(split_y, split_x), heading=heading, eps=0.15, memory=memory
    )

    scenic_time = 0
    dull_time = 0
    time_center = 0
    scenic_time_barrier = 0
    dull_time_barrier = 0
    choice_commit = None
    decided_commit = False
    commitment_threshold = 3
    decision_deadline = T
    lane_history = []

    choice_entry = None
    decided_entry = False

    s_nov, d_nov, delta_nov = split_novelty(memory, env)
    valid = True
    if phase == "post" and familiarize_side in ("scenic", "dull", "both"):
        valid = validate_delta_novelty(delta_nov, familiarize_side, NOVELTY_TOL)
        if not valid:
            print(
                "WARNING: delta novelty check failed "
                f"(model={model}, seed={seed}, phase={phase}, morning={morning_idx}, "
                f"familiarize_side={familiarize_side}, delta={delta_nov:+.4f})"
            )

    valence_scenic = []
    valence_dull = []
    arousal_scenic = []
    arousal_dull = []
    split_pred_I_scenic = float("nan")
    split_pred_I_dull = float("nan")
    split_pred_valence_scenic = float("nan")
    split_pred_valence_dull = float("nan")
    split_pred_arousal_scenic = float("nan")
    split_pred_arousal_dull = float("nan")

    for t in range(T):
        if agent.x == env.scenic_x:
            scenic_time += 1
            lane_history.append("scenic")
        elif agent.x == env.dull_x:
            dull_time += 1
            lane_history.append("dull")
        else:
            time_center += 1
            lane_history.append("center")

        if env.barrier_start <= agent.y < env.barrier_end:
            if agent.x == env.scenic_x:
                scenic_time_barrier += 1
            elif agent.x == env.dull_x:
                dull_time_barrier += 1

        if choice_commit is None and t <= decision_deadline and len(lane_history) >= commitment_threshold:
            recent = lane_history[-commitment_threshold:]
            if all(l == "scenic" for l in recent):
                choice_commit = "scenic"
                decided_commit = True
            elif all(l == "dull" for l in recent):
                choice_commit = "dull"
                decided_commit = True

        choice_entry, decided_entry = update_choice_entry(
            choice_entry, decided_entry, env, agent.y, agent.x
        )

        I_total, *_ = cw.compute_I_affect(env, agent.y, agent.x, agent.heading)
        if hasattr(agent.net, "_update_ipsundrum_sensor"):
            agent.net._update_ipsundrum_sensor(float(I_total), rng=agent.rng)
        else:
            agent.net.set_sensor_value("Ns", float(np.clip(0.5 + 0.5 * I_total, 0.0, 1.0)))
        agent.net.step()

        # Counterfactual internal diagnostic at the split: what does the model predict
        # (via its own forward dynamics) for the scenic-turn vs dull-turn?
        #
        # This avoids "dull" readouts being underpowered for HB when it almost never
        # chooses dull in post trials.
        if t == 0:
            try:
                rng_state = copy.deepcopy(agent.rng.bit_generator.state)
                base_state = dict(getattr(agent.net, "_ipsundrum_state", {}))
                base_state["g"] = float(base_state.get("g", getattr(agent.b.params, "g", 1.0)))
                forward_model = cw.select_forward_model(model=model)
                probe_horizon = 5

                eval_left = cw._CORRIDOR_ADAPTER.eval_action(
                    env, agent.y, agent.x, agent.heading, "turn_left"
                )
                eval_right = cw._CORRIDOR_ADAPTER.eval_action(
                    env, agent.y, agent.x, agent.heading, "turn_right"
                )
                I_left, *_ = cw.compute_I_affect(env, eval_left.pred_y, eval_left.pred_x, eval_left.pred_heading)
                I_right, *_ = cw.compute_I_affect(env, eval_right.pred_y, eval_right.pred_x, eval_right.pred_heading)

                def rollout(pred_I: float) -> dict:
                    rg = np.random.default_rng()
                    rg.bit_generator.state = copy.deepcopy(rng_state)
                    s_pred = dict(base_state)
                    for _ in range(probe_horizon):
                        s_pred = forward_model(s_pred, agent.b.params, agent.b.affect, float(pred_I), rng=rg)
                    return s_pred

                s_left = rollout(float(I_left))
                s_right = rollout(float(I_right))

                val_left = float(s_left.get("valence", np.nan))
                aro_left = float(s_left.get("arousal", np.nan))
                val_right = float(s_right.get("valence", np.nan))
                aro_right = float(s_right.get("arousal", np.nan))

                # Map turn actions to scenic/dull based on which direction moves closer
                # to the configured scenic lane.
                left_dist = abs(int(eval_left.pred_x) - int(env.scenic_x))
                right_dist = abs(int(eval_right.pred_x) - int(env.scenic_x))
                if left_dist <= right_dist:
                    split_pred_I_scenic = float(I_left)
                    split_pred_I_dull = float(I_right)
                    split_pred_valence_scenic = val_left
                    split_pred_valence_dull = val_right
                    split_pred_arousal_scenic = aro_left
                    split_pred_arousal_dull = aro_right
                else:
                    split_pred_I_scenic = float(I_right)
                    split_pred_I_dull = float(I_left)
                    split_pred_valence_scenic = val_right
                    split_pred_valence_dull = val_left
                    split_pred_arousal_scenic = aro_right
                    split_pred_arousal_dull = aro_left
            except Exception:
                pass

        valence, arousal = read_affect_state(agent)
        if agent.x == env.scenic_x:
            valence_scenic.append(valence)
            arousal_scenic.append(arousal)
        elif agent.x == env.dull_x:
            valence_dull.append(valence)
            arousal_dull.append(arousal)

        if agent.y >= env.goal_y:
            break

        explore_eps = 0.50 if (scenic_time + dull_time) == 0 else agent.eps
        policy_action = None
        if update_memory:
            policy_action = cw.choose_action_feelings(
                agent,
                horizon=5,
                curiosity=True,
                w_progress=w_progress,
                w_epistemic=0.0,
                beauty_weight=0.0,
                use_beauty_term=False,
                novelty_scale=novelty_scale,
                **(score_kwargs or {}),
            )
        if agent.rng.random() < explore_eps:
            action = agent.rng.choice(["forward", "turn_left", "turn_right"])
            if update_memory and hasattr(agent, "policy"):
                agent.policy.state.last_action = action
        else:
            action = (
                policy_action
                if policy_action is not None
                else cw.choose_action_feelings(
                    agent,
                    horizon=5,
                    curiosity=False,
                    w_progress=w_progress,
                    w_epistemic=0.0,
                    beauty_weight=0.0,
                    use_beauty_term=False,
                    novelty_scale=novelty_scale,
                    **(score_kwargs or {}),
                )
            )
        agent.y, agent.x, agent.heading = env.step(agent.y, agent.x, action, agent.heading)

    if choice_commit is None:
        choice_commit = "none"
        decided_commit = False

    scenic_time_share = scenic_time / max(1, scenic_time + dull_time)
    scenic_time_share_barrier = scenic_time_barrier / max(
        1, scenic_time_barrier + dull_time_barrier
    )
    mean_valence_scenic = safe_nanmean(valence_scenic)
    mean_valence_dull = safe_nanmean(valence_dull)
    mean_arousal_scenic = safe_nanmean(arousal_scenic)
    mean_arousal_dull = safe_nanmean(arousal_dull)

    return FamiliarityResult(
        model=model,
        seed=seed,
        scenic_side=scenic_side,
        phase=phase,
        familiarize_side=familiarize_side,
        morning_idx=morning_idx,
        valid=bool(valid),
        decided=bool(decided_entry),
        choice_entry=choice_entry,
        choice_commit=choice_commit,
        decided_commit=bool(decided_commit),
        scenic_choice=(choice_entry == "scenic"),
        scenic_time=scenic_time,
        dull_time=dull_time,
        scenic_time_share=scenic_time_share,
        scenic_time_barrier=scenic_time_barrier,
        dull_time_barrier=dull_time_barrier,
        scenic_time_share_barrier=scenic_time_share_barrier,
        split_delta_novelty=delta_nov,
        split_scenic_novelty=s_nov,
        split_dull_novelty=d_nov,
        split_pred_I_scenic=float(split_pred_I_scenic),
        split_pred_I_dull=float(split_pred_I_dull),
        split_pred_valence_scenic=float(split_pred_valence_scenic),
        split_pred_valence_dull=float(split_pred_valence_dull),
        split_pred_arousal_scenic=float(split_pred_arousal_scenic),
        split_pred_arousal_dull=float(split_pred_arousal_dull),
        mean_valence_scenic=mean_valence_scenic,
        mean_valence_dull=mean_valence_dull,
        mean_arousal_scenic=mean_arousal_scenic,
        mean_arousal_dull=mean_arousal_dull,
    )


def run_familiarity_control(
    models=tuple(MODEL_ORDER),
    seeds=tuple(range(5)),
    post_repeats: int = N_POST_REPEAT,
    outdir: str = "results/familiarity",
    n_fam: int = N_FAM,
    fam_steps: int = FAM_STEPS_PER_EP,
):
    print("=" * 70)
    print("QUALIAPHILIA FAMILIARITY CONTROL (Designed Exposure)")
    print("=" * 70)

    all_results = []
    familiarize_sides = ["none", "scenic", "dull", "both"]

    for model in models:
        for seed in seeds:
            scenic_side = "left" if seed % 2 == 0 else "right"
            for familiarize_side in familiarize_sides:
                print(f"Running {model}, seed={seed}, familiarization={familiarize_side}...")

                env = QualiaphiliaCorridorWorld(H=18, W=18, seed=seed, scenic_side=scenic_side)
                env.use_beauty_term = False
                memory = {}

                res_base = run_choice_episode(
                    env,
                    model,
                    seed,
                    scenic_side,
                    familiarize_side,
                    memory,
                    phase="baseline",
                    morning_idx=0,
                    update_memory=False,
                )
                all_results.append(asdict(res_base))

                if familiarize_side != "none":
                    start_y = env.barrier_start
                    for i in range(int(n_fam)):
                        if familiarize_side == "both":
                            side = "scenic" if i % 2 == 0 else "dull"
                        else:
                            side = familiarize_side
                        start_x = env.scenic_x if side == "scenic" else env.dull_x
                        scripted_familiarize(
                            env,
                            memory,
                            start_pose=(start_y, start_x, 2),
                            steps=fam_steps,
                        )

                for morning_idx in range(1, int(post_repeats) + 1):
                    res_post = run_choice_episode(
                        env,
                        model,
                        seed,
                        scenic_side,
                        familiarize_side,
                        memory,
                        phase="post",
                        morning_idx=morning_idx,
                        update_memory=True,
                    )
                    all_results.append(asdict(res_post))

    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(all_results)
    df, _ = order_models(df)
    df = df.sort_values(["model", "seed", "familiarize_side", "phase", "morning_idx"])
    df_out = df.copy()
    if "model" in df_out.columns:
        df_out["model"] = df_out["model"].map(canonical_model_display)
        df_out, _ = order_models(df_out, order=MODEL_DISPLAY_ORDER)
        df_out = df_out.sort_values(["model", "seed", "familiarize_side", "phase", "morning_idx"])
    df_out.to_csv(os.path.join(outdir, "episodes_improved.csv"), index=False)

    df_valid = df[df["valid"] == True].copy() if "valid" in df.columns else df
    df_decided = df_valid[df_valid["decided"] == True].copy()

    summary = df_decided.groupby(
        ["model", "familiarize_side", "phase", "morning_idx"],
        sort=False,
        observed=False,
    ).agg(
        scenic_rate_entry=("scenic_choice", "mean"),
        scenic_time_share_barrier=("scenic_time_share_barrier", "mean"),
        split_delta_novelty=("split_delta_novelty", "mean"),
        mean_valence_scenic=("mean_valence_scenic", "mean"),
        mean_valence_dull=("mean_valence_dull", "mean"),
        mean_arousal_scenic=("mean_arousal_scenic", "mean"),
        mean_arousal_dull=("mean_arousal_dull", "mean"),
        n=("seed", "count"),
    ).reset_index()
    summary, _ = order_models(summary)
    summary_out = summary.copy()
    if "model" in summary_out.columns:
        summary_out["model"] = summary_out["model"].map(canonical_model_display)
        summary_out, _ = order_models(summary_out, order=MODEL_DISPLAY_ORDER)
    summary_out.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    print("\n" + "=" * 70)
    print("RESULTS (Decided Trials Only)")
    print("=" * 70)
    print(summary_out)

    post = df_decided[df_decided["phase"] == "post"].copy()
    post_scenic = post[post["familiarize_side"] == "scenic"]
    neg = post_scenic[post_scenic["split_delta_novelty"] < 0]

    print("\nPost P(scenic | split_delta_novelty < 0), scenic familiarization:")
    for morning_idx in sorted(post_scenic["morning_idx"].unique()):
        print(f"  morning {morning_idx}:")
        for model in models:
            subset = neg[(neg["model"] == model) & (neg["morning_idx"] == morning_idx)]
            rate = subset["scenic_choice"].mean() if len(subset) > 0 else np.nan
            print(f"    {model:18s}: {rate if np.isfinite(rate) else float('nan'):.3f}")

    repeat_summary = post.groupby(["model", "familiarize_side"], sort=False, observed=False).agg(
        repeat_scenic_rate=("scenic_choice", "mean"),
        n=("seed", "count"),
        mornings=("morning_idx", "nunique"),
    ).reset_index()
    repeat_summary, _ = order_models(repeat_summary)

    print("\nRepeat scenic rate across mornings (post):")
    print(repeat_summary)

    print("\nMedian delta_novelty by familiarize_side (post):")
    for side in ["scenic", "dull"]:
        subset = post[post["familiarize_side"] == side]
        med = float(np.median(subset["split_delta_novelty"])) if len(subset) > 0 else float("nan")
        print(f"{side:8s}: {med:+.4f}")

    side_bias_rows = []
    side_subset = post[post["familiarize_side"] == "scenic"].copy()
    for model in models:
        sub = side_subset[side_subset["model"] == model]
        left = sub[sub["scenic_side"] == "left"]
        right = sub[sub["scenic_side"] == "right"]
        side_bias_rows.append({
            "model": model,
            "scenic_rate_left": left["scenic_choice"].mean() if len(left) > 0 else np.nan,
            "scenic_rate_right": right["scenic_choice"].mean() if len(right) > 0 else np.nan,
            "n_left": int(len(left)),
            "n_right": int(len(right)),
        })

    side_bias_df = pd.DataFrame(side_bias_rows)
    side_bias_df, _ = order_models(side_bias_df)
    side_bias_out = side_bias_df.copy()
    if "model" in side_bias_out.columns:
        side_bias_out["model"] = side_bias_out["model"].map(canonical_model_display)
        side_bias_out, _ = order_models(side_bias_out, order=MODEL_DISPLAY_ORDER)
    side_bias_out.to_csv(os.path.join(outdir, "side_bias.csv"), index=False)
    print("\nSide-bias check (post, scenic-familiar):")
    print(side_bias_out)

    decided_rates = (
        df_valid.groupby(["model", "phase"], sort=False, observed=False)["decided"]
        .mean()
        .reset_index()
    )
    decided_rates, _ = order_models(decided_rates)
    decided_rates_out = decided_rates.copy()
    if "model" in decided_rates_out.columns:
        decided_rates_out["model"] = decided_rates_out["model"].map(canonical_model_display)
        decided_rates_out, _ = order_models(decided_rates_out, order=MODEL_DISPLAY_ORDER)
    print("\nDecided rate by model/phase:")
    print(decided_rates_out)

    return df, summary, side_bias_df, decided_rates


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Familiarity control assay")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--post_repeats", type=int, default=N_POST_REPEAT, help="Post repeats per seed")
    parser.add_argument("--outdir", type=str, default="results/familiarity", help="Output directory")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model list (optional)")
    args = parser.parse_args()

    if args.models.strip():
        model_list = tuple(canonical_model_id(m) for m in args.models.split(",") if m.strip())
    else:
        model_list = tuple(MODEL_ORDER)

    run_familiarity_control(
        models=model_list,
        seeds=tuple(range(args.seeds)),
        post_repeats=args.post_repeats,
        outdir=args.outdir,
    )
