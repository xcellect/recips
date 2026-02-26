"""Exploratory play assay with clarified operational definition.

Operational definition (toy GridWorld):
Play-like exploration = self-initiated exploration in the absence of extrinsic
reward/goals, characterized by:
  (1) sustained exploration (non-trivial coverage),
  (2) structured scanning behavior (turning/gaze sampling, not just locomotion),
  (3) not explained by freezing/indecision (control for dwell),
  (4) optional persistence under mild costs or constraints.

Aligned with Humphrey's exploratory play discussion (Sentience ch. 19-20).

Limitation: engineered curiosity is a gameable behavioral indicator. Following
"Identifying Indicators of Consciousness in AI systems," we treat this as a
credence-shifting mechanistic probe (not evidence of consciousness) and report
internal-process traces (valence/arousal/bb_err) to support attribution.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict, deque

import matplotlib
import numpy as np
import pandas as pd

import experiments.gridworld_exp as gw
from experiments.evaluation_harness import EvalAgent


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from utils.plot_style import apply_times_style  # noqa: E402


apply_times_style()


ACTIONS = ("forward", "turn_left", "turn_right", "stay")

EPISODE_STEPS = 200
SEEDS = 5
SENSORY_BINS = 8
HAZARD_SMELL_THRESH = 0.35
ASSAY_VARIANT = os.getenv("EXPLORATORY_PLAY_VARIANT", "neutral_texture")
CYCLE_LENGTHS = (2, 3, 4, 5, 6)
CYCLE_MIN_REPEATS = 3
CYCLE_TAIL_STEPS = 150


def internal_actuator_mode(net) -> bool:
    p = getattr(net, "nodes", {}).get("P")
    ne = getattr(net, "nodes", {}).get("Ne")
    if p is None or ne is None:
        return False
    threshold = getattr(p, "efference_threshold", None)
    if threshold is None:
        return False
    return float(ne.activation) >= float(threshold)

MODEL_CONFIGS = [
    {
        "name": "random",
        "label": "Random",
        "model": "recon",
        "policy": "random",
        "curiosity": False,
        "w_epistemic": 0.0,
    },
    {
        "name": "Recon_epistemic",
        "label": "Recon+epistemic",
        "model": "recon",
        "policy": "feelings",
        "curiosity": False,
        "w_epistemic": 0.35,
    },
    {
        "name": "Recon_curiosity",
        "label": "Recon+curiosity",
        "model": "recon",
        "policy": "feelings",
        "curiosity": True,
        "w_epistemic": 0.35,
    },
    {
        "name": "Ipsundrum_curiosity",
        "label": "Ipsundrum+curiosity",
        "model": "humphrey",
        "policy": "feelings",
        "curiosity": True,
        "w_epistemic": 0.35,
    },
    {
        "name": "Ipsundrum_Affect_no_curiosity",
        "label": "Ipsundrum+affect (no curiosity)",
        "model": "humphrey_barrett",
        "policy": "feelings",
        "curiosity": False,
        "w_epistemic": 0.35,
    },
    {
        "name": "Ipsundrum_Affect_curiosity",
        "label": "Ipsundrum+affect+curiosity",
        "model": "humphrey_barrett",
        "policy": "feelings",
        "curiosity": True,
        "w_epistemic": 0.35,
    },
]
MODEL_LABELS = {cfg["name"]: cfg["label"] for cfg in MODEL_CONFIGS}
MODEL_ORDER_FULL = [cfg["name"] for cfg in MODEL_CONFIGS]
TRAJECTORY_MODELS = (
    "Recon_curiosity",
    "Ipsundrum_curiosity",
    "Ipsundrum_Affect_curiosity",
)

ABLATION_MODEL_CONFIGS = [
    {
        "name": "Ipsundrum_Affect",
        "label": "Ipsundrum+affect (full)",
        "model": "humphrey_barrett",
        "policy": "feelings",
        "curiosity": True,
        "w_epistemic": 0.35,
    },
    {
        "name": "Ipsundrum_Affect_readout_only",
        "label": "Ipsundrum+affect (readout-only)",
        "model": "humphrey_barrett_readout_only",
        "policy": "feelings",
        "curiosity": True,
        "w_epistemic": 0.35,
    },
    {
        "name": "Ipsundrum_Affect_modulation_only",
        "label": "Ipsundrum+affect (modulation-only)",
        "model": "humphrey_barrett_modulation_only",
        "policy": "feelings",
        "curiosity": True,
        "w_epistemic": 0.35,
    },
]


# -----------------
# Metric helpers
# -----------------

def safe_mean(values):
    if not values:
        return np.nan
    return float(np.nanmean(values))


def shannon_entropy(counts):
    total = float(sum(counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = float(count) / total
        entropy -= p * np.log2(p)
    return float(entropy)


def action_entropy(actions, action_space=ACTIONS):
    counts = Counter(actions)
    total = len(actions)
    if total == 0:
        return 0.0
    entropy = 0.0
    for action in action_space:
        count = counts.get(action, 0)
        if count <= 0:
            continue
        p = float(count) / float(total)
        entropy -= p * np.log2(p)
    return float(entropy)


def is_boundary(env, y, x):
    return y == 0 or y == env.H - 1 or x == 0 or x == env.W - 1


def cone_mean(field, env, y, x, heading, radius=5, fov_deg=70):
    dy0, dx0 = [(-1, 0), (0, 1), (1, 0), (0, -1)][heading]
    ang0 = np.arctan2(dy0, dx0)
    fov = np.deg2rad(fov_deg)
    w_sum = 0.0
    total = 0.0
    for rr in range(1, radius + 1):
        for yy in range(-rr, rr + 1):
            for xx in range(-rr, rr + 1):
                ny, nx = y + yy, x + xx
                if not env.in_bounds(ny, nx):
                    continue
                d = np.sqrt(xx * xx + yy * yy)
                if d < 1e-9 or d > rr:
                    continue
                ang = np.arctan2(yy, xx)
                da = (ang - ang0 + np.pi) % (2 * np.pi) - np.pi
                if abs(da) <= 0.5 * fov and d <= radius:
                    w = 1.0 / (d + 1e-6)
                    total += w * field[ny, nx]
                    w_sum += w
    if w_sum <= 1e-12:
        return 0.0
    return float(total / w_sum)


def run_lengths(seq):
    if not seq:
        return []
    lengths = []
    current = seq[0]
    run = 1
    for item in seq[1:]:
        if item == current:
            run += 1
        else:
            lengths.append(run)
            current = item
            run = 1
    lengths.append(run)
    return lengths


def summarize_lengths(lengths, percentiles=(50, 90, 99)):
    if not lengths:
        return {p: 0.0 for p in ["mean", "median", "p90", "p99", "max"]}
    arr = np.asarray(lengths, dtype=float)
    stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }
    if 50 in percentiles:
        stats["median"] = float(np.percentile(arr, 50))
    if 90 in percentiles:
        stats["p90"] = float(np.percentile(arr, 90))
    if 99 in percentiles:
        stats["p99"] = float(np.percentile(arr, 99))
    return stats


def cycle_metrics(state_seq, lengths=CYCLE_LENGTHS, min_repeats=CYCLE_MIN_REPEATS, tail=CYCLE_TAIL_STEPS):
    if not state_seq:
        return 0, 0
    seq = state_seq[-tail:] if len(state_seq) > tail else state_seq
    total_score = 0
    best_score = 0
    best_L = 0
    for L in lengths:
        i = 0
        score_L = 0
        while i + L * min_repeats <= len(seq):
            # Count repeats beyond the first occurrence of an L-length pattern.
            pattern = seq[i:i + L]
            repeats = 1
            while i + (repeats + 1) * L <= len(seq) and seq[i + repeats * L:i + (repeats + 1) * L] == pattern:
                repeats += 1
            if repeats >= min_repeats:
                score_L += (repeats - 1)
                i += repeats * L
            else:
                i += 1
        total_score += score_L
        if score_L > best_score:
            best_score = score_L
            best_L = L
    return total_score, best_L


# -----------------
# Main experiment
# -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploratory play assay")
    parser.add_argument("--profile", choices=["quick", "paper"], default=None, help="Preset seeds/steps")
    parser.add_argument("--seeds", type=int, default=None, help="Number of seeds")
    parser.add_argument("--steps", type=int, default=None, help="Episode steps")
    parser.add_argument("--outdir", type=str, default="results/exploratory-play", help="Output directory")
    parser.add_argument("--config_set", choices=["default", "ablation"], default="default", help="Model config set")
    parser.add_argument("--tag", type=str, default="", help="Tag for output filenames")
    parser.add_argument("--assay_variant", type=str, default=None, help="Override EXPLORATORY_PLAY_VARIANT")
    args = parser.parse_args()

    if args.profile == "paper":
        default_seeds = 20
        default_steps = EPISODE_STEPS
    elif args.profile == "quick":
        default_seeds = 5
        default_steps = EPISODE_STEPS
    else:
        default_seeds = SEEDS
        default_steps = EPISODE_STEPS

    SEEDS = int(args.seeds if args.seeds is not None else default_seeds)
    EPISODE_STEPS = int(args.steps if args.steps is not None else default_steps)
    if args.assay_variant:
        ASSAY_VARIANT = args.assay_variant

    if args.config_set == "ablation":
        MODEL_CONFIGS = list(ABLATION_MODEL_CONFIGS)
        TRAJECTORY_MODELS = tuple(cfg["name"] for cfg in MODEL_CONFIGS)
    else:
        MODEL_CONFIGS = list(MODEL_CONFIGS)
        TRAJECTORY_MODELS = (
            "Recon_curiosity",
            "Ipsundrum_curiosity",
            "Ipsundrum_Affect_curiosity",
        )
    MODEL_LABELS = {cfg["name"]: cfg["label"] for cfg in MODEL_CONFIGS}
    MODEL_ORDER_FULL = [cfg["name"] for cfg in MODEL_CONFIGS]

    tag = args.tag or (args.profile if args.profile else args.config_set)
    tag_suffix = f"_{tag}" if tag else ""

    print("=" * 70)
    print("EXPLORATORY PLAY ASSAY (clarified)")
    print("=" * 70)

    results = []
    trace_rows = []
    coverage_curves = defaultdict(list)
    occupancy_maps = {}
    trajectory_runs = {}
    hazard_map = None

    for cfg in MODEL_CONFIGS:
        for seed in range(SEEDS):
            print(f"Running {cfg['name']}, seed={seed}...")

            env = gw.GridWorld(H=18, W=18, seed=seed)
            env.beauty[:, :] = 0
            env.smell_b[:, :] = 0
            K = gw.gaussian_kernel(radius=3, sigma=1.2)
            env.smell_b = gw.conv2_same(env.beauty, K)
            assay_variant = ASSAY_VARIANT
            if assay_variant == "neutral_texture" and not hasattr(env, "texture"):
                assay_variant = "hazard_only"

            agent = EvalAgent(
                env,
                model=cfg["model"],
                seed=seed,
                start=(env.H // 2, env.W // 2),
                heading=1,
                eps=0.0,
            )

            visited_states = set()
            visited_viewpoints = set()
            headings_per_cell = defaultdict(set)
            sensory_vectors = []
            neutral_vectors = []
            viewpoint_counts = Counter()
            cell_seq = []
            viewpoint_seq = []
            travel_distance = 0.0
            last_pos = (agent.y, agent.x)
            start_pos = (agent.y, agent.x)

            action_hist = Counter()
            actions = []
            coverage_curve = []
            revisit_count = 0
            turn_in_place_count = 0
            blocked_forward_count = 0
            true_stall_count = 0
            scan_events = 0
            hazard_contacts = 0
            boundary_steps = 0
            internal_actuator_steps = 0

            turn_window = deque(maxlen=3)
            pos_window = deque(maxlen=3)

            precision_eff_samples = []
            alpha_eff_samples = []
            valence_samples = []
            arousal_samples = []
            Ns_samples = []
            bb_err_samples = []

            occupancy = np.zeros((env.H, env.W), dtype=int)
            record_traj = seed == 0 and cfg["name"] in TRAJECTORY_MODELS
            traj_positions = []
            traj_scan_positions = []
            if record_traj and hazard_map is None:
                hazard_map = env.hazard.copy()

            for t in range(EPISODE_STEPS):
                occupancy[agent.y, agent.x] += 1
                visited_states.add((agent.y, agent.x))
                viewpoint = (agent.y, agent.x, agent.heading)
                cell = (agent.y, agent.x)
                if is_boundary(env, agent.y, agent.x):
                    boundary_steps += 1
                cell_seq.append(cell)
                viewpoint_seq.append(viewpoint)
                viewpoint_counts[viewpoint] += 1
                if record_traj:
                    traj_positions.append(cell)
                if viewpoint in visited_viewpoints:
                    revisit_count += 1
                visited_viewpoints.add(viewpoint)
                headings_per_cell[(agent.y, agent.x)].add(agent.heading)
                coverage_curve.append(len(visited_viewpoints))

                I, touch, smell, vision = gw.compute_I_affect(env, agent.y, agent.x, agent.heading)
                smell_h = float(env.smell_h[agent.y, agent.x]) if hasattr(env, "smell_h") else 0.0
                smell_b = float(env.smell_b[agent.y, agent.x]) if hasattr(env, "smell_b") else 0.0
                sensory_vectors.append([float(touch), smell_h, smell_b, float(vision)])
                texture_cell = np.nan
                texture_cone = np.nan
                if assay_variant == "neutral_texture" and hasattr(env, "texture"):
                    texture_cell = float(env.texture[agent.y, agent.x])
                    texture_cone = cone_mean(env.texture, env, agent.y, agent.x, agent.heading)
                    neutral_vectors.append([texture_cell, texture_cone])
                if touch > 0.0 or smell_h >= HAZARD_SMELL_THRESH:
                    hazard_contacts += 1

                if hasattr(agent.net, "_update_ipsundrum_sensor"):
                    agent.net._update_ipsundrum_sensor(float(I), rng=agent.rng)
                    if hasattr(agent.net, "_ipsundrum_state"):
                        state = agent.net._ipsundrum_state
                        precision_eff_samples.append(float(state.get("precision_eff", 1.0)))
                        alpha_eff_samples.append(float(state.get("alpha_eff", np.nan)))
                else:
                    agent.net.set_sensor_value("Ns", float(np.clip(0.5 + 0.5 * I, 0.0, 1.0)))

                agent.net.step()

                internal_mode = internal_actuator_mode(agent.net)
                internal_actuator_steps += int(internal_mode)

                state = getattr(agent.net, "_ipsundrum_state", {})
                valence = float(state.get("valence", np.nan)) if isinstance(state, dict) else np.nan
                arousal = float(state.get("arousal", np.nan)) if isinstance(state, dict) else np.nan
                Ns = np.nan
                if "Ns" in getattr(agent.net, "nodes", {}):
                    Ns = float(agent.net.get("Ns").activation)
                elif isinstance(state, dict) and "Ns" in state:
                    Ns = float(state.get("Ns", np.nan))
                bb_err = np.nan
                if isinstance(state, dict) and getattr(agent.b.affect, "enabled", False):
                    bb_model = float(state.get("bb_model", np.nan))
                    setpoint = float(getattr(agent.b.affect, "setpoint", 0.0))
                    if not np.isnan(bb_model):
                        bb_err = abs(bb_model - setpoint)

                valence_samples.append(valence)
                arousal_samples.append(arousal)
                Ns_samples.append(Ns)
                bb_err_samples.append(bb_err)

                if cfg["policy"] == "random":
                    action = agent.rng.choice(ACTIONS)
                else:
                    action = gw.choose_action_feelings(
                        agent,
                        horizon=3,
                        curiosity=cfg["curiosity"],
                        w_epistemic=cfg["w_epistemic"],
                    )

                trace_rows.append(
                    {
                        "model": cfg["name"],
                        "base_model": cfg["model"],
                        "seed": seed,
                        "step": t,
                        "assay_variant": assay_variant,
                        "y": agent.y,
                        "x": agent.x,
                        "heading": agent.heading,
                        "action": action,
                        "I_total": float(I),
                        "touch": float(touch),
                        "smell": float(smell),
                        "smell_h": smell_h,
                        "smell_b": smell_b,
                        "vision": float(vision),
                        "texture_cell": texture_cell,
                        "texture_cone": texture_cone,
                        "is_boundary": is_boundary(env, agent.y, agent.x),
                        "valence": valence,
                        "arousal": arousal,
                        "Ns": Ns,
                        "internal_actuator_mode": int(internal_mode),
                        "bb_err": bb_err,
                        "precision_eff": precision_eff_samples[-1] if precision_eff_samples else np.nan,
                        "alpha_eff": alpha_eff_samples[-1] if alpha_eff_samples else np.nan,
                    }
                )

                prev_state = (agent.y, agent.x, agent.heading)
                agent.y, agent.x, agent.heading = env.step(agent.y, agent.x, action, agent.heading)
                travel_distance += abs(agent.y - last_pos[0]) + abs(agent.x - last_pos[1])
                last_pos = (agent.y, agent.x)

                action_hist[action] += 1
                actions.append(action)

                same_pos = (agent.y, agent.x) == (prev_state[0], prev_state[1])
                same_state = (agent.y, agent.x, agent.heading) == prev_state
                if action in ("turn_left", "turn_right") and same_pos:
                    turn_in_place_count += 1
                if action == "forward" and same_pos:
                    blocked_forward_count += 1
                if same_state:
                    true_stall_count += 1

                turn_in_place = action in ("turn_left", "turn_right") and (agent.y, agent.x) == (prev_state[0], prev_state[1])
                turn_window.append(turn_in_place)
                pos_window.append((agent.y, agent.x))
                # Scan event: >=2 turns-in-place within a 3-step window at the same location.
                if len(turn_window) == 3 and sum(turn_window) >= 2 and len(set(pos_window)) == 1:
                    scan_events += 1
                    if record_traj:
                        traj_scan_positions.append((agent.y, agent.x))

            if record_traj:
                trajectory_runs[cfg["name"]] = {
                    "positions": traj_positions,
                    "scan_positions": traj_scan_positions,
                }

            unique_viewpoints = len(visited_viewpoints)
            unique_states = len(visited_states)
            scan_depth = float(np.mean([len(h) for h in headings_per_cell.values()])) if headings_per_cell else 1.0

            quantized = [
                tuple(int(np.clip(x, 0, 1) * (SENSORY_BINS - 1)) for x in v)
                for v in sensory_vectors
            ]
            sensory_counts = Counter(quantized)
            sensory_entropy = shannon_entropy(sensory_counts)
            neutral_sensory_entropy = np.nan
            if neutral_vectors:
                neutral_quantized = [
                    tuple(int(np.clip(x, 0, 1) * (SENSORY_BINS - 1)) for x in v)
                    for v in neutral_vectors
                ]
                neutral_counts = Counter(neutral_quantized)
                neutral_sensory_entropy = shannon_entropy(neutral_counts)
            viewpoint_entropy = shannon_entropy(viewpoint_counts)

            cell_stats = summarize_lengths(run_lengths(cell_seq))
            dwell_mean = cell_stats["mean"]
            dwell_median = cell_stats["median"]
            dwell_p90 = cell_stats["p90"]
            dwell_p99 = cell_stats["p99"]
            dwell_max = cell_stats["max"]

            vp_stats = summarize_lengths(run_lengths(viewpoint_seq))
            vp_dwell_mean = vp_stats["mean"]
            vp_dwell_p90 = vp_stats["p90"]
            vp_dwell_max = vp_stats["max"]

            total_actions = len(actions)
            move_ratio = float(action_hist.get("forward", 0)) / float(total_actions or 1)
            turn_ratio = float(action_hist.get("turn_left", 0) + action_hist.get("turn_right", 0)) / float(total_actions or 1)
            turn_in_place_rate = float(turn_in_place_count) / float(total_actions or 1)
            blocked_forward_rate = float(blocked_forward_count) / float(total_actions or 1)
            true_stall_rate = float(true_stall_count) / float(total_actions or 1)
            revisit_rate = float(revisit_count) / float(total_actions or 1)
            boundary_hugging_fraction = float(boundary_steps) / float(total_actions or 1)
            internal_actuator_fraction = float(internal_actuator_steps) / float(total_actions or 1)

            net_displacement = abs(agent.y - start_pos[0]) + abs(agent.x - start_pos[1])
            tortuosity = float(travel_distance) / float(net_displacement + 1e-6)
            cycle_score, cycle_best_L = cycle_metrics(viewpoint_seq)

            total_visits = int(occupancy.sum())
            if total_visits <= 0:
                max_occupancy_fraction = 0.0
                occupancy_entropy = 0.0
            else:
                max_occupancy_fraction = float(occupancy.max()) / float(total_visits)
                probs = occupancy.ravel().astype(float) / float(total_visits)
                probs = probs[probs > 0]
                occupancy_entropy = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0

            results.append(
                {
                    "model": cfg["name"],
                    "base_model": cfg["model"],
                    "seed": seed,
                    "policy": cfg["policy"],
                    "curiosity": cfg["curiosity"],
                    "w_epistemic": cfg["w_epistemic"],
                    "assay_variant": assay_variant,
                    "unique_viewpoints": unique_viewpoints,
                    "scan_depth": scan_depth,
                    "sensory_entropy": sensory_entropy,
                    "neutral_sensory_entropy": neutral_sensory_entropy,
                    "viewpoint_entropy": viewpoint_entropy,
                    "unique_states": unique_states,
                    "travel_distance": travel_distance,
                    "net_displacement": net_displacement,
                    "tortuosity": tortuosity,
                    "cycle_score": cycle_score,
                    "cycle_best_L": cycle_best_L,
                    "action_entropy": action_entropy(actions),
                    "move_ratio": move_ratio,
                    "turn_ratio": turn_ratio,
                    "turn_in_place_rate": turn_in_place_rate,
                    "blocked_forward_rate": blocked_forward_rate,
                    "true_stall_rate": true_stall_rate,
                    "revisit_rate": revisit_rate,
                    "scan_events": scan_events,
                    "hazard_contacts": hazard_contacts,
                    "boundary_hugging_fraction": boundary_hugging_fraction,
                    "internal_actuator_fraction": internal_actuator_fraction,
                    "max_occupancy_fraction": max_occupancy_fraction,
                    "occupancy_entropy": occupancy_entropy,
                    "dwell_mean": dwell_mean,
                    "dwell_median": dwell_median,
                    "dwell_p90": dwell_p90,
                    "dwell_p99": dwell_p99,
                    "dwell_max": dwell_max,
                    "vp_dwell_mean": vp_dwell_mean,
                    "vp_dwell_p90": vp_dwell_p90,
                    "vp_dwell_max": vp_dwell_max,
                    "mean_valence": safe_mean(valence_samples),
                    "mean_arousal": safe_mean(arousal_samples),
                    "mean_Ns": safe_mean(Ns_samples),
                    "mean_bb_err": safe_mean(bb_err_samples),
                    "mean_precision_eff": safe_mean(precision_eff_samples),
                    "mean_alpha_eff": safe_mean(alpha_eff_samples),
                }
            )

            coverage_curves[cfg["name"]].append(coverage_curve)
            if seed == 0 and cfg["name"] not in occupancy_maps:
                occupancy_maps[cfg["name"]] = occupancy.copy()


    df = pd.DataFrame(results)
    if not df.empty:
        df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER_FULL, ordered=True)
        df = df.sort_values(["model", "seed"])
        summary = df.groupby("model", sort=False, observed=False).mean(numeric_only=True)
    else:
        summary = pd.DataFrame()

    print("\nRESULTS (mean across seeds):")
    if not summary.empty:
        summary_view = summary.reindex(MODEL_ORDER_FULL)
        print(
            summary_view[[
                "unique_states",
                "travel_distance",
                "unique_viewpoints",
                "scan_depth",
                "sensory_entropy",
                "neutral_sensory_entropy",
                "viewpoint_entropy",
                "action_entropy",
                "move_ratio",
                "turn_ratio",
                "turn_in_place_rate",
                "blocked_forward_rate",
                "true_stall_rate",
                "revisit_rate",
                "scan_events",
                "hazard_contacts",
                "boundary_hugging_fraction",
                "max_occupancy_fraction",
                "occupancy_entropy",
                "dwell_p90",
                "vp_dwell_p90",
                "tortuosity",
                "cycle_score",
                "mean_valence",
                "mean_arousal",
                "mean_bb_err",
            ]]
        )
    else:
        print("(no results)")

    # --- Outputs ---

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"exploratory_play_clarified{tag_suffix}.csv")
    df.to_csv(csv_path, index=False)

    trace_df = pd.DataFrame(trace_rows)
    trace_path = os.path.join(out_dir, f"exploratory_play_clarified_trace{tag_suffix}.csv")
    if not trace_df.empty:
        trace_df.to_csv(trace_path, index=False)

    if not summary.empty:
        summary.to_csv(os.path.join(out_dir, f"summary{tag_suffix}.csv"))

    # Plot coverage over time and key summary bars
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ax_cov, ax_ent, ax_rev, ax_scan, ax_occ, ax_blank = axes.flat

    colors = plt.cm.tab20(np.linspace(0, 1, len(MODEL_ORDER_FULL)))
    steps = np.arange(1, EPISODE_STEPS + 1)
    for idx, model_name in enumerate(MODEL_ORDER_FULL):
        curves = coverage_curves.get(model_name, [])
        if not curves:
            continue
        mean_curve = np.mean(np.array(curves), axis=0)
        ax_cov.plot(steps, mean_curve, label=MODEL_LABELS.get(model_name, model_name), color=colors[idx])

    ax_cov.set_title("Coverage over time (mean)")
    ax_cov.set_xlabel("Step")
    ax_cov.set_ylabel("Unique viewpoints")
    ax_cov.legend(fontsize=8, frameon=False)

    x = np.arange(len(MODEL_ORDER_FULL))
    labels = [MODEL_LABELS.get(m, m) for m in MODEL_ORDER_FULL]

    if not summary.empty:
        summary_view = summary.reindex(MODEL_ORDER_FULL)
        ax_ent.bar(x, summary_view["action_entropy"], color="steelblue", edgecolor="black")
        ax_rev.bar(x, summary_view["revisit_rate"], color="seagreen", edgecolor="black")
        ax_scan.bar(x, summary_view["scan_events"], color="sandybrown", edgecolor="black")
        ax_occ.bar(x, summary_view["occupancy_entropy"], color="slateblue", edgecolor="black")

    for ax, title, ylabel in (
        (ax_ent, "Action entropy", "bits"),
        (ax_rev, "Revisit rate", "fraction"),
        (ax_scan, "Scan events", "count"),
        (ax_occ, "Occupancy entropy", "bits"),
    ):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")

    ax_blank.axis("off")

    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"fig_exploratory_play_clarified{tag_suffix}.png")
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    # Optional occupancy heatmaps (representative seed=0)
    if occupancy_maps:
        n_models = len(MODEL_ORDER_FULL)
        ncols = 3
        nrows = int(np.ceil(n_models / ncols))
        fig_h, axes_h = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes_h = np.array(axes_h).reshape(-1)
        for idx, model_name in enumerate(MODEL_ORDER_FULL):
            ax = axes_h[idx]
            occ = occupancy_maps.get(model_name)
            if occ is None:
                ax.axis("off")
                continue
            im = ax.imshow(occ, cmap="viridis", interpolation="nearest")
            ax.set_title(MODEL_LABELS.get(model_name, model_name))
            ax.set_xticks([])
            ax.set_yticks([])
            fig_h.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes_h[n_models:]:
            ax.axis("off")

        fig_h.tight_layout()
        heatmap_path = os.path.join(out_dir, f"fig_exploratory_play_clarified_heatmaps{tag_suffix}.png")
        fig_h.savefig(heatmap_path, dpi=160)
        plt.close(fig_h)

    # Trajectory figure (paper models, seed=0)
    if hazard_map is not None and trajectory_runs:
        traj_order = list(TRAJECTORY_MODELS)
        label_map = {
            "Recon_curiosity": "Recon",
            "Ipsundrum_curiosity": "Ipsundrum",
            "Ipsundrum_Affect_curiosity": "Ipsundrum+affect",
        }
        fig_t, axes_t = plt.subplots(2, 2, figsize=(7.2, 7.2))
        axes_t = axes_t.reshape(-1)
        H, W = hazard_map.shape
        for idx, model_name in enumerate(traj_order):
            ax = axes_t[idx]
            ax.imshow(hazard_map, cmap="Reds", alpha=0.25, interpolation="nearest")
            traj = trajectory_runs.get(model_name)
            if not traj or not traj.get("positions"):
                ax.axis("off")
                continue
            positions = traj["positions"]
            xs = [p[1] for p in positions]
            ys = [p[0] for p in positions]
            ax.plot(xs, ys, color="black", linewidth=1.2, alpha=0.85)
            if traj.get("scan_positions"):
                scan_x = [p[1] for p in traj["scan_positions"]]
                scan_y = [p[0] for p in traj["scan_positions"]]
                ax.scatter(scan_x, scan_y, s=14, color="tab:blue", alpha=0.7)
            ax.scatter(xs[0], ys[0], s=50, color="green", edgecolor="black", linewidth=0.5, zorder=3)
            ax.scatter(xs[-1], ys[-1], s=60, color="red", marker="X", edgecolor="black", linewidth=0.5, zorder=3)
            ax.set_title(label_map.get(model_name, model_name))
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        for ax in axes_t[len(traj_order):]:
            ax.axis("off")

        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor="green",
                   markeredgecolor="black", label="start"),
            Line2D([0], [0], marker="X", linestyle="None", markersize=8, markerfacecolor="red",
                   markeredgecolor="black", label="end"),
        ]
        fig_t.legend(
            handles=legend_handles,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02),
            frameon=True,
            fontsize=9,
        )

        fig_t.tight_layout()
        traj_path = os.path.join(out_dir, f"fig_exploratory_play_trajectories{tag_suffix}.png")
        fig_t.savefig(traj_path, dpi=220)
        plt.close(fig_t)

    # Summary output (for publication_figures compatibility)
    dwell_path = None
    expected = {"Recon_curiosity", "Ipsundrum_curiosity", "Ipsundrum_Affect_curiosity"}
    expected_mapped = {"Recon", "Ipsundrum", "Ipsundrum+affect"}
    if "model" in df.columns and expected.issubset(set(df["model"].unique())):
        summary_rows = df[df["model"].isin(sorted(expected))].copy()
        summary_map = {
            "Recon_curiosity": "Recon",
            "Ipsundrum_curiosity": "Ipsundrum",
            "Ipsundrum_Affect_curiosity": "Ipsundrum+affect",
        }
        summary_rows["model"] = summary_rows["model"].map(summary_map)
    else:
        summary_rows = df.copy()

    if not summary_rows.empty:
        summary_rows = summary_rows.sort_values(["model", "seed"])
        summary_rows[
            [
                "model",
                "seed",
                "unique_viewpoints",
                "scan_depth",
                "sensory_entropy",
                "neutral_sensory_entropy",
                "viewpoint_entropy",
                "unique_states",
                "travel_distance",
                "net_displacement",
                "tortuosity",
                "cycle_score",
                "cycle_best_L",
                "mean_precision_eff",
                "mean_alpha_eff",
                "internal_actuator_fraction",
                "scan_events",
                "revisit_rate",
                "turn_in_place_rate",
                "true_stall_rate",
                "blocked_forward_rate",
                "max_occupancy_fraction",
                "occupancy_entropy",
                "dwell_p90",
                "dwell_max",
                "vp_dwell_p90",
                "vp_dwell_max",
                "boundary_hugging_fraction",
                "hazard_contacts",
                "assay_variant",
            ]
        ].to_csv(os.path.join(out_dir, "final_viewpoints.csv"), index=False)

        if expected_mapped.issubset(set(summary_rows["model"].unique())):
            dwell_order = ["Recon", "Ipsundrum", "Ipsundrum+affect"]
            dwell_labels = ["Recon", "Ipsundrum", "Ipsundrum+affect"]
            dwell_data = [summary_rows[summary_rows["model"] == m]["dwell_p90"].values for m in dwell_order]
            fig_d, ax_d = plt.subplots(figsize=(7, 3.8))
            dwell_colors = plt.cm.Set2(np.linspace(0, 1, len(dwell_order)))
            all_vals = np.concatenate([d for d in dwell_data if len(d) > 0]) if dwell_data else np.array([])
            for idx, data in enumerate(dwell_data, start=1):
                if data.size == 0:
                    continue
                rng = np.random.default_rng(100 + idx)
                jitter = rng.normal(0, 0.06, size=len(data))
                ax_d.scatter(
                    np.full(len(data), idx) + jitter,
                    data,
                    s=45,
                    alpha=0.7,
                    color=dwell_colors[idx - 1],
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=2,
                )
                median = float(np.median(data))
                q25, q75 = np.percentile(data, [25, 75])
                mean = float(np.mean(data))
                ax_d.plot([idx - 0.2, idx + 0.2], [median, median], color="black", linewidth=2.0, zorder=3)
                ax_d.plot([idx, idx], [q25, q75], color="black", linewidth=2.0, zorder=3)
                ax_d.scatter([idx], [mean], marker="D", s=35, color="black", zorder=4)

            ax_d.set_xticks(range(1, len(dwell_labels) + 1))
            ax_d.set_xticklabels(dwell_labels)
            ax_d.set_ylabel("Dwell p90 (steps)")
            ax_d.set_title("Dwell-time distribution (cell)")
            ax_d.grid(axis="y", alpha=0.3)
            if all_vals.size:
                ymin = max(0.0, float(np.min(all_vals)) - 0.5)
                ymax = float(np.max(all_vals)) + 0.5
                ax_d.set_ylim(ymin, ymax)
                if ymax - ymin <= 6:
                    ax_d.set_yticks(np.arange(np.floor(ymin), np.ceil(ymax) + 1))
            fig_d.tight_layout()
            dwell_path = os.path.join(out_dir, f"fig_exploratory_play_dwell{tag_suffix}.png")
            fig_d.savefig(dwell_path, dpi=200)
            plt.close(fig_d)

    print(f"\nOK: Results saved to {csv_path}")
    if not trace_df.empty:
        print(f"OK: Trace saved to {trace_path}")
    print(f"OK: Figure saved to {fig_path}")
    if occupancy_maps:
        print(f"OK: Heatmaps saved to {heatmap_path}")
    if hazard_map is not None and trajectory_runs:
        print(f"OK: Trajectories saved to {traj_path}")
    if dwell_path:
        print(f"OK: Dwell figure saved to {dwell_path}")
    print("\nDone.")
