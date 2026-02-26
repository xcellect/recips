"""
Pain-Tail Assay: Measure lingering behavioral effects after hazard contact removed.

Tests Humphrey's "thick moment" prediction: phenomenally conscious agents should
show behavioral persistence (lingering caution) after a noxious stimulus is removed,
while non-conscious agents should immediately return to baseline.
"""
import argparse
import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

from utils.plot_style import apply_times_style

apply_times_style()

import experiments.gridworld_exp as gw
from experiments.evaluation_harness import EvalAgent, build_model_network, MODEL_ORDER, order_models
from core.recon_core import ScriptState
from utils.model_naming import MODEL_DISPLAY_ORDER, canonical_model_display


@dataclass
class PainTailResult:
    model: str
    seed: int
    tail_duration: int  # steps until turn_rate < 0.2
    ns_baseline: float
    ns_peak: float
    ns_half_life: int  # steps to decay to 50% of peak
    ns_auc: float  # sum of Ns over post window
    ns_auc_above_baseline: float  # sum of max(0, Ns - baseline) over post window
    arousal_integral: float  # cumulative arousal
    mean_turn_rate_tail: float  # avg turn rate in first 20 steps
    mean_forward_rate_tail: float  # avg forward rate in first 20 steps


def baseline_corrected_half_life(
    ns_trace: List[float],
    *,
    ns_baseline: float,
    ns_peak: float,
    max_steps: int,
    eps: float = 1e-12,
) -> int:
    """
    Construct-valid half-life for Ns:
      time to decay halfway back to baseline, i.e.
        Ns <= baseline + 0.5 * (peak - baseline).

    This avoids the strict-inequality artifact when baseline is ~0.5 and
    the trace returns exactly to baseline (common in non-affect variants).
    """
    peak = float(ns_peak)
    base = float(ns_baseline)
    delta = peak - base
    if not np.isfinite(delta) or delta <= 0.0:
        return 0
    target = base + 0.5 * delta
    for t, ns_val in enumerate(ns_trace):
        if float(ns_val) <= target + float(eps):
            return int(t)
    return int(max_steps)


def run_pain_tail_trial(
    model: str,
    seed: int,
    post_stimulus_steps: int = 50
) -> PainTailResult:
    """
    Run one pain-tail trial.
    
    Protocol:
      1. Create gridworld environment
      2. Place agent at safe location
      3. Force ONE hazard contact (teleport onto hazard for 1 step)
      4. Teleport back to adjacent safe cell
      5. Record behavior for post_stimulus_steps
    """
    env = gw.GridWorld(H=18, W=18, seed=seed)
    
    # Find a hazard cell and adjacent safe cell
    hazard_locs = np.argwhere(env.hazard > 0)
    if len(hazard_locs) == 0:
        # Fallback: manually place one hazard
        env.hazard[9, 9] = 1.0
        hazard_y, hazard_x = 9, 9
    else:
        hazard_y, hazard_x = hazard_locs[0]
    
    # Find adjacent safe cell
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        safe_y, safe_x = hazard_y + dy, hazard_x + dx
        if env.in_bounds(safe_y, safe_x) and env.hazard[safe_y, safe_x] == 0:
            break
    else:
        # Fallback
        safe_y, safe_x = hazard_y, hazard_x + 2
    
    # Create agent at safe position
    agent = EvalAgent(
        env, model=model, seed=seed,
        start=(safe_y, safe_x), heading=1,  # facing right
        eps=0.0  # no exploration noise
    )

    # Snapshot pre-stimulus internal state so we can compute a baseline under the
    # *post-stimulus* (hazard-removed) sensory input. This keeps the half-life
    # metric from being confounded by baseline offsets in Ns.
    pre_state = dict(getattr(agent.net, "_ipsundrum_state", {}))
    pre_state["g"] = float(pre_state.get("g", getattr(agent.b.params, "g", 1.0)))
    
    # --- FORCE HAZARD CONTACT ---
    # Teleport to hazard
    agent.y, agent.x = hazard_y, hazard_x
    
    # Compute sensory input and update physiology
    I_hazard, I_touch, *_ = gw.compute_I_affect(env, agent.y, agent.x, agent.heading)
    assert I_touch > 0.5, "Should be on hazard"
    
    if hasattr(agent.net, '_update_ipsundrum_sensor'):
        agent.net._update_ipsundrum_sensor(float(I_hazard), rng=agent.rng)
    else:
        agent.net.set_sensor_value('Ns', float(np.clip(0.5 + 0.5*I_hazard, 0.0, 1.0)))
    
    agent.net.step()

    # Record peak Ns
    st = agent.read_state()
    ns_peak = float(st.get('node_Ns', 0))

    # Remove the noxious stimulus from the environment to isolate lingering internal effects.
    env.hazard[hazard_y, hazard_x] = 0.0
    K = gw.gaussian_kernel(radius=3, sigma=1.2)
    env.smell_h = gw.conv2_same(env.hazard, K)

    # Baseline Ns under the hazard-removed sensory input, starting from the pre-stimulus state.
    # This approximates the equilibrium Ns for the "safe" context without carrying over the
    # hazard contact itself.
    agent.y, agent.x = safe_y, safe_x
    I_safe_baseline, *_ = gw.compute_I_affect(env, agent.y, agent.x, agent.heading)
    forward_model = gw.select_forward_model(model=model)
    rg = np.random.default_rng(seed + 1337)
    s_base = copy.deepcopy(pre_state)
    for _ in range(20):
        s_base = forward_model(s_base, agent.b.params, agent.b.affect, float(I_safe_baseline), rng=rg)
    ns_baseline = float(s_base.get("Ns", 0.5))
    
    # --- REMOVE STIMULUS: teleport back to safe cell ---
    agent.y, agent.x = safe_y, safe_x
    
    # --- MEASURE TAIL BEHAVIOR ---
    # NOTE: Keep agent STATIONARY to measure pure internal dynamics decay
    # (moving could re-trigger hazards or change sensory input unpredictably)
    actions: List[str] = []
    ns_trace: List[float] = []
    na_trace: List[float] = []
    
    for t in range(post_stimulus_steps):
        # Sensory input (now safe, agent stationary)
        I_safe, I_touch, *_ = gw.compute_I_affect(env, agent.y, agent.x, agent.heading)
        assert I_touch < 0.1, f"Should be safe at step {t}"
        
        # Physiology update
        if hasattr(agent.net, '_update_ipsundrum_sensor'):
            agent.net._update_ipsundrum_sensor(float(I_safe), rng=agent.rng)
        else:
            agent.net.set_sensor_value('Ns', float(np.clip(0.5 + 0.5*I_safe, 0.0, 1.0)))
        
        agent.net.step()
        
        # Record internal state
        st = agent.read_state()
        ns_trace.append(float(st.get('node_Ns', 0)))
        na_trace.append(float(st.get('node_Na', 0)))
        
        # Choose action (but don't execute - measure planning preference only)
        action = gw.choose_action_feelings(agent, horizon=3)
        actions.append(action)
        
        # DO NOT MOVE - we're measuring internal decay/persistence
        # (agent stays at safe_y, safe_x)
    
    # --- COMPUTE METRICS ---
    # Tail duration: when does turn_rate drop below 0.2?
    turn_rate_window = 5
    tail_duration = post_stimulus_steps
    for t in range(turn_rate_window, len(actions)):
        recent = actions[t-turn_rate_window:t]
        turn_count = sum(1 for a in recent if a in ('turn_left', 'turn_right'))
        if turn_count / turn_rate_window < 0.2:
            tail_duration = t
            break
    
    # Ns half-life
    ns_half_life = baseline_corrected_half_life(
        ns_trace,
        ns_baseline=ns_baseline,
        ns_peak=float(ns_peak),
        max_steps=post_stimulus_steps,
    )
    
    # Arousal integral
    arousal_integral = float(np.sum(na_trace))

    # Persistence AUC (baseline-corrected)
    ns_arr = np.asarray(ns_trace, dtype=float)
    ns_auc = float(np.sum(ns_arr))
    ns_auc_above_baseline = float(np.sum(np.maximum(0.0, ns_arr - float(ns_baseline))))
    
    # Mean rates in first 20 steps
    first_20 = actions[:20]
    n = max(1, len(first_20))
    mean_turn_rate = sum(1 for a in first_20 if a in ('turn_left', 'turn_right')) / n
    mean_forward_rate = sum(1 for a in first_20 if a == 'forward') / n
    
    return PainTailResult(
        model=model,
        seed=seed,
        tail_duration=tail_duration,
        ns_baseline=float(ns_baseline),
        ns_peak=float(ns_peak),
        ns_half_life=ns_half_life,
        ns_auc=float(ns_auc),
        ns_auc_above_baseline=float(ns_auc_above_baseline),
        arousal_integral=arousal_integral,
        mean_turn_rate_tail=mean_turn_rate,
        mean_forward_rate_tail=mean_forward_rate
    )


def run_pain_tail_sweep(
    models: Tuple[str, ...] = tuple(MODEL_ORDER),
    seeds: Tuple[int, ...] = tuple(range(10)),
    post_stimulus_steps: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run pain-tail assay across models and seeds."""
    results: List[Dict[str, Any]] = []
    
    for model in models:
        for seed in seeds:
            print(f"Running pain-tail: model={model}, seed={seed}")
            res = run_pain_tail_trial(model, seed, post_stimulus_steps=post_stimulus_steps)
            results.append(asdict(res))
    
    df = pd.DataFrame(results)
    df, _ = order_models(df)
    df = df.sort_values(["model", "seed"])
    
    # Aggregate by model
    summary = df.groupby('model', as_index=False, sort=False, observed=False).agg(
        mean_tail_duration=('tail_duration', 'mean'),
        std_tail_duration=('tail_duration', 'std'),
        mean_ns_peak=('ns_peak', 'mean'),
        mean_ns_half_life=('ns_half_life', 'mean'),
        mean_ns_auc_above_baseline=('ns_auc_above_baseline', 'mean'),
        std_ns_auc_above_baseline=('ns_auc_above_baseline', 'std'),
        mean_arousal_integral=('arousal_integral', 'mean'),
        mean_turn_rate_tail=('mean_turn_rate_tail', 'mean'),
        mean_forward_rate_tail=('mean_forward_rate_tail', 'mean'),
        n=('seed', 'count')
    )
    summary, _ = order_models(summary)
    
    return df, summary


def plot_pain_tail_results(summary: pd.DataFrame):
    """Generate comparison plots for pain-tail assay."""
    summary = summary.copy()
    if "model" in summary.columns:
        summary["model"] = summary["model"].map(canonical_model_display)
    summary, _ = order_models(summary, order=MODEL_DISPLAY_ORDER)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Pain-Tail Assay: Lingering Behavioral Effects After Hazard Removal", fontsize=14, fontweight='bold')
    
    models = summary['model'].values
    x = np.arange(len(models))
    width = 0.6
    
    # Tail duration
    ax = axes[0, 0]
    ax.bar(x, summary['mean_tail_duration'], width, yerr=summary['std_tail_duration'],
           capsize=5, color='steelblue', edgecolor='black')
    ax.set_ylabel('Tail Duration (steps)')
    ax.set_title('How long does caution persist?')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Ns half-life
    ax = axes[0, 1]
    ax.bar(
        x,
        summary['mean_ns_auc_above_baseline'],
        width,
        yerr=summary['std_ns_auc_above_baseline'],
        capsize=5,
        color='coral',
        edgecolor='black',
    )
    ax.set_ylabel('Ns AUC Above Baseline')
    ax.set_title('How much salience persists after removal?')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Arousal integral
    ax = axes[1, 0]
    ax.bar(x, summary['mean_arousal_integral'], width, color='gold', edgecolor='black')
    ax.set_ylabel('Cumulative Arousal')
    ax.set_title('Total affective response')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Turn rate (first 20 steps)
    ax = axes[1, 1]
    ax.bar(x, summary['mean_turn_rate_tail'], width, color='mediumseagreen', edgecolor='black')
    ax.set_ylabel('Mean Turn Rate (first 20 steps)')
    ax.set_title('Post-stimulus scan/vigilance')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.axhline(0.2, color='red', linestyle='--', label='baseline threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pain-tail assay")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--post_steps", type=int, default=50, help="Post-stimulus steps")
    parser.add_argument("--outdir", type=str, default="results/pain-tail", help="Output directory")
    args = parser.parse_args()

    print("="*60)
    print("PAIN-TAIL ASSAY")
    print("="*60)
    
    df, summary = run_pain_tail_sweep(
        models=tuple(MODEL_ORDER),
        seeds=tuple(range(args.seeds)),
        post_stimulus_steps=args.post_steps,
    )

    # Save results
    os.makedirs(args.outdir, exist_ok=True)
    df_out = df.copy()
    summary_out = summary.copy()
    if "model" in df_out.columns:
        df_out["model"] = df_out["model"].map(canonical_model_display)
        df_out, _ = order_models(df_out, order=MODEL_DISPLAY_ORDER)
        df_out = df_out.sort_values(["model", "seed"])
    if "model" in summary_out.columns:
        summary_out["model"] = summary_out["model"].map(canonical_model_display)
        summary_out, _ = order_models(summary_out, order=MODEL_DISPLAY_ORDER)
    df_out.to_csv(os.path.join(args.outdir, "episodes.csv"), index=False)
    summary_out.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary_out.to_string(index=False))
    
    # Plot
    fig = plot_pain_tail_results(summary_out)
    fig.savefig(os.path.join(args.outdir, "results.png"), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {os.path.join(args.outdir, 'results.png')}")
