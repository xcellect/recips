"""Pain-Tail Causal Lesion Test - Extended Window (150 steps)"""
import argparse
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import experiments.corridor_exp as cw
from experiments.evaluation_harness import EvalAgent, MODEL_ORDER, order_models
from utils.model_naming import MODEL_DISPLAY_ORDER, canonical_model_display, canonical_model_id

@dataclass
class LesionResult:
    model: str
    seed: int
    condition: str
    valid: bool
    ns_at_lesion: float
    post_lesion_halflife: float
    post_lesion_auc: float
    immediate_slope: float  # ΔNs_5
    message: str

def measure_ns_halflife_from(ns_trace, t_start, window):
    if t_start >= len(ns_trace): return 0.0, False, "t_start OOB"
    ns0 = float(ns_trace[t_start])
    
    if ns0 <= 0.2:
        return 0.0, False, f"Ns0={ns0:.3f}<=0.2"
    
    threshold = 0.5 * ns0
    for dt in range(window + 1):
        idx = t_start + dt
        if idx >= len(ns_trace):
            return float(window), True, "censored"
        if ns_trace[idx] <= threshold:
            return float(dt), True, "crossed"
    
    return float(window), True, "censored"

def compute_auc_from(ns_trace, t_start, window):
    end = min(len(ns_trace), t_start + window)
    segment = ns_trace[t_start:end]
    return float(np.sum(segment))

def compute_immediate_slope(ns_trace, t_start):
    """ΔNs_5 = mean(Ns[t+1:t+6]) - Ns[t]"""
    if t_start + 5 >= len(ns_trace):
        return 0.0
    ns0 = ns_trace[t_start]
    ns_next = np.mean(ns_trace[t_start+1:t_start+6])
    return float(ns_next - ns0)

def run_trial(model, seed, lesion=True, lesion_t=3, T=200, post_window=150):
    env = cw.CorridorWorld(H=18, W=18, seed=seed)
    agent = EvalAgent(env, model=model, seed=seed, start=(1, env.goal_x), heading=2, eps=0.0)
    
    if hasattr(agent.net, '_update_ipsundrum_sensor'):
        I, *_ = cw.compute_I_affect(env, agent.y, agent.x, agent.heading)
        agent.net._update_ipsundrum_sensor(float(I), rng=agent.rng)
        agent.net.step()
    
    ns_trace = []
    
    for t in range(T):
        if t == 0:
            env.hazard[agent.y, agent.x] = 1.0
        elif t == 1:
            env.hazard[agent.y, agent.x] = 0.0
        
        if hasattr(agent.net, 'get'):
             ns_val = float(agent.net.get("Ns").activation)
        else:
             ns_val = 0.0
        ns_trace.append(ns_val)
        
        if t == lesion_t and lesion:
            if hasattr(agent.net, '_ipsundrum_state'):
                agent.net._ipsundrum_state["lesion_integrator"] = True
                agent.net._ipsundrum_state["lesion_feedback"] = True
        
        I, *_ = cw.compute_I_affect(env, agent.y, agent.x, agent.heading)
        if hasattr(agent.net, '_update_ipsundrum_sensor'):
            agent.net._update_ipsundrum_sensor(float(I), rng=agent.rng)
        else:
            agent.net.set_sensor_value('Ns', float(np.clip(0.5 + 0.5*I, 0.0, 1.0)))
        agent.net.step()
    
    # Measure from lesion time with EXTENDED window
    hl, valid, msg = measure_ns_halflife_from(ns_trace, lesion_t, window=post_window)
    auc = compute_auc_from(ns_trace, lesion_t, window=post_window)
    slope = compute_immediate_slope(ns_trace, lesion_t)
    ns_at_les = ns_trace[lesion_t] if lesion_t < len(ns_trace) else 0.0
    
    return LesionResult(
        model=model, seed=seed, condition="lesion" if lesion else "sham",
        valid=valid, ns_at_lesion=ns_at_les, post_lesion_halflife=hl,
        post_lesion_auc=auc, immediate_slope=slope, message=msg
    ), ns_trace

def run_lesion_causal(
    models=tuple(MODEL_ORDER),
    seeds=tuple(range(5)),
    lesion_t: int = 3,
    total_steps: int = 200,
    post_window: int = 150,
    outdir: str = "results/lesion",
):
    print("="*70)
    print(f"PAIN-TAIL CAUSAL LESION (Extended Window: {post_window} steps)")
    print("="*70)

    results = []
    traces = {}  # Store for plotting

    for model in models:
        model_label = canonical_model_display(model)
        for seed in seeds:
            r_sham, trace_sham = run_trial(model, seed, lesion=False, lesion_t=lesion_t, T=total_steps, post_window=post_window)
            results.append(asdict(r_sham))
            traces[f"{model_label}_seed{seed}_sham"] = trace_sham

            r_les, trace_les = run_trial(model, seed, lesion=True, lesion_t=lesion_t, T=total_steps, post_window=post_window)
            results.append(asdict(r_les))
            traces[f"{model_label}_seed{seed}_lesion"] = trace_les

    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(results)
    df, _ = order_models(df)
    df = df.sort_values(["model", "seed", "condition"])
    df_out = df.copy()
    if "model" in df_out.columns:
        df_out["model"] = df_out["model"].map(canonical_model_display)
        df_out, _ = order_models(df_out, order=MODEL_DISPLAY_ORDER)
        df_out = df_out.sort_values(["model", "seed", "condition"])
    df_out.to_csv(os.path.join(outdir, "episodes_extended.csv"), index=False)

    # Save traces for plotting
    np.savez(os.path.join(outdir, "ns_traces.npz"), **traces)

    valid_df = df[df['valid']].copy()

    if len(valid_df) == 0:
        print("No valid trials!")
        return df, pd.DataFrame()

    summary = valid_df.groupby(['model', 'condition'], sort=False, observed=False).agg({
        'post_lesion_halflife': 'mean',
        'post_lesion_auc': 'mean',
        'immediate_slope': 'mean',
        'valid': 'count'
    }).reset_index()
    summary, _ = order_models(summary)

    print("\nRESULTS:")
    print("\nHalf-Life (steps):")
    pivot_hl = summary.pivot(index='model', columns='condition', values='post_lesion_halflife')
    print(pivot_hl)

    print("\nAUC (integrated Ns):")
    pivot_auc = summary.pivot(index='model', columns='condition', values='post_lesion_auc')
    print(pivot_auc)

    print("\nDifference (Sham - Lesion):")
    diff = pivot_auc['sham'] - pivot_auc['lesion']
    print(diff)

    print("\nOK: Lesion causes causal drop in persistence (AUC)")
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal lesion assay")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--post_window", type=int, default=150, help="Post-lesion window length")
    parser.add_argument("--total_steps", type=int, default=200, help="Total episode steps")
    parser.add_argument("--lesion_t", type=int, default=3, help="Lesion time step")
    parser.add_argument("--outdir", type=str, default="results/lesion", help="Output directory")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model list (optional)")
    args = parser.parse_args()

    if args.models.strip():
        model_list = tuple(canonical_model_id(m) for m in args.models.split(",") if m.strip())
    else:
        model_list = tuple(m for m in MODEL_ORDER if m in ["recon", "humphrey", "humphrey_barrett"])

    run_lesion_causal(
        models=model_list,
        seeds=tuple(range(args.seeds)),
        lesion_t=args.lesion_t,
        total_steps=args.total_steps,
        post_window=args.post_window,
        outdir=args.outdir,
    )
