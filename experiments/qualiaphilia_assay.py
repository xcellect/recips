"""
Qualiaphilia Assay: Preference for sensory-rich experiences.

Tests Humphrey's prediction: phenomenally conscious agents value rich sensory
experiences for their own sake, choosing "scenic" routes over "dull" routes
even when both are equally safe and lead to the same goal.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import copy

from utils.plot_style import apply_times_style

apply_times_style()

import experiments.corridor_exp as cw
from experiments.evaluation_harness import EvalAgent, MODEL_ORDER, order_models
from utils.model_naming import MODEL_DISPLAY_ORDER, canonical_model_display
from core.recon_core import ScriptState


class QualiaphiliaCorridorWorld(cw.CorridorWorld):
    """
    Modified corridor with two safe paths:
    - Left lane (x=6): SCENIC - rich sensory features (varying beauty/odor)
    - Right lane (x=12): DULL - uniform gray (constant low beauty/odor)
    
    Both paths are equally safe and reach the same goal.
    """
    def __init__(self, H=18, W=18, seed=0, scenic_side='left'):
        super().__init__(H, W, seed)
        
        # Store which side is scenic
        self.scenic_side = scenic_side
        self.scenic_x = 6 if scenic_side == 'left' else 12
        self.dull_x = 12 if scenic_side == 'left' else 6
        
        # Clear existing beauty/hazard, rebuild
        self.hazard = np.zeros((H, W), dtype=float)
        self.beauty = np.zeros((H, W), dtype=float)
        
        # Corridor geometry (same as parent)
        self.blocked = np.ones((H, W), dtype=bool)
        x0, x1 = 6, 13
        self.blocked[:, x0:x1] = False
        
        # Wall barrier spanning most of corridor (forces lane choice)
        # Leave only left (x=6) and right (x=12) lanes passable
        barrier_start = 3  # start barrier early
        barrier_end = H - 4  # end before goal region
        self.barrier_start = barrier_start
        self.barrier_end = barrier_end
        
        for y in range(barrier_start, barrier_end):
            for x in range(x0+1, x1-1):  # center columns
                self.blocked[y, x] = True

        # No hazard anywhere in Qualiaphilia (both routes equally safe)
        self.hazard = np.zeros((H, W), dtype=float)
        
        # Goal at bottom center (reachable from both lanes)
        self.goal_y, self.goal_x = H - 2, (x0 + x1) // 2
        split_y = barrier_start - 1
        split_x = self.goal_x
        split_heading = 2  # down
        self.split_pose = (split_y, split_x, split_heading)
        self.bump_penalty = 0.30
        self.stay_penalty = 0.60
        self.forward_prior = 0.00
        self.use_beauty_term = False
        
        # SCENIC PATH: rich sensory features
        rng = np.random.default_rng(seed + 100)  # different seed for scenic features
        for y in range(H):
            if self.is_free(y, self.scenic_x):
                # Varying beauty with "flowers" - some cells very beautiful
                if rng.random() < 0.3:  # 30% of cells are "flowers"
                    self.beauty[y, self.scenic_x] = rng.uniform(0.6, 1.0)
                else:
                    self.beauty[y, self.scenic_x] = rng.uniform(0.2, 0.5)
                
                # Add some visual variety to adjacent cells (visible in cone)
                adj = self.scenic_x + (1 if scenic_side == 'left' else -1)
                if self.is_free(y, adj):
                    self.beauty[y, adj] = rng.uniform(0.1, 0.4)
        
        # DULL PATH: uniform low beauty
        for y in range(H):
            if self.is_free(y, self.dull_x):
                self.beauty[y, self.dull_x] = 0.15  # constant dull gray
                adj = self.dull_x + (-1 if scenic_side == 'left' else 1)
                if self.is_free(y, adj):
                    self.beauty[y, adj] = 0.15
        
        # Goal region beauty (both paths converge here)
        for y in range(self.goal_y-1, self.goal_y+2):
            for x in range(x0, x1):
                if self.is_free(y, x):
                    self.beauty[y, x] = 0.8  # goal is attractive
        
        # Recalculate smell fields
        K = cw.gaussian_kernel(radius=3, sigma=1.2)
        self.smell_h = cw.conv2_same(self.hazard, K)
        self.smell_b = cw.conv2_same(self.beauty, K)


@dataclass
class QualiaphiliaResult:
    model: str
    seed: int
    scenic_side: str  # 'left' or 'right'
    choice_entry: Optional[str]  # 'scenic', 'dull', or None
    decided: bool
    choice_commit: str  # debug: 'scenic', 'dull', or 'none'
    decided_commit: bool
    time_scenic: int  # timesteps spent in scenic lane
    time_dull: int  # timesteps spent in dull lane
    time_center: int  # timesteps in center/goal area
    scenic_time_share: float
    time_scenic_barrier: int
    time_dull_barrier: int
    scenic_time_share_barrier: float
    success: bool  # reached goal
    hazard_contacts: int
    mean_valence_scenic: float
    mean_valence_dull: float
    mean_arousal_scenic: float
    mean_arousal_dull: float


def route_entry_choice(env: QualiaphiliaCorridorWorld, y: int, x: int) -> Optional[str]:
    if y >= env.barrier_start:
        if x == env.scenic_x:
            return "scenic"
        if x == env.dull_x:
            return "dull"
    return None


def update_choice_entry(
    choice_entry: Optional[str],
    decided_entry: bool,
    env: QualiaphiliaCorridorWorld,
    y: int,
    x: int
) -> Tuple[Optional[str], bool]:
    if decided_entry:
        return choice_entry, decided_entry
    entry = route_entry_choice(env, y, x)
    if entry is None:
        return choice_entry, decided_entry
    return entry, True


def safe_nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def read_affect_state(agent: EvalAgent) -> Tuple[float, float]:
    st = agent.read_state()
    valence = st.get("node_Nv", np.nan)
    arousal = st.get("node_Na", np.nan)
    return float(valence), float(arousal)


def run_qualiaphilia_trial(
    model: str,
    seed: int,
    scenic_side: str = 'left',
    T: int = 80,
    w_progress: float = 0.20
) -> QualiaphiliaResult:
    """
    Run one qualiaphilia trial.
    
    Protocol:
      1. Create corridor with scenic vs dull paths
      2. Start agent at top center
      3. Run for T steps
      4. Measure which path agent takes and time spent in each
    """
    env = QualiaphiliaCorridorWorld(H=18, W=18, seed=seed, scenic_side=scenic_side)
    env.use_beauty_term = False
    
    # Start at split point, facing down
    split_y, split_x, heading = env.split_pose
    start = (split_y, split_x)
    
    agent = EvalAgent(
        env, model=model, seed=seed,
        start=start, heading=heading, eps=0.15
    )
    
    # Track path taken
    time_scenic = 0
    time_dull = 0
    time_center = 0
    time_scenic_barrier = 0
    time_dull_barrier = 0
    hazard_contacts = 0
    valence_scenic: List[float] = []
    valence_dull: List[float] = []
    arousal_scenic: List[float] = []
    arousal_dull: List[float] = []
    
    # Determine which path was chosen based on first divergence (debug)
    choice_commit = None  # will be set when agent commits to a lane
    decided_commit = False
    commitment_threshold = 3  # must spend 3 consecutive steps in a lane
    decision_deadline = T  # steps from split to make a commitment
    
    lane_history: List[str] = []  # 'scenic', 'dull', or 'center'

    # Route-entry choice: first lane entered at barrier start
    choice_entry: Optional[str] = None
    decided_entry = False
    
    for t in range(T):
        # Sensory input
        I_total, I_touch, *_ = cw.compute_I_affect(env, agent.y, agent.x, agent.heading)
        if I_touch > 0.5:
            hazard_contacts += 1
        
        # Physiology update
        if hasattr(agent.net, '_update_ipsundrum_sensor'):
            agent.net._update_ipsundrum_sensor(float(I_total), rng=agent.rng)
        else:
            agent.net.set_sensor_value('Ns', float(np.clip(0.5 + 0.5*I_total, 0.0, 1.0)))
        
        agent.net.step()

        valence, arousal = read_affect_state(agent)
        
        # Track position
        if agent.x == env.scenic_x:
            time_scenic += 1
            lane_history.append('scenic')
            valence_scenic.append(valence)
            arousal_scenic.append(arousal)
        elif agent.x == env.dull_x:
            time_dull += 1
            lane_history.append('dull')
            valence_dull.append(valence)
            arousal_dull.append(arousal)
        else:
            time_center += 1
            lane_history.append('center')

        if env.barrier_start <= agent.y < env.barrier_end:
            if agent.x == env.scenic_x:
                time_scenic_barrier += 1
            elif agent.x == env.dull_x:
                time_dull_barrier += 1
        
        # Determine choice (first sustained commitment)
        if choice_commit is None and t <= decision_deadline and len(lane_history) >= commitment_threshold:
            recent = lane_history[-commitment_threshold:]
            if all(l == 'scenic' for l in recent):
                choice_commit = "scenic"
                decided_commit = True
            elif all(l == 'dull' for l in recent):
                choice_commit = "dull"
                decided_commit = True

        # Determine route-entry choice (first step into barrier region)
        choice_entry, decided_entry = update_choice_entry(
            choice_entry, decided_entry, env, agent.y, agent.x
        )
        
        # Check goal
        if agent.y >= env.goal_y:
            break
        
        # Choose action
        explore_eps = 0.50 if (time_scenic + time_dull) == 0 else agent.eps
        if agent.rng.random() < explore_eps:
            action = agent.rng.choice(["forward", "turn_left", "turn_right"])
        else:
            action = cw.choose_action_feelings(
                agent,
                horizon=5,
                curiosity=False,
                w_progress=w_progress,
                w_epistemic=0.0,
                beauty_weight=0.0,
                use_beauty_term=False,
            )
        
        # Move
        agent.y, agent.x, agent.heading = env.step(agent.y, agent.x, action, agent.heading)
    
    # Final commitment determination if not committed
    if choice_commit is None:
        choice_commit = "none"
        decided_commit = False
    
    success = (agent.y >= env.goal_y)
    scenic_time_share = time_scenic / max(1, time_scenic + time_dull)
    scenic_time_share_barrier = time_scenic_barrier / max(
        1, time_scenic_barrier + time_dull_barrier
    )
    
    mean_valence_scenic = safe_nanmean(valence_scenic)
    mean_valence_dull = safe_nanmean(valence_dull)
    mean_arousal_scenic = safe_nanmean(arousal_scenic)
    mean_arousal_dull = safe_nanmean(arousal_dull)

    return QualiaphiliaResult(
        model=model,
        seed=seed,
        scenic_side=scenic_side,
        choice_entry=choice_entry,
        decided=bool(decided_entry),
        choice_commit=choice_commit,
        decided_commit=bool(decided_commit),
        time_scenic=time_scenic,
        time_dull=time_dull,
        time_center=time_center,
        scenic_time_share=scenic_time_share,
        time_scenic_barrier=time_scenic_barrier,
        time_dull_barrier=time_dull_barrier,
        scenic_time_share_barrier=scenic_time_share_barrier,
        success=success,
        hazard_contacts=hazard_contacts,
        mean_valence_scenic=mean_valence_scenic,
        mean_valence_dull=mean_valence_dull,
        mean_arousal_scenic=mean_arousal_scenic,
        mean_arousal_dull=mean_arousal_dull,
    )


def run_qualiaphilia_sweep(
    models: Tuple[str, ...] = tuple(MODEL_ORDER),
    seeds: Tuple[int, ...] = tuple(range(10)),
    counterbalance: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run qualiaphilia assay across models and seeds.
    
    If counterbalance=True, alternate scenic_side to control for spatial biases.
    """
    results: List[Dict[str, Any]] = []
    
    for model in models:
        for i, seed in enumerate(seeds):
            # Counterbalance scenic side
            scenic_side = 'left' if (i % 2 == 0) else 'right' if counterbalance else 'left'
            
            print(f"Running qualiaphilia: model={model}, seed={seed}, scenic={scenic_side}")
            res = run_qualiaphilia_trial(model, seed, scenic_side=scenic_side)
            results.append(asdict(res))
    
    df = pd.DataFrame(results)
    df, _ = order_models(df)
    df = df.sort_values(["model", "seed"])
    
    decided_df = df[df["decided"] == True].copy()
    scenic_rate_entry = decided_df.groupby("model", sort=False, observed=False)["choice_entry"].apply(
        lambda s: (s == "scenic").mean()
    ).rename("scenic_rate_entry")
    scenic_rate_commit = df[df["decided_commit"] == True].groupby("model", sort=False, observed=False)["choice_commit"].apply(
        lambda s: (s == "scenic").mean()
    ).rename("scenic_rate_commit")

    # Aggregate by model
    summary = df.groupby('model', as_index=False, sort=False, observed=False).agg(
        decided_rate=('decided', 'mean'),
        mean_time_scenic=('time_scenic', 'mean'),
        mean_time_dull=('time_dull', 'mean'),
        mean_time_center=('time_center', 'mean'),
        mean_scenic_time_share=('scenic_time_share', 'mean'),
        mean_time_scenic_barrier=('time_scenic_barrier', 'mean'),
        mean_time_dull_barrier=('time_dull_barrier', 'mean'),
        scenic_time_share_barrier_mean=('scenic_time_share_barrier', 'mean'),
        success_rate=('success', 'mean'),
        mean_hazards=('hazard_contacts', 'mean'),
        mean_valence_scenic=('mean_valence_scenic', 'mean'),
        mean_valence_dull=('mean_valence_dull', 'mean'),
        mean_arousal_scenic=('mean_arousal_scenic', 'mean'),
        mean_arousal_dull=('mean_arousal_dull', 'mean'),
        n=('seed', 'count')
    )
    summary = summary.merge(scenic_rate_entry, on="model", how="left")
    summary = summary.merge(scenic_rate_commit, on="model", how="left")
    summary, _ = order_models(summary)
    
    return df, summary


def plot_qualiaphilia_results(summary: pd.DataFrame):
    """Generate comparison plots for qualiaphilia assay."""
    summary = summary.copy()
    if "model" in summary.columns:
        summary["model"] = summary["model"].map(canonical_model_display)
    summary, _ = order_models(summary, order=MODEL_DISPLAY_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Build decided_text first so we can add it below suptitle
    decided_rates = []
    for _, row in summary.iterrows():
        decided_rates.append(f"{row['model']}: {row['decided_rate'] * 100:.0f}%")
    decided_text = "Decided rate: " + ", ".join(decided_rates)
    
    models = summary['model'].values
    x = np.arange(len(models))
    width = 0.6
    
    # Scenic preference (main result)
    ax = axes[0]
    bars = ax.bar(x, summary['scenic_rate_entry'] * 100, width, 
                   color='steelblue', edgecolor='black')
    ax.axhline(50, color='red', linestyle='--', linewidth=2, label='Chance (50%)')
    ax.set_ylabel('Scenic Choice Rate (%)')
    ax.set_title('Do agents prefer sensory-rich routes?', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylim(0, 110)  # Extra headroom for labels
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars - position inside bars for tall ones
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 50:
            # Inside bar for tall bars
            ax.text(bar.get_x() + bar.get_width()/2., height - 8,
                    f'{height:.1f}%', ha='center', va='top', fontsize=9, fontweight='bold', color='white')
        else:
            # Above bar for short bars
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Time allocation
    ax = axes[1]
    scenic_time = summary['mean_time_scenic_barrier'].values
    dull_time = summary['mean_time_dull_barrier'].values
    
    ax.bar(x - width/4, scenic_time, width/2, label='Scenic lane', color='coral', edgecolor='black')
    ax.bar(x + width/4, dull_time, width/2, label='Dull lane', color='gray', edgecolor='black')
    ax.set_ylabel('Mean Time (steps)')
    ax.set_title('Barrier-segment time')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Apply layout FIRST, then add titles on top
    plt.tight_layout()
    fig.subplots_adjust(top=0.82)
    
    # Now add suptitle and decided_text after layout is finalized
    fig.suptitle(
        "Baseline qualiaphilia (curiosity off; no familiarity manipulation)",
        fontsize=13,
        fontweight='bold',
        y=0.99
    )
    fig.text(0.5, 0.93, decided_text, ha='center', va='top', fontsize=9)
    
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualiaphilia corridor assay")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--outdir", type=str, default="results/qualiaphilia", help="Output directory")
    parser.add_argument("--no_counterbalance", action="store_true", help="Disable side counterbalancing")
    args = parser.parse_args()

    print("="*60)
    print("QUALIAPHILIA ASSAY")
    print("="*60)
    
    df, summary = run_qualiaphilia_sweep(
        models=tuple(MODEL_ORDER),
        seeds=tuple(range(args.seeds)),
        counterbalance=not args.no_counterbalance,
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
    fig = plot_qualiaphilia_results(summary_out)
    fig.savefig(os.path.join(args.outdir, "results.png"), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {os.path.join(args.outdir, 'results.png')}")
    
    # Statistical note
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("Chance = 50% (random choice between paths)")
    print("Evidence for qualiaphilia: preference significantly > 50%")
    for _, row in summary_out.iterrows():
        pref = row['scenic_rate_entry'] * 100
        diff = pref - 50
        decided = row['decided_rate'] * 100
        print(f"  {row['model']:20s}: {pref:5.1f}% ({diff:+4.1f}% vs chance), decided={decided:5.1f}%")
