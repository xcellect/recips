"""
Run full quantitative sweeps and generate plots for paper.
"""
import argparse
import os
import experiments.evaluation_harness as eh
import pandas as pd
import matplotlib.pyplot as plt
from utils.model_naming import MODEL_DISPLAY_ORDER


def run_goal_directed_sweeps(
    seeds=tuple(range(10)),
    horizons=(1, 2, 3, 5, 10, 20),
    outdir: str = "results/goal-directed",
):
    # Run both sweeps with reasonable parameters
    models = list(eh.MODEL_ORDER)
    print("="*60)
    print("RUNNING GRIDWORLD SWEEP")
    print("="*60)
    df_g, summ_g = eh.sweep_gridworld(
        horizons=horizons,
        seeds=tuple(seeds),
        models=tuple(models),
        T=250,
        eps=0.10
    )

    print("\n" + "="*60)
    print("RUNNING CORRIDOR SWEEP")
    print("="*60)
    df_c, summ_c = eh.sweep_corridor(
        horizons=horizons,
        seeds=tuple(seeds),
        models=tuple(models),
        T=80, # shorter because corridor is linear
        eps=0.08
    )

    # Save CSVs
    os.makedirs(outdir, exist_ok=True)
    df_g_out = eh.canonicalize_model_column_display(df_g)
    summ_g_out = eh.canonicalize_model_column_display(summ_g)
    df_c_out = eh.canonicalize_model_column_display(df_c)
    summ_c_out = eh.canonicalize_model_column_display(summ_c)
    df_g_out, _ = eh.order_models(df_g_out, order=MODEL_DISPLAY_ORDER)
    summ_g_out, _ = eh.order_models(summ_g_out, order=MODEL_DISPLAY_ORDER)
    df_c_out, _ = eh.order_models(df_c_out, order=MODEL_DISPLAY_ORDER)
    summ_c_out, _ = eh.order_models(summ_c_out, order=MODEL_DISPLAY_ORDER)
    df_g_out.to_csv(os.path.join(outdir, "gridworld_episodes.csv"), index=False)
    summ_g_out.to_csv(os.path.join(outdir, "gridworld_summary.csv"), index=False)
    df_c_out.to_csv(os.path.join(outdir, "corridor_episodes.csv"), index=False)
    summ_c_out.to_csv(os.path.join(outdir, "corridor_summary.csv"), index=False)

    print("\n" + "="*60)
    print("GRIDWORLD SUMMARY")
    print("="*60)
    print(summ_g_out.sort_values(["model", "horizon"]).to_string(index=False))

    print("\n" + "="*60)
    print("CORRIDOR SUMMARY")
    print("="*60)
    print(summ_c_out.sort_values(["model", "horizon"]).to_string(index=False))

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    # Gridworld plots
    fig = eh.plot_metric(
        summ_g, "gridworld", "mean_hazards", "Gridworld: Hazard Contacts vs Horizon", show=False
    )
    fig.savefig(os.path.join(outdir, "gridworld_hazards.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = eh.plot_metric(
        summ_g, "gridworld", "success_rate", "Gridworld: Success Rate vs Horizon", show=False
    )
    fig.savefig(os.path.join(outdir, "gridworld_success.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = eh.plot_metric(
        summ_g, "gridworld", "mean_time", "Gridworld: Time-to-Goal vs Horizon", show=False
    )
    fig.savefig(os.path.join(outdir, "gridworld_time.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Corridor plots
    fig = eh.plot_metric(
        summ_c, "corridor", "mean_hazards", "Corridor: Hazard Contacts vs Horizon", show=False
    )
    fig.savefig(os.path.join(outdir, "corridor_hazards.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = eh.plot_metric(
        summ_c, "corridor", "success_rate", "Corridor: Success Rate vs Horizon", show=False
    )
    fig.savefig(os.path.join(outdir, "corridor_success.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = eh.plot_metric(
        summ_c, "corridor", "mean_time", "Corridor: Time-to-Goal vs Horizon", show=False
    )
    fig.savefig(os.path.join(outdir, "corridor_time.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("\nAll plots saved!")
    print("\nCSVs saved:")
    print("  - gridworld_episodes.csv, gridworld_summary.csv")
    print("  - corridor_episodes.csv, corridor_summary.csv")

    return df_g, summ_g, df_c, summ_c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Goal-directed sweeps")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--outdir", type=str, default="results/goal-directed", help="Output directory")
    parser.add_argument("--horizons", type=str, default="1,2,3,5,10,20", help="Comma-separated horizons")
    args = parser.parse_args()

    horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())
    run_goal_directed_sweeps(seeds=tuple(range(args.seeds)), horizons=horizons, outdir=args.outdir)
