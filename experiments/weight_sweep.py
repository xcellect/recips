"""Optional coarse weight/novelty sweep for familiarity control."""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments.familiarity_control import (
    QualiaphiliaCorridorWorld,
    run_choice_episode,
    scripted_familiarize,
    N_FAM,
    FAM_STEPS_PER_EP,
)


def run_weight_sweep(
    seeds: Tuple[int, ...],
    post_repeats: int,
    novelty_scales: Tuple[float, ...],
    weight_scales: Tuple[float, ...],
    outdir: str,
):
    rows: List[Dict[str, float]] = []
    for novelty_scale in novelty_scales:
        for weight_scale in weight_scales:
            score_kwargs = {
                "w_valence": 2.0 * weight_scale,
                "w_arousal": -1.2 * weight_scale,
                "w_ns": -0.8 * weight_scale,
                "w_bb_err": -0.4 * weight_scale,
            }
            for seed in seeds:
                scenic_side = "left" if seed % 2 == 0 else "right"
                per_seed = {}
                for familiarize_side in ("scenic", "dull"):
                    env = QualiaphiliaCorridorWorld(H=18, W=18, seed=seed, scenic_side=scenic_side)
                    env.use_beauty_term = False
                    memory = {}

                    start_y = env.barrier_start
                    for i in range(int(N_FAM)):
                        side = familiarize_side
                        start_x = env.scenic_x if side == "scenic" else env.dull_x
                        scripted_familiarize(
                            env,
                            memory,
                            start_pose=(start_y, start_x, 2),
                            steps=FAM_STEPS_PER_EP,
                        )

                    scenic_choices = []
                    for morning_idx in range(1, int(post_repeats) + 1):
                        res_post = run_choice_episode(
                            env,
                            "humphrey_barrett",
                            seed,
                            scenic_side,
                            familiarize_side,
                            memory,
                            phase="post",
                            morning_idx=morning_idx,
                            update_memory=True,
                            score_kwargs=score_kwargs,
                            novelty_scale=novelty_scale,
                        )
                        scenic_choices.append(float(res_post.scenic_choice))

                    per_seed[familiarize_side] = float(np.mean(scenic_choices)) if scenic_choices else float("nan")
                    rows.append({
                        "novelty_scale": novelty_scale,
                        "weight_scale": weight_scale,
                        "seed": seed,
                        "familiarize_side": familiarize_side,
                        "scenic_choice_rate": per_seed[familiarize_side],
                    })

                if "scenic" in per_seed and "dull" in per_seed:
                    rows.append({
                        "novelty_scale": novelty_scale,
                        "weight_scale": weight_scale,
                        "seed": seed,
                        "familiarize_side": "delta_dull_minus_scenic",
                        "scenic_choice_rate": per_seed["dull"] - per_seed["scenic"],
                    })

    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "weight_sweep_rows.csv"), index=False)

    if not df.empty:
        summary = (
            df[df["familiarize_side"] == "delta_dull_minus_scenic"]
            .groupby(["novelty_scale", "weight_scale"], sort=False, observed=False)["scenic_choice_rate"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(os.path.join(outdir, "weight_sweep_summary.csv"), index=False)
    print("OK: weight sweep saved to", outdir)
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional weight sweep")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--post_repeats", type=int, default=1, help="Post repeats per seed")
    parser.add_argument("--novelty_scales", type=str, default="0.25,0.5,0.75", help="Comma-separated novelty scales")
    parser.add_argument("--weight_scales", type=str, default="0.5,1.0,1.5", help="Comma-separated weight scales")
    parser.add_argument("--outdir", type=str, default="results/ablations/weight-sweep", help="Output directory")
    args = parser.parse_args()

    novelty_scales = tuple(float(x) for x in args.novelty_scales.split(",") if x.strip())
    weight_scales = tuple(float(x) for x in args.weight_scales.split(",") if x.strip())
    run_weight_sweep(
        seeds=tuple(range(args.seeds)),
        post_repeats=args.post_repeats,
        novelty_scales=novelty_scales,
        weight_scales=weight_scales,
        outdir=args.outdir,
    )
