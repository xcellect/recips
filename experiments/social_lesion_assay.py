from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from experiments.social_corridor import LOAD_PRESETS, run_corridor_experiment
from experiments.social_foodshare import run_foodshare_experiment


def run_social_lesion_assay(profile: str = "quick", outdir: str = "results/social-lesions"):
    os.makedirs(outdir, exist_ok=True)
    all_summaries: List[pd.DataFrame] = []
    for env_name, runner in (("foodshare", run_foodshare_experiment),):
        for lesion_mode in ("sham", "coupling_off", "shuffle_partner"):
            _, summary = runner(
                conditions=("social_affective_direct", "social_full_direct"),
                lambda_affective=1.0,
                profile=profile,
                outdir=os.path.join(outdir, f"{env_name}-{lesion_mode}"),
                lesion_mode=lesion_mode,
            )
            summary = summary.copy()
            summary["task"] = env_name
            summary["metabolic_load"] = "low"
            all_summaries.append(summary)
    for lesion_mode in ("sham", "coupling_off", "shuffle_partner"):
        _, summary = run_corridor_experiment(
            conditions=("social_affective_direct", "social_full_direct"),
            lambda_affective=1.0,
            profile=profile,
            outdir=os.path.join(outdir, f"corridor-{lesion_mode}"),
            lesion_mode=lesion_mode,
            metabolic_load="low",
        )
        summary = summary.copy()
        summary["task"] = "corridor"
        all_summaries.append(summary)
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    sweep_rows: List[Dict[str, float | str]] = []
    for load in LOAD_PRESETS:
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            _, summary = run_corridor_experiment(
                conditions=("social_affective_direct",),
                lambda_affective=lam,
                profile=profile,
                outdir=os.path.join(outdir, f"sweep-{load}-{lam:.2f}"),
                lesion_mode="none",
                metabolic_load=load,
            )
            sweep_rows.append(
                {
                    "metabolic_load": load,
                    "lambda_affective": lam,
                    "mutual_viability": float(summary["mutual_viability"].mean()),
                    "help_rate_when_partner_distressed": float(summary["help_rate_when_partner_distressed"].mean()),
                    "episode_joint_longevity": float(summary["episode_joint_longevity"].mean()),
                    "self_cost_of_help": float(summary["self_cost_of_help"].mean()),
                }
            )
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(outdir, "coupling_sweep.csv"), index=False)
    return combined, sweep


if __name__ == "__main__":
    run_social_lesion_assay(profile="quick")
