from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd

from analysis.social_summary import cliffs_delta, summarize_by_condition


def compute_social_claims(summary_df: pd.DataFrame) -> Dict[str, float | str]:
    claims: Dict[str, float | str] = {}
    affective = summary_df[summary_df["condition"] == "social_affective_direct"]
    cognitive = summary_df[summary_df["condition"] == "social_cognitive_direct"]
    none = summary_df[summary_df["condition"] == "social_none"]
    full = summary_df[summary_df["condition"] == "social_full_direct"]
    if not affective.empty and not cognitive.empty:
        claims["help_rate_affective_minus_cognitive"] = float(
            affective["help_rate_when_partner_distressed"].mean() - cognitive["help_rate_when_partner_distressed"].mean()
        )
        claims["help_rate_affective_vs_cognitive_cliffs_delta"] = cliffs_delta(
            affective["help_rate_when_partner_distressed"],
            cognitive["help_rate_when_partner_distressed"],
        )
    if not affective.empty and not none.empty:
        claims["mutual_viability_affective_minus_none"] = float(
            affective["mutual_viability"].mean() - none["mutual_viability"].mean()
        )
        if "partner_recovery_rate" in summary_df.columns:
            claims["partner_recovery_affective_minus_none"] = float(
                affective["partner_recovery_rate"].mean() - none["partner_recovery_rate"].mean()
            )
    lesion = summary_df[summary_df["lesion_mode"] == "coupling_off"] if "lesion_mode" in summary_df.columns else pd.DataFrame()
    sham = summary_df[summary_df["lesion_mode"] == "sham"] if "lesion_mode" in summary_df.columns else pd.DataFrame()
    if not lesion.empty and not sham.empty:
        claims["lesion_minus_sham_help_rate"] = float(
            lesion["help_rate_when_partner_distressed"].mean() - sham["help_rate_when_partner_distressed"].mean()
        )
    if not full.empty and not affective.empty:
        claims["full_minus_affective_rescue_latency"] = float(
            full["rescue_latency"].mean() - affective["rescue_latency"].mean()
        )
    return claims


def compute_task_claims(summary_df: pd.DataFrame) -> Dict[str, Dict[str, float | str]]:
    key = "env_name" if "env_name" in summary_df.columns else "task"
    out: Dict[str, Dict[str, float | str]] = {}
    for name, grp in summary_df.groupby(key):
        out[str(name)] = compute_social_claims(grp)
    return out


def build_social_artifacts(
    *,
    foodshare_summary_csv: str = "results/social-foodshare/summary.csv",
    corridor_summary_csv: str = "results/social-corridor/summary.csv",
    lesion_summary_csv: str = "results/social-lesions/summary.csv",
    coupling_sweep_csv: str = "results/social-lesions/coupling_sweep.csv",
    outdir: str = "results/social-paper",
):
    os.makedirs(outdir, exist_ok=True)
    foodshare = pd.read_csv(foodshare_summary_csv)
    corridor = pd.read_csv(corridor_summary_csv)
    lesion = pd.read_csv(lesion_summary_csv)
    sweep = pd.read_csv(coupling_sweep_csv)

    headline = pd.concat([foodshare, corridor], ignore_index=True)
    headline_summary = summarize_by_condition(
        headline,
        [
            "help_rate_when_partner_distressed",
            "partner_recovery_rate",
            "mutual_viability",
            "rescue_latency",
            "self_cost_of_help",
            "episode_joint_longevity",
            "partner_final_energy",
            "partner_peak_energy",
            "self_final_energy",
            "joint_homeostatic_margin",
        ],
    )
    headline_summary.to_csv(os.path.join(outdir, "headline_summary.csv"), index=False)
    lesion_summary = summarize_by_condition(
        lesion,
        [
            "help_rate_when_partner_distressed",
            "partner_recovery_rate",
            "mutual_viability",
            "rescue_latency",
            "partner_final_energy",
            "joint_homeostatic_margin",
        ],
    )
    lesion_summary.to_csv(os.path.join(outdir, "lesion_summary.csv"), index=False)
    sweep.to_csv(os.path.join(outdir, "coupling_sweep.csv"), index=False)

    claims = {
        "headline": compute_social_claims(headline),
        "headline_by_task": compute_task_claims(headline),
        "lesions": compute_social_claims(lesion),
        "lesions_by_task": compute_task_claims(lesion),
    }
    with open(os.path.join(outdir, "claims.json"), "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2, sort_keys=True)
    return headline_summary, lesion_summary, sweep, claims


if __name__ == "__main__":
    build_social_artifacts()
