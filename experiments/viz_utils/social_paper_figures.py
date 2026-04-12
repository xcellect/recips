from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.social_exact_solver import solve_foodshare_state
from utils.plot_style import apply_times_style


def _apply_publication_style() -> None:
    apply_times_style()
    plt.rcParams.update(
        {
            # Keep all figure text comfortably above common publication minimums.
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.titleweight": "semibold",
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "figure.dpi": 180,
            "axes.linewidth": 0.9,
        }
    )


def _friendly_condition_name(condition: str) -> str:
    mapping = {
        "social_none": "none",
        "social_cognitive_direct": "cognitive",
        "social_affective_direct": "affective",
        "social_full_direct": "full",
    }
    return mapping.get(condition, condition)


def _foodshare_switch_curve() -> tuple[np.ndarray, np.ndarray, float]:
    lam_values = np.linspace(0.0, 1.0, 21)
    deltas = []
    for lam in lam_values:
        _, scores = solve_foodshare_state(
            0.55,
            0.20,
            condition="social_affective_direct",
            lambda_affective=float(lam),
        )
        deltas.append(float(scores["PASS"] - scores["EAT"]))
    deltas_arr = np.asarray(deltas, dtype=float)

    sign_change = np.where(np.signbit(deltas_arr[:-1]) != np.signbit(deltas_arr[1:]))[0]
    if sign_change.size:
        idx = int(sign_change[0])
        x0, x1 = lam_values[idx], lam_values[idx + 1]
        y0, y1 = deltas_arr[idx], deltas_arr[idx + 1]
        switch_point = float(x0 - y0 * (x1 - x0) / max(1e-12, (y1 - y0)))
    else:
        switch_point = float(lam_values[int(np.argmin(np.abs(deltas_arr)))])
    return lam_values, deltas_arr, switch_point


def create_social_summary_figure(
    *,
    headline_csv: Path,
    lesion_csv: Path,
    sweep_csv: Path,
    out_pdf: Path,
    out_png: Path | None = None,
) -> None:
    _apply_publication_style()
    headline = pd.read_csv(headline_csv)
    lesion = pd.read_csv(lesion_csv)
    sweep = pd.read_csv(sweep_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # Panel A: exact one-step switch
    lam_values, delta_score, switch_point = _foodshare_switch_curve()
    ax_a.plot(lam_values, delta_score, color="#1f77b4", linewidth=2.2)
    ax_a.axhline(0.0, color="#1f77b4", linewidth=1.1, alpha=0.85)
    ax_a.axvline(switch_point, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.95)
    ax_a.annotate(
        rf"$\lambda^* = {switch_point:.2f}$",
        xy=(switch_point, 0.02),
        xytext=(0.74, 0.86),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 1.0},
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#1f77b4", "alpha": 0.9},
    )
    ax_a.text(0.08, 0.84, "EAT optimal", transform=ax_a.transAxes, fontsize=10)
    ax_a.text(0.74, 0.14, "PASS optimal", transform=ax_a.transAxes, fontsize=10)
    ax_a.set_title("A. FoodShareToy exact switch point")
    ax_a.set_xlabel(r"$\lambda_{\mathrm{affective}}$")
    ax_a.set_ylabel(r"$\Delta$ score (PASS - EAT)")
    ax_a.set_ylim(min(-0.42, float(delta_score.min() - 0.03)), max(0.18, float(delta_score.max() + 0.03)))

    # Panel B: corridor outcomes by condition
    corridor = headline[
        (headline["env_name"] == "SocialCorridorWorld")
        & (headline["lesion_mode"] == "none")
    ].copy()
    condition_order = ["social_none", "social_cognitive_direct", "social_affective_direct", "social_full_direct"]
    corridor["condition"] = pd.Categorical(corridor["condition"], condition_order, ordered=True)
    corridor = corridor.sort_values("condition")
    labels = [_friendly_condition_name(c) for c in corridor["condition"]]
    x = np.arange(len(corridor))
    width = 0.24
    ax_b.bar(x - width, corridor["help_rate_when_partner_distressed_mean"], width=width, label="help", color="#1f77b4")
    ax_b.bar(x, corridor["partner_recovery_rate_mean"], width=width, label="recovery", color="#ff7f0e")
    ax_b.bar(x + width, corridor["mutual_viability_mean"], width=width, label="viability", color="#2ca02c")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylim(0.0, 1.03)
    ax_b.set_ylabel("mean value")
    ax_b.set_title("B. Corridor outcomes by condition")
    # Stack entries vertically (one after another), inside the axes.
    ax_b.legend(loc="upper left", ncol=1, frameon=False)

    # Panel C: lesions (using both tasks)
    lesion_use = lesion[
        lesion["condition"].isin(["social_affective_direct"])
        & lesion["lesion_mode"].isin(["sham", "coupling_off", "shuffle_partner"])
        & lesion["env_name"].isin(["FoodShareToy", "SocialCorridorWorld"])
    ].copy()
    lesion_order = ["sham", "coupling_off", "shuffle_partner"]
    lesion_use["lesion_mode"] = pd.Categorical(lesion_use["lesion_mode"], lesion_order, ordered=True)
    lesion_use = lesion_use.sort_values(["lesion_mode", "env_name"])
    envs = ["FoodShareToy", "SocialCorridorWorld"]
    env_colors = {"FoodShareToy": "#1f77b4", "SocialCorridorWorld": "#ff7f0e"}
    x = np.arange(len(lesion_order))
    width = 0.36
    for i, env in enumerate(envs):
        vals = []
        for lm in lesion_order:
            sub = lesion_use[(lesion_use["env_name"] == env) & (lesion_use["lesion_mode"] == lm)]
            vals.append(float(sub["help_rate_when_partner_distressed_mean"].iloc[0]) if len(sub) else np.nan)
        offset = (i - 0.5) * width
        label = "foodshare" if env == "FoodShareToy" else "corridor"
        ax_c.bar(x + offset, vals, width=width, color=env_colors[env], label=label)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["sham", "coupling off", "shuffle"])
    ax_c.set_ylim(0.0, 1.05)
    ax_c.set_ylabel("help rate")
    ax_c.set_title("C. Causal lesions abolish helping")
    ax_c.legend(loc="upper right", frameon=False)

    # Panel D: coupling sweep by load
    load_order = ["low", "medium", "high"]
    load_colors = {"low": "#1f77b4", "medium": "#ff7f0e", "high": "#2ca02c"}
    for load in load_order:
        sub = sweep[sweep["metabolic_load"] == load].sort_values("lambda_affective")
        xvals = sub["lambda_affective"].to_numpy(dtype=float)
        yvals = sub["mutual_viability"].to_numpy(dtype=float)
        ax_d.plot(xvals, yvals, marker="o", linewidth=1.9, color=load_colors[load], label=load)
        helping = sub["help_rate_when_partner_distressed"].to_numpy(dtype=float) > 0.0
        if np.any(helping):
            ax_d.scatter(xvals[helping], yvals[helping], s=60, color=load_colors[load], zorder=4)
    ax_d.text(0.02, 0.95, "filled markers: helping present", transform=ax_d.transAxes, va="top", fontsize=10)
    ax_d.set_xlabel(r"$\lambda_{\mathrm{affective}}$")
    ax_d.set_ylabel("mutual viability")
    ax_d.set_title("D. Load-dependent coupling sweep")
    ax_d.legend(loc="center right", frameon=False)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.18, linestyle="-")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build social paper summary Figure 1.")
    parser.add_argument(
        "--social-paper-dir",
        default="results/social-paper-paper",
        help="Directory containing headline_summary.csv, lesion_summary.csv, coupling_sweep.csv.",
    )
    parser.add_argument(
        "--out-pdf",
        default="../alife_social_paper/figures/fig_summary.pdf",
        help="Output PDF path for the summary figure.",
    )
    parser.add_argument(
        "--out-png",
        default="../alife_social_paper/figures/fig_summary.png",
        help="Optional PNG output path (set empty string to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    social_dir = Path(args.social_paper_dir)
    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png) if args.out_png else None
    create_social_summary_figure(
        headline_csv=social_dir / "headline_summary.csv",
        lesion_csv=social_dir / "lesion_summary.csv",
        sweep_csv=social_dir / "coupling_sweep.csv",
        out_pdf=out_pdf,
        out_png=out_png,
    )
    print(f"Saved {out_pdf}")
    if out_png is not None:
        print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
