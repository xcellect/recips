"""Create publication and debug figures for consciousness tests."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analysis.stats import bootstrap_mean_ci
from experiments.evaluation_harness import order_models
from utils.model_naming import MODEL_DISPLAY_ORDER, canonical_model_display
from utils.plot_style import apply_times_style

# Publication style
apply_times_style()
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 150

print("Creating publication figures...")

os.makedirs("results/publication-figures", exist_ok=True)
os.makedirs("results/publication-figures/debug", exist_ok=True)

COLORS = {
    "Recon": "#1f77b4",
    "Ipsundrum": "#ff7f0e",
    "Ipsundrum+affect": "#2ca02c",
}


def bootstrap_ci(values, n_boot=2000, ci=95, seed=0):
    result = bootstrap_mean_ci(values, n_boot=n_boot, ci=ci, seed=seed)
    return result.mean, (result.ci_low, result.ci_high)


def format_n(df):
    if "seed" not in df.columns:
        return "N = unknown"
    counts = df.groupby("model", observed=False)["seed"].nunique().values
    if len(counts) == 0:
        return "N = 0"
    uniq = sorted(set(int(c) for c in counts))
    if len(uniq) == 1:
        return f"N = {uniq[0]} seeds per model"
    return f"N = {uniq[0]}-{uniq[-1]} seeds per model"


def model_labels(models):
    return list(models)


def color_for_model(model: str) -> str:
    return COLORS.get(model, "#333333")


def ordered_models(df):
    df = df.copy()
    if "model" in df.columns:
        df["model"] = df["model"].map(canonical_model_display)
    df, ordered = order_models(df, order=MODEL_DISPLAY_ORDER)
    present = [m for m in ordered if m in df["model"].values]
    return df, present


def add_footer(fig, n_text):
    footer = f"{n_text}; error bars show 95% bootstrap CI"
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9)


def seed_values(df, model, value_col, extra_filters=None):
    sub = df[df["model"] == model]
    if extra_filters:
        for key, val in extra_filters.items():
            sub = sub[sub[key] == val]
    if "seed" in sub.columns:
        return sub.groupby("seed", sort=False, observed=False)[value_col].mean().values
    return sub[value_col].values


# ==================================================================
# FIGURE 1: PLAY (Entropy + Coverage + Disambiguators)
# ==================================================================

df_play = pd.read_csv("results/exploratory-play/final_viewpoints.csv")
df_play, models_play = ordered_models(df_play)


def pick_entropy_col(df):
    for col in ("neutral_sensory_entropy", "viewpoint_entropy", "sensory_entropy"):
        if col in df.columns and df[col].notna().any():
            return col
    return "sensory_entropy"


def pick_disambig_col(df):
    for col in ("cycle_score", "tortuosity", "dwell_p90"):
        if col in df.columns and df[col].notna().any():
            return col
    return "dwell_p90"


def metric_label(col):
    labels = {
        "neutral_sensory_entropy": "Neutral texture entropy (bits)",
        "viewpoint_entropy": "Viewpoint entropy (bits)",
        "sensory_entropy": "Sensory entropy (bits)",
        "unique_viewpoints": "Unique viewpoints",
        "scan_events": "Scan events",
        "cycle_score": "Cycle score",
        "tortuosity": "Tortuosity",
        "dwell_p90": "Dwell p90 (steps)",
    }
    return labels.get(col, col.replace("_", " ").title())


def entropy_panel_title(col):
    if col == "neutral_sensory_entropy":
        return "A. Neutral sensory diversity"
    if col == "viewpoint_entropy":
        return "A. Viewpoint diversity"
    return "A. Sensory diversity"


def disambig_panel_title(col):
    if col == "cycle_score":
        return "D. Limit-cycle control"
    if col == "tortuosity":
        return "D. Path loopiness"
    if col == "dwell_p90":
        return "D. Dwell-time control"
    return "D. Control metric"


def plot_ci_panel(ax, col, title, debug=False, seed=0):
    for idx, model in enumerate(models_play):
        data = df_play[df_play["model"] == model][col]
        mean, (low, high) = bootstrap_ci(data)
        yerr = np.array([[mean - low], [high - mean]])
        ax.errorbar(idx, mean, yerr=yerr, fmt="o", color=color_for_model(model), capsize=4, zorder=3)
        if debug:
            rng = np.random.default_rng(seed)
            x_jit = idx + rng.normal(0, 0.05, size=len(data))
            ax.scatter(x_jit, data, color=color_for_model(model), s=28, alpha=0.7, zorder=2)
    ax.set_ylabel(metric_label(col))
    ax.set_xticks(range(len(models_play)))
    ax.set_xticklabels(model_labels(models_play), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)


def plot_play_figure(out_path, debug=False):
    entropy_col = pick_entropy_col(df_play)
    disambig_col = pick_disambig_col(df_play)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    plot_ci_panel(ax_a, entropy_col, entropy_panel_title(entropy_col), debug=debug, seed=0)
    plot_ci_panel(ax_b, "unique_viewpoints", "B. Viewpoint coverage", debug=debug, seed=1)
    plot_ci_panel(ax_c, "scan_events", "C. Scan events", debug=debug, seed=2)
    plot_ci_panel(ax_d, disambig_col, disambig_panel_title(disambig_col), debug=debug, seed=3)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    add_footer(fig, format_n(df_play))
    fig.savefig(out_path, dpi=300 if not debug else 200, bbox_inches="tight")
    plt.close(fig)


plot_play_figure("results/publication-figures/fig1_play.png", debug=False)
plot_play_figure("results/publication-figures/debug/fig1_play_debug.png", debug=True)
print("OK: Figure 1 (play) saved")


# ==================================================================
# FIGURE 2: FAMILIARITY CONTROL (Post + Delta Novelty)
# ==================================================================

df_fam_all = pd.read_csv("results/familiarity/episodes_improved.csv")
if "valid" in df_fam_all.columns:
    df_fam_all = df_fam_all[df_fam_all["valid"] == True].copy()
df_fam_post = df_fam_all[(df_fam_all["phase"] == "post") & (df_fam_all["decided"] == True)].copy()
df_fam_base = df_fam_all[(df_fam_all["phase"] == "baseline") & (df_fam_all["decided"] == True)].copy()
df_fam_base = df_fam_base[df_fam_base["familiarize_side"] == "none"].copy()
df_fam_post, models_fam = ordered_models(df_fam_post)


def plot_familiarity_figure(out_path, debug=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        "Familiarity control (post-familiarization; curiosity on)",
        fontsize=12,
        fontweight="bold",
    )

    # Panel A: Post-familiarization scenic choice
    df_bar = df_fam_post[df_fam_post["familiarize_side"].isin(["scenic", "both"])].copy()
    df_bar, models_bar = ordered_models(df_bar)
    fam_groups = ["scenic", "both"]
    x = np.arange(len(models_bar))
    width = 0.36
    for j, fam_side in enumerate(fam_groups):
        means = []
        ci_low = []
        ci_high = []
        for model in models_bar:
            data = seed_values(df_bar, model, "scenic_choice", {"familiarize_side": fam_side}).astype(float)
            mean, (low, high) = bootstrap_ci(data)
            means.append(mean)
            ci_low.append(mean - low)
            ci_high.append(high - mean)
        offset = (j - 0.5) * width
        hatch = "//" if fam_side == "both" else None
        ax1.bar(
            x + offset,
            means,
            width,
            color=[color_for_model(m) for m in models_bar],
            alpha=0.75,
            edgecolor="black",
            hatch=hatch,
            label=f"Familiarize: {fam_side}"
        )
        ax1.errorbar(x + offset, means, yerr=[ci_low, ci_high], fmt="none",
                     ecolor="black", capsize=4, zorder=3)
        if debug:
            rng = np.random.default_rng(2 + j)
            for i, model in enumerate(models_bar):
                data = seed_values(df_bar, model, "scenic_choice", {"familiarize_side": fam_side}).astype(float)
                x_jit = i + offset + rng.normal(0, 0.05, size=len(data))
                ax1.scatter(x_jit, data, color="black", s=18, alpha=0.6, zorder=4)

    ax1.set_ylabel("Scenic choice rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels(models_bar), rotation=25, ha='right')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("A. Post-familiarization choice (controls)")
    ax1.legend(loc="lower right")

    # Panel B: Delta novelty scatter with choice markers
    df_scatter = df_fam_post[df_fam_post["familiarize_side"] == "scenic"].copy()
    df_scatter, models_scatter = ordered_models(df_scatter)
    for i, model in enumerate(models_scatter):
        data = df_scatter[df_scatter["model"] == model]
        scenic = data[data["scenic_choice"] == True]
        dull = data[data["scenic_choice"] == False]
        rng = np.random.default_rng(i + 3)
        y_s = i + rng.normal(0, 0.06, size=len(scenic))
        y_d = i + rng.normal(0, 0.06, size=len(dull))
        ax2.scatter(scenic["split_delta_novelty"], y_s, marker="*", s=120,
                    color=color_for_model(model), alpha=0.8, label=None)
        ax2.scatter(dull["split_delta_novelty"], y_d, marker="x", s=70,
                    color=color_for_model(model), alpha=0.8, label=None)

        # Highlight scenic choices when delta < 0
        highlight = scenic[scenic["split_delta_novelty"] < 0]
        if len(highlight) > 0:
            y_h = i + rng.normal(0, 0.04, size=len(highlight))
            ax2.scatter(highlight["split_delta_novelty"], y_h, marker="o", s=160,
                        facecolors="none", edgecolors="red", linewidth=1.5, zorder=4)

    ax2.axvline(0, color="black", linestyle="-", linewidth=1.2)
    ax2.set_xlabel("Delta novelty (scenic - dull)")
    ax2.set_yticks(range(len(models_scatter)))
    ax2.set_yticklabels(model_labels(models_scatter))
    ax2.grid(axis="x", alpha=0.3)
    ax2.set_title("B. Choice vs novelty delta (scenic familiar)")

    star = Line2D([0], [0], marker="*", color="w", label="Scenic choice (*)",
                  markerfacecolor="black", markersize=10)
    cross = Line2D([0], [0], marker="x", color="gray", label="Dull choice (x)", markersize=8)
    ring = Line2D([0], [0], marker="o", color="red", label="Scenic when delta < 0",
                  markerfacecolor="none", markersize=8)
    ax2.legend(handles=[star, cross, ring], loc="center right", fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26, top=0.86)
    n_mornings = int(df_fam_post["morning_idx"].nunique()) if "morning_idx" in df_fam_post.columns else 0
    footer = f"{format_n(df_bar)}; post only; {n_mornings} mornings/seed"
    add_footer(fig, footer)
    fig.savefig(out_path, dpi=300 if not debug else 200, bbox_inches="tight")
    plt.close(fig)


def plot_familiarity_figure_supp(out_path, debug=False):
    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(13, 4), gridspec_kw={"width_ratios": [1.0, 1.35, 1.35]}
    )
    fig.suptitle(
        "Familiarity control supplement (baseline vs post)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.93,
        "Interaction headline: Recon/Ipsundrum switch away from scenic after familiarity; affect models persist.",
        ha="center",
        va="top",
        fontsize=9,
    )

    # Panel A: Baseline scenic choice
    df_base = df_fam_base.copy()
    df_base, models_base = ordered_models(df_base)
    x0 = np.arange(len(models_base))
    means = []
    ci_low = []
    ci_high = []
    for model in models_base:
        data = seed_values(df_base, model, "scenic_choice").astype(float)
        mean, (low, high) = bootstrap_ci(data)
        means.append(mean)
        ci_low.append(mean - low)
        ci_high.append(high - mean)
    ax0.bar(x0, means, color=[color_for_model(m) for m in models_base], alpha=0.75, edgecolor="black")
    ax0.errorbar(x0, means, yerr=[ci_low, ci_high], fmt="none", ecolor="black", capsize=4, zorder=3)
    if debug:
        rng = np.random.default_rng(10)
        for i, model in enumerate(models_base):
            data = seed_values(df_base, model, "scenic_choice").astype(float)
            x_jit = i + rng.normal(0, 0.05, size=len(data))
            ax0.scatter(x_jit, data, color="black", s=18, alpha=0.6, zorder=4)
    ax0.set_ylabel("Scenic choice rate")
    ax0.set_xticks(x0)
    ax0.set_xticklabels(model_labels(models_base), rotation=25, ha="right")
    ax0.set_ylim(0, 1.05)
    ax0.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax0.grid(axis="y", alpha=0.3)
    ax0.set_title("A. Baseline (curiosity off; no familiarity)")

    # Panel B: Post-familiarization scenic choice
    df_bar = df_fam_post[df_fam_post["familiarize_side"].isin(["scenic", "both"])].copy()
    df_bar, models_bar = ordered_models(df_bar)
    fam_groups = ["scenic", "both"]
    x = np.arange(len(models_bar))
    width = 0.36
    for j, fam_side in enumerate(fam_groups):
        means = []
        ci_low = []
        ci_high = []
        for model in models_bar:
            data = seed_values(df_bar, model, "scenic_choice", {"familiarize_side": fam_side}).astype(float)
            mean, (low, high) = bootstrap_ci(data)
            means.append(mean)
            ci_low.append(mean - low)
            ci_high.append(high - mean)
        offset = (j - 0.5) * width
        hatch = "//" if fam_side == "both" else None
        ax1.bar(
            x + offset,
            means,
            width,
            color=[color_for_model(m) for m in models_bar],
            alpha=0.75,
            edgecolor="black",
            hatch=hatch,
            label=f"Familiarize: {fam_side}"
        )
        ax1.errorbar(x + offset, means, yerr=[ci_low, ci_high], fmt="none",
                     ecolor="black", capsize=4, zorder=3)
        if debug:
            rng = np.random.default_rng(2 + j)
            for i, model in enumerate(models_bar):
                data = seed_values(df_bar, model, "scenic_choice", {"familiarize_side": fam_side}).astype(float)
                x_jit = i + offset + rng.normal(0, 0.05, size=len(data))
                ax1.scatter(x_jit, data, color="black", s=18, alpha=0.6, zorder=4)

    ax1.set_ylabel("Scenic choice rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels(models_bar), rotation=25, ha='right')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("B. Post-familiarization (curiosity on)")
    ax1.legend(loc="lower right")

    # Panel C: Delta novelty scatter with choice markers
    df_scatter = df_fam_post[df_fam_post["familiarize_side"] == "scenic"].copy()
    df_scatter, models_scatter = ordered_models(df_scatter)
    for i, model in enumerate(models_scatter):
        data = df_scatter[df_scatter["model"] == model]
        scenic = data[data["scenic_choice"] == True]
        dull = data[data["scenic_choice"] == False]
        rng = np.random.default_rng(i + 3)
        y_s = i + rng.normal(0, 0.06, size=len(scenic))
        y_d = i + rng.normal(0, 0.06, size=len(dull))
        ax2.scatter(scenic["split_delta_novelty"], y_s, marker="*", s=120,
                    color=color_for_model(model), alpha=0.8, label=None)
        ax2.scatter(dull["split_delta_novelty"], y_d, marker="x", s=70,
                    color=color_for_model(model), alpha=0.8, label=None)

        # Highlight scenic choices when delta < 0
        highlight = scenic[scenic["split_delta_novelty"] < 0]
        if len(highlight) > 0:
            y_h = i + rng.normal(0, 0.04, size=len(highlight))
            ax2.scatter(highlight["split_delta_novelty"], y_h, marker="o", s=160,
                        facecolors="none", edgecolors="red", linewidth=1.5, zorder=4)

    ax2.axvline(0, color="black", linestyle="-", linewidth=1.2)
    ax2.set_xlabel("Delta novelty (scenic - dull)")
    ax2.set_yticks(range(len(models_scatter)))
    ax2.set_yticklabels(model_labels(models_scatter))
    ax2.grid(axis="x", alpha=0.3)
    ax2.set_title("C. Choice vs novelty delta (scenic familiar)")

    star = Line2D([0], [0], marker="*", color="w", label="Scenic choice (*)",
                  markerfacecolor="black", markersize=10)
    cross = Line2D([0], [0], marker="x", color="gray", label="Dull choice (x)", markersize=8)
    ring = Line2D([0], [0], marker="o", color="red", label="Scenic when delta < 0",
                  markerfacecolor="none", markersize=8)
    ax2.legend(handles=[star, cross, ring], loc="center right", fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26, top=0.78)
    n_mornings = int(df_fam_post["morning_idx"].nunique()) if "morning_idx" in df_fam_post.columns else 0
    footer = (
        f"{format_n(df_bar)}; baseline uses curiosity off/no familiarity; "
        f"post uses curiosity on with novelty competition; {n_mornings} mornings/seed."
    )
    add_footer(fig, footer)
    fig.savefig(out_path, dpi=300 if not debug else 200, bbox_inches="tight")
    plt.close(fig)


plot_familiarity_figure("results/publication-figures/fig2.png", debug=False)
plot_familiarity_figure("results/publication-figures/debug/fig2_debug.png", debug=True)
plot_familiarity_figure_supp("results/publication-figures/fig2_supp.png", debug=False)
plot_familiarity_figure_supp("results/publication-figures/debug/fig2_supp_debug.png", debug=True)
print("OK: Figure 2 saved")


# ==================================================================
# FIGURE 3: LESION TEST (Ns Trace + AUC Drop)
# ==================================================================

traces = np.load("results/lesion/ns_traces.npz")
df_lesion = pd.read_csv("results/lesion/episodes_extended.csv")
df_lesion = df_lesion[df_lesion["valid"] == True].copy()
df_lesion, models_lesion = ordered_models(df_lesion)


def auc_diffs_by_model(df):
    diffs = {}
    for model in models_lesion:
        subset = df[df["model"] == model]
        sham = subset[subset["condition"] == "sham"]["post_lesion_auc"]
        lesion = subset[subset["condition"] == "lesion"]["post_lesion_auc"]
        if "seed" in subset.columns:
            sham_by_seed = subset[subset["condition"] == "sham"].set_index("seed")["post_lesion_auc"]
            lesion_by_seed = subset[subset["condition"] == "lesion"].set_index("seed")["post_lesion_auc"]
            common = sham_by_seed.index.intersection(lesion_by_seed.index)
            diff = (sham_by_seed.loc[common] - lesion_by_seed.loc[common]).values
        else:
            diff = sham.values - lesion.values
        diffs[model] = diff
    return diffs


def plot_lesion_figure(out_path, debug=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Representative trace (Ipsundrum, seed 0)
    model_to_plot = "Ipsundrum"
    seed_to_plot = 0
    sham_trace = traces[f"{model_to_plot}_seed{seed_to_plot}_sham"]
    lesion_trace = traces[f"{model_to_plot}_seed{seed_to_plot}_lesion"]
    time = np.arange(len(sham_trace))
    ax1.plot(time, sham_trace, label="Sham", color="#1f77b4", linewidth=2)
    ax1.plot(time, lesion_trace, label="Lesion", color="#d62728", linewidth=2, linestyle="--")
    ax1.axvline(3, color="black", linestyle=":", linewidth=2, label="Lesion time")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel(r"Internal state ($N^s$)")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1.0)
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_title(r"A. $N^s$ time series (Ipsundrum, seed 0)")

    # Panel B: AUC difference (sham - lesion)
    diffs = auc_diffs_by_model(df_lesion)
    means = []
    ci_low = []
    ci_high = []
    for model in models_lesion:
        mean, (low, high) = bootstrap_ci(diffs[model])
        means.append(mean)
        ci_low.append(mean - low)
        ci_high.append(high - mean)

    x = np.arange(len(models_lesion))
    ax2.bar(x, means, color=[color_for_model(m) for m in models_lesion], alpha=0.7, edgecolor="black")
    ax2.errorbar(x, means, yerr=[ci_low, ci_high], fmt="none", ecolor="black", capsize=4)
    if debug:
        rng = np.random.default_rng(4)
        for i, model in enumerate(models_lesion):
            data = diffs[model]
            x_jit = i + rng.normal(0, 0.05, size=len(data))
            ax2.scatter(x_jit, data, color="black", s=25, alpha=0.7, zorder=3)

    ax2.set_ylabel("AUC drop (sham - lesion)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels(models_lesion))
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("B. Causal effect on persistence")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    add_footer(fig, format_n(df_lesion))
    fig.savefig(out_path, dpi=300 if not debug else 200, bbox_inches="tight")
    plt.close(fig)


plot_lesion_figure("results/publication-figures/fig3.png", debug=False)
plot_lesion_figure("results/publication-figures/debug/fig3_debug.png", debug=True)
print("OK: Figure 3 saved")

print("\n" + "=" * 70)
print("ALL FIGURES CREATED SUCCESSFULLY")
print("=" * 70)
