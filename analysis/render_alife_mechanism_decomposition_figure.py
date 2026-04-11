"""Render the ALife mechanism-decomposition summary figure for paper-2-v1.

This figure is paper-facing rather than a generic data exploration chart, so
the values are the exact summary numbers reported in the manuscript and the
annotation offsets are tuned for print-scale legibility.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils.plot_style import apply_times_style


OUT_PNG = Path("docs-alife/paper-2-v1/fig_mechanism_decomposition.png")
OUT_PDF = Path("docs-alife/paper-2-v1/fig_mechanism_decomposition.pdf")

COLORS = {
    "Full": "#1f77b4",
    "Readout-only": "#ff7f0e",
    "Modulation-only": "#2ca02c",
}

FAMILIARITY = {
    "Full": {"rates": [0.911, 0.911], "novelty": [0.186, -0.405]},
    "Readout-only": {"rates": [0.960, 0.960], "novelty": [0.184, -0.407]},
    "Modulation-only": {"rates": [0.878, 0.858], "novelty": [0.210, -0.370]},
}

PLAY = {
    "Full": {"unique_viewpoints": 136.4, "cycle_score": 7.5, "scan_events": 33.9},
    "Readout-only": {"unique_viewpoints": 131.9, "cycle_score": 9.9, "scan_events": 36.0},
    "Modulation-only": {"unique_viewpoints": 155.3, "cycle_score": 0.2, "scan_events": 34.8},
}

LESION = {
    "Full": 27.63,
    "Readout-only": 19.65,
    "Modulation-only": 27.63,
}

PAIN_TAIL = {
    "Full": {"ns_auc": 0.158, "turn_rate": 0.89},
    "Readout-only": {"ns_auc": 0.234, "turn_rate": 0.78},
    "Modulation-only": {"ns_auc": 0.158, "turn_rate": 0.94},
}


def apply_style() -> None:
    apply_times_style()
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def bubble_size(scan_events: float) -> float:
    return 18.0 * scan_events


def add_panel_label(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=8)
    ax.grid(True, alpha=0.22, linewidth=0.8)


def render() -> None:
    apply_style()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=2 / 72, h_pad=2 / 72, wspace=0.08, hspace=0.10)

    ax_a, ax_b, ax_c, ax_d = axes.flat

    # Panel A: familiarity control.
    x = np.array([0, 1], dtype=float)
    xlabels = ["Dull\nfamiliarized", "Scenic\nfamiliarized"]
    fam_offsets = {
        # Keep left-side novelty labels inside the plotting area.
        "Full": [(24, 10), (-30, -12)],
        "Readout-only": [(24, 10), (-30, -12)],
        "Modulation-only": [(24, 10), (-30, -12)],
    }
    for model, values in FAMILIARITY.items():
        y = values["rates"]
        ax_a.plot(
            x,
            y,
            marker="o",
            markersize=7.5,
            linewidth=2.3,
            color=COLORS[model],
            label=model,
            zorder=3,
        )
        for idx, (xv, yv, dnov) in enumerate(zip(x, y, values["novelty"])):
            dx, dy = fam_offsets[model][idx]
            ax_a.annotate(
                f"\u0394nov={dnov:+.2f}",
                xy=(xv, yv),
                xytext=(dx, dy),
                textcoords="offset points",
                color="#222222",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
                zorder=4,
            )
    ax_a.set_ylim(0.75, 1.00)
    ax_a.set_ylabel("Scenic-entry rate")
    ax_a.set_xticks(x, xlabels)
    ax_a.legend(loc="lower left", frameon=False)
    add_panel_label(ax_a, "A. Familiarity control")

    # Panel B: exploratory play.
    play_legend_handles: list[Line2D] = []
    for model, values in PLAY.items():
        ax_b.scatter(
            values["unique_viewpoints"],
            values["cycle_score"],
            s=bubble_size(values["scan_events"]),
            color=COLORS[model],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.85,
            zorder=3,
        )
        play_legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=COLORS[model],
                markeredgecolor="white",
                markeredgewidth=1.0,
                markersize=9,
                label=model,
            )
        )
    ax_b.text(
        137.2,
        10.05,
        "Bubble area encodes scan events",
        fontsize=10,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        ha="left",
        va="top",
    )
    ax_b.set_xlim(131.0, 156.5)
    ax_b.set_ylim(-0.2, 10.3)
    ax_b.set_xlabel("Unique viewpoints")
    ax_b.set_ylabel("Cycle score")
    ax_b.legend(handles=play_legend_handles, loc="upper right", frameon=False)
    add_panel_label(ax_b, "B. Exploratory play")

    # Panel C: lesion persistence cost.
    lesion_models = list(LESION.keys())
    lesion_x = np.arange(len(lesion_models))
    lesion_y = [LESION[m] for m in lesion_models]
    bars = ax_c.bar(
        lesion_x,
        lesion_y,
        color=[COLORS[m] for m in lesion_models],
        alpha=0.95,
        width=0.8,
        edgecolor="white",
        linewidth=1.0,
    )
    for bar, val in zip(bars, lesion_y):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.45,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            color="#222222",
        )
    ax_c.set_xticks(lesion_x, lesion_models)
    ax_c.set_ylabel("Sham-lesion AUC drop")
    ax_c.set_ylim(0, 33.5)
    add_panel_label(ax_c, "C. Lesion persistence cost")

    # Panel D: pain-tail.
    pain_legend_handles: list[Line2D] = []
    for model, values in PAIN_TAIL.items():
        ax_d.scatter(
            values["ns_auc"],
            values["turn_rate"],
            s=220,
            color=COLORS[model],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.85,
            zorder=3,
        )
        pain_legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=COLORS[model],
                markeredgecolor="white",
                markeredgewidth=1.0,
                markersize=9,
                label=model,
            )
        )
    ax_d.set_xlim(0.154, 0.238)
    ax_d.set_ylim(0.775, 0.947)
    ax_d.set_xlabel(r"Baseline-corrected $N^s$ AUC")
    ax_d.set_ylabel("Turn rate in tail")
    ax_d.legend(handles=pain_legend_handles, loc="lower left", frameon=False)
    add_panel_label(ax_d, "D. Pain-tail (exploratory)")

    fig.savefig(OUT_PNG, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    render()
