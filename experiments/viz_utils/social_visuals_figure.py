from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.viz_utils.social_viz import (
    _draw_corridor_world,
    _draw_foodshare_world,
    simulate_corridor,
    simulate_foodshare,
)
from utils.plot_style import apply_times_style


CORRIDOR_TIMES = (0, 3, 9, 17)


def _apply_publication_style() -> None:
    apply_times_style()
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 180,
        }
    )


def _select_corridor_frames(condition: str) -> list:
    frames = simulate_corridor(condition, seed=0, horizon=18)
    selected = []
    for t in CORRIDOR_TIMES:
        if t >= len(frames):
            raise ValueError(f"Requested frame t={t} but only {len(frames)} corridor frames were produced")
        selected.append(frames[t])
    return selected


def create_social_visuals_figure(*, out_pdf: Path, out_png: Path | None = None) -> None:
    _apply_publication_style()

    food_left = simulate_foodshare("social_none", seed=0)[0]
    food_right = simulate_foodshare("social_affective_direct", seed=0)[0]
    corridor_left = _select_corridor_frames("social_none")
    corridor_right = _select_corridor_frames("social_affective_direct")

    fig = plt.figure(figsize=(13.2, 8.1))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 1.0], hspace=0.3, wspace=0.06)

    top = gs[0, :].subgridspec(1, 2, wspace=0.12)
    ax_food_left = fig.add_subplot(top[0, 0])
    ax_food_right = fig.add_subplot(top[0, 1])
    _draw_foodshare_world(ax_food_left, food_left)
    _draw_foodshare_world(ax_food_right, food_right)
    ax_food_left.set_title("FoodShareToy: self-only baseline")
    ax_food_right.set_title("FoodShareToy: affective coupling")

    fig.text(0.5, 0.965, "Representative trajectories from the validated runs", ha="center", va="top", fontsize=16)

    for col, frame in enumerate(corridor_left):
        ax = fig.add_subplot(gs[1, col])
        _draw_corridor_world(ax, frame)
        ax.set_title(f"baseline t={frame.t}")

    for col, frame in enumerate(corridor_right):
        ax = fig.add_subplot(gs[2, col])
        _draw_corridor_world(ax, frame)
        ax.set_title(f"coupled t={frame.t}")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the social visuals figure for the ALife paper.")
    parser.add_argument(
        "--out-pdf",
        default="../alife_social_paper/figures/fig_visuals.pdf",
        help="Output PDF path for the visuals figure.",
    )
    parser.add_argument(
        "--out-png",
        default="../alife_social_paper/figures/fig_visuals.png",
        help="Optional PNG output path (set empty string to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png) if args.out_png else None
    create_social_visuals_figure(out_pdf=out_pdf, out_png=out_png)
    print(f"Saved {out_pdf}")
    if out_png is not None:
        print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
