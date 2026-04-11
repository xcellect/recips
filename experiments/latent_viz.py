from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plot_style import apply_times_style

apply_times_style()


def _latent_columns(df: pd.DataFrame) -> List[str]:
    prefixes = ("z_", "p_", "workspace_", "selector_weights_", "sel_")
    return [c for c in df.columns if c.startswith(prefixes)]


def _choose_axes(latent_cols: Sequence[str]) -> List[str]:
    preferred = [c for c in latent_cols if c.startswith(("p_", "workspace_", "z_"))]
    return list(preferred[:2] if len(preferred) >= 2 else latent_cols[:2])


def plot_latent_trajectories(df: pd.DataFrame, outdir: str, tag: str = "latent") -> None:
    latent_cols = _latent_columns(df)
    if len(latent_cols) < 2:
        return
    axes_cols = _choose_axes(latent_cols)
    fig, ax = plt.subplots(figsize=(6, 5))
    group_cols = [c for c in ("model", "arch_seed", "env_seed", "seed") if c in df.columns]
    if not group_cols:
        group_cols = [None]
    if group_cols == [None]:
        ax.plot(df[axes_cols[0]], df[axes_cols[1]], alpha=0.8)
    else:
        for _, sub in df.groupby(group_cols, sort=False, observed=False):
            ax.plot(sub[axes_cols[0]], sub[axes_cols[1]], alpha=0.45)
    ax.set_xlabel(axes_cols[0])
    ax.set_ylabel(axes_cols[1])
    ax.set_title("Latent Trajectories")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"{tag}_trajectories.png"), dpi=200)
    fig.savefig(os.path.join(outdir, f"{tag}_trajectories.pdf"))
    plt.close(fig)


def plot_selector_weights(df: pd.DataFrame, outdir: str, tag: str = "latent") -> None:
    selector_cols = [c for c in df.columns if c.startswith("sel_") or c.startswith("selector_weights_")]
    if not selector_cols:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(selector_cols))
    means = [float(pd.to_numeric(df[c], errors="coerce").mean()) for c in selector_cols]
    ax.bar(x, means, color="#457b9d")
    ax.set_xticks(x)
    ax.set_xticklabels(selector_cols, rotation=20, ha="right")
    ax.set_ylabel("Mean weight")
    ax.set_title("Selector Weights")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"{tag}_selector_weights.png"), dpi=200)
    fig.savefig(os.path.join(outdir, f"{tag}_selector_weights.pdf"))
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot latent-state trajectories from flattened logs")
    parser.add_argument("csv", type=str, help="Input CSV with flattened latent columns")
    parser.add_argument("--outdir", type=str, default="results/latent-viz", help="Output directory")
    parser.add_argument("--tag", type=str, default="latent", help="Output filename stem")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    plot_latent_trajectories(df, args.outdir, tag=args.tag)
    plot_selector_weights(df, args.outdir, tag=args.tag)


if __name__ == "__main__":
    main()
