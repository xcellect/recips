from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.stats import bootstrap_mean_ci
from experiments.evaluation_harness import build_model_network
from core.model_factory import flatten_latent_state
from utils.model_naming import canonical_model_display, canonical_model_id


def _motifs() -> List[Tuple[float, float, float, float]]:
    return [
        (0.8, 0.9, 0.4, -0.2),
        (0.2, 0.0, 0.5, 0.3),
        (-0.6, -0.1, -0.7, 0.1),
        (0.0, 0.0, 0.2, 0.7),
    ]


def _phi(state: Dict[str, object]) -> np.ndarray:
    if "p" in state:
        return np.asarray(state["p"], dtype=float)
    if "workspace" in state:
        return np.asarray(state["workspace"], dtype=float)
    return np.asarray([float(state.get("internal", state.get("Ns", 0.0)))], dtype=float)


def _state_from_net(net: object) -> Dict[str, object]:
    st = getattr(net, "_ipsundrum_state", None)
    if isinstance(st, dict):
        return st
    nodes = getattr(net, "nodes", {})
    if "Ns" in nodes:
        return {"Ns": float(net.get("Ns").activation)}  # type: ignore[attr-defined]
    return {}


def _scaled_obs(motif: Tuple[float, float, float, float], lam: float) -> Tuple[float, float, float, float]:
    total, touch, smell, vision = motif
    return (lam * total, lam * touch, lam * smell, lam * vision)


def run_probe(model: str, arch_seed: int, env_seed: int) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    _, net = build_model_network(model, arch_seed=arch_seed)
    net.start_root(True)
    motifs = _motifs()
    rng = np.random.default_rng(env_seed)
    motif_order = rng.permutation(len(motifs))
    lam_grid = np.linspace(0.0, 1.0, 9)
    pre = _phi(_state_from_net(net)).copy()
    up = []
    down = []
    dw_norm = []
    open_vals = []
    trace_rows: List[Dict[str, float]] = []
    def record_row(step_idx: int, phase: str, lam: float, st: Dict[str, object]) -> None:
        row = {
            "model": canonical_model_id(model),
            "arch_seed": int(arch_seed),
            "env_seed": int(env_seed),
            "step": int(step_idx),
            "phase": phase,
            "lambda": float(lam),
            "phi_norm": float(np.linalg.norm(_phi(st))),
            "plasticity_open": float(st.get("plasticity_open", 0.0)),
            "delta_w_norm": float(st.get("delta_w_norm", 0.0)),
        }
        row.update(flatten_latent_state(st))
        trace_rows.append(row)

    for lam in lam_grid:
        motif = motifs[int(motif_order[len(up) % len(motif_order)])]
        obs = _scaled_obs(motif, float(lam))
        net._update_ipsundrum_sensor(obs[0], rng=np.random.default_rng(env_seed), obs_components=obs)  # type: ignore[attr-defined]
        st = _state_from_net(net)
        up.append(_phi(st).copy())
        dw_norm.append(float(st.get("delta_w_norm", 0.0)))
        open_vals.append(float(st.get("plasticity_open", 0.0)))
        record_row(len(trace_rows), "up", float(lam), st)
    for lam in lam_grid[::-1]:
        motif = motifs[int(motif_order[len(down) % len(motif_order)])]
        obs = _scaled_obs(motif, float(lam))
        net._update_ipsundrum_sensor(obs[0], rng=np.random.default_rng(env_seed), obs_components=obs)  # type: ignore[attr-defined]
        st = _state_from_net(net)
        down.append(_phi(st).copy())
        dw_norm.append(float(st.get("delta_w_norm", 0.0)))
        open_vals.append(float(st.get("plasticity_open", 0.0)))
        record_row(len(trace_rows), "down", float(lam), st)
    post = _phi(_state_from_net(net)).copy()
    H = float(np.mean([np.linalg.norm(a - b) for a, b in zip(up, down[::-1])]))
    result = {
        "model": canonical_model_id(model),
        "arch_seed": int(arch_seed),
        "env_seed": int(env_seed),
        "hysteresis": H,
        "residue": float(np.linalg.norm(post - pre)),
        "plasticity_open_mean": float(np.mean(open_vals)),
        "delta_w_norm_mean": float(np.mean(dw_norm)),
    }
    return result, trace_rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in df.groupby("model", sort=False):
        for metric in ("hysteresis", "residue", "plasticity_open_mean", "delta_w_norm_mean"):
            result = bootstrap_mean_ci(sub[metric].values)
            rows.append({
                "model": model,
                "model_display": canonical_model_display(model),
                "metric": metric,
                "mean": result.mean,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "n": result.n,
            })
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, outdir: str) -> None:
    metric = summary[summary["metric"] == "hysteresis"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(metric["model_display"], metric["mean"], color="#355c7d")
    ax.set_ylabel("Hysteresis")
    ax.set_title("Hysteresis Probe")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "hysteresis_probe.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "hysteresis_probe.pdf"))
    plt.close(fig)


def plot_trace_summary(trace_df: pd.DataFrame, outdir: str) -> None:
    if trace_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, sub in trace_df.groupby("model", sort=False):
        curve = sub.groupby(["phase", "lambda"], sort=False, observed=False)["phi_norm"].mean().reset_index()
        up = curve[curve["phase"] == "up"]
        down = curve[curve["phase"] == "down"]
        ax.plot(up["lambda"], up["phi_norm"], label=f"{canonical_model_display(model)} up")
        ax.plot(down["lambda"], down["phi_norm"], linestyle="--", label=f"{canonical_model_display(model)} down")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Mean latent norm")
    ax.set_title("Hysteresis Trace Summary")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "hysteresis_trace_summary.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "hysteresis_trace_summary.pdf"))
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the hysteresis probe assay")
    parser.add_argument("--outdir", default="results/hysteresis-probe")
    parser.add_argument("--models", default="perspective,perspective_plastic,gw_lite,humphrey_barrett,humphrey,recon")
    parser.add_argument("--arch-seeds", type=int, default=4)
    parser.add_argument("--env-seeds", type=int, default=4)
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    models = [canonical_model_id(m) for m in args.models.split(",") if m.strip()]
    rows = []
    trace_rows: List[Dict[str, float]] = []
    for model in models:
        for a in range(args.arch_seeds):
            for e in range(args.env_seeds):
                result, trace = run_probe(model, a, e)
                rows.append(result)
                trace_rows.extend(trace)
    df = pd.DataFrame(rows)
    trace_df = pd.DataFrame(trace_rows)
    summary = summarize(df)
    df.to_csv(os.path.join(args.outdir, "episodes.csv"), index=False)
    trace_df.to_csv(os.path.join(args.outdir, "trace.csv"), index=False)
    summary.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    plot_summary(summary, args.outdir)
    plot_trace_summary(trace_df, args.outdir)


if __name__ == "__main__":
    main()
