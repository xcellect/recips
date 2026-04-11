from __future__ import annotations

import argparse
import os
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.stats import bootstrap_mean_ci
from experiments.evaluation_harness import build_model_network
from core.model_factory import flatten_latent_state
from utils.model_naming import canonical_model_display, canonical_model_id


def _selector_focus(state: Dict[str, object]) -> float:
    weights = np.asarray(state.get("selector_weights", np.full(4, 0.25)), dtype=float)
    return float(np.max(weights))


def _state_from_net(net: object) -> Dict[str, object]:
    st = getattr(net, "_ipsundrum_state", None)
    if isinstance(st, dict):
        return dict(st)
    nodes = getattr(net, "nodes", {})
    if "Ns" in nodes:
        return {"Ns": float(net.get("Ns").activation)}  # type: ignore[attr-defined]
    return {}


def _run_sequence(model: str, arch_seed: int, env_seed: int, condition: str) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    _, net = build_model_network(model, arch_seed=arch_seed)
    net.start_root(True)
    clean_seq = [
        (0.5, 0.0, 0.6, 0.7),
        (0.4, 0.0, 0.5, 0.6),
        (0.3, 0.0, 0.4, 0.5),
    ]
    if condition == "conflict":
        seq = [(t, touch, smell, -vision) for t, touch, smell, vision in clean_seq]
    elif condition == "dropout":
        seq = [(t, touch, smell, 0.0) for t, touch, smell, vision in clean_seq]
    else:
        seq = clean_seq
    ns_vals = []
    focus_vals = []
    trace_rows: List[Dict[str, float]] = []
    for obs in seq:
        net._update_ipsundrum_sensor(obs[0], rng=np.random.default_rng(env_seed), obs_components=obs)  # type: ignore[attr-defined]
        st = _state_from_net(net)
        ns_vals.append(float(st.get("Ns", 0.0)))
        focus_vals.append(_selector_focus(st))
        row = {
            "model": canonical_model_id(model),
            "arch_seed": int(arch_seed),
            "env_seed": int(env_seed),
            "condition": condition,
            "step": int(len(trace_rows)),
            "Ns": float(st.get("Ns", 0.0)),
            "selector_focus": float(_selector_focus(st)),
        }
        row.update(flatten_latent_state(st))
        trace_rows.append(row)
    result = {
        "model": canonical_model_id(model),
        "arch_seed": int(arch_seed),
        "env_seed": int(env_seed),
        "condition": condition,
        "mean_ns": float(np.mean(ns_vals)),
        "selector_focus": float(np.mean(focus_vals)),
    }
    return result, trace_rows


def run_assay(model: str, arch_seed: int, env_seed: int) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    clean, clean_trace = _run_sequence(model, arch_seed, env_seed, "clean")
    conflict, conflict_trace = _run_sequence(model, arch_seed, env_seed, "conflict")
    dropout, dropout_trace = _run_sequence(model, arch_seed, env_seed, "dropout")
    robustness = clean["mean_ns"] - 0.5 * (abs(clean["mean_ns"] - conflict["mean_ns"]) + abs(clean["mean_ns"] - dropout["mean_ns"]))
    result = {
        "model": clean["model"],
        "arch_seed": int(arch_seed),
        "env_seed": int(env_seed),
        "robustness": float(robustness),
        "selector_focus_conflict": float(conflict["selector_focus"]),
        "degradation_conflict": float(clean["mean_ns"] - conflict["mean_ns"]),
    }
    return result, clean_trace + conflict_trace + dropout_trace


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in df.groupby("model", sort=False):
        for metric in ("robustness", "selector_focus_conflict", "degradation_conflict"):
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
    metric = summary[summary["metric"] == "robustness"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(metric["model_display"], metric["mean"], color="#6c5b7b")
    ax.set_ylabel("Robustness proxy")
    ax.set_title("Multimodal Conflict Assay")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "multimodal_conflict.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "multimodal_conflict.pdf"))
    plt.close(fig)


def plot_trace_summary(trace_df: pd.DataFrame, outdir: str) -> None:
    if trace_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for (model, condition), sub in trace_df.groupby(["model", "condition"], sort=False):
        curve = sub.groupby("step", sort=False, observed=False)["selector_focus"].mean().reset_index()
        ax.plot(curve["step"], curve["selector_focus"], marker="o", label=f"{canonical_model_display(model)} {condition}")
    ax.set_xlabel("step")
    ax.set_ylabel("selector focus")
    ax.set_title("Multimodal Conflict Trace Summary")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "multimodal_conflict_trace_summary.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "multimodal_conflict_trace_summary.pdf"))
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the multimodal conflict assay")
    parser.add_argument("--outdir", default="results/multimodal-conflict")
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
                result, trace = run_assay(model, a, e)
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
