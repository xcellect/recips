from __future__ import annotations

import argparse
import os
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.stats import bootstrap_mean_ci
from core.driver.active_perception import score_internal
from experiments.evaluation_harness import build_model_network
from experiments.gridworld_exp import select_forward_model
from core.model_factory import flatten_latent_state
from utils.model_naming import canonical_model_display, canonical_model_id


CTX_A = (0.7, 0.0, 0.8, 0.2)
CTX_B = (-0.7, 0.0, -0.8, -0.2)
NEUTRAL = (0.0, 0.0, 0.0, 0.0)
FORK = (0.1, 0.0, 0.1, 0.1)
LEFT = (0.5, 0.0, 0.4, 0.6)
RIGHT = (-0.5, 0.0, -0.4, -0.6)


def _phi(state: Dict[str, object]) -> np.ndarray:
    if "p" in state:
        return np.asarray(state["p"], dtype=float)
    if "workspace" in state:
        return np.asarray(state["workspace"], dtype=float)
    return np.asarray([float(state.get("internal", state.get("Ns", 0.0)))], dtype=float)


def _state_from_net(net: object) -> Dict[str, object]:
    st = getattr(net, "_ipsundrum_state", None)
    if isinstance(st, dict):
        return dict(st)
    nodes = getattr(net, "nodes", {})
    if "Ns" in nodes:
        return {"Ns": float(net.get("Ns").activation)}  # type: ignore[attr-defined]
    return {}


def _score_candidate(model: str, state: dict, obs: Tuple[float, float, float, float], arch_seed: int) -> float:
    builder, _ = build_model_network(model, arch_seed=arch_seed)
    forward = select_forward_model(model=model)
    pred = forward(state, builder.params, builder.affect, obs[0], rng=np.random.default_rng(0), obs_components=obs)
    return float(score_internal(pred, builder.affect, current_I=0.0, predicted_I=obs[0]))


def run_trial(model: str, arch_seed: int, env_seed: int, delay: int, context: str) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    builder, net = build_model_network(model, arch_seed=arch_seed)
    net.start_root(True)
    cue = CTX_A if context == "A" else CTX_B
    correct = LEFT if context == "A" else RIGHT
    incorrect = RIGHT if context == "A" else LEFT

    trace_rows: List[Dict[str, float]] = []
    def record(stage: str, step_idx: int) -> None:
        st = _state_from_net(net)
        row = {
            "model": canonical_model_id(model),
            "arch_seed": int(arch_seed),
            "env_seed": int(env_seed),
            "delay": int(delay),
            "context": context,
            "stage": stage,
            "step": int(step_idx),
            "phi_norm": float(np.linalg.norm(_phi(st))),
        }
        row.update(flatten_latent_state(st))
        trace_rows.append(row)

    net._update_ipsundrum_sensor(cue[0], rng=np.random.default_rng(env_seed), obs_components=cue)  # type: ignore[attr-defined]
    record("cue", 0)
    for _ in range(delay):
        net._update_ipsundrum_sensor(NEUTRAL[0], rng=np.random.default_rng(env_seed), obs_components=NEUTRAL)  # type: ignore[attr-defined]
        record("delay", len(trace_rows))
    net._update_ipsundrum_sensor(FORK[0], rng=np.random.default_rng(env_seed), obs_components=FORK)  # type: ignore[attr-defined]
    state = _state_from_net(net)
    record("fork", len(trace_rows))
    sep_anchor = _phi(state)
    correct_score = _score_candidate(model, state, correct, arch_seed)
    incorrect_score = _score_candidate(model, state, incorrect, arch_seed)
    result = {
        "model": canonical_model_id(model),
        "arch_seed": int(arch_seed),
        "env_seed": int(env_seed),
        "delay": int(delay),
        "context": context,
        "success": float(correct_score > incorrect_score),
        "fork_score_margin": float(correct_score - incorrect_score),
        "fork_latent_norm": float(np.linalg.norm(sep_anchor)),
    }
    return result, trace_rows


def run_assay(model: str, arch_seed: int, env_seed: int, delay: int) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    a, a_trace = run_trial(model, arch_seed, env_seed, delay, "A")
    b, b_trace = run_trial(model, arch_seed, env_seed, delay, "B")
    result = {
        "model": a["model"],
        "arch_seed": int(arch_seed),
        "env_seed": int(env_seed),
        "delay": int(delay),
        "success_rate": float(0.5 * (a["success"] + b["success"])),
        "fork_score_margin": float(0.5 * (a["fork_score_margin"] + b["fork_score_margin"])),
        "R": float(np.linalg.norm(np.asarray([a["fork_latent_norm"]]) - np.asarray([b["fork_latent_norm"]]))),
    }
    return result, a_trace + b_trace


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, delay), sub in df.groupby(["model", "delay"], sort=False):
        for metric in ("success_rate", "fork_score_margin", "R"):
            result = bootstrap_mean_ci(sub[metric].values)
            rows.append({
                "model": model,
                "model_display": canonical_model_display(model),
                "delay": int(delay),
                "metric": metric,
                "mean": result.mean,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "n": result.n,
            })
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, outdir: str) -> None:
    metric = summary[summary["metric"] == "success_rate"].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, sub in metric.groupby("model_display", sort=False):
        ax.plot(sub["delay"], sub["mean"], marker="o", label=model)
    ax.set_xlabel("Delay")
    ax.set_ylabel("Success rate")
    ax.set_title("Context Fork Assay")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "context_fork.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "context_fork.pdf"))
    plt.close(fig)


def plot_trace_summary(trace_df: pd.DataFrame, outdir: str) -> None:
    if trace_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, sub in trace_df.groupby("model", sort=False):
        curve = sub.groupby(["delay", "stage"], sort=False, observed=False)["phi_norm"].mean().reset_index()
        for delay, dsub in curve.groupby("delay", sort=False):
            ax.plot(range(len(dsub)), dsub["phi_norm"], marker="o", label=f"{canonical_model_display(model)} d={delay}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["cue", "delay", "fork"])
    ax.set_ylabel("Mean latent norm")
    ax.set_title("Context Fork Trace Summary")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "context_fork_trace_summary.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "context_fork_trace_summary.pdf"))
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the context fork assay")
    parser.add_argument("--outdir", default="results/context-fork")
    parser.add_argument("--models", default="perspective,perspective_plastic,gw_lite,humphrey_barrett,humphrey,recon")
    parser.add_argument("--arch-seeds", type=int, default=4)
    parser.add_argument("--env-seeds", type=int, default=4)
    parser.add_argument("--delays", default="6,12,18")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    models = [canonical_model_id(m) for m in args.models.split(",") if m.strip()]
    delays = [int(x) for x in args.delays.split(",") if x.strip()]
    rows = []
    trace_rows: List[Dict[str, float]] = []
    for model in models:
        for a in range(args.arch_seeds):
            for e in range(args.env_seeds):
                for d in delays:
                    result, trace = run_assay(model, a, e, d)
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
