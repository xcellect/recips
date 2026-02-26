"""Compute paper claims and emit JSON/TeX/Markdown outputs."""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.stats import bootstrap_mean_ci
from utils.model_naming import canonical_model_id


@dataclass
class Claim:
    claim_id: str
    value: Any
    ci: Optional[Tuple[float, float]] = None
    n: Optional[int] = None
    claim_type: str = "scalar"
    digits: Optional[int] = 2
    meta: Optional[Dict[str, Any]] = None


class ClaimsRegistry:
    def __init__(self) -> None:
        self._claims: Dict[str, Callable[["ClaimsContext"], Claim]] = {}

    def register(self, claim_id: str) -> Callable[[Callable[["ClaimsContext"], Claim]], Callable[["ClaimsContext"], Claim]]:
        def decorator(func: Callable[["ClaimsContext"], Claim]) -> Callable[["ClaimsContext"], Claim]:
            if claim_id in self._claims:
                raise ValueError(f"Duplicate claim id: {claim_id}")
            self._claims[claim_id] = func
            return func
        return decorator

    def compute(self, ctx: "ClaimsContext") -> Dict[str, Claim]:
        results: Dict[str, Claim] = {}
        for cid, func in self._claims.items():
            claim = func(ctx)
            results[cid] = claim
        return results


@dataclass
class ClaimsContext:
    familiarity: pd.DataFrame
    play: pd.DataFrame
    play_clarified: pd.DataFrame
    pain_tail: pd.DataFrame
    lesion: pd.DataFrame
    goal_corridor: pd.DataFrame
    goal_gridworld: pd.DataFrame
    goal_corridor_episodes: pd.DataFrame
    goal_gridworld_episodes: pd.DataFrame
    config: Dict[str, Any]


REGISTRY = ClaimsRegistry()


# ---------------------------
# Helpers
# ---------------------------

def _maybe_filter_valid_decided(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "valid" in out.columns:
        out = out[out["valid"] == True]
    if "decided" in out.columns:
        out = out[out["decided"] == True]
    return out


def _seed_means(df: pd.DataFrame, value_col: str) -> np.ndarray:
    if "seed" in df.columns:
        return df.groupby("seed", sort=False, observed=False)[value_col].mean().values
    return df[value_col].values


def _mean_ci(values: Iterable[float]) -> Tuple[float, float, float, int]:
    result = bootstrap_mean_ci(values)
    return result.mean, result.ci_low, result.ci_high, result.n


def _paired_diffs(df: pd.DataFrame, model: str, value_col: str, cond_col: str, cond_a: str, cond_b: str) -> np.ndarray:
    sub = df[df["model"] == model]
    if "seed" not in sub.columns:
        return (sub[sub[cond_col] == cond_b][value_col].values - sub[sub[cond_col] == cond_a][value_col].values)
    a = sub[sub[cond_col] == cond_a].groupby("seed", sort=False, observed=False)[value_col].mean()
    b = sub[sub[cond_col] == cond_b].groupby("seed", sort=False, observed=False)[value_col].mean()
    common = a.index.intersection(b.index)
    return (b.loc[common] - a.loc[common]).values


def _median_per_seed(df: pd.DataFrame, value_col: str) -> Tuple[float, int]:
    if "seed" not in df.columns:
        vals = df[value_col].values
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size else float("nan"), int(vals.size)
    med = df.groupby("seed", sort=False, observed=False)[value_col].median()
    med = med[np.isfinite(med.values)]
    return float(np.median(med.values)) if med.size else float("nan"), int(med.size)


def _select_play_clarified(path_root: str) -> str:
    candidates = [
        p for p in glob.glob(os.path.join(path_root, "exploratory_play_clarified*.csv"))
        if "trace" not in os.path.basename(p)
    ]
    if not candidates:
        raise FileNotFoundError("No exploratory_play_clarified*.csv found")
    preferred = [c for c in candidates if c.endswith("_paper.csv")]
    if preferred:
        return preferred[0]
    preferred = [c for c in candidates if c.endswith("_quick.csv")]
    if preferred:
        return preferred[0]
    exact = [c for c in candidates if c.endswith("exploratory_play_clarified.csv")]
    if exact:
        return exact[0]
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _entropy_column(df: pd.DataFrame) -> str:
    for col in ("neutral_sensory_entropy", "viewpoint_entropy", "sensory_entropy"):
        if col in df.columns and df[col].notna().any():
            return col
    return "sensory_entropy"


def _claim_mean_ci(claim_id: str, values: Iterable[float], digits: int, meta: Dict[str, Any]) -> Claim:
    mean, low, high, n = _mean_ci(values)
    return Claim(claim_id=claim_id, value=mean, ci=(low, high), n=n, claim_type="mean_ci", digits=digits, meta=meta)


def _claim_scalar(claim_id: str, value: float, n: Optional[int], digits: int, meta: Dict[str, Any]) -> Claim:
    return Claim(claim_id=claim_id, value=value, ci=None, n=n, claim_type="scalar", digits=digits, meta=meta)


def _claim_range(claim_id: str, low: float, high: float, digits: int, meta: Dict[str, Any]) -> Claim:
    return Claim(claim_id=claim_id, value=(low, high), ci=None, n=None, claim_type="range", digits=digits, meta=meta)


def _claim_bool(claim_id: str, value: bool, meta: Dict[str, Any]) -> Claim:
    return Claim(claim_id=claim_id, value=bool(value), ci=None, n=None, claim_type="boolean", digits=None, meta=meta)


# ---------------------------
# Familiarity control claims
# ---------------------------

@REGISTRY.register("fam_post_scenic_entry_recon_scenic")
def _fam_recon_scenic(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["familiarize_side"] == "scenic") & (df["model"] == "recon")]
    values = _seed_means(sub, "scenic_choice")
    return _claim_mean_ci("fam_post_scenic_entry_recon_scenic", values, 2, {"source": "results/familiarity/episodes_improved.csv"})


@REGISTRY.register("fam_post_scenic_entry_humphrey_scenic")
def _fam_humphrey_scenic(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["familiarize_side"] == "scenic") & (df["model"] == "humphrey")]
    values = _seed_means(sub, "scenic_choice")
    return _claim_mean_ci("fam_post_scenic_entry_humphrey_scenic", values, 2, {"source": "results/familiarity/episodes_improved.csv"})


@REGISTRY.register("fam_post_scenic_entry_hb_scenic")
def _fam_hb_scenic(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["familiarize_side"] == "scenic") & (df["model"] == "humphrey_barrett")]
    values = _seed_means(sub, "scenic_choice")
    return _claim_mean_ci("fam_post_scenic_entry_hb_scenic", values, 2, {"source": "results/familiarity/episodes_improved.csv"})


@REGISTRY.register("fam_post_scenic_entry_recon_dull")
def _fam_recon_dull(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["familiarize_side"] == "dull") & (df["model"] == "recon")]
    values = _seed_means(sub, "scenic_choice")
    return _claim_mean_ci("fam_post_scenic_entry_recon_dull", values, 2, {"source": "results/familiarity/episodes_improved.csv"})


@REGISTRY.register("fam_post_scenic_entry_humphrey_dull")
def _fam_humphrey_dull(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["familiarize_side"] == "dull") & (df["model"] == "humphrey")]
    values = _seed_means(sub, "scenic_choice")
    return _claim_mean_ci("fam_post_scenic_entry_humphrey_dull", values, 2, {"source": "results/familiarity/episodes_improved.csv"})


@REGISTRY.register("fam_post_scenic_entry_hb_dull")
def _fam_hb_dull(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["familiarize_side"] == "dull") & (df["model"] == "humphrey_barrett")]
    values = _seed_means(sub, "scenic_choice")
    return _claim_mean_ci("fam_post_scenic_entry_hb_dull", values, 2, {"source": "results/familiarity/episodes_improved.csv"})


@REGISTRY.register("fam_delta_scenic_entry_recon")
def _fam_delta_recon(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    values = _paired_diffs(df[df["phase"] == "post"], "recon", "scenic_choice", "familiarize_side", "scenic", "dull")
    return _claim_mean_ci("fam_delta_scenic_entry_recon", values, 2, {"source": "results/familiarity/episodes_improved.csv", "definition": "dull - scenic"})


@REGISTRY.register("fam_delta_scenic_entry_humphrey")
def _fam_delta_humphrey(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    values = _paired_diffs(df[df["phase"] == "post"], "humphrey", "scenic_choice", "familiarize_side", "scenic", "dull")
    return _claim_mean_ci("fam_delta_scenic_entry_humphrey", values, 2, {"source": "results/familiarity/episodes_improved.csv", "definition": "dull - scenic"})


@REGISTRY.register("fam_delta_scenic_entry_hb")
def _fam_delta_hb(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    values = _paired_diffs(df[df["phase"] == "post"], "humphrey_barrett", "scenic_choice", "familiarize_side", "scenic", "dull")
    return _claim_mean_ci("fam_delta_scenic_entry_hb", values, 2, {"source": "results/familiarity/episodes_improved.csv", "definition": "dull - scenic"})


@REGISTRY.register("fam_median_delta_novelty_hb_scenic")
def _fam_median_delta_novelty(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["familiarize_side"] == "scenic") & (df["model"] == "humphrey_barrett")]
    median, n = _median_per_seed(sub, "split_delta_novelty")
    return _claim_scalar("fam_median_delta_novelty_hb_scenic", median, n, 2, {"source": "results/familiarity/episodes_improved.csv"})


@REGISTRY.register("fam_valence_scenic_hb")
def _fam_valence_scenic(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["model"] == "humphrey_barrett")]
    vals = _seed_means(sub, "mean_valence_scenic")
    vals = vals[np.isfinite(vals)]
    return _claim_mean_ci("fam_valence_scenic_hb", vals, 2, {"source": "results/familiarity/episodes_improved.csv"})


@REGISTRY.register("fam_valence_dull_hb")
def _fam_valence_dull(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["model"] == "humphrey_barrett")]
    if "seed" in sub.columns:
        seed_vals = []
        for seed, group in sub.groupby("seed", sort=False, observed=False):
            vals = group["mean_valence_dull"].values
            vals = vals[np.isfinite(vals)]
            if vals.size:
                seed_vals.append(float(np.mean(vals)))
        values = np.asarray(seed_vals, dtype=float)
    else:
        values = sub["mean_valence_dull"].values
    return _claim_mean_ci("fam_valence_dull_hb", values, 2, {"source": "results/familiarity/episodes_improved.csv", "note": "seeds with dull visits"})


@REGISTRY.register("fam_arousal_scenic_hb")
def _fam_arousal_scenic(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["model"] == "humphrey_barrett")]
    vals = _seed_means(sub, "mean_arousal_scenic")
    vals = vals[np.isfinite(vals)]
    return _claim_mean_ci("fam_arousal_scenic_hb", vals, 2, {"source": "results/familiarity/episodes_improved.csv"})


@REGISTRY.register("fam_arousal_dull_hb")
def _fam_arousal_dull(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["model"] == "humphrey_barrett")]
    if "seed" in sub.columns:
        seed_vals = []
        for seed, group in sub.groupby("seed", sort=False, observed=False):
            vals = group["mean_arousal_dull"].values
            vals = vals[np.isfinite(vals)]
            if vals.size:
                seed_vals.append(float(np.mean(vals)))
        values = np.asarray(seed_vals, dtype=float)
    else:
        values = sub["mean_arousal_dull"].values
    return _claim_mean_ci("fam_arousal_dull_hb", values, 2, {"source": "results/familiarity/episodes_improved.csv", "note": "seeds with dull visits"})


@REGISTRY.register("fam_probe_valence_scenic_hb")
def _fam_probe_valence_scenic_hb(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["model"] == "humphrey_barrett")]
    col = "split_pred_valence_scenic"
    if col not in sub.columns:
        return _claim_mean_ci(
            "fam_probe_valence_scenic_hb",
            np.asarray([]),
            2,
            {"source": "results/familiarity/episodes_improved.csv", "missing": col},
        )
    vals = _seed_means(sub, col)
    vals = vals[np.isfinite(vals)]
    return _claim_mean_ci(
        "fam_probe_valence_scenic_hb",
        vals,
        2,
        {"source": "results/familiarity/episodes_improved.csv", "definition": "horizon-5 split rollout"},
    )


@REGISTRY.register("fam_probe_valence_dull_hb")
def _fam_probe_valence_dull_hb(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["model"] == "humphrey_barrett")]
    col = "split_pred_valence_dull"
    if col not in sub.columns:
        return _claim_mean_ci(
            "fam_probe_valence_dull_hb",
            np.asarray([]),
            2,
            {"source": "results/familiarity/episodes_improved.csv", "missing": col},
        )
    vals = _seed_means(sub, col)
    vals = vals[np.isfinite(vals)]
    return _claim_mean_ci(
        "fam_probe_valence_dull_hb",
        vals,
        2,
        {"source": "results/familiarity/episodes_improved.csv", "definition": "horizon-5 split rollout"},
    )


@REGISTRY.register("fam_probe_arousal_scenic_hb")
def _fam_probe_arousal_scenic_hb(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["model"] == "humphrey_barrett")]
    col = "split_pred_arousal_scenic"
    if col not in sub.columns:
        return _claim_mean_ci(
            "fam_probe_arousal_scenic_hb",
            np.asarray([]),
            2,
            {"source": "results/familiarity/episodes_improved.csv", "missing": col},
        )
    vals = _seed_means(sub, col)
    vals = vals[np.isfinite(vals)]
    return _claim_mean_ci(
        "fam_probe_arousal_scenic_hb",
        vals,
        2,
        {"source": "results/familiarity/episodes_improved.csv", "definition": "horizon-5 split rollout"},
    )


@REGISTRY.register("fam_probe_arousal_dull_hb")
def _fam_probe_arousal_dull_hb(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    sub = df[(df["phase"] == "post") & (df["model"] == "humphrey_barrett")]
    col = "split_pred_arousal_dull"
    if col not in sub.columns:
        return _claim_mean_ci(
            "fam_probe_arousal_dull_hb",
            np.asarray([]),
            2,
            {"source": "results/familiarity/episodes_improved.csv", "missing": col},
        )
    vals = _seed_means(sub, col)
    vals = vals[np.isfinite(vals)]
    return _claim_mean_ci(
        "fam_probe_arousal_dull_hb",
        vals,
        2,
        {"source": "results/familiarity/episodes_improved.csv", "definition": "horizon-5 split rollout"},
    )


@REGISTRY.register("fam_n_seeds")
def _fam_n_seeds(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    n_seeds = int(df["seed"].nunique()) if "seed" in df.columns else 0
    return _claim_scalar("fam_n_seeds", float(n_seeds), n_seeds, 0, {"source": "results/familiarity/episodes_improved.csv"})


@REGISTRY.register("fam_post_repeats")
def _fam_post_repeats(ctx: ClaimsContext) -> Claim:
    df = _maybe_filter_valid_decided(ctx.familiarity)
    post = df[df["phase"] == "post"]
    repeats = int(post["morning_idx"].nunique()) if "morning_idx" in post.columns else 0
    return _claim_scalar("fam_post_repeats", float(repeats), repeats, 0, {"source": "results/familiarity/episodes_improved.csv"})


# ---------------------------
# Exploratory play claims
# ---------------------------

@REGISTRY.register("play_unique_viewpoints_recon")
def _play_unique_viewpoints_recon(ctx: ClaimsContext) -> Claim:
    sub = ctx.play[ctx.play["model"] == "recon"]
    return _claim_mean_ci("play_unique_viewpoints_recon", sub["unique_viewpoints"].values, 0, {"source": "results/exploratory-play/final_viewpoints.csv"})


@REGISTRY.register("play_unique_viewpoints_humphrey")
def _play_unique_viewpoints_humphrey(ctx: ClaimsContext) -> Claim:
    sub = ctx.play[ctx.play["model"] == "humphrey"]
    return _claim_mean_ci("play_unique_viewpoints_humphrey", sub["unique_viewpoints"].values, 0, {"source": "results/exploratory-play/final_viewpoints.csv"})


@REGISTRY.register("play_unique_viewpoints_hb")
def _play_unique_viewpoints_hb(ctx: ClaimsContext) -> Claim:
    sub = ctx.play[ctx.play["model"] == "humphrey_barrett"]
    return _claim_mean_ci("play_unique_viewpoints_hb", sub["unique_viewpoints"].values, 0, {"source": "results/exploratory-play/final_viewpoints.csv"})


@REGISTRY.register("play_unique_viewpoints_range")
def _play_unique_viewpoints_range(ctx: ClaimsContext) -> Claim:
    means = []
    for model in ("recon", "humphrey", "humphrey_barrett"):
        sub = ctx.play[ctx.play["model"] == model]
        means.append(float(np.mean(sub["unique_viewpoints"].values)))
    low = float(np.min(means)) if means else float("nan")
    high = float(np.max(means)) if means else float("nan")
    return _claim_range("play_unique_viewpoints_range", low, high, 0, {"source": "results/exploratory-play/final_viewpoints.csv"})


@REGISTRY.register("play_neutral_entropy_recon")
def _play_entropy_recon(ctx: ClaimsContext) -> Claim:
    col = _entropy_column(ctx.play)
    sub = ctx.play[ctx.play["model"] == "recon"]
    return _claim_mean_ci("play_neutral_entropy_recon", sub[col].values, 2, {"source": "results/exploratory-play/final_viewpoints.csv", "column": col})


@REGISTRY.register("play_neutral_entropy_humphrey")
def _play_entropy_humphrey(ctx: ClaimsContext) -> Claim:
    col = _entropy_column(ctx.play)
    sub = ctx.play[ctx.play["model"] == "humphrey"]
    return _claim_mean_ci("play_neutral_entropy_humphrey", sub[col].values, 2, {"source": "results/exploratory-play/final_viewpoints.csv", "column": col})


@REGISTRY.register("play_neutral_entropy_hb")
def _play_entropy_hb(ctx: ClaimsContext) -> Claim:
    col = _entropy_column(ctx.play)
    sub = ctx.play[ctx.play["model"] == "humphrey_barrett"]
    return _claim_mean_ci("play_neutral_entropy_hb", sub[col].values, 2, {"source": "results/exploratory-play/final_viewpoints.csv", "column": col})


@REGISTRY.register("play_neutral_entropy_range")
def _play_entropy_range(ctx: ClaimsContext) -> Claim:
    col = _entropy_column(ctx.play)
    means = []
    for model in ("recon", "humphrey", "humphrey_barrett"):
        sub = ctx.play[ctx.play["model"] == model]
        means.append(float(np.mean(sub[col].values)))
    low = float(np.min(means)) if means else float("nan")
    high = float(np.max(means)) if means else float("nan")
    return _claim_range("play_neutral_entropy_range", low, high, 2, {"source": "results/exploratory-play/final_viewpoints.csv", "column": col})


@REGISTRY.register("play_internal_actuator_recon_zero")
def _play_internal_actuator_recon_zero(ctx: ClaimsContext) -> Claim:
    col = "internal_actuator_fraction"
    if col not in ctx.play.columns:
        return _claim_bool("play_internal_actuator_recon_zero", False, {"source": "results/exploratory-play/final_viewpoints.csv", "missing": col})
    sub = ctx.play[ctx.play["model"] == "recon"]
    vals = sub[col].values.astype(float) if len(sub) else np.asarray([])
    ok = bool(vals.size > 0 and np.allclose(vals, 0.0))
    return _claim_bool("play_internal_actuator_recon_zero", ok, {"source": "results/exploratory-play/final_viewpoints.csv"})


@REGISTRY.register("play_internal_actuator_ipsundrum_nontrivial")
def _play_internal_actuator_ipsundrum_nontrivial(ctx: ClaimsContext) -> Claim:
    col = "internal_actuator_fraction"
    threshold = 0.05
    if col not in ctx.play.columns:
        return _claim_bool(
            "play_internal_actuator_ipsundrum_nontrivial",
            False,
            {"source": "results/exploratory-play/final_viewpoints.csv", "missing": col},
        )
    ok = True
    for model in ("humphrey", "humphrey_barrett"):
        sub = ctx.play[ctx.play["model"] == model]
        mean = float(np.mean(sub[col].values)) if len(sub) else float("nan")
        if not np.isfinite(mean) or mean <= threshold:
            ok = False
    return _claim_bool(
        "play_internal_actuator_ipsundrum_nontrivial",
        ok,
        {"source": "results/exploratory-play/final_viewpoints.csv", "threshold": threshold},
    )


@REGISTRY.register("play_scan_events_recon")
def _play_scan_recon(ctx: ClaimsContext) -> Claim:
    sub = ctx.play[ctx.play["model"] == "recon"]
    return _claim_mean_ci("play_scan_events_recon", sub["scan_events"].values, 1, {"source": "results/exploratory-play/final_viewpoints.csv"})


@REGISTRY.register("play_scan_events_humphrey")
def _play_scan_humphrey(ctx: ClaimsContext) -> Claim:
    sub = ctx.play[ctx.play["model"] == "humphrey"]
    return _claim_mean_ci("play_scan_events_humphrey", sub["scan_events"].values, 1, {"source": "results/exploratory-play/final_viewpoints.csv"})


@REGISTRY.register("play_scan_events_hb")
def _play_scan_hb(ctx: ClaimsContext) -> Claim:
    sub = ctx.play[ctx.play["model"] == "humphrey_barrett"]
    return _claim_mean_ci("play_scan_events_hb", sub["scan_events"].values, 1, {"source": "results/exploratory-play/final_viewpoints.csv"})


@REGISTRY.register("play_cycle_score_hb")
def _play_cycle_hb(ctx: ClaimsContext) -> Claim:
    sub = ctx.play[ctx.play["model"] == "humphrey_barrett"]
    return _claim_mean_ci("play_cycle_score_hb", sub["cycle_score"].values, 1, {"source": "results/exploratory-play/final_viewpoints.csv"})


@REGISTRY.register("play_action_entropy_hb_curiosity")
def _play_action_entropy_hb(ctx: ClaimsContext) -> Claim:
    sub = ctx.play_clarified[ctx.play_clarified["model"] == "humphrey_barrett_curiosity"]
    return _claim_mean_ci("play_action_entropy_hb_curiosity", sub["action_entropy"].values, 2, {"source": ctx.play_clarified.attrs.get("source", "exploratory_play_clarified")})


@REGISTRY.register("play_action_entropy_random")
def _play_action_entropy_random(ctx: ClaimsContext) -> Claim:
    sub = ctx.play_clarified[ctx.play_clarified["model"] == "random"]
    return _claim_mean_ci("play_action_entropy_random", sub["action_entropy"].values, 2, {"source": ctx.play_clarified.attrs.get("source", "exploratory_play_clarified")})


@REGISTRY.register("play_dwell_p90_hb_curiosity")
def _play_dwell_hb(ctx: ClaimsContext) -> Claim:
    sub = ctx.play_clarified[ctx.play_clarified["model"] == "humphrey_barrett_curiosity"]
    return _claim_mean_ci("play_dwell_p90_hb_curiosity", sub["dwell_p90"].values, 1, {"source": ctx.play_clarified.attrs.get("source", "exploratory_play_clarified")})


@REGISTRY.register("play_dwell_p90_random")
def _play_dwell_random(ctx: ClaimsContext) -> Claim:
    sub = ctx.play_clarified[ctx.play_clarified["model"] == "random"]
    return _claim_mean_ci("play_dwell_p90_random", sub["dwell_p90"].values, 2, {"source": ctx.play_clarified.attrs.get("source", "exploratory_play_clarified")})


# ---------------------------
# Pain-tail claims
# ---------------------------

@REGISTRY.register("pain_ns_half_life_recon")
def _pain_ns_recon(ctx: ClaimsContext) -> Claim:
    sub = ctx.pain_tail[ctx.pain_tail["model"] == "recon"]
    return _claim_mean_ci("pain_ns_half_life_recon", sub["ns_half_life"].values, 0, {"source": "results/pain-tail/episodes.csv"})


@REGISTRY.register("pain_ns_half_life_humphrey")
def _pain_ns_humphrey(ctx: ClaimsContext) -> Claim:
    sub = ctx.pain_tail[ctx.pain_tail["model"] == "humphrey"]
    return _claim_mean_ci("pain_ns_half_life_humphrey", sub["ns_half_life"].values, 0, {"source": "results/pain-tail/episodes.csv"})


@REGISTRY.register("pain_ns_half_life_hb")
def _pain_ns_hb(ctx: ClaimsContext) -> Claim:
    sub = ctx.pain_tail[ctx.pain_tail["model"] == "humphrey_barrett"]
    return _claim_mean_ci("pain_ns_half_life_hb", sub["ns_half_life"].values, 0, {"source": "results/pain-tail/episodes.csv"})


@REGISTRY.register("pain_tail_duration_recon")
def _pain_tail_recon(ctx: ClaimsContext) -> Claim:
    sub = ctx.pain_tail[ctx.pain_tail["model"] == "recon"]
    return _claim_mean_ci("pain_tail_duration_recon", sub["tail_duration"].values, 0, {"source": "results/pain-tail/episodes.csv"})


@REGISTRY.register("pain_tail_duration_humphrey")
def _pain_tail_humphrey(ctx: ClaimsContext) -> Claim:
    sub = ctx.pain_tail[ctx.pain_tail["model"] == "humphrey"]
    return _claim_mean_ci("pain_tail_duration_humphrey", sub["tail_duration"].values, 0, {"source": "results/pain-tail/episodes.csv"})


@REGISTRY.register("pain_tail_duration_hb")
def _pain_tail_hb(ctx: ClaimsContext) -> Claim:
    sub = ctx.pain_tail[ctx.pain_tail["model"] == "humphrey_barrett"]
    return _claim_mean_ci("pain_tail_duration_hb", sub["tail_duration"].values, 0, {"source": "results/pain-tail/episodes.csv"})


@REGISTRY.register("pain_ns_half_life_saturation_recon")
def _pain_saturation_recon(ctx: ClaimsContext) -> Claim:
    post_steps_raw = ctx.config.get("window_lengths", {}).get("pain_tail_post_stimulus_steps", np.nan)
    post_steps = float(post_steps_raw) if post_steps_raw is not None else float("nan")
    if not np.isfinite(post_steps):
        post_steps = float(np.nanmax(ctx.pain_tail["ns_half_life"].values))
    post_steps = int(post_steps)
    sub = ctx.pain_tail[ctx.pain_tail["model"] == "recon"]
    frac = float(np.mean(sub["ns_half_life"].values == post_steps)) if len(sub) else float("nan")
    return _claim_scalar("pain_ns_half_life_saturation_recon", frac, len(sub), 2, {"source": "results/pain-tail/episodes.csv", "post_steps": int(post_steps)})


@REGISTRY.register("pain_ns_half_life_saturation_humphrey")
def _pain_saturation_humphrey(ctx: ClaimsContext) -> Claim:
    post_steps_raw = ctx.config.get("window_lengths", {}).get("pain_tail_post_stimulus_steps", np.nan)
    post_steps = float(post_steps_raw) if post_steps_raw is not None else float("nan")
    if not np.isfinite(post_steps):
        post_steps = float(np.nanmax(ctx.pain_tail["ns_half_life"].values))
    post_steps = int(post_steps)
    sub = ctx.pain_tail[ctx.pain_tail["model"] == "humphrey"]
    frac = float(np.mean(sub["ns_half_life"].values == post_steps)) if len(sub) else float("nan")
    return _claim_scalar("pain_ns_half_life_saturation_humphrey", frac, len(sub), 2, {"source": "results/pain-tail/episodes.csv", "post_steps": int(post_steps)})


@REGISTRY.register("pain_ns_half_life_saturation_hb")
def _pain_saturation_hb(ctx: ClaimsContext) -> Claim:
    post_steps_raw = ctx.config.get("window_lengths", {}).get("pain_tail_post_stimulus_steps", np.nan)
    post_steps = float(post_steps_raw) if post_steps_raw is not None else float("nan")
    if not np.isfinite(post_steps):
        post_steps = float(np.nanmax(ctx.pain_tail["ns_half_life"].values))
    post_steps = int(post_steps)
    sub = ctx.pain_tail[ctx.pain_tail["model"] == "humphrey_barrett"]
    frac = float(np.mean(sub["ns_half_life"].values == post_steps)) if len(sub) else float("nan")
    return _claim_scalar("pain_ns_half_life_saturation_hb", frac, len(sub), 2, {"source": "results/pain-tail/episodes.csv", "post_steps": int(post_steps)})


# ---------------------------
# Lesion claims
# ---------------------------

@REGISTRY.register("lesion_auc_drop_recon")
def _lesion_drop_recon(ctx: ClaimsContext) -> Claim:
    sub = ctx.lesion[ctx.lesion["model"] == "recon"]
    sham = sub[sub["condition"] == "sham"].set_index("seed")["post_lesion_auc"]
    les = sub[sub["condition"] == "lesion"].set_index("seed")["post_lesion_auc"]
    common = sham.index.intersection(les.index)
    diffs = (sham.loc[common] - les.loc[common]).values
    return _claim_mean_ci("lesion_auc_drop_recon", diffs, 2, {"source": "results/lesion/episodes_extended.csv"})


@REGISTRY.register("lesion_auc_drop_humphrey")
def _lesion_drop_humphrey(ctx: ClaimsContext) -> Claim:
    sub = ctx.lesion[ctx.lesion["model"] == "humphrey"]
    sham = sub[sub["condition"] == "sham"].set_index("seed")["post_lesion_auc"]
    les = sub[sub["condition"] == "lesion"].set_index("seed")["post_lesion_auc"]
    common = sham.index.intersection(les.index)
    diffs = (sham.loc[common] - les.loc[common]).values
    return _claim_mean_ci("lesion_auc_drop_humphrey", diffs, 2, {"source": "results/lesion/episodes_extended.csv"})


@REGISTRY.register("lesion_auc_drop_hb")
def _lesion_drop_hb(ctx: ClaimsContext) -> Claim:
    sub = ctx.lesion[ctx.lesion["model"] == "humphrey_barrett"]
    sham = sub[sub["condition"] == "sham"].set_index("seed")["post_lesion_auc"]
    les = sub[sub["condition"] == "lesion"].set_index("seed")["post_lesion_auc"]
    common = sham.index.intersection(les.index)
    diffs = (sham.loc[common] - les.loc[common]).values
    return _claim_mean_ci("lesion_auc_drop_hb", diffs, 2, {"source": "results/lesion/episodes_extended.csv"})


@REGISTRY.register("lesion_auc_drop_pct_humphrey")
def _lesion_pct_humphrey(ctx: ClaimsContext) -> Claim:
    sub = ctx.lesion[ctx.lesion["model"] == "humphrey"]
    sham = sub[sub["condition"] == "sham"].set_index("seed")["post_lesion_auc"]
    les = sub[sub["condition"] == "lesion"].set_index("seed")["post_lesion_auc"]
    common = sham.index.intersection(les.index)
    pct = (sham.loc[common] - les.loc[common]) / sham.loc[common] * 100.0
    return _claim_mean_ci("lesion_auc_drop_pct_humphrey", pct.values, 1, {"source": "results/lesion/episodes_extended.csv"})


@REGISTRY.register("lesion_auc_drop_pct_hb")
def _lesion_pct_hb(ctx: ClaimsContext) -> Claim:
    sub = ctx.lesion[ctx.lesion["model"] == "humphrey_barrett"]
    sham = sub[sub["condition"] == "sham"].set_index("seed")["post_lesion_auc"]
    les = sub[sub["condition"] == "lesion"].set_index("seed")["post_lesion_auc"]
    common = sham.index.intersection(les.index)
    pct = (sham.loc[common] - les.loc[common]) / sham.loc[common] * 100.0
    return _claim_mean_ci("lesion_auc_drop_pct_hb", pct.values, 1, {"source": "results/lesion/episodes_extended.csv"})


@REGISTRY.register("lesion_recon_auc_drop_near_zero")
def _lesion_recon_near_zero(ctx: ClaimsContext) -> Claim:
    claim = _lesion_drop_recon(ctx)
    mean = float(claim.value)
    ci_low, ci_high = claim.ci if claim.ci else (float("nan"), float("nan"))
    near_zero = abs(mean) < 1e-3 or (ci_low <= 0.0 <= ci_high)
    return _claim_bool("lesion_recon_auc_drop_near_zero", near_zero, {"source": "results/lesion/episodes_extended.csv", "mean": mean, "ci": claim.ci})


# ---------------------------
# Goal-directed robustness
# ---------------------------

def _as_success_float(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.astype(float).values
    s = series.astype(str).str.lower()
    return s.isin(["true", "1", "yes", "y", "t"]).astype(float).values


def _goal_metric_mean_ci(df: pd.DataFrame, model: str, horizon: int, value_col: str) -> Tuple[float, float, float, int]:
    sub = df[(df["model"] == model) & (df["horizon"] == horizon)].copy()
    if value_col == "success":
        sub[value_col] = _as_success_float(sub[value_col])
    values = _seed_means(sub, value_col)
    return _mean_ci(values)


def _goal_metric_mean_ci_all(df: pd.DataFrame, model: str, value_col: str) -> Tuple[float, float, float, int]:
    sub = df[df["model"] == model].copy()
    if value_col == "success":
        sub[value_col] = _as_success_float(sub[value_col])
    values = _seed_means(sub, value_col)
    return _mean_ci(values)


@REGISTRY.register("goal_corridor_hazards_max_hb")
def _goal_corridor_hazards(ctx: ClaimsContext) -> Claim:
    sub = ctx.goal_corridor[ctx.goal_corridor["model"] == "humphrey_barrett"]
    max_val = float(np.max(sub["mean_hazards"].values)) if len(sub) else float("nan")
    return _claim_scalar("goal_corridor_hazards_max_hb", max_val, len(sub), 2, {"source": "results/goal-directed/corridor_summary.csv"})


@REGISTRY.register("goal_corridor_hazards_zero_hb")
def _goal_corridor_hazards_zero(ctx: ClaimsContext) -> Claim:
    sub = ctx.goal_corridor[ctx.goal_corridor["model"] == "humphrey_barrett"]
    ok = bool(np.allclose(sub["mean_hazards"].values, 0.0)) if len(sub) else False
    return _claim_bool("goal_corridor_hazards_zero_hb", ok, {"source": "results/goal-directed/corridor_summary.csv"})


@REGISTRY.register("goal_directed_seeds")
def _goal_directed_seeds(ctx: ClaimsContext) -> Claim:
    # Derive from actual sweep outputs when available so the paper updates
    # correctly even if only the goal-directed sweep is re-run.
    seeds_c = set(ctx.goal_corridor_episodes["seed"].unique()) if len(ctx.goal_corridor_episodes) else set()
    seeds_g = set(ctx.goal_gridworld_episodes["seed"].unique()) if len(ctx.goal_gridworld_episodes) else set()
    seeds = seeds_c | seeds_g
    if seeds:
        n = len(seeds)
        meta = {
            "source": [
                "results/goal-directed/corridor_episodes.csv",
                "results/goal-directed/gridworld_episodes.csv",
            ],
            "corridor_n_unique_seeds": len(seeds_c),
            "gridworld_n_unique_seeds": len(seeds_g),
        }
        return _claim_scalar("goal_directed_seeds", float(n), n, 0, meta)

    seeds_cfg = ctx.config.get("seeds", {}).get("goal_directed", [])
    n = len(seeds_cfg)
    return _claim_scalar("goal_directed_seeds", float(n), n, 0, {"source": "results/config_metadata.json"})


@REGISTRY.register("goal_directed_horizons")
def _goal_directed_horizons(ctx: ClaimsContext) -> Claim:
    # Like goal_directed_seeds, derive from actual sweep output when possible.
    hs_c = set(ctx.goal_corridor_episodes["horizon"].unique()) if len(ctx.goal_corridor_episodes) else set()
    hs_g = set(ctx.goal_gridworld_episodes["horizon"].unique()) if len(ctx.goal_gridworld_episodes) else set()
    horizons = sorted(hs_c | hs_g)
    if horizons:
        value = ",".join(str(int(h)) for h in horizons)
        return Claim(
            claim_id="goal_directed_horizons",
            value=value,
            ci=None,
            n=len(horizons),
            claim_type="scalar",
            digits=None,
            meta={
                "source": [
                    "results/goal-directed/corridor_episodes.csv",
                    "results/goal-directed/gridworld_episodes.csv",
                ],
                "corridor_unique_horizons": sorted(int(h) for h in hs_c),
                "gridworld_unique_horizons": sorted(int(h) for h in hs_g),
            },
        )

    horizons_cfg = ctx.config.get("horizons", {}).get("goal_directed", [])
    value = ",".join(str(h) for h in horizons_cfg)
    return Claim(claim_id="goal_directed_horizons", value=value, ci=None, n=len(horizons_cfg), claim_type="scalar", digits=None, meta={"source": "results/config_metadata.json"})


@REGISTRY.register("goal_directed_horizon_count")
def _goal_directed_horizon_count(ctx: ClaimsContext) -> Claim:
    hs_c = set(ctx.goal_corridor_episodes["horizon"].unique()) if len(ctx.goal_corridor_episodes) else set()
    hs_g = set(ctx.goal_gridworld_episodes["horizon"].unique()) if len(ctx.goal_gridworld_episodes) else set()
    horizons = sorted(hs_c | hs_g)
    if not horizons:
        horizons = list(ctx.config.get("horizons", {}).get("goal_directed", []))
    n = len(horizons)
    meta = {"source": "results/config_metadata.json"}
    if hs_c or hs_g:
        meta = {
            "source": [
                "results/goal-directed/corridor_episodes.csv",
                "results/goal-directed/gridworld_episodes.csv",
            ],
            "corridor_unique_horizons": sorted(int(h) for h in hs_c),
            "gridworld_unique_horizons": sorted(int(h) for h in hs_g),
        }
    return _claim_scalar("goal_directed_horizon_count", float(n), n, 0, meta)


@REGISTRY.register("goal_directed_paper_horizon")
def _goal_directed_paper_horizon(ctx: ClaimsContext) -> Claim:
    # Prefer an explicit override in config metadata; otherwise fall back to the
    # canonical paper horizon (5) if present in the sweep, or the first horizon.
    paper_cfg = ctx.config.get("paper_horizon") if isinstance(ctx.config, dict) else None
    if isinstance(paper_cfg, dict):
        horizon = paper_cfg.get("goal_directed")
    else:
        horizon = paper_cfg

    horizons = set()
    if len(ctx.goal_corridor_episodes):
        horizons |= set(ctx.goal_corridor_episodes["horizon"].unique())
    if len(ctx.goal_gridworld_episodes):
        horizons |= set(ctx.goal_gridworld_episodes["horizon"].unique())
    if not horizons:
        horizons = set(ctx.config.get("horizons", {}).get("goal_directed", []))

    if horizon is None:
        if 5 in horizons:
            horizon = 5
        elif horizons:
            horizon = sorted(horizons)[0]
        else:
            horizon = 5

    return _claim_scalar(
        "goal_directed_paper_horizon",
        float(horizon),
        None,
        0,
        {"source": "results/config_metadata.json", "definition": "paper rollout horizon for goal-directed table"},
    )


@REGISTRY.register("goal_corridor_h5_hazards_recon")
def _goal_corridor_h5_hazards_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_corridor_episodes, "recon", 5, "hazard_contacts")
    return Claim("goal_corridor_h5_hazards_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_h5_hazards_humphrey")
def _goal_corridor_h5_hazards_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_corridor_episodes, "humphrey", 5, "hazard_contacts")
    return Claim("goal_corridor_h5_hazards_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_h5_hazards_hb")
def _goal_corridor_h5_hazards_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_corridor_episodes, "humphrey_barrett", 5, "hazard_contacts")
    return Claim("goal_corridor_h5_hazards_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_h5_time_recon")
def _goal_corridor_h5_time_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_corridor_episodes, "recon", 5, "time_to_goal")
    return Claim("goal_corridor_h5_time_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_h5_time_humphrey")
def _goal_corridor_h5_time_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_corridor_episodes, "humphrey", 5, "time_to_goal")
    return Claim("goal_corridor_h5_time_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_h5_time_hb")
def _goal_corridor_h5_time_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_corridor_episodes, "humphrey_barrett", 5, "time_to_goal")
    return Claim("goal_corridor_h5_time_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_h5_success_recon")
def _goal_corridor_h5_success_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_corridor_episodes, "recon", 5, "success")
    return Claim("goal_corridor_h5_success_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_h5_success_humphrey")
def _goal_corridor_h5_success_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_corridor_episodes, "humphrey", 5, "success")
    return Claim("goal_corridor_h5_success_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_h5_success_hb")
def _goal_corridor_h5_success_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_corridor_episodes, "humphrey_barrett", 5, "success")
    return Claim("goal_corridor_h5_success_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_gridworld_h5_hazards_recon")
def _goal_gridworld_h5_hazards_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_gridworld_episodes, "recon", 5, "hazard_contacts")
    return Claim("goal_gridworld_h5_hazards_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_h5_hazards_humphrey")
def _goal_gridworld_h5_hazards_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_gridworld_episodes, "humphrey", 5, "hazard_contacts")
    return Claim("goal_gridworld_h5_hazards_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_h5_hazards_hb")
def _goal_gridworld_h5_hazards_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_gridworld_episodes, "humphrey_barrett", 5, "hazard_contacts")
    return Claim("goal_gridworld_h5_hazards_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_h5_time_recon")
def _goal_gridworld_h5_time_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_gridworld_episodes, "recon", 5, "time_to_goal")
    return Claim("goal_gridworld_h5_time_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_h5_time_humphrey")
def _goal_gridworld_h5_time_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_gridworld_episodes, "humphrey", 5, "time_to_goal")
    return Claim("goal_gridworld_h5_time_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_h5_time_hb")
def _goal_gridworld_h5_time_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_gridworld_episodes, "humphrey_barrett", 5, "time_to_goal")
    return Claim("goal_gridworld_h5_time_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_h5_success_recon")
def _goal_gridworld_h5_success_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_gridworld_episodes, "recon", 5, "success")
    return Claim("goal_gridworld_h5_success_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_h5_success_humphrey")
def _goal_gridworld_h5_success_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_gridworld_episodes, "humphrey", 5, "success")
    return Claim("goal_gridworld_h5_success_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_h5_success_hb")
def _goal_gridworld_h5_success_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci(ctx.goal_gridworld_episodes, "humphrey_barrett", 5, "success")
    return Claim("goal_gridworld_h5_success_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_corridor_all_hazards_recon")
def _goal_corridor_all_hazards_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_corridor_episodes, "recon", "hazard_contacts")
    return Claim("goal_corridor_all_hazards_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_all_hazards_humphrey")
def _goal_corridor_all_hazards_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_corridor_episodes, "humphrey", "hazard_contacts")
    return Claim("goal_corridor_all_hazards_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_all_hazards_hb")
def _goal_corridor_all_hazards_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_corridor_episodes, "humphrey_barrett", "hazard_contacts")
    return Claim("goal_corridor_all_hazards_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_all_time_recon")
def _goal_corridor_all_time_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_corridor_episodes, "recon", "time_to_goal")
    return Claim("goal_corridor_all_time_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_all_time_humphrey")
def _goal_corridor_all_time_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_corridor_episodes, "humphrey", "time_to_goal")
    return Claim("goal_corridor_all_time_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_all_time_hb")
def _goal_corridor_all_time_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_corridor_episodes, "humphrey_barrett", "time_to_goal")
    return Claim("goal_corridor_all_time_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_all_success_recon")
def _goal_corridor_all_success_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_corridor_episodes, "recon", "success")
    return Claim("goal_corridor_all_success_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_all_success_humphrey")
def _goal_corridor_all_success_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_corridor_episodes, "humphrey", "success")
    return Claim("goal_corridor_all_success_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_corridor_all_success_hb")
def _goal_corridor_all_success_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_corridor_episodes, "humphrey_barrett", "success")
    return Claim("goal_corridor_all_success_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/corridor_episodes.csv"})


@REGISTRY.register("goal_gridworld_all_hazards_recon")
def _goal_gridworld_all_hazards_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_gridworld_episodes, "recon", "hazard_contacts")
    return Claim("goal_gridworld_all_hazards_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_all_hazards_humphrey")
def _goal_gridworld_all_hazards_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_gridworld_episodes, "humphrey", "hazard_contacts")
    return Claim("goal_gridworld_all_hazards_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_all_hazards_hb")
def _goal_gridworld_all_hazards_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_gridworld_episodes, "humphrey_barrett", "hazard_contacts")
    return Claim("goal_gridworld_all_hazards_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_all_time_recon")
def _goal_gridworld_all_time_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_gridworld_episodes, "recon", "time_to_goal")
    return Claim("goal_gridworld_all_time_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_all_time_humphrey")
def _goal_gridworld_all_time_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_gridworld_episodes, "humphrey", "time_to_goal")
    return Claim("goal_gridworld_all_time_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_all_time_hb")
def _goal_gridworld_all_time_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_gridworld_episodes, "humphrey_barrett", "time_to_goal")
    return Claim("goal_gridworld_all_time_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_all_success_recon")
def _goal_gridworld_all_success_recon(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_gridworld_episodes, "recon", "success")
    return Claim("goal_gridworld_all_success_recon", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_all_success_humphrey")
def _goal_gridworld_all_success_humphrey(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_gridworld_episodes, "humphrey", "success")
    return Claim("goal_gridworld_all_success_humphrey", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})


@REGISTRY.register("goal_gridworld_all_success_hb")
def _goal_gridworld_all_success_hb(ctx: ClaimsContext) -> Claim:
    mean, low, high, n = _goal_metric_mean_ci_all(ctx.goal_gridworld_episodes, "humphrey_barrett", "success")
    return Claim("goal_gridworld_all_success_hb", mean, (low, high), n, "mean_ci", 2, {"source": "results/goal-directed/gridworld_episodes.csv"})

@REGISTRY.register("headline_seeds")
def _headline_seeds(ctx: ClaimsContext) -> Claim:
    seeds = ctx.config.get("seeds", {}).get("familiarity", [])
    n = len(seeds)
    return _claim_scalar("headline_seeds", float(n), n, 0, {"source": "results/config_metadata.json"})


@REGISTRY.register("pain_post_steps")
def _pain_post_steps(ctx: ClaimsContext) -> Claim:
    post_steps_raw = ctx.config.get("window_lengths", {}).get("pain_tail_post_stimulus_steps", np.nan)
    post_steps = float(post_steps_raw) if post_steps_raw is not None else float("nan")
    if not np.isfinite(post_steps):
        post_steps = float(np.nanmax(ctx.pain_tail["ns_half_life"].values))
    return _claim_scalar("pain_post_steps", float(post_steps), None, 0, {"source": "results/config_metadata.json"})


# ---------------------------
# I/O helpers
# ---------------------------


def load_context(results_dir: str) -> ClaimsContext:
    def _canonicalize_models(df: pd.DataFrame) -> pd.DataFrame:
        if "model" not in df.columns:
            return df
        out = df.copy()
        out["model"] = out["model"].map(canonical_model_id)
        return out

    fam_path = os.path.join(results_dir, "familiarity", "episodes_improved.csv")
    play_path = os.path.join(results_dir, "exploratory-play", "final_viewpoints.csv")
    play_clarified_path = _select_play_clarified(os.path.join(results_dir, "exploratory-play"))
    pain_path = os.path.join(results_dir, "pain-tail", "episodes.csv")
    lesion_path = os.path.join(results_dir, "lesion", "episodes_extended.csv")
    goal_corridor_path = os.path.join(results_dir, "goal-directed", "corridor_summary.csv")
    goal_gridworld_path = os.path.join(results_dir, "goal-directed", "gridworld_summary.csv")
    goal_corridor_eps_path = os.path.join(results_dir, "goal-directed", "corridor_episodes.csv")
    goal_gridworld_eps_path = os.path.join(results_dir, "goal-directed", "gridworld_episodes.csv")
    config_path = os.path.join(results_dir, "config_metadata.json")

    ctx = ClaimsContext(
        familiarity=_canonicalize_models(pd.read_csv(fam_path)),
        play=_canonicalize_models(pd.read_csv(play_path)),
        play_clarified=_canonicalize_models(pd.read_csv(play_clarified_path)),
        pain_tail=_canonicalize_models(pd.read_csv(pain_path)),
        lesion=_canonicalize_models(pd.read_csv(lesion_path)),
        goal_corridor=_canonicalize_models(pd.read_csv(goal_corridor_path)),
        goal_gridworld=_canonicalize_models(pd.read_csv(goal_gridworld_path)),
        goal_corridor_episodes=_canonicalize_models(pd.read_csv(goal_corridor_eps_path)),
        goal_gridworld_episodes=_canonicalize_models(pd.read_csv(goal_gridworld_eps_path)),
        config=json.load(open(config_path, "r", encoding="utf-8")) if os.path.exists(config_path) else {},
    )
    ctx.play_clarified.attrs["source"] = play_clarified_path
    return ctx


def format_value(value: Any, digits: Optional[int]) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "nan"
    if isinstance(value, (tuple, list)):
        raise ValueError("format_value expects scalar")
    if digits is None:
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _claim_passes(claim: Claim) -> bool:
    if claim.claim_type == "boolean":
        return bool(claim.value)
    if claim.claim_type == "range":
        if not isinstance(claim.value, (tuple, list)) or len(claim.value) != 2:
            return False
        low, high = claim.value
        if not (_is_finite_number(low) and _is_finite_number(high)):
            return False
        return float(low) <= float(high)
    value_ok = True
    if isinstance(claim.value, (np.floating, np.integer, float, int)):
        value_ok = _is_finite_number(claim.value)
    elif isinstance(claim.value, str):
        value_ok = bool(claim.value.strip())
    elif isinstance(claim.value, (tuple, list)):
        value_ok = len(claim.value) > 0
    elif claim.value is None:
        value_ok = False
    if not value_ok:
        return False
    if claim.ci is None:
        return True
    if len(claim.ci) != 2:
        return False
    low, high = claim.ci
    if not (_is_finite_number(low) and _is_finite_number(high)):
        return False
    return float(low) <= float(high)


def write_claims_tex(claims: Dict[str, Claim], out_path: str) -> None:
    lines = []
    lines.append("% Auto-generated claims macros")
    lines.append("\\providecommand{\\Claim}[1]{\\csname claim@#1\\endcsname}")
    for cid, claim in claims.items():
        if claim.claim_type == "range" and isinstance(claim.value, (tuple, list)):
            low, high = claim.value
            lines.append(f"\\expandafter\\newcommand\\csname claim@{cid}_low\\endcsname{{{format_value(low, claim.digits)}}}")
            lines.append(f"\\expandafter\\newcommand\\csname claim@{cid}_high\\endcsname{{{format_value(high, claim.digits)}}}")
            continue
        lines.append(f"\\expandafter\\newcommand\\csname claim@{cid}\\endcsname{{{format_value(claim.value, claim.digits)}}}")
        if claim.ci is not None:
            low, high = claim.ci
            lines.append(f"\\expandafter\\newcommand\\csname claim@{cid}_ci_low\\endcsname{{{format_value(low, claim.digits)}}}")
            lines.append(f"\\expandafter\\newcommand\\csname claim@{cid}_ci_high\\endcsname{{{format_value(high, claim.digits)}}}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_claims_json(claims: Dict[str, Claim], out_path: str) -> None:
    def serialize(val: Any) -> Any:
        if isinstance(val, (np.floating, np.integer)):
            return float(val)
        if isinstance(val, (tuple, list)):
            return [serialize(v) for v in val]
        if isinstance(val, bool):
            return bool(val)
        return val

    payload = {}
    for cid, claim in claims.items():
        payload[cid] = {
            "value": serialize(claim.value),
            "ci": serialize(list(claim.ci)) if claim.ci is not None else None,
            "n": claim.n,
            "type": claim.claim_type,
            "digits": claim.digits,
            "meta": claim.meta or {},
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_claims_md(claims: Dict[str, Claim], out_path: str) -> None:
    lines = ["# Claims report", "", "| Claim ID | Value | CI | n | PASS/FAIL |", "|---|---|---|---:|:---:|"]
    for cid, claim in claims.items():
        if claim.claim_type == "range" and isinstance(claim.value, (tuple, list)):
            low, high = claim.value
            val = f"{format_value(low, claim.digits)}--{format_value(high, claim.digits)}"
            ci_text = ""
        else:
            val = format_value(claim.value, claim.digits)
            if claim.ci is not None:
                ci_text = f"[{format_value(claim.ci[0], claim.digits)}, {format_value(claim.ci[1], claim.digits)}]"
            else:
                ci_text = ""
        pass_fail = "PASS" if _claim_passes(claim) else "FAIL"
        n_text = str(claim.n) if claim.n is not None else ""
        lines.append(f"| {cid} | {val} | {ci_text} | {n_text} | {pass_fail} |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def check_macro_coverage(tex_path: str, claims: Dict[str, Claim]) -> None:
    with open(tex_path, "r", encoding="utf-8") as f:
        tex = f.read()
    used = []
    start = 0
    while True:
        idx = tex.find("\\Claim{", start)
        if idx == -1:
            break
        end = tex.find("}", idx)
        if end == -1:
            break
        used.append(tex[idx + len("\\Claim{"):end])
        start = end + 1
    missing = [cid for cid in used if cid not in claims and not cid.endswith("_ci_low") and not cid.endswith("_ci_high") and not cid.endswith("_low") and not cid.endswith("_high")]
    if missing:
        raise ValueError(f"Missing claim IDs in registry: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paper claims")
    parser.add_argument("--outdir", type=str, default="results/paper", help="Output directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--tex", type=str, default="docs/paper-3-v7.tex", help="Paper tex for macro coverage")
    args = parser.parse_args()

    ctx = load_context(args.results_dir)
    claims = REGISTRY.compute(ctx)

    os.makedirs(args.outdir, exist_ok=True)
    write_claims_json(claims, os.path.join(args.outdir, "claims.json"))
    write_claims_tex(claims, os.path.join(args.outdir, "claims.tex"))
    write_claims_md(claims, os.path.join(args.outdir, "claims.md"))

    if os.path.exists(args.tex):
        check_macro_coverage(args.tex, claims)


if __name__ == "__main__":
    main()
