from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def bootstrap_ci(values: Iterable[float], n_boot: int = 1000, seed: int = 0) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots.append(float(np.mean(sample)))
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    xa = np.asarray(list(a), dtype=float)
    xb = np.asarray(list(b), dtype=float)
    if xa.size == 0 or xb.size == 0:
        return float("nan")
    wins = 0
    losses = 0
    for av in xa:
        wins += int(np.sum(av > xb))
        losses += int(np.sum(av < xb))
    return float((wins - losses) / (xa.size * xb.size))


def summarize_by_condition(df: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby([c for c in ("env_name", "condition", "lesion_mode") if c in df.columns], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {}
        group_cols = [c for c in ("env_name", "condition", "lesion_mode") if c in df.columns]
        for col, value in zip(group_cols, keys):
            row[col] = value
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(grp[metric].mean())
            lo, hi = bootstrap_ci(grp[metric])
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)
