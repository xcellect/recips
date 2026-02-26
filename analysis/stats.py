"""Shared statistical utilities for paper claims and figures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap_mean_ci(values: Iterable[float], n_boot: int = 2000, ci: float = 95, seed: int = 0) -> BootstrapResult:
    """Bootstrap CI for the mean over independent seed-level values."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = np.mean(sample)
    alpha = (100 - ci) / 2.0
    low = float(np.percentile(means, alpha))
    high = float(np.percentile(means, 100 - alpha))
    return BootstrapResult(float(np.mean(arr)), low, high, int(arr.size))


def format_ci(result: BootstrapResult, digits: int = 2) -> Tuple[str, str, str]:
    fmt = f"{{:.{digits}f}}"
    return fmt.format(result.mean), fmt.format(result.ci_low), fmt.format(result.ci_high)
