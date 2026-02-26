from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
import numpy as np

from .recon_network import Network


@dataclass
class Trace:
    ticks: List[int]
    Ns: List[float]
    Ne: List[float]
    P_state: List[str]


def run_episode(net: Network, stimulus: Callable[[int], float], steps: int, seed: int = 0) -> Trace:
    rng = np.random.default_rng(seed)
    ticks, ns, ne, ps = [], [], [], []

    for _ in range(steps):
        t = net.tick
        I_ext = float(stimulus(t))

        if hasattr(net, "_update_ipsundrum_sensor"):
            net._update_ipsundrum_sensor(I_ext, rng=rng)  # type: ignore[attr-defined]
        else:
            net.set_sensor_value("Ns", I_ext)

        net.step()

        ns.append(float(net.get("Ns").activation))
        ne.append(float(net.get("Ne").activation) if "Ne" in net.nodes else 0.0)
        ps.append(net.get("P").script_state.name if "P" in net.nodes else net.get("R").script_state.name)
        ticks.append(net.tick)

    return Trace(ticks=ticks, Ns=ns, Ne=ne, P_state=ps)


def phenomenal_duration(trace: Trace, threshold: float = 0.5) -> int:
    return int(sum(1 for x in trace.Ns if x >= threshold))


def recurrence_peaks(trace: Trace, threshold: float = 0.5) -> int:
    count, above = 0, False
    for x in trace.Ns:
        if x >= threshold and not above:
            count += 1
            above = True
        if x < threshold:
            above = False
    return count


def spectrum_power(signal: List[float]) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    fft = np.fft.rfft(x)
    return (fft.real**2 + fft.imag**2)


def signature(trace: Trace) -> np.ndarray:
    x = np.asarray(trace.Ns, dtype=float)
    mean = float(x.mean())
    std = float(x.std())
    dur = float(phenomenal_duration(trace, threshold=0.5))

    p = spectrum_power(trace.Ns)[1:]  # drop DC
    psum = float(p.sum())
    centroid = 0.0 if psum <= 0 else float(((p / psum) * np.arange(len(p))).sum())

    return np.array([mean, std, dur, centroid], dtype=float)
