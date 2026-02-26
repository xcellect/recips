from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return k


def conv2_same(A: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = A.shape
    r = K.shape[0] // 2
    Ap = np.pad(A, ((r, r), (r, r)), mode="edge")
    out = np.zeros_like(A, dtype=float)
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(Ap[y:y + 2 * r + 1, x:x + 2 * r + 1] * K)
    return out


class GridWorld:
    def __init__(self, H: int = 18, W: int = 18, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.H, self.W = H, W
        self.hazard_penalty = 0.10

        # hazards sparse binary
        self.hazard = (rng.random((H, W)) < 0.08).astype(float)

        # beauty as smooth hills
        beauty = rng.random((H, W)) * 0.25
        cy, cx = H // 3, W // 3
        for y in range(H):
            for x in range(W):
                d = math.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                beauty[y, x] += math.exp(-0.5 * (d / 2.2) ** 2) * 0.9
        self.beauty = np.clip(beauty, 0, 1)

        # smell fields
        K = gaussian_kernel(radius=3, sigma=1.2)
        self.smell_h = conv2_same(self.hazard, K)   # hazard odor
        self.smell_b = conv2_same(self.beauty, K)   # beauty odor

        # neutral texture (assay-only; not used in affect)
        self.texture = rng.random((H, W))
        self.texture_blur = conv2_same(self.texture, K)
        self.bump_penalty = 0.20

    def in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.H and 0 <= x < self.W

    def step(self, y: int, x: int, action: str, heading: int) -> Tuple[int, int, int]:
        # heading: 0 up,1 right,2 down,3 left
        if action == "turn_left":
            heading = (heading - 1) % 4
        elif action == "turn_right":
            heading = (heading + 1) % 4
        elif action == "forward":
            dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][heading]
            ny, nx = y + dy, x + dx
            if self.in_bounds(ny, nx):
                y, x = ny, nx
        elif action == "stay":
            pass
        else:
            raise ValueError(action)
        return y, x, heading

    def touch(self, y: int, x: int) -> float:
        return float(self.hazard[y, x])

    def smell(self, y: int, x: int) -> float:
        # signed: hazard odor - beauty odor
        return float(self.smell_h[y, x] - 0.6 * self.smell_b[y, x])

    def smell_components(self, y: int, x: int) -> Tuple[float, float]:
        return float(self.smell_h[y, x]), float(self.smell_b[y, x])

    def vision_cone_features(self, y: int, x: int, heading: int, radius: int = 5, fov_deg: int = 70) -> Tuple[float, float]:
        dy0, dx0 = [(-1, 0), (0, 1), (1, 0), (0, -1)][heading]
        ang0 = math.atan2(dy0, dx0)
        fov = math.radians(fov_deg)

        hz_sum = 0.0
        bt_sum = 0.0
        w_sum = 0.0

        for rr in range(1, radius + 1):
            for yy in range(-rr, rr + 1):
                for xx in range(-rr, rr + 1):
                    ny, nx = y + yy, x + xx
                    if not self.in_bounds(ny, nx):
                        continue
                    d = math.sqrt(xx * xx + yy * yy)
                    if d < 1e-9 or d > rr:
                        continue
                    ang = math.atan2(yy, xx)
                    da = (ang - ang0 + math.pi) % (2 * math.pi) - math.pi
                    if abs(da) <= 0.5 * fov and d <= radius:
                        w = 1.0 / (d + 1e-6)
                        hz_sum += w * self.hazard[ny, nx]
                        bt_sum += w * self.beauty[ny, nx]
                        w_sum += w

        if w_sum <= 1e-12:
            return 0.0, 0.0
        return hz_sum / w_sum, bt_sum / w_sum
