from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    k /= max(1e-12, k.sum())
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


class CorridorWorld:
    """
    A corridor with a mid-way hazardous "puddle" that can be avoided by shifting lanes.
    - touch: hazard[y,x]
    - smell: gaussian-blurred hazard minus beauty odor
    - vision: cone-aggregated hazard minus beauty
    """
    def __init__(self, H: int = 18, W: int = 18, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.H, self.W = H, W
        self.hazard_penalty = 0.10

        # --- geometry: a vertical corridor centered in the grid ---
        self.blocked = np.ones((H, W), dtype=bool)
        x0, x1 = 6, 12  # open corridor columns [6..11]
        self.blocked[:, x0:x1] = False

        # --- fields ---
        self.hazard = np.zeros((H, W), dtype=float)
        self.beauty = np.zeros((H, W), dtype=float)

        # goal (bright region)
        self.goal_y, self.goal_x = H - 2, (x0 + x1) // 2  # near bottom center

        # beauty: gaussian hill around goal + mild noise (only in corridor)
        sigma_goal = 3.0
        for y in range(H):
            for x in range(W):
                if self.blocked[y, x]:
                    continue
                d = math.sqrt((y - self.goal_y) ** 2 + (x - self.goal_x) ** 2)
                self.beauty[y, x] = math.exp(-0.5 * (d / sigma_goal) ** 2)

        self.beauty += (rng.random((H, W)) * 0.05) * (~self.blocked)
        self.beauty = np.clip(self.beauty, 0.0, 1.0)

        # hazard "puddle" around the middle, center lanes (avoidable on left/right edge lanes)
        mid = H // 2
        for y in range(mid - 1, mid + 2):          # 3 rows
            for x in range(x0 + 1, x1 - 1):        # interior cols (leave edge lanes safe)
                if not self.blocked[y, x]:
                    self.hazard[y, x] = 1.0

        # OPTIONAL: "temptation" bump of beauty *near* the hazard side
        # (makes the trade-off sharper: it looks good but hurts)
        tempt_cy, tempt_cx = mid, x1 - 2
        for y in range(H):
            for x in range(W):
                if self.blocked[y, x]:
                    continue
                d = math.sqrt((y - tempt_cy) ** 2 + (x - tempt_cx) ** 2)
                self.beauty[y, x] += 0.55 * math.exp(-0.5 * (d / 2.0) ** 2)
        self.beauty = np.clip(self.beauty, 0.0, 1.0)

        # smell fields (same mechanism as gridworld)
        K = gaussian_kernel(radius=3, sigma=1.2)
        self.smell_h = conv2_same(self.hazard, K)
        self.smell_b = conv2_same(self.beauty, K)

    def in_bounds(self, y: int, x: int) -> bool:
        return (0 <= y < self.H) and (0 <= x < self.W)

    def is_free(self, y: int, x: int) -> bool:
        return self.in_bounds(y, x) and (not self.blocked[y, x])

    def step(self, y: int, x: int, action: str, heading: int) -> Tuple[int, int, int]:
        # heading: 0 up,1 right,2 down,3 left
        if action == "turn_left":
            heading = (heading - 1) % 4
        elif action == "turn_right":
            heading = (heading + 1) % 4
        elif action == "forward":
            dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][heading]
            ny, nx = y + dy, x + dx
            if self.is_free(ny, nx):
                y, x = ny, nx
        elif action == "stay":
            pass
        else:
            raise ValueError(action)
        return y, x, heading

    def touch(self, y: int, x: int) -> float:
        return float(self.hazard[y, x]) if self.is_free(y, x) else 0.0

    def smell(self, y: int, x: int) -> float:
        if not self.is_free(y, x):
            return 0.0
        # signed: hazard odor - beauty odor
        return float(self.smell_h[y, x] - 0.6 * self.smell_b[y, x])

    def smell_components(self, y: int, x: int) -> Tuple[float, float]:
        if not self.is_free(y, x):
            return 0.0, 0.0
        return float(self.smell_h[y, x]), float(self.smell_b[y, x])

    def vision_cone_features(self, y: int, x: int, heading: int, radius: int = 5, fov_deg: int = 70) -> Tuple[float, float]:
        """
        Returns (vision_hazard, vision_beauty) aggregated over a cone ahead.
        We ignore blocked cells (walls don't contribute to vision features).
        """
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
                    if not self.is_free(ny, nx):
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
