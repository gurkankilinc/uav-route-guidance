"""Look-ahead guard against flying into a no-fly zone.

The planners work on a map with a fixed safety margin and know nothing about
how tightly the aircraft can turn. When a route is replanned from the
current position it can start with a sharp turn right next to a zone, and
the aircraft sweeps through the zone because it needs a 40-100 m turn radius.

The guard rolls the aircraft's own turn dynamics forward (constant bank for a
few seconds). If the bank the guidance asks for would take it closer than
`margin` to a zone, it flies the maximum bank to whichever side keeps it
clear. The border of the area is treated like a zone edge. It sits below the
planner and is the same for every planner.
"""

from __future__ import annotations

import math

import numpy as np

from .config import G, VehicleParams


def rollout(x: float, y: float, psi: float, phi0: float, v: float, phi_cmd: float,
            wind, veh: VehicleParams, horizon: float = 6.0, dt: float = 0.25) -> np.ndarray:
    """Positions after `horizon` s of holding bank `phi_cmd` (roll response modelled as first order)."""
    n = int(round(horizon / dt))
    out = np.empty((n, 2))
    wx, wy = float(wind[0]), float(wind[1])
    for i in range(n):
        phi = phi_cmd + (phi0 - phi_cmd) * math.exp(-(i * dt) / veh.roll_tau)
        psi += G * math.tan(phi) / v * dt
        x += (v * math.sin(psi) + wx) * dt
        y += (v * math.cos(psi) + wy) * dt
        out[i] = (x, y)
    return out


def path_clearance(path: np.ndarray, C: np.ndarray, R: np.ndarray, bounds=None) -> float:
    """Smallest distance from the path to a zone edge or, if `bounds` is given, to the area border."""
    best = math.inf
    if len(R):
        d = np.hypot(path[:, None, 0] - C[None, :, 0], path[:, None, 1] - C[None, :, 1]) - R[None, :]
        best = float(d.min())
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        best = min(best, float(min((path[:, 0] - x0).min(), (x1 - path[:, 0]).min(),
                                   (path[:, 1] - y0).min(), (y1 - path[:, 1]).min())))
    return best


class ZoneGuard:
    def __init__(self, veh: VehicleParams, bounds=None, margin: float = 3.0, release: float = 12.0, horizon: float = 6.0):
        self.veh, self.bounds, self.margin, self.release, self.horizon = veh, bounds, margin, release, horizon
        self.side = 0.0          # +1 / -1 while overriding, 0 when idle

    def reset(self) -> None:
        self.side = 0.0

    def __call__(self, s, wind, phi_cmd: float, C: np.ndarray, R: np.ndarray) -> tuple[float, bool]:
        """Returns (bank command to use, True if the guard overrode the guidance)."""
        if len(R) == 0 and self.bounds is None:
            return phi_cmd, False

        def clr(cmd: float) -> float:
            return path_clearance(rollout(s.x, s.y, s.psi, s.phi, s.v, cmd, wind, self.veh, self.horizon), C, R, self.bounds)

        c0 = clr(phi_cmd)
        if self.side == 0.0 and c0 >= self.margin:
            return phi_cmd, False
        if self.side != 0.0 and c0 >= self.margin + self.release:
            self.side = 0.0
            return phi_cmd, False
        pm = self.veh.roll_max
        if self.side != 0.0 and clr(self.side * pm) >= self.margin:
            return self.side * pm, True
        cr, cl = clr(pm), clr(-pm)
        best_side, best = (1.0, cr) if cr >= cl else (-1.0, cl)
        if best > c0:
            self.side = best_side
            return best_side * pm, True
        self.side = 0.0
        return phi_cmd, False
