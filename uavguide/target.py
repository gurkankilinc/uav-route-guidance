"""8.2 - moving target and the approach speed profile.

The target flies at constant ground speed on one of three paths:

    line    straight; near a wall it turns away with radius `turn_radius`, like
            a real aircraft would (so it never leaves, and never reverses at once)
    circle  clockwise circle of `circle_radius` around `start`
    fixed   does not move (a plain waypoint; turns the pursuit into a goto)

The pursuer's speed is a function of the distance d to the target:

    v(d) = v_t + (v_max - v_t) * tanh((d - d_follow) / d0)

far away it flies at (almost) v_max, at d = d_follow it matches the target
speed, closer than that it slows down further. The value is a *ground* speed;
`pursuit.py` converts it to the airspeed command using the wind the guidance
believes in.
"""

from __future__ import annotations

import math

import numpy as np

from .config import ApproachParams, TargetParams, VehicleParams


class MovingTarget:
    def __init__(
        self,
        params: TargetParams,
        bounds: tuple[float, float, float, float],
        start=(300.0, 300.0),
        heading_deg: float = 45.0,
        margin: float = 20.0,
        turn_radius: float = 60.0,
        horizon: float = 600.0,
        dt: float = 0.05,
    ):
        if params.mode not in ("line", "circle", "fixed"):
            raise ValueError(f"unknown target mode {params.mode!r}")
        self.p = params
        x0, y0, x1, y1 = bounds
        self.lo = np.array([x0 + margin, y0 + margin])
        self.hi = np.array([x1 - margin, y1 - margin])
        self.start = (float(start[0]), float(start[1]))
        self.heading = math.radians(heading_deg)
        self.turn_radius = float(turn_radius)
        self._dt = dt
        self._track = self._integrate_line(horizon, dt) if params.mode == "line" else None

    @property
    def speed(self) -> float:
        return 0.0 if self.p.mode == "fixed" else self.p.speed

    def _integrate_line(self, horizon: float, dt: float):
        """Straight flight; when the point `2 turn_radius + 20` m ahead would leave the box,
        turn towards the centre at v / turn_radius until the heading no longer points at the wall."""
        v, r = self.p.speed, self.turn_radius
        centre = 0.5 * (self.lo + self.hi)
        n = int(math.ceil(horizon / dt)) + 2
        pos = np.empty((n, 2)); vel = np.empty((n, 2))
        x, y, psi = self.start[0], self.start[1], self.heading
        turning = 0.0
        look = 2.0 * r + 20.0
        for i in range(n):
            pos[i] = (x, y)
            vel[i] = (v * math.sin(psi), v * math.cos(psi))
            ahead = np.array([x + look * math.sin(psi), y + look * math.cos(psi)])
            outside = bool(np.any(ahead < self.lo) or np.any(ahead > self.hi))
            if outside and turning == 0.0:
                to_c = centre - np.array([x, y])
                side = math.sin(psi) * to_c[1] - math.cos(psi) * to_c[0]   # >0: centre is to the left
                turning = -1.0 if side >= 0.0 else 1.0           # +1 = right turn (clockwise)
            elif turning != 0.0 and not outside:
                # keep turning until the heading no longer points towards a nearby wall
                v_now = np.array([vel[i, 0], vel[i, 1]])
                near = (np.array([x, y]) < self.lo + look) & (v_now < -0.1 * v) \
                    | (np.array([x, y]) > self.hi - look) & (v_now > 0.1 * v)
                if not near.any():
                    turning = 0.0
            psi += turning * v / r * dt
            x += v * math.sin(psi) * dt
            y += v * math.cos(psi) * dt
        return pos, vel

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity) in (east, north) at time t."""
        mode = self.p.mode
        if mode == "fixed":
            return np.array(self.start), np.zeros(2)
        v = self.p.speed
        if mode == "line":
            pos, vel = self._track
            f = min(max(t, 0.0) / self._dt, len(pos) - 2.0)
            i = int(f)
            w = f - i
            return (1 - w) * pos[i] + w * pos[i + 1], (1 - w) * vel[i] + w * vel[i + 1]
        r = self.p.circle_radius
        th = self.heading + v / r * t               # clockwise: theta grows
        pos = np.array([self.start[0] + r * math.sin(th), self.start[1] + r * math.cos(th)])
        vel = np.array([v * math.cos(th), -v * math.sin(th)])
        return pos, vel


def approach_speed(d: float, v_target: float, vehicle: VehicleParams, approach: ApproachParams) -> float:
    """Desired ground speed of the pursuer at distance d (not yet clipped to the flight envelope)."""
    v_t = max(v_target, vehicle.v_stall)             # a fixed goal still needs flying speed
    return v_t + (vehicle.v_max - v_t) * math.tanh((d - approach.d_follow) / approach.d0)
