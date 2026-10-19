"""Closed-loop route following: vehicle model + L1 guidance + wind.

`follow_route` flies a polyline the way ArduPlane flies a mission: the
active leg is A -> B, and the next leg starts when the aircraft is within
the L1 turn distance of B or has passed it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .config import Config
from .guidance import L1Guidance
from .sensors import TrueGroundVelocity, VelocitySource
from .simplify import point_segment_distance
from .vehicle import FixedWing, VehicleState
from .world import World

WindFn = Callable[[float], np.ndarray]


@dataclass
class SimLog:
    t: np.ndarray
    pos: np.ndarray            # (N, 2) east, north
    psi: np.ndarray
    phi: np.ndarray            # actual roll
    phi_cmd: np.ndarray        # commanded roll
    v: np.ndarray              # airspeed
    xtrack_right: np.ndarray   # signed, w.r.t. the active leg, positive = right
    leg: np.ndarray            # index of the active leg
    wind: np.ndarray           # (N, 2) true wind
    vel_used: np.ndarray       # (N, 2) ground velocity the guidance was given
    finished: bool

    def route_distance(self, route) -> np.ndarray:
        """Unsigned distance from every logged position to the whole route polyline."""
        route = np.asarray(route, dtype=float)
        d = np.full(len(self.pos), np.inf)
        for a, b in zip(route[:-1], route[1:]):
            d = np.minimum(d, point_segment_distance(self.pos, a, b))
        return d

    def obstacle_clearance(self, world: World) -> np.ndarray:
        if not world.obstacles:
            return np.full(len(self.pos), np.inf)
        c = np.array([(o.x, o.y) for o in world.obstacles])
        r = np.array([o.r for o in world.obstacles])
        d = np.hypot(self.pos[:, None, 0] - c[None, :, 0], self.pos[:, None, 1] - c[None, :, 1])
        return np.min(d - r[None, :], axis=1)


def constant_wind(vec) -> WindFn:
    w = np.asarray(vec, dtype=float)
    return lambda t: w


def _heading_of(a, b) -> float:
    return math.atan2(b[0] - a[0], b[1] - a[1])   # clockwise from north


def _turn_angle_deg(a, b, c) -> float:
    h1, h2 = _heading_of(a, b), _heading_of(b, c)
    return math.degrees((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)


def follow_route(
    route,
    cfg: Config | None = None,
    wind: WindFn | None = None,
    source: VelocitySource | None = None,
    start: VehicleState | None = None,
    t_max: float | None = None,
    seed: int = 0,
    stop_at_end: bool = True,
) -> SimLog:
    """Fly `route` (N >= 2 waypoints, east/north metres) and log everything."""
    cfg = cfg or Config()
    route = np.asarray(route, dtype=float)
    if len(route) < 2:
        raise ValueError("route needs at least two waypoints")
    wind = wind or constant_wind((0.0, 0.0))
    source = source or TrueGroundVelocity()
    rng = np.random.default_rng(seed)
    source.reset(rng)

    dt = cfg.sim.dt
    t_max = cfg.sim.t_max if t_max is None else t_max
    model = FixedWing(cfg.vehicle)
    guid = L1Guidance(cfg.guidance, cfg.vehicle)
    v_cmd = cfg.vehicle.v_cruise
    s = start or VehicleState(x=route[0, 0], y=route[0, 1],
                              psi=_heading_of(route[0], route[1]), v=v_cmd)

    n = int(round(t_max / dt)) + 1
    T = np.empty(n); P = np.empty((n, 2)); PSI = np.empty(n); PHI = np.empty(n)
    PHIC = np.empty(n); V = np.empty(n); XT = np.empty(n); LEG = np.empty(n, dtype=int)
    W = np.empty((n, 2)); VU = np.empty((n, 2))

    leg, last = 0, len(route) - 2
    finished = False
    k = 0
    for k in range(n):
        t = k * dt
        w = np.asarray(wind(t), dtype=float)
        vel = source.ground_velocity(t, s, w)
        a, b = route[leg], route[leg + 1]
        out = guid.update_waypoint(s.pos, vel, a, b, dt)

        T[k] = t; P[k] = (s.x, s.y); PSI[k] = s.psi; PHI[k] = s.phi; PHIC[k] = out.roll_cmd
        V[k] = s.v; XT[k] = out.xtrack_right; LEG[k] = leg; W[k] = w; VU[k] = vel

        # leg switching, ArduPlane style: inside the turn distance, or past B
        gs = float(np.hypot(*vel))
        ab = b - a
        along = float((s.pos - a) @ ab) / max(float(np.hypot(*ab)), 1e-9)
        dist_b = float(np.hypot(*(b - s.pos)))
        if leg < last:
            turn = _turn_angle_deg(a, b, route[leg + 2])
            if dist_b <= guid.turn_distance(cfg.guidance.wp_accept_radius, turn, gs) \
                    or along >= float(np.hypot(*ab)):
                leg += 1
        elif stop_at_end and (dist_b <= cfg.guidance.wp_accept_radius or along >= float(np.hypot(*ab))):
            finished = True
            break

        s = model.step(s, out.roll_cmd, v_cmd, w, dt)

    m = k + 1
    return SimLog(t=T[:m], pos=P[:m], psi=PSI[:m], phi=PHI[:m], phi_cmd=PHIC[:m], v=V[:m],
                  xtrack_right=XT[:m], leg=LEG[:m], wind=W[:m], vel_used=VU[:m], finished=finished)


def straight_line_test(offset_right: float, cfg: Config | None = None, wind: WindFn | None = None,
                       source: VelocitySource | None = None, v: float | None = None,
                       t_max: float = 120.0, seed: int = 0) -> SimLog:
    """Track due north from (0, 0); the aircraft starts `offset_right` m east of it, heading north."""
    cfg = cfg or Config()
    v = cfg.vehicle.v_cruise if v is None else v
    route = np.array([[0.0, 0.0], [0.0, 10_000.0]])
    start = VehicleState(x=offset_right, y=0.0, psi=0.0, v=v)
    if v != cfg.vehicle.v_cruise:
        cfg = replace(cfg, vehicle=replace(cfg.vehicle, v_cruise=v))
    return follow_route(route, cfg, wind=wind, source=source, start=start, t_max=t_max,
                        seed=seed, stop_at_end=False)
