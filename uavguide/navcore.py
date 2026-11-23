"""The navigation core: everything that decides where the aircraft goes, with no simulation in it.

One call per control tick:

    cmd = navigator.update(state, target, zones, dt)

`state` is what an autopilot reports about the aircraft, `target` and `zones` are what the mission
knows, `cmd` is a bank angle and an airspeed. Nothing else crosses the boundary, so the same
`Navigator` can be driven by the built-in simulation (`pursuit.py`), by ArduPlane SITL through
MAVLink, or by a companion computer on the real aircraft.

Inside, four pieces run in a fixed order:

  1. aim point: `d_follow` behind the target (the target itself if it stands still),
  2. steering, one of two kinds:
       - route: plan with A* / Theta* (+ Douglas-Peucker), replan periodically or when the known
         zones change, follow the route with the L1 guidance; the last waypoint tracks the aim,
       - field: a potential field gives the course and the bank directly, no route,
  3. speed: the approach profile v(d), corrected for the wind the aircraft believes in, and
     slowed down for a tight turn when the target is close but not ahead,
  4. guard (optional): the look-ahead check of `safety.py` may override the bank.

The wind is not an input. The core derives it as ground velocity minus air velocity, which is all
an aircraft without a wind sensor can do; a caller with a better estimate can pass it in.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .config import Config
from .guidance import L1Guidance
from .planner import GridMap, PlanningError
from .planners import PLANNERS, free_point
from .potential import FIELD_METHODS, PotentialField
from .safety import ZoneGuard
from .target import approach_speed
from .world import Obstacle, World

# every method the core can run: route planners (L1 flies the route) and potential fields (no route)
METHODS = list(PLANNERS) + list(FIELD_METHODS)


@dataclass(frozen=True)
class NavState:
    """The aircraft as its autopilot reports it. Position east/north in m, angles in rad."""

    t: float
    x: float
    y: float
    psi: float                 # heading, clockwise from north
    phi: float                 # bank
    v: float                   # airspeed
    ground_vel: np.ndarray     # (east, north) m/s, e.g. GNSS velocity

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def air_velocity(self) -> np.ndarray:
        return self.v * np.array([math.sin(self.psi), math.cos(self.psi)])


@dataclass(frozen=True)
class NavTarget:
    pos: np.ndarray
    vel: np.ndarray
    speed: float | None = None     # nominal speed if the caller knows it; default |vel|


@dataclass(frozen=True)
class NavZones:
    """The no-fly zones the aircraft currently knows about, and the border of the area."""

    centres: np.ndarray        # (N, 2)
    radii: np.ndarray          # (N,)
    bounds: tuple[float, float, float, float]


@dataclass
class ReplanEvent:
    t: float
    reason: str                # "initial" | "periodic" | "detected"; "field" = a potential-field snapshot
                               # (route = where the field points, plan_ms = mean field time per tick)
    ok: bool
    route: np.ndarray
    plan_ms: float
    expanded: int


@dataclass
class NavCommand:
    roll: float                # rad, bank command
    airspeed: float            # m/s
    xtrack_right: float        # m, signed cross-track error on the active leg (nan for a field)
    guard_active: bool         # the look-ahead guard overrode the steering this tick
    event: ReplanEvent | None  # set on the ticks where a route was planned / a field snapshot was taken


class Navigator:
    def __init__(self, method: str, cfg: Config | None = None, replan_interval: float | None = 1.0,
                 guard: bool = True, plan_latency: float = 0.0):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        self.method = method
        self.cfg = cfg or Config()
        self.replan_interval = replan_interval
        self.use_guard = guard
        self.plan_latency = plan_latency   # s between asking for a route and being able to fly it
        self._field_params = FIELD_METHODS.get(method)
        self._plan = PLANNERS.get(method)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        cfg = self.cfg
        self._l1 = L1Guidance(cfg.guidance, cfg.vehicle)
        self._guard: ZoneGuard | None = None
        self._field: PotentialField | None = None
        self._tick = 0
        self._guard_roll, self._guard_on = 0.0, False
        self._zones_key: tuple | None = None
        self._world = World((0.0, 0.0, 1.0, 1.0), [])
        self._grid: GridMap | None = None
        self.route: np.ndarray | None = None
        self.leg = 0
        self._last_plan_t = -math.inf
        self._pending: tuple[float, np.ndarray] | None = None   # (time it becomes usable, route)
        self._field_ms, self._field_n = 0.0, 0

    def _sync_zones(self, zones: NavZones) -> bool:
        """Rebuild the planning map when the known zones changed. True if they did (and it is not the first call)."""
        key = (zones.bounds, zones.centres.tobytes(), zones.radii.tobytes())
        if key == self._zones_key:
            return False
        first = self._zones_key is None
        self._zones_key = key
        pp = self.cfg.planner
        if self._plan is not None:
            self._world = World(zones.bounds, [Obstacle(float(c[0]), float(c[1]), float(r))
                                               for c, r in zip(zones.centres, zones.radii)])
            self._grid = GridMap(self._world, pp.grid_res, pp.safety_margin)
        if self._field_params is not None and (self._field is None or self._field.bounds != zones.bounds):
            self._field = PotentialField(self._field_params, self.cfg.vehicle, pp.safety_margin, zones.bounds)
        g = self.cfg.guard
        if self._guard is None or self._guard.bounds != zones.bounds:
            self._guard = ZoneGuard(self.cfg.vehicle, bounds=zones.bounds, margin=g.margin, release=g.release,
                                    horizon=g.horizon)
        return not first

    # ------------------------------------------------------------------ steering
    def _steer_field(self, st: NavState, aim, zones: NavZones, changed: bool):
        t0 = time.perf_counter()
        roll = self._field.roll_command(st, st.ground_vel, aim, zones.centres, zones.radii)
        self._field_ms += (time.perf_counter() - t0) * 1e3
        self._field_n += 1
        event = None
        show_every = self.replan_interval or 2.0       # how often a field snapshot is logged for display
        if changed or st.t - self._last_plan_t >= show_every - 1e-9:
            course = math.atan2(st.ground_vel[0], st.ground_vel[1])
            line = self._field.streamline(st.pos, aim, course, zones.centres, zones.radii, v=st.v)
            event = ReplanEvent(st.t, "field", True, line, self._field_ms / max(self._field_n, 1), 0)
            self._field_ms, self._field_n, self._last_plan_t = 0.0, 0, st.t
        return roll, math.nan, event

    def _steer_route(self, st: NavState, aim, changed: bool, dt: float):
        pp = self.cfg.planner
        pos = st.pos
        if self._pending is not None and st.t >= self._pending[0] - 1e-9:
            self.route, self.leg = self._pending[1], 0   # the route computed a while ago is ready now
            self._pending = None
        reason = "detected" if changed else None
        if self._pending is not None:
            reason = None                                # still computing: no new request
        elif self.route is None:
            reason = "initial"
        elif reason is None and self.replan_interval is not None \
                and st.t - self._last_plan_t >= self.replan_interval - 1e-9:
            reason = "periodic"

        goal = free_point(self._world, aim, pp.safety_margin, 1.5 * pp.grid_res)
        event = None
        if reason is not None:
            begin = free_point(self._world, pos, pp.safety_margin, 1.5 * pp.grid_res)
            try:
                out = self._plan(self._grid, self._world, tuple(begin), tuple(goal), pp)
                route = out.route.copy()
                if len(route) < 2:                     # start and goal in the same cell
                    route = np.vstack([route, route])
                route[0] = pos                         # the leg starts where the aircraft is (was, if late)
                if self.plan_latency > 0.0 and self.route is not None:
                    self._pending = (st.t + self.plan_latency, route)   # keep flying the old route meanwhile
                else:
                    self.route, self.leg = route, 0
                event = ReplanEvent(st.t, reason, True, route.copy(), out.plan_ms, out.expanded)
            except PlanningError:
                if self.route is None:                 # nothing to fall back on: fly straight
                    self.route, self.leg = np.array([pos, goal]), 0
                event = ReplanEvent(st.t, reason, False, self.route.copy(), 0.0, 0)
            self._last_plan_t = st.t

        self.route[-1] = goal                          # the end of the route follows the target
        a, b = self.route[self.leg], self.route[self.leg + 1]
        out_g = self._l1.update_waypoint(pos, st.ground_vel, a, b, dt)
        return out_g.roll_cmd, out_g.xtrack_right, event

    def _advance_leg(self, st: NavState) -> None:
        """ArduPlane-style leg switching; the last leg never ends, it chases the target."""
        route = self.route
        if route is None or self.leg >= len(route) - 2:
            return
        gs = float(np.hypot(*st.ground_vel))
        if self._l1.leg_complete(self.cfg.guidance.wp_accept_radius, st.pos, gs,
                                 route[self.leg], route[self.leg + 1], route[self.leg + 2]):
            self.leg += 1

    # ------------------------------------------------------------------ the tick
    def update(self, st: NavState, target: NavTarget, zones: NavZones, dt: float,
               wind=None) -> NavCommand:
        cfg = self.cfg
        veh, app, pur = cfg.vehicle, cfg.approach, cfg.pursuit
        changed = self._sync_zones(zones)
        tp = np.asarray(target.pos, dtype=float)
        tv = np.asarray(target.vel, dtype=float)
        tspeed = float(np.hypot(*tv))
        # aim d_follow behind the target, so the aircraft joins from behind instead of flying through it
        aim = tp - tv / tspeed * app.d_follow if tspeed > 1e-6 else tp
        vel = np.asarray(st.ground_vel, dtype=float)

        if self._field_params is not None:
            roll, xtrack, event = self._steer_field(st, aim, zones, changed)
        else:
            roll, xtrack, event = self._steer_route(st, aim, changed, dt)

        d_vec = tp - st.pos
        d = float(np.hypot(*d_vec))
        w_hat = vel - st.air_velocity()                # wind as the aircraft believes it
        nominal = tspeed if target.speed is None else target.speed
        g_des = approach_speed(d, nominal, veh, app)
        v_cmd = float(np.clip(g_des - float(w_hat @ (d_vec / max(d, 1e-6))), veh.v_stall, veh.v_max))
        slow = pur.turn_back_factor * veh.v_stall
        if d < pur.turn_back_range and float(vel @ d_vec) < pur.turn_back_cos * float(np.hypot(*vel)) * d:
            v_cmd = min(v_cmd, slow)                   # target not ahead: slow down for a tight turn

        if self.use_guard:
            every = max(1, int(round(cfg.guard.period / dt)))
            if self._tick % every == 0:
                w = w_hat if wind is None else np.asarray(wind, dtype=float)
                self._guard_roll, self._guard_on = self._guard(st, w, roll, zones.centres, zones.radii)
            if self._guard_on:
                roll = self._guard_roll
                v_cmd = min(v_cmd, slow)
        self._tick += 1

        if self._field_params is None:
            self._advance_leg(st)
        return NavCommand(roll=roll, airspeed=v_cmd, xtrack_right=xtrack,
                          guard_active=self.use_guard and self._guard_on, event=event)
