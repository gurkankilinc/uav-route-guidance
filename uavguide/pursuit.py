"""8.2 - closed-loop pursuit of a moving target with route replanning.

Each tick of the 50 Hz loop

  1. the target moves, the aircraft may detect zones inside its sensor radius,
  2. the route is (re)planned from the current position when it is the first
     tick, when a new zone was detected, or when `replan_interval` has passed,
  3. the last waypoint follows the target live, so a route planned a second
     ago still ends where the target is now. It is not the target itself but
     the point `d_follow` behind it: aiming at the target would make the
     aircraft fly through it and then need a 120 m loop to come back,
  4. the L1 guidance flies the route and the speed command follows the
     approach profile v(d) of `target.py`,
  5. the physical (not inflated) clearance to every real zone is logged.

The aircraft is "caught up" once it stays within `capture_radius` of the
target for `hold_time` seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .guidance import L1Guidance
from .planner import GridMap, PlanningError
from .planners import PLANNERS, free_point
from .safety import ZoneGuard
from .sensors import TrueGroundVelocity, VelocitySource
from .target import MovingTarget, approach_speed
from .vehicle import FixedWing, VehicleState
from .world import World


@dataclass(frozen=True)
class PursuitScenario:
    world: World                       # the real no-fly zones
    target: MovingTarget
    start: VehicleState
    wind: tuple[float, float] = (0.0, 0.0)   # (east, north) m/s, constant
    detect_radius: float | None = None  # None: all zones known from the start; else a zone is
                                        # learned when its edge is closer than this


@dataclass(frozen=True)
class PursuitPolicy:
    replan_interval: float | None = 1.0  # s between replans; None = plan once at the start
    guard: bool = True                   # look-ahead zone guard (see safety.py), same for every planner


@dataclass
class ReplanEvent:
    t: float
    reason: str                # "initial" | "periodic" | "detected"
    ok: bool
    route: np.ndarray
    plan_ms: float
    expanded: int


@dataclass
class PursuitLog:
    planner: str
    t: np.ndarray
    pos: np.ndarray            # (N, 2)
    psi: np.ndarray
    phi: np.ndarray
    phi_cmd: np.ndarray
    v: np.ndarray              # airspeed
    v_cmd: np.ndarray
    target: np.ndarray         # (N, 2)
    dist: np.ndarray           # distance aircraft - target
    xtrack_right: np.ndarray
    clearance: np.ndarray      # physical clearance to the nearest real zone
    known: np.ndarray          # number of zones known at each tick
    replans: list[ReplanEvent] = field(default_factory=list)
    captured: bool = False
    capture_t: float = math.nan


TURN_BACK_COS = 0.5      # the target counts as "ahead" within +-60 deg of the ground track
TURN_BACK_RANGE = 150.0  # m, only this close does a tight turn matter (far away a wide one is fine)
TURN_BACK_FACTOR = 1.2   # airspeed for turning back, times the stall speed (R = V^2 / (g tan phi))


def _heading_of(a, b) -> float:
    return math.atan2(b[0] - a[0], b[1] - a[1])


def _turn_angle_deg(a, b, c) -> float:
    return math.degrees((_heading_of(b, c) - _heading_of(a, b) + math.pi) % (2 * math.pi) - math.pi)


def pursue(
    scenario: PursuitScenario,
    planner: str,
    policy: PursuitPolicy | None = None,
    cfg: Config | None = None,
    source: VelocitySource | None = None,
    seed: int = 0,
) -> PursuitLog:
    cfg = cfg or Config()
    policy = policy or PursuitPolicy()
    plan_fn = PLANNERS[planner]
    veh, pp, app = cfg.vehicle, cfg.planner, cfg.approach
    tgt = scenario.target
    if tgt.speed and not (veh.v_stall < tgt.speed < veh.v_max):
        raise ValueError("target speed must lie between stall and maximum speed of the aircraft")

    rng = np.random.default_rng(seed)
    source = source or TrueGroundVelocity()
    source.reset(rng)

    dt = cfg.sim.dt
    n = int(round(cfg.sim.t_max / dt)) + 1
    world = scenario.world
    obstacles = list(world.obstacles)
    C = np.array([(o.x, o.y) for o in obstacles], dtype=float).reshape(-1, 2)
    R = np.array([o.r for o in obstacles], dtype=float)
    known = np.full(len(obstacles), scenario.detect_radius is None)

    model = FixedWing(veh)
    guid = L1Guidance(cfg.guidance, veh)
    guard = ZoneGuard(veh, bounds=world.bounds)
    guard_every = max(1, int(round(0.1 / cfg.sim.dt)))
    phi_cmd_held, guard_on = 0.0, False
    s = scenario.start
    wind = np.asarray(scenario.wind, dtype=float)

    T = np.empty(n); P = np.empty((n, 2)); PSI = np.empty(n); PHI = np.empty(n)
    PHIC = np.empty(n); V = np.empty(n); VC = np.empty(n); TG = np.empty((n, 2))
    D = np.empty(n); XT = np.empty(n); CL = np.empty(n); KN = np.empty(n, dtype=int)

    events: list[ReplanEvent] = []
    route: np.ndarray | None = None
    leg = 0
    last_plan_t = -math.inf
    grid: GridMap | None = None
    world_k = World(world.bounds, [])
    need_grid = True
    hold_start: float | None = None
    captured, capture_t = False, math.nan

    k = 0
    for k in range(n):
        t = k * dt
        tp, tv = tgt.state(t)
        pos = s.pos
        tspeed = float(np.hypot(*tv))
        # aim d_follow behind the target, so the aircraft joins from behind instead of flying through it
        aim = tp - tv / tspeed * app.d_follow if tspeed > 1e-6 else tp

        d_obs = np.hypot(C[:, 0] - s.x, C[:, 1] - s.y) - R if len(R) else np.empty(0)
        clearance = float(d_obs.min()) if len(R) else math.inf

        reason = None
        if scenario.detect_radius is not None and len(R):
            new = (~known) & (d_obs <= scenario.detect_radius)
            if new.any():
                known |= new
                need_grid = True
                reason = "detected"
        if route is None:
            reason = "initial"
        elif reason is None and policy.replan_interval is not None \
                and t - last_plan_t >= policy.replan_interval - 1e-9:
            reason = "periodic"

        if need_grid:
            world_k = World(world.bounds, [o for o, kn in zip(obstacles, known) if kn])
            grid = GridMap(world_k, pp.grid_res, pp.safety_margin)
            need_grid = False

        goal = free_point(world_k, aim, pp.safety_margin, 1.5 * pp.grid_res)

        if reason is not None:
            begin = free_point(world_k, pos, pp.safety_margin, 1.5 * pp.grid_res)
            try:
                out = plan_fn(grid, world_k, tuple(begin), tuple(goal), pp)
                new_route = out.route.copy()
                if len(new_route) < 2:                 # start and goal in the same cell
                    new_route = np.vstack([new_route, new_route])
                new_route[0] = pos                     # the leg starts where the aircraft is
                route, leg, ok = new_route, 0, True
                events.append(ReplanEvent(t, reason, True, route.copy(), out.plan_ms, out.expanded))
            except PlanningError:
                ok = False
                if route is None:                      # nothing to fall back on: fly straight
                    route, leg = np.array([pos, goal]), 0
                events.append(ReplanEvent(t, reason, False, route.copy(), 0.0, 0))
            last_plan_t = t

        route[-1] = goal                               # the end of the route follows the target
        vel = source.ground_velocity(t, s, wind)
        a, b = route[leg], route[leg + 1]
        out_g = guid.update_waypoint(pos, vel, a, b, dt)

        d_vec = tp - pos
        d = float(np.hypot(*d_vec))
        w_hat = vel - s.air_velocity()                 # wind as the guidance believes it
        g_des = approach_speed(d, tgt.speed, veh, app)
        v_cmd = float(np.clip(g_des - float(w_hat @ (d_vec / max(d, 1e-6))), veh.v_stall, veh.v_max))
        if d < TURN_BACK_RANGE and float(vel @ d_vec) < TURN_BACK_COS * float(np.hypot(*vel)) * d:
            v_cmd = min(v_cmd, TURN_BACK_FACTOR * veh.v_stall)   # target not ahead: slow down for a tight turn

        phi_cmd = out_g.roll_cmd
        if policy.guard:
            if k % guard_every == 0:
                phi_cmd_held, guard_on = guard(s, wind, out_g.roll_cmd, C[known], R[known])
            if guard_on:
                phi_cmd = phi_cmd_held
                v_cmd = min(v_cmd, TURN_BACK_FACTOR * veh.v_stall)

        T[k] = t; P[k] = pos; PSI[k] = s.psi; PHI[k] = s.phi; PHIC[k] = phi_cmd
        V[k] = s.v; VC[k] = v_cmd; TG[k] = tp; D[k] = d; XT[k] = out_g.xtrack_right
        CL[k] = clearance; KN[k] = int(known.sum())

        # leg switching (ArduPlane style); the last leg never ends, it chases the target
        last = len(route) - 2
        if leg < last:
            gs = float(np.hypot(*vel))
            ab = b - a
            ab_len = float(np.hypot(*ab))
            along = float((pos - a) @ ab) / max(ab_len, 1e-9)
            turn = _turn_angle_deg(a, b, route[leg + 2])
            if float(np.hypot(*(b - pos))) <= guid.turn_distance(cfg.guidance.wp_accept_radius, turn, gs) \
                    or along >= ab_len:
                leg += 1

        if d <= app.capture_radius:
            if hold_start is None:
                hold_start = t
            if t - hold_start >= app.hold_time:
                captured, capture_t = True, hold_start
                break
        else:
            hold_start = None

        s = model.step(s, phi_cmd, v_cmd, wind, dt)

    m = k + 1
    return PursuitLog(
        planner=planner, t=T[:m], pos=P[:m], psi=PSI[:m], phi=PHI[:m], phi_cmd=PHIC[:m],
        v=V[:m], v_cmd=VC[:m], target=TG[:m], dist=D[:m], xtrack_right=XT[:m],
        clearance=CL[:m], known=KN[:m], replans=events, captured=captured, capture_t=capture_t)
