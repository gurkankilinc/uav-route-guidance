"""8.2 - closed-loop pursuit of a moving target: the simulation around the navigation core.

All decisions (route planning, replanning, L1 or potential-field steering, speed profile, guard)
live in `navcore.Navigator`. This module is only the world the navigator flies in. Each tick of
the 50 Hz loop

  1. the target moves and the aircraft may detect zones inside its sensor radius,
  2. the navigator gets the aircraft state, the target and the known zones, and returns a bank
     and an airspeed command,
  3. the vehicle model flies that command in the true wind,
  4. the physical (not inflated) clearance to every real zone is logged.

The aircraft is "caught up" once it stays within `capture_radius` of the target for `hold_time`
seconds. That is a scoring rule, not a navigation decision, so it lives here.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .navcore import METHODS, NavState, NavTarget, NavZones, Navigator, ReplanEvent
from .sensors import AirDataOnly, NoisyGnssVelocity, TrueGroundVelocity, VelocitySource
from .target import MovingTarget
from .vehicle import FixedWing, VehicleState
from .wind import WindEstimateSource
from .world import World

__all__ = ["METHODS", "PursuitScenario", "PursuitPolicy", "PursuitLog", "ReplanEvent", "pursue"]


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
    guard: bool = True                   # look-ahead zone guard (see safety.py), same for every method
    # how far from ideal the link between the core and the aircraft is (defaults: ideal)
    control_rate_hz: float | None = None  # how often the core runs; None = every simulation tick (50 Hz)
    command_delay: float = 0.0           # s from computing a command to the aircraft acting on it
    plan_latency: float = 0.0            # s a new route takes to compute; the old one is flown meanwhile
    velocity_source: str = "true"        # "true" | "gnss" | "wind_est" | "air_only" (see make_velocity_source)


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
    guard_time: float = 0.0    # s the look-ahead guard was overriding the steering


def make_velocity_source(name: str, cfg: Config) -> VelocitySource:
    """Where the core's ground velocity comes from.

    true      exact (the reference)
    gnss      noisy 5 Hz GNSS velocity, held between samples
    wind_est  air data + a Kalman wind estimate fed by that GNSS velocity
    air_only  air data only, i.e. the wind is ignored
    """
    if name == "true":
        return TrueGroundVelocity()
    if name == "gnss":
        return NoisyGnssVelocity(cfg.sensors)
    if name == "wind_est":
        return WindEstimateSource(cfg.wind_estimator, cfg.sensors, cfg.vehicle.v_cruise)
    if name == "air_only":
        return AirDataOnly(cfg.sensors)
    raise ValueError(f"unknown velocity source {name!r}")


def pursue(
    scenario: PursuitScenario,
    planner: str,
    policy: PursuitPolicy | None = None,
    cfg: Config | None = None,
    source: VelocitySource | None = None,
    seed: int = 0,
    recorder=None,
) -> PursuitLog:
    cfg = cfg or Config()
    policy = policy or PursuitPolicy()
    nav = Navigator(planner, cfg, replan_interval=policy.replan_interval, guard=policy.guard,
                    plan_latency=policy.plan_latency)
    if recorder is not None:
        nav = recorder.wrap(nav)
    veh, app = cfg.vehicle, cfg.approach
    tgt = scenario.target
    if tgt.speed and not (veh.v_stall < tgt.speed < veh.v_max):
        raise ValueError("target speed must lie between stall and maximum speed of the aircraft")

    rng = np.random.default_rng(seed)
    source = source or make_velocity_source(policy.velocity_source, cfg)
    source.reset(rng)

    dt = cfg.sim.dt
    ctl_every = 1 if policy.control_rate_hz is None else max(1, int(round(1.0 / (policy.control_rate_hz * dt))))
    dt_ctl = ctl_every * dt
    in_flight: deque = deque()            # commands on their way to the aircraft: (arrival time, command)
    roll_now, speed_now = 0.0, veh.v_cruise   # what the aircraft does until the first command arrives
    xtrack_now, guard_now = math.nan, False
    n = int(round(cfg.sim.t_max / dt)) + 1
    world = scenario.world
    C = np.array([(o.x, o.y) for o in world.obstacles], dtype=float).reshape(-1, 2)
    R = np.array([o.r for o in world.obstacles], dtype=float)
    known = np.full(len(R), scenario.detect_radius is None)
    zones = NavZones(C[known], R[known], world.bounds)

    model = FixedWing(veh)
    s = scenario.start
    wind = np.asarray(scenario.wind, dtype=float)

    T = np.empty(n); P = np.empty((n, 2)); PSI = np.empty(n); PHI = np.empty(n)
    PHIC = np.empty(n); V = np.empty(n); VC = np.empty(n); TG = np.empty((n, 2))
    D = np.empty(n); XT = np.empty(n); CL = np.empty(n); KN = np.empty(n, dtype=int)

    events: list[ReplanEvent] = []
    hold_start: float | None = None
    captured, capture_t = False, math.nan
    guard_ticks = 0

    k = 0
    for k in range(n):
        t = k * dt
        tp, tv = tgt.state(t)

        # sensor model: which zones does the aircraft know about?
        d_obs = np.hypot(C[:, 0] - s.x, C[:, 1] - s.y) - R if len(R) else np.empty(0)
        clearance = float(d_obs.min()) if len(R) else math.inf
        if scenario.detect_radius is not None and len(R):
            new = (~known) & (d_obs <= scenario.detect_radius)
            if new.any():
                known |= new
                zones = NavZones(C[known], R[known], world.bounds)

        vel = source.ground_velocity(t, s, wind)
        if k % ctl_every == 0:
            state = NavState(t=t, x=s.x, y=s.y, psi=s.psi, phi=s.phi, v=s.v, ground_vel=vel)
            cmd = nav.update(state, NavTarget(tp, tv, speed=tgt.speed), zones, dt_ctl)
            if cmd.event is not None:
                events.append(cmd.event)
            in_flight.append((t + policy.command_delay, cmd))
        while in_flight and in_flight[0][0] <= t + 1e-9:
            c = in_flight.popleft()[1]
            roll_now, speed_now, xtrack_now, guard_now = c.roll, c.airspeed, c.xtrack_right, c.guard_active
        guard_ticks += guard_now

        d = float(np.hypot(*(tp - s.pos)))
        T[k] = t; P[k] = s.pos; PSI[k] = s.psi; PHI[k] = s.phi; PHIC[k] = roll_now
        V[k] = s.v; VC[k] = speed_now; TG[k] = tp; D[k] = d; XT[k] = xtrack_now
        CL[k] = clearance; KN[k] = int(known.sum())

        if d <= app.capture_radius:
            if hold_start is None:
                hold_start = t
            if t - hold_start >= app.hold_time:
                captured, capture_t = True, hold_start
                break
        else:
            hold_start = None

        s = model.step(s, roll_now, speed_now, wind, dt)

    m = k + 1
    return PursuitLog(
        planner=planner, t=T[:m], pos=P[:m], psi=PSI[:m], phi=PHI[:m], phi_cmd=PHIC[:m],
        v=V[:m], v_cmd=VC[:m], target=TG[:m], dist=D[:m], xtrack_right=XT[:m],
        clearance=CL[:m], known=KN[:m], replans=events, captured=captured, capture_t=capture_t,
        guard_time=guard_ticks * dt)
