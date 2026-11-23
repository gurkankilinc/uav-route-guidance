"""Reproducible pursuit scenarios, for the comparison and for the app."""

from __future__ import annotations

import math

import numpy as np

from .config import Config, TargetParams
from .planners import free_point
from .pursuit import PursuitScenario
from .target import MovingTarget
from .vehicle import VehicleState
from .world import Obstacle, World, random_world

START = (30.0, 30.0)

# Seeds are split so that nothing is tuned on the scenarios it is judged on.
#   TUNE_SEEDS  for choosing gains and constants (scripts/tune_potential.py uses the first 20)
#   TEST_SEEDS  for reported results only; never look at individual runs here while tuning
# Seeds 0-19 were used for both during early development (v1), which is why they are in neither set.
TUNE_SEEDS = range(100, 200)
TEST_SEEDS = range(200, 300)


def scale_zones(world: World, k: float) -> World:
    """Every zone radius multiplied by k (zone size is a scenario setting, fixed during a flight)."""
    return World(world.bounds, [Obstacle(o.x, o.y, o.r * k) for o in world.obstacles])


def make_scenario(
    seed: int,
    cfg: Config | None = None,
    zone_scale: float = 1.0,
    n_zones: int = 12,
    target_mode: str = "line",
    target_speed: float = 17.0,
    wind_speed: float = 5.0,
    wind_from_deg: float | None = None,
    detect_radius: float | None = None,
) -> PursuitScenario:
    cfg = cfg or Config()
    rng = np.random.default_rng(seed + 10_000)
    world = scale_zones(random_world(seed, n=n_zones, bounds=cfg.planner.area), zone_scale)
    margin = cfg.planner.safety_margin
    x0, y0, x1, y1 = world.bounds

    start = free_point(world, START, margin + 10.0)

    # target start / circle centre: away from the aircraft, inside the area
    for _ in range(1000):
        p = rng.uniform((x0 + 120, y0 + 120), (x1 - 120, y1 - 120))
        if np.hypot(*(p - start)) >= 250.0:
            break
    heading = float(rng.uniform(0.0, 360.0))
    tp = TargetParams(speed=target_speed, mode=target_mode, circle_radius=cfg.target.circle_radius)
    if target_mode == "circle":
        r = tp.circle_radius + 20.0
        p = np.clip(p, (x0 + r, y0 + r), (x1 - r, y1 - r))
    target = MovingTarget(tp, world.bounds, start=tuple(p), heading_deg=heading)

    if wind_from_deg is None:
        wind_from_deg = float(rng.uniform(0.0, 360.0))
    a = math.radians(wind_from_deg)
    wind = (-wind_speed * math.sin(a), -wind_speed * math.cos(a))

    t0, _ = target.state(0.0)
    psi0 = math.atan2(t0[0] - start[0], t0[1] - start[1])
    return PursuitScenario(world=world, target=target,
                           start=VehicleState(float(start[0]), float(start[1]), psi0, cfg.vehicle.v_cruise),
                           wind=wind, detect_radius=detect_radius)


def stress_scenarios(cfg: Config | None = None, wind: tuple[float, float] = (0.0, 0.0)) -> dict[str, PursuitScenario]:
    """Hand-built cases aimed at the known weak points of potential fields (Koren & Borenstein 1991,
    Ge & Cui 2000). The aircraft always starts on the west side heading east.

    behind_zone   a large zone sits exactly on the line between aircraft and a fixed target:
                  attraction and repulsion are collinear, the classic local minimum
    u_trap        zones form a cup that opens towards the aircraft, the target is behind its bottom
    narrow_gap    the straight line passes through a 30 m gap between two zones (10 m once inflated)
    orbit_zone    the target circles a zone 25 m outside its edge, so the aim point is always
                  close to a zone (goal non-reachable with obstacles nearby)
    """
    cfg = cfg or Config()
    b = cfg.planner.area
    v0 = cfg.vehicle.v_cruise

    def fixed(x, y):
        return MovingTarget(TargetParams(mode="fixed"), b, start=(x, y))

    def start(x, y, heading_deg=90.0):
        return VehicleState(x, y, math.radians(heading_deg), v0)

    def world(zones):
        return World(b, [Obstacle(float(x), float(y), float(r)) for x, y, r in zones])

    u_zones = [(380, y, 22) for y in (210, 250, 290, 330, 370, 410)] \
        + [(x, 190, 22) for x in (300, 340)] + [(x, 430, 22) for x in (300, 340)]
    orbit = MovingTarget(TargetParams(speed=17.0, mode="circle", circle_radius=35.0 + 25.0), b,
                         start=(400.0, 300.0), heading_deg=270.0)
    return {
        "behind_zone": PursuitScenario(world([(300, 300, 45)]), fixed(480, 300), start(80, 300), wind),
        "u_trap": PursuitScenario(world(u_zones), fixed(500, 310), start(80, 310), wind),
        "narrow_gap": PursuitScenario(world([(300, 255, 30), (300, 345, 30)]), fixed(520, 300),
                                      start(80, 300), wind),
        "orbit_zone": PursuitScenario(world([(400, 300, 35)]), orbit, start(80, 300), wind),
    }
