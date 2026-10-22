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
