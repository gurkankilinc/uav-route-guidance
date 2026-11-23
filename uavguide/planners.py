"""Planner registry: every route algorithm the comparison can run, behind one signature.

    fn(grid, world, start, goal, params) -> PlanOutcome

`grid` is the inflated occupancy grid of the zones the aircraft currently
knows about (built once per set of known zones, not once per replan) and
`world` holds the same zones with their physical radii, for the
Douglas-Peucker clearance check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import PlannerParams
from .planner import GridMap, astar
from .shortcut import los_shortcut
from .simplify import douglas_peucker
from .theta_star import theta_star
from .world import World


@dataclass
class PlanOutcome:
    route: np.ndarray       # (N, 2), start..goal
    expanded: int           # search nodes expanded
    plan_ms: float          # wall-clock time of search + simplification


PlanFn = Callable[[GridMap, World, tuple, tuple, PlannerParams], PlanOutcome]


def _make(search, simplify: bool | str) -> PlanFn:
    def run(grid, world, start, goal, params):
        t0 = time.perf_counter()
        res = search(grid, start, goal)
        route = res.path
        if simplify == "los":
            # full margin to the zones and to the border, so the shortcut never eats into it
            m = params.safety_margin
            route, _ = los_shortcut(
                route, segment_ok=lambda a, b: world.segment_clear(a, b, m) and world.segment_inside(a, b, m))
        elif simplify:
            route, _ = douglas_peucker(
                route, params.dp_epsilon,
                segment_ok=lambda a, b: world.segment_clear(a, b, params.dp_clearance) and world.segment_inside(a, b))
        return PlanOutcome(np.asarray(route, dtype=float), res.expanded, (time.perf_counter() - t0) * 1e3)
    return run


PLANNERS: dict[str, PlanFn] = {
    "A*": _make(astar, False),
    "A* + DP": _make(astar, True),
    "A* + LOS": _make(astar, "los"),
    "Theta*": _make(theta_star, False),
    "Theta* + DP": _make(theta_star, True),
}


def free_point(world: World, p, margin: float, extra: float = 3.0) -> np.ndarray:
    """Nearest point to `p` that is outside every inflated zone and inside the area.

    A moving target can fly over a no-fly zone; the pursuer then aims at the
    closest point it is allowed to be at.
    """
    p = np.asarray(p, dtype=float).copy()
    x0, y0, x1, y1 = world.bounds
    lo, hi = np.array([x0 + 1.0, y0 + 1.0]), np.array([x1 - 1.0, y1 - 1.0])
    p = np.clip(p, lo, hi)
    for _ in range(10):
        moved = False
        for o in world.obstacles:
            c = np.array([o.x, o.y])
            d = p - c
            n = float(np.hypot(*d))
            if n < o.r + margin:
                u = d / n if n > 1e-9 else np.array([1.0, 0.0])
                p = c + u * (o.r + margin + extra)
                moved = True
        p = np.clip(p, lo, hi)
        if world.polygon is not None and world.border_clearance(p) < margin + extra:
            # outside (or too close to) the polygon border: go to the nearest border point, then inwards
            q = world.nearest_border_point(p)
            d = (q - p) if not world.contains(*p) else (p - q)
            n = float(np.hypot(*d))
            u = d / n if n > 1e-9 else (world.polygon.mean(axis=0) - q) / max(float(np.hypot(*(world.polygon.mean(axis=0) - q))), 1e-9)
            p = q + u * (margin + extra)
            moved = True
        if not moved:
            break
    return p
