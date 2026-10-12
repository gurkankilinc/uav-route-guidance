"""8.1 - Obstacle-avoiding route on an inflated occupancy grid (A*).

"Optimal" here means: shortest path on an 8-connected grid of the
inflated map. Octile distance is used as the heuristic, which is
admissible and consistent on that graph, so A* returns the optimum.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .world import World

SQRT2 = math.sqrt(2.0)
_NEIGHBORS = [(di, dj, math.hypot(di, dj))
              for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


class PlanningError(RuntimeError):
    pass


class GridMap:
    """Boolean occupancy grid; obstacles inflated by `margin` metres."""

    def __init__(self, world: World, resolution: float, margin: float):
        self.res = float(resolution)
        self.x0, self.y0, x1, y1 = world.bounds
        self.nx = int(math.floor((x1 - self.x0) / self.res)) + 1
        self.ny = int(math.floor((y1 - self.y0) / self.res)) + 1
        xs = self.x0 + np.arange(self.nx) * self.res
        ys = self.y0 + np.arange(self.ny) * self.res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        occ = np.zeros((self.nx, self.ny), dtype=bool)
        for o in world.obstacles:
            occ |= (X - o.x) ** 2 + (Y - o.y) ** 2 <= (o.r + margin) ** 2
        self.occ = occ

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(round((x - self.x0) / self.res)), int(round((y - self.y0) / self.res)))

    def to_world(self, i: int, j: int) -> tuple[float, float]:
        return (self.x0 + i * self.res, self.y0 + j * self.res)

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny

    def is_free(self, i: int, j: int) -> bool:
        return self.in_bounds(i, j) and not self.occ[i, j]


@dataclass
class PlanResult:
    path: np.ndarray        # (N, 2) world coordinates, start..goal
    length: float           # m
    expanded: int           # nodes popped from the open set


def _octile(i: int, j: int, gi: int, gj: int) -> float:
    dx, dy = abs(i - gi), abs(j - gj)
    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)


def astar(grid: GridMap, start, goal) -> PlanResult:
    s = grid.to_cell(*start)
    g = grid.to_cell(*goal)
    if not grid.is_free(*s):
        raise PlanningError(f"start {start} is inside an inflated obstacle or out of bounds")
    if not grid.is_free(*g):
        raise PlanningError(f"goal {goal} is inside an inflated obstacle or out of bounds")

    cost = np.full((grid.nx, grid.ny), np.inf)
    parent = np.full((grid.nx, grid.ny, 2), -1, dtype=np.int32)
    closed = np.zeros((grid.nx, grid.ny), dtype=bool)
    occ = grid.occ

    cost[s] = 0.0
    heap = [(_octile(*s, *g), 0.0, s)]
    expanded = 0
    while heap:
        _, c, (i, j) = heapq.heappop(heap)
        if closed[i, j]:
            continue
        closed[i, j] = True
        expanded += 1
        if (i, j) == g:
            break
        for di, dj, step in _NEIGHBORS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < grid.nx and 0 <= nj < grid.ny) or occ[ni, nj] or closed[ni, nj]:
                continue
            # no corner cutting: a diagonal move needs both side cells free
            if di and dj and (occ[i + di, j] or occ[i, j + dj]):
                continue
            nc = c + step
            if nc < cost[ni, nj]:
                cost[ni, nj] = nc
                parent[ni, nj] = (i, j)
                heapq.heappush(heap, (nc + _octile(ni, nj, *g), nc, (ni, nj)))
    else:
        raise PlanningError("no collision-free route exists between start and goal")

    cells = [g]
    while cells[-1] != s:
        pi, pj = parent[cells[-1]]
        cells.append((int(pi), int(pj)))
    cells.reverse()

    path = np.array([grid.to_world(i, j) for i, j in cells], dtype=float)
    path[0] = start   # replace snapped endpoints with the exact ones
    path[-1] = goal
    length = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
    return PlanResult(path=path, length=length, expanded=expanded)


def plan_route(world: World, start, goal, resolution: float, margin: float) -> PlanResult:
    return astar(GridMap(world, resolution, margin), start, goal)
