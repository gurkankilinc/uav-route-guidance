"""Theta* - any-angle path planning on the same inflated grid as A*.

A* is restricted to 8 grid directions, which makes routes slightly longer
than necessary and produces hundreds of waypoints. Theta* (Daniel et al.,
2010) fixes that with one change: when a node is expanded, its neighbour may
take the *grandparent* as parent if there is a collision-free line of sight
from the grandparent. Paths therefore bend only where obstacles force them to.

`line_of_sight` is exact (integer grid traversal) and conservative: a
segment that passes exactly through the corner of an occupied cell is
treated as blocked, matching the "no corner cutting" rule used in `planner`.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from .planner import GridMap, PlanningError, PlanResult, _NEIGHBORS
from .world import World


def line_of_sight(occ: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if the straight segment between cell centres a and b crosses only free cells."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return not occ[x0, y0]
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    adx, ady = abs(dx), abs(dy)
    ix, iy = x0, y0
    nx = ny = 0  # cell boundaries crossed so far along x / y
    while True:
        if occ[ix, iy]:
            return False
        if (ix, iy) == (x1, y1):
            return True
        # compare when the ray reaches the next x- and y-boundary, in integers
        tx = (2 * nx + 1) * ady if adx else math.inf
        ty = (2 * ny + 1) * adx if ady else math.inf
        if tx == ty:                      # exactly through a cell corner
            if occ[ix + sx, iy] or occ[ix, iy + sy]:
                return False
            ix += sx
            iy += sy
            nx += 1
            ny += 1
        elif tx < ty:
            ix += sx
            nx += 1
        else:
            iy += sy
            ny += 1


def theta_star(grid: GridMap, start, goal) -> PlanResult:
    s = grid.to_cell(*start)
    g = grid.to_cell(*goal)
    if not grid.is_free(*s):
        raise PlanningError(f"start {start} is inside an inflated obstacle or out of bounds")
    if not grid.is_free(*g):
        raise PlanningError(f"goal {goal} is inside an inflated obstacle or out of bounds")

    occ = grid.occ
    cost = np.full((grid.nx, grid.ny), np.inf)
    parent = np.full((grid.nx, grid.ny, 2), -1, dtype=np.int32)
    closed = np.zeros((grid.nx, grid.ny), dtype=bool)

    def h(i: int, j: int) -> float:       # Euclidean: consistent for any-angle moves
        return math.hypot(i - g[0], j - g[1])

    cost[s] = 0.0
    parent[s] = s
    heap = [(h(*s), 0.0, s)]
    expanded = 0
    while heap:
        _, c, (i, j) = heapq.heappop(heap)
        if closed[i, j]:
            continue
        closed[i, j] = True
        expanded += 1
        if (i, j) == g:
            break
        pi, pj = int(parent[i, j, 0]), int(parent[i, j, 1])
        for di, dj, step in _NEIGHBORS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < grid.nx and 0 <= nj < grid.ny) or occ[ni, nj] or closed[ni, nj]:
                continue
            if di and dj and (occ[i + di, j] or occ[i, j + dj]):
                continue
            if line_of_sight(occ, (pi, pj), (ni, nj)):          # path 2: via grandparent
                nc = cost[pi, pj] + math.hypot(ni - pi, nj - pj)
                np_ = (pi, pj)
            else:                                               # path 1: via current node
                nc = c + step
                np_ = (i, j)
            if nc < cost[ni, nj]:
                cost[ni, nj] = nc
                parent[ni, nj] = np_
                heapq.heappush(heap, (nc + h(ni, nj), nc, (ni, nj)))
    else:
        raise PlanningError("no collision-free route exists between start and goal")

    cells = [g]
    while cells[-1] != s:
        qi, qj = parent[cells[-1]]
        cells.append((int(qi), int(qj)))
    cells.reverse()

    path = np.array([grid.to_world(i, j) for i, j in cells], dtype=float)
    path[0] = start
    path[-1] = goal
    length = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
    return PlanResult(path=path, length=length, expanded=expanded)


def plan_route_theta(world: World, start, goal, resolution: float, margin: float) -> PlanResult:
    return theta_star(GridMap(world, resolution, margin), start, goal)
