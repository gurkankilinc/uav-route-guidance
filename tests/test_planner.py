import math

import numpy as np
import pytest

from uavguide.planner import GridMap, PlanningError, astar, plan_route
from uavguide.world import Obstacle, World, default_world, random_world

BOUNDS = (0.0, 0.0, 600.0, 600.0)


def test_empty_world_gives_straight_diagonal():
    w = World(BOUNDS, [])
    res = plan_route(w, (0, 0), (300, 300), resolution=2.0, margin=10.0)
    assert math.isclose(res.length, math.hypot(300, 300), rel_tol=1e-6)


def test_route_keeps_safety_margin():
    w = default_world()
    margin, res_m = 10.0, 2.0
    res = plan_route(w, (30, 30), (570, 570), resolution=res_m, margin=margin)
    # grid snapping can lose at most half a cell diagonal of clearance
    assert w.path_clearance(res.path) >= margin - res_m * math.sqrt(2) / 2


def test_route_is_not_shorter_than_straight_line():
    w = default_world()
    res = plan_route(w, (30, 30), (570, 570), resolution=2.0, margin=10.0)
    assert res.length >= math.hypot(540, 540)


def test_wall_blocks_route():
    wall = [Obstacle(300.0, y, 12.0) for y in np.arange(0, 601, 10)]
    with pytest.raises(PlanningError):
        plan_route(World(BOUNDS, wall), (30, 300), (570, 300), resolution=2.0, margin=10.0)


def test_start_inside_obstacle_is_rejected():
    w = World(BOUNDS, [Obstacle(100.0, 100.0, 20.0)])
    with pytest.raises(PlanningError):
        plan_route(w, (100, 100), (500, 500), resolution=2.0, margin=10.0)


def test_no_diagonal_corner_cutting():
    w = World(BOUNDS, [])
    grid = GridMap(w, 1.0, 0.0)
    grid.occ[:] = False
    grid.occ[5, 4] = True   # two blocked cells touching only at a corner
    grid.occ[4, 5] = True
    res = astar(grid, (4, 4), (5, 5))
    assert len(res.path) > 2   # must go around, not squeeze through the corner


@pytest.mark.parametrize("seed", range(10))
def test_random_worlds_are_solved_safely(seed):
    w = random_world(seed)
    try:
        res = plan_route(w, (30, 30), (570, 570), resolution=2.0, margin=10.0)
    except PlanningError:
        pytest.skip("random field happened to block the route")
    assert w.path_clearance(res.path) >= 10.0 - math.sqrt(2)
