import math

import numpy as np
import pytest

from uavguide.planner import GridMap, PlanningError, plan_route
from uavguide.theta_star import line_of_sight, plan_route_theta, theta_star
from uavguide.world import Obstacle, World, default_world, random_world

BOUNDS = (0.0, 0.0, 600.0, 600.0)
START, GOAL = (30.0, 30.0), (570.0, 570.0)
RES, MARGIN = 2.0, 10.0


def test_empty_world_is_a_single_straight_segment():
    res = plan_route_theta(World(BOUNDS, []), (0, 0), (300, 100), RES, MARGIN)
    assert len(res.path) == 2
    assert math.isclose(res.length, math.hypot(300, 100), rel_tol=1e-9)


def test_line_of_sight_blocked_by_cell_in_the_way():
    occ = np.zeros((10, 10), dtype=bool)
    occ[5, 5] = True
    assert not line_of_sight(occ, (0, 0), (9, 9))
    assert line_of_sight(occ, (0, 0), (9, 0))


def test_line_of_sight_does_not_squeeze_between_corner_touching_cells():
    occ = np.zeros((10, 10), dtype=bool)
    occ[5, 4] = True
    occ[4, 5] = True               # touch only at a corner
    assert not line_of_sight(occ, (4, 4), (5, 5))


def test_line_of_sight_is_symmetric():
    rng = np.random.default_rng(1)
    occ = rng.random((40, 40)) < 0.12
    occ[0, 0] = occ[39, 39] = False
    for _ in range(200):
        a = tuple(int(v) for v in rng.integers(0, 40, 2))
        b = tuple(int(v) for v in rng.integers(0, 40, 2))
        if occ[a] or occ[b]:
            continue
        assert line_of_sight(occ, a, b) == line_of_sight(occ, b, a)


def test_default_scenario_is_shorter_and_sparser_than_astar():
    w = default_world()
    a = plan_route(w, START, GOAL, RES, MARGIN)
    t = plan_route_theta(w, START, GOAL, RES, MARGIN)
    assert t.length <= a.length + 1e-6
    assert t.length >= math.hypot(540, 540) - 1e-6
    assert len(t.path) < len(a.path) / 10


def test_route_keeps_safety_margin():
    w = default_world()
    t = plan_route_theta(w, START, GOAL, RES, MARGIN)
    assert w.path_clearance(t.path) >= MARGIN - RES * math.sqrt(2) / 2


def test_wall_blocks_route():
    wall = [Obstacle(300.0, y, 12.0) for y in np.arange(0, 601, 10)]
    with pytest.raises(PlanningError):
        plan_route_theta(World(BOUNDS, wall), (30, 300), (570, 300), RES, MARGIN)


def test_start_inside_obstacle_is_rejected():
    w = World(BOUNDS, [Obstacle(100.0, 100.0, 20.0)])
    with pytest.raises(PlanningError):
        plan_route_theta(w, (100, 100), (500, 500), RES, MARGIN)


@pytest.mark.parametrize("seed", range(10))
def test_random_worlds_are_solved_safely_and_never_longer_than_astar(seed):
    w = random_world(seed)
    try:
        a = plan_route(w, START, GOAL, RES, MARGIN)
    except PlanningError:
        pytest.skip("random field happened to block the route")
    t = plan_route_theta(w, START, GOAL, RES, MARGIN)
    assert w.path_clearance(t.path) >= MARGIN - RES * math.sqrt(2) / 2
    assert t.length <= a.length + 1e-6
