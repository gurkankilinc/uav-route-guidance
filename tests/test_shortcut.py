import numpy as np
import pytest

from uavguide.config import PlannerParams
from uavguide.planner import GridMap, astar
from uavguide.planners import PLANNERS
from uavguide.shortcut import corner_cut, los_shortcut, margin_for
from uavguide.world import Obstacle, World, random_world

START, GOAL = (30.0, 30.0), (570.0, 570.0)


def _length(p):
    return float(np.sum(np.hypot(*np.diff(np.asarray(p), axis=0).T)))


def test_straight_line_collapses_to_two_points():
    pts = np.column_stack([np.linspace(0, 100, 51), np.zeros(51)])
    out, keep = los_shortcut(pts, lambda a, b: True)
    assert len(out) == 2 and keep.sum() == 2
    assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])


def test_nothing_visible_keeps_every_point():
    pts = np.column_stack([np.arange(6.0), np.zeros(6)])
    out, keep = los_shortcut(pts, lambda a, b: False)
    assert keep.all() and len(out) == 6


def test_empty_and_single_point():
    out, keep = los_shortcut(np.empty((0, 2)), lambda a, b: True)
    assert len(out) == 0 and len(keep) == 0
    out, _ = los_shortcut([[1.0, 2.0]], lambda a, b: True)
    assert out.tolist() == [[1.0, 2.0]]


@pytest.mark.parametrize("seed", range(8))
def test_shortcut_is_safe_shorter_and_keeps_the_margin(seed):
    world = random_world(seed)
    p = PlannerParams()
    grid = GridMap(world, p.grid_res, p.safety_margin)
    raw = astar(grid, START, GOAL).path
    los = PLANNERS["A* + LOS"](grid, world, START, GOAL, p).route
    dp = PLANNERS["A* + DP"](grid, world, START, GOAL, p).route
    assert np.allclose(los[0], START) and np.allclose(los[-1], GOAL)
    assert _length(los) <= _length(raw) + 1e-9
    assert _length(los) <= _length(dp) + 1e-6          # at least as short as DP ...
    assert len(los) <= len(dp)                          # ... with no more waypoints
    # a little slack: neighbouring grid cells may sit a fraction of a cell inside the margin
    assert world.path_clearance(los) >= p.safety_margin - p.grid_res


def test_corner_cut_and_margin_values():
    assert corner_cut(58.0, 0.0) == pytest.approx(0.0)
    assert corner_cut(58.0, 90.0) == pytest.approx(58.0 * (2 ** 0.5 - 1.0))
    assert corner_cut(58.0, 30.0) < corner_cut(58.0, 60.0)
    assert margin_for(58.0, 45.0, 5.0) == pytest.approx(5.0 + corner_cut(58.0, 45.0))


def test_los_planner_respects_a_polygon_border():
    poly = [[405, 0], [0, 131], [235, 929], [654, 801]]
    world = World.from_polygon(poly, [Obstacle(330.0, 465.0, 80.0)])
    p = PlannerParams(grid_res=5.0)
    grid = GridMap(world, p.grid_res, p.safety_margin)
    route = PLANNERS["A* + LOS"](grid, world, (330.0, 200.0), (330.0, 750.0), p).route
    assert all(world.contains(x, y) for x, y in route)
    assert world.path_clearance(route) >= p.safety_margin - p.grid_res
    for a, b in zip(route[:-1], route[1:]):
        assert world.segment_inside(a, b)
