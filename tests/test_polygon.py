import numpy as np
import pytest

from uavguide.planner import GridMap, PlanningError, astar
from uavguide.planners import free_point
from uavguide.world import Obstacle, World, points_in_polygon, points_polygon_distance

SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]
# an L: the notch (50..100, 50..100) is outside
ELL = [[0, 0], [100, 0], [100, 50], [50, 50], [50, 100], [0, 100]]


def test_rectangle_world_is_unchanged():
    w = World((0.0, 0.0, 100.0, 100.0))
    assert w.polygon is None
    assert w.contains(50, 50) and not w.contains(101, 50)
    assert w.border_clearance((10.0, 50.0)) == pytest.approx(10.0)
    assert w.segment_inside((10, 10), (90, 90), 5.0)
    assert not w.segment_inside((10, 10), (99, 90), 5.0)


def test_from_polygon_sets_the_bounding_box():
    w = World.from_polygon(ELL)
    assert w.bounds == (0.0, 0.0, 100.0, 100.0)
    with pytest.raises(ValueError):
        World.from_polygon([[0, 0], [1, 1]])


def test_contains_and_border_clearance_on_a_concave_polygon():
    w = World.from_polygon(ELL)
    assert w.contains(25, 75) and w.contains(75, 25)
    assert not w.contains(75, 75)                       # inside the bounding box, outside the area
    assert w.border_clearance((25.0, 25.0)) == pytest.approx(25.0)
    assert w.border_clearance((75.0, 75.0)) == pytest.approx(-25.0)


def test_segment_inside_detects_a_cut_across_the_notch():
    w = World.from_polygon(ELL)
    assert w.segment_inside((25, 25), (25, 90))
    assert not w.segment_inside((25, 90), (90, 25))     # both ends inside, the middle is not
    assert not w.segment_inside((25, 25), (25, 97), 5.0)  # too close to the border


def test_vectorised_helpers_match_the_scalar_ones():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-20, 120, size=(500, 2))
    poly = np.array(ELL, dtype=float)
    w = World.from_polygon(ELL)
    inside = points_in_polygon(pts, poly)
    dist = points_polygon_distance(pts, poly)
    for p, i, d in zip(pts, inside, dist):
        assert w.contains(*p) == bool(i) or abs(d) < 1e-9
        assert abs(w.border_clearance(p)) == pytest.approx(d)


def test_grid_blocks_cells_outside_or_near_the_border():
    w = World.from_polygon(ELL)
    g = GridMap(w, 2.0, 6.0)
    assert g.is_free(*g.to_cell(25, 25))
    assert not g.is_free(*g.to_cell(75, 75))            # in the notch
    assert not g.is_free(*g.to_cell(2, 50))             # closer than the margin to the border
    route = astar(g, (25.0, 85.0), (85.0, 25.0)).path   # has to go around the notch
    assert all(w.contains(x, y) for x, y in route)
    assert min(w.border_clearance(p) for p in route) >= 6.0 - 1e-9


def test_planning_into_the_notch_fails_loudly():
    g = GridMap(World.from_polygon(ELL), 2.0, 6.0)
    with pytest.raises(PlanningError):
        astar(g, (25.0, 25.0), (75.0, 75.0))


def test_free_point_pulls_a_point_into_the_polygon():
    w = World.from_polygon(ELL, [Obstacle(25.0, 25.0, 10.0)])
    p = free_point(w, (75.0, 75.0), margin=6.0)
    assert w.contains(*p) and w.border_clearance(p) >= 6.0
    assert w.clearance(p) >= 6.0
    q = free_point(w, (25.0, 25.0), margin=6.0)         # inside the zone
    assert w.clearance(q) >= 6.0 and w.border_clearance(q) >= 6.0
