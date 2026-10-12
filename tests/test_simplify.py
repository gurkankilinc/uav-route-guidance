import numpy as np

from uavguide.planner import plan_route
from uavguide.simplify import douglas_peucker, point_segment_distance
from uavguide.world import Obstacle, World, default_world


def test_collinear_points_collapse_to_endpoints():
    pts = np.column_stack([np.linspace(0, 100, 51), np.zeros(51)])
    out, keep = douglas_peucker(pts, epsilon=0.1)
    assert len(out) == 2 and keep[0] and keep[-1]


def test_corner_is_kept():
    pts = np.array([[0, 0], [50, 0], [100, 0], [100, 50], [100, 100]], dtype=float)
    out, _ = douglas_peucker(pts, epsilon=1.0)
    assert out.tolist() == [[0, 0], [100, 0], [100, 100]]


def test_error_bound_holds():
    rng = np.random.default_rng(0)
    pts = np.cumsum(rng.normal(size=(300, 2)), axis=0)
    eps = 2.0
    out, keep = douglas_peucker(pts, eps)
    idx = np.flatnonzero(keep)
    for s, e in zip(idx[:-1], idx[1:]):
        if e - s > 1:
            assert point_segment_distance(pts[s + 1:e], pts[s], pts[e]).max() <= eps


def test_zero_epsilon_keeps_every_turn():
    pts = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=float)
    out, _ = douglas_peucker(pts, epsilon=0.0)
    assert len(out) == 4


def test_plain_dp_can_cut_an_obstacle_but_safe_dp_does_not():
    # route bends around an obstacle; a large epsilon shortcuts straight through it
    obs = World((0, 0, 200, 200), [Obstacle(100.0, 50.0, 20.0)])
    pts = np.array([[0, 50], [60, 50], [80, 85], [120, 85], [140, 50], [200, 50]], dtype=float)
    plain, _ = douglas_peucker(pts, epsilon=50.0)
    assert not obs.segment_clear(plain[0], plain[1])
    safe, _ = douglas_peucker(pts, epsilon=50.0, segment_ok=lambda a, b: obs.segment_clear(a, b))
    assert all(obs.segment_clear(a, b) for a, b in zip(safe[:-1], safe[1:]))


def test_astar_route_simplifies_hard_and_stays_safe():
    w = default_world()
    route = plan_route(w, (30, 30), (570, 570), resolution=2.0, margin=10.0).path
    out, _ = douglas_peucker(route, 5.0, segment_ok=lambda a, b: w.segment_clear(a, b, 5.0))
    assert len(out) < len(route) / 10
    assert w.path_clearance(out) >= 5.0
