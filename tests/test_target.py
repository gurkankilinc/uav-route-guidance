import math

import numpy as np
import pytest

from uavguide.config import ApproachParams, TargetParams, VehicleParams
from uavguide.target import MovingTarget, approach_speed

BOUNDS = (0.0, 0.0, 600.0, 600.0)


def test_line_target_moves_at_constant_speed_and_stays_inside():
    tgt = MovingTarget(TargetParams(speed=17.0, mode="line"), BOUNDS, start=(300, 300), heading_deg=30)
    pts = np.array([tgt.state(t)[0] for t in np.arange(0, 600, 0.5)])
    assert pts.min() >= 0.0 and pts.max() <= 600.0
    for t in (0.0, 13.0, 77.7, 301.2):
        assert np.hypot(*tgt.state(t)[1]) == pytest.approx(17.0)


def test_line_target_velocity_matches_position_derivative():
    tgt = MovingTarget(TargetParams(speed=17.0, mode="line"), BOUNDS, start=(100, 500), heading_deg=140)
    for t in (3.0, 40.0, 95.0, 210.0):
        h = 0.01
        num = (tgt.state(t + h)[0] - tgt.state(t - h)[0]) / (2 * h)
        assert num == pytest.approx(tgt.state(t)[1], abs=0.5)


def test_line_target_turns_smoothly_at_the_wall():
    tgt = MovingTarget(TargetParams(speed=20.0, mode="line"), BOUNDS, start=(500, 300), heading_deg=90)
    turn_rate = 20.0 / tgt.turn_radius
    prev = math.atan2(*tgt.state(0.0)[1])
    peak = 0.0
    for t in np.arange(0.1, 60.0, 0.1):
        h = math.atan2(*tgt.state(t)[1])
        step = abs((h - prev + math.pi) % (2 * math.pi) - math.pi)
        peak = max(peak, step / 0.1)
        prev = h
    assert peak > 0.5 * turn_rate                 # it did turn around ...
    assert peak <= turn_rate * 1.05               # ... but never faster than its turn radius allows


def test_circle_target_keeps_its_radius_and_speed():
    tgt = MovingTarget(TargetParams(speed=17.0, mode="circle", circle_radius=120.0), BOUNDS, start=(300, 300))
    for t in (0.0, 5.0, 31.0, 123.0):
        pos, vel = tgt.state(t)
        assert np.hypot(*(pos - [300, 300])) == pytest.approx(120.0)
        assert np.hypot(*vel) == pytest.approx(17.0)
        # velocity is tangential
        assert (pos - [300, 300]) @ vel == pytest.approx(0.0, abs=1e-6)


def test_circle_is_clockwise():
    tgt = MovingTarget(TargetParams(speed=17.0, mode="circle", circle_radius=120.0), BOUNDS,
                       start=(300, 300), heading_deg=0.0)
    pos0, _ = tgt.state(0.0)                # north of the centre
    pos1, _ = tgt.state(1.0)
    assert pos0[1] > 300 and pos1[0] > pos0[0]   # moves east: clockwise seen from above


def test_fixed_target_does_not_move():
    tgt = MovingTarget(TargetParams(mode="fixed"), BOUNDS, start=(450, 120))
    pos, vel = tgt.state(99.0)
    assert tuple(pos) == (450.0, 120.0) and not vel.any()


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        MovingTarget(TargetParams(mode="zigzag"), BOUNDS)


def test_approach_speed_profile():
    veh, app = VehicleParams(), ApproachParams()
    assert approach_speed(app.d_follow, 17.0, veh, app) == pytest.approx(17.0)
    assert approach_speed(2000.0, 17.0, veh, app) == pytest.approx(veh.v_max, abs=1e-3)
    d = np.linspace(0, 800, 50)
    v = [approach_speed(x, 17.0, veh, app) for x in d]
    assert all(b >= a for a, b in zip(v, v[1:]))        # monotonic in distance
    assert approach_speed(0.0, 17.0, veh, app) < 17.0   # slower than the target when closer than d_follow
    # a fixed goal is approached at flying speed, never below stall
    assert approach_speed(app.d_follow, 0.0, veh, app) == pytest.approx(veh.v_stall)
