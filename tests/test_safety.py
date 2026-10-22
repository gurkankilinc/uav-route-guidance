import math

import numpy as np
import pytest

from uavguide.config import G, VehicleParams
from uavguide.safety import ZoneGuard, path_clearance, rollout
from uavguide.vehicle import VehicleState

VEH = VehicleParams()
NO_WIND = (0.0, 0.0)
BOUNDS = (0.0, 0.0, 600.0, 600.0)


def test_rollout_straight_flight_follows_the_heading():
    p = rollout(100, 100, math.radians(90), 0.0, 20.0, 0.0, NO_WIND, VEH, horizon=4.0, dt=0.25)
    assert p[-1, 0] == pytest.approx(100 + 20 * 4.0, abs=1e-6)
    assert p[-1, 1] == pytest.approx(100.0, abs=1e-6)


def test_rollout_steady_turn_has_the_expected_radius():
    phi = math.radians(30)
    p = rollout(0, 0, 0.0, phi, 20.0, phi, NO_WIND, VEH, horizon=12.0, dt=0.05)
    r = 20.0 ** 2 / (G * math.tan(phi))
    centre = np.array([r, 0.0])                      # right turn from heading north: centre to the east
    assert np.hypot(*(p - centre).T) == pytest.approx(r, rel=0.03)


def test_rollout_drifts_with_the_wind():
    p = rollout(0, 0, 0.0, 0.0, 20.0, 0.0, (5.0, 0.0), VEH, horizon=4.0, dt=0.25)
    assert p[-1, 0] == pytest.approx(20.0, abs=1e-6)


def test_path_clearance_zone_and_border():
    path = np.array([[100.0, 100.0], [200.0, 100.0]])
    C, R = np.array([[200.0, 140.0]]), np.array([10.0])
    assert path_clearance(path, C, R) == pytest.approx(30.0)
    assert path_clearance(path, C, R, BOUNDS) == pytest.approx(30.0)       # border is 100 m away
    assert path_clearance(path, np.empty((0, 2)), np.empty(0), BOUNDS) == pytest.approx(100.0)


def test_guard_is_silent_when_the_way_is_clear():
    g = ZoneGuard(VEH, BOUNDS)
    s = VehicleState(300, 100, 0.0, 20.0)
    cmd, active = g(s, NO_WIND, 0.0, np.array([[500.0, 500.0]]), np.array([20.0]))
    assert (cmd, active) == (0.0, False)


def test_guard_turns_away_from_a_zone_dead_ahead():
    g = ZoneGuard(VEH, BOUNDS)
    s = VehicleState(300, 100, 0.0, 20.0)                       # heading north
    cmd, active = g(s, NO_WIND, 0.0, np.array([[300.0, 160.0]]), np.array([25.0]))
    assert active and abs(cmd) == pytest.approx(VEH.roll_max)


def test_guard_keeps_the_aircraft_inside_the_area():
    g = ZoneGuard(VEH, BOUNDS)
    s = VehicleState(300, 570, 0.0, 20.0)                       # 30 m from the north border, heading at it
    cmd, active = g(s, NO_WIND, 0.0, np.empty((0, 2)), np.empty(0))
    assert active and abs(cmd) == pytest.approx(VEH.roll_max)
