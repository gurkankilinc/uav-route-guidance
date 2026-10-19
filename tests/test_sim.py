import math

import numpy as np
import pytest

from uavguide.config import Config, WindParams
from uavguide.sensors import AirDataOnly, TrueGroundVelocity
from uavguide.sim import constant_wind, follow_route, straight_line_test
from uavguide.theta_star import plan_route_theta
from uavguide.world import Obstacle, World, default_world

CFG = Config()


def test_15m_offset_converges_with_small_overshoot():
    log = straight_line_test(15.0, t_max=60.0)
    assert math.degrees(log.phi_cmd[0]) == pytest.approx(-11.80, abs=0.01)
    assert abs(log.xtrack_right[log.t >= 40.0]).max() < 0.5
    assert log.xtrack_right.min() > -1.0          # overshoot below 1 m (~2.5 %)


def test_response_in_time_is_speed_independent():
    slow = straight_line_test(15.0, v=12.0, t_max=40.0)
    fast = straight_line_test(15.0, v=26.0, t_max=40.0)
    assert np.max(np.abs(slow.xtrack_right - fast.xtrack_right)) < 0.1


def test_crosswind_with_ground_velocity_leaves_no_steady_error():
    log = straight_line_test(0.0, wind=constant_wind(CFG.wind.vector()), t_max=120.0)
    assert abs(log.xtrack_right[log.t > 60.0].mean()) < 0.1


def test_crosswind_with_air_data_only_leaves_l1_sin_beta_error():
    log = straight_line_test(0.0, wind=constant_wind(CFG.wind.vector()),
                             source=AirDataOnly(CFG.sensors), t_max=120.0)
    beta = math.asin(CFG.wind.speed / CFG.vehicle.v_cruise)
    expected = CFG.guidance.l1_distance(CFG.vehicle.v_cruise) * math.sin(beta)   # ~20.3 m
    assert log.xtrack_right[log.t > 60.0].mean() == pytest.approx(expected, abs=0.3)


def test_route_is_flown_leg_by_leg_to_the_end():
    route = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=float)
    log = follow_route(route, CFG, t_max=200.0)
    assert log.finished
    assert log.leg.max() == len(route) - 2
    assert np.all(np.diff(log.leg) >= 0)


def test_log_helpers():
    route = np.array([[0, 0], [0, 500]], dtype=float)
    log = follow_route(route, CFG, source=TrueGroundVelocity(), t_max=10.0)
    assert log.route_distance(route).max() < 1e-6
    w = World((0, 0, 600, 600), [Obstacle(50.0, 100.0, 10.0)])
    assert log.obstacle_clearance(w).min() == pytest.approx(40.0, abs=1e-6)


def test_theta_star_route_keeps_clearance_when_flown_in_any_wind_direction():
    world = default_world()
    pc = CFG.planner
    route = plan_route_theta(world, (30.0, 30.0), (570.0, 570.0), pc.grid_res, pc.safety_margin).path
    for wind_from in range(0, 360, 45):
        wind = constant_wind(WindParams(5.0, float(wind_from)).vector())
        log = follow_route(route, CFG, wind=wind, t_max=200.0)
        assert log.finished
        assert log.obstacle_clearance(world).min() > 5.0, f"wind from {wind_from} deg"
