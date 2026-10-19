import math

import numpy as np
import pytest

from uavguide.config import Config, WindEstimatorParams
from uavguide.sensors import AirDataOnly
from uavguide.sim import constant_wind, straight_line_test
from uavguide.wind import WindEstimateSource, WindEstimator, downwind_turn_radius

CFG = Config()


def make_source():
    return WindEstimateSource(CFG.wind_estimator, CFG.sensors, CFG.vehicle.v_cruise)


def test_kalman_update_math():
    est = WindEstimator(WindEstimatorParams(initial_sigma=1.0), CFG.sensors)
    est.R = np.eye(2)
    est.update(np.array([2.0, 0.0]))
    assert est.w == pytest.approx([1.0, 0.0])
    assert est.P == pytest.approx(0.5 * np.eye(2))


def test_estimate_converges_to_constant_wind():
    src = make_source()
    straight_line_test(0.0, wind=constant_wind(CFG.wind.vector()), source=src, t_max=60.0)
    est = src.estimator
    assert np.hypot(*(est.w - np.array(CFG.wind.vector()))) < 0.3
    assert est.sigma < 0.2


def test_estimate_follows_a_wind_change():
    w1, w2 = np.array([5.0, 0.0]), np.array([0.0, -6.0])
    src = make_source()
    straight_line_test(0.0, wind=lambda t: w1 if t < 60.0 else w2, source=src, t_max=120.0)
    assert np.hypot(*(src.estimator.w - w2)) < 0.5


def test_estimated_wind_removes_the_air_data_crosstrack_error():
    wind = constant_wind(CFG.wind.vector())
    blind = straight_line_test(0.0, wind=wind, source=AirDataOnly(CFG.sensors), t_max=120.0)
    aware = straight_line_test(0.0, wind=wind, source=make_source(), t_max=120.0)
    tail = blind.t > 60.0
    assert blind.xtrack_right[tail].mean() > 15.0
    assert abs(aware.xtrack_right[tail].mean()) < 1.0


def test_same_seed_same_result():
    wind = constant_wind(CFG.wind.vector())
    a = straight_line_test(0.0, wind=wind, source=make_source(), t_max=20.0, seed=3)
    b = straight_line_test(0.0, wind=wind, source=make_source(), t_max=20.0, seed=3)
    assert np.array_equal(a.pos, b.pos)


def test_downwind_turn_radius():
    assert downwind_turn_radius(20.0, 0.0, CFG.vehicle.roll_max) == pytest.approx(CFG.vehicle.min_turn_radius())
    assert downwind_turn_radius(20.0, 5.0, CFG.vehicle.roll_max) == pytest.approx(
        25.0 ** 2 / (9.81 * math.tan(CFG.vehicle.roll_max)))
