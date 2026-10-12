import math

from uavguide.config import Config


def test_min_turn_radius_at_cruise():
    # 20^2 / (9.81 * tan 35deg) = 58.2 m
    assert math.isclose(Config().vehicle.min_turn_radius(), 58.2, abs_tol=0.1)


def test_l1_distance_matches_arduplane_formula():
    # AP_L1_Control: L1 = (1/pi) * zeta * T * V = 0.75 * 17 * 20 / pi = 81.2 m
    assert math.isclose(Config().guidance.l1_distance(20.0), 81.17, abs_tol=0.01)


def test_l1_gain_matches_arduplane():
    # K_L1 = 4 * zeta^2 = 2.25 for zeta = 0.75
    assert math.isclose(Config().guidance.k_l1, 2.25)


def test_west_wind_blows_east():
    wx, wy = Config().wind.vector()
    assert wx > 4.99 and abs(wy) < 1e-9


def test_target_speed_is_matchable():
    c = Config()
    assert c.vehicle.v_stall < c.target.speed < c.vehicle.v_max
