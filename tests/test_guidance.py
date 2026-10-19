import math
from dataclasses import replace

import pytest

from uavguide.config import Config
from uavguide.guidance import L1Guidance, initial_roll_for_offset

CFG = Config()
A, B = (0.0, 0.0), (0.0, 1000.0)          # track due north


def guidance(**overrides):
    gp = replace(CFG.guidance, **overrides)
    return L1Guidance(gp, CFG.vehicle)


def roll_deg(pos, vel, a=A, b=B, g=None):
    g = g or guidance()
    return math.degrees(g.update_waypoint(pos, vel, a, b, 0.02).roll_cmd)


@pytest.mark.parametrize("v", [12.0, 20.0, 26.0])
def test_task_example_15m_right_needs_11_8_deg_left_at_any_speed(v):
    assert roll_deg((15.0, 0.0), (0.0, v)) == pytest.approx(-11.80, abs=0.01)


@pytest.mark.parametrize("v", [12.0, 20.0, 26.0])
def test_5m_right_needs_4_deg_left_at_any_speed(v):
    assert roll_deg((5.0, 0.0), (0.0, v)) == pytest.approx(-3.98, abs=0.01)


def test_closed_form_matches_guidance():
    for e in (-30.0, -5.0, 0.0, 5.0, 15.0, 30.0):
        expected = math.degrees(initial_roll_for_offset(e, CFG.guidance))
        assert roll_deg((e, 0.0), (0.0, 20.0)) == pytest.approx(expected, abs=1e-9)


def test_left_of_track_rolls_right_and_reports_negative_xtrack():
    out = guidance().update_waypoint((-15.0, 0.0), (0.0, 20.0), A, B, 0.02)
    assert out.roll_cmd > 0
    assert out.xtrack_right == pytest.approx(-15.0)


def test_heading_away_from_track_is_corrected():
    # on the track but flying 10 deg to the right of it -> turn left
    v = (20 * math.sin(math.radians(10)), 20 * math.cos(math.radians(10)))
    assert roll_deg((0.0, 0.0), v) < 0


def test_large_offset_saturates_nu1_and_roll_limit():
    out = guidance().update_waypoint((500.0, 0.0), (0.0, 20.0), A, B, 0.02)
    assert abs(out.roll_cmd) <= CFG.vehicle.roll_max + 1e-12


def test_far_behind_A_flies_to_A_and_far_past_B_flies_back():
    g = guidance()
    assert g.update_waypoint((0.0, -500.0), (0.0, 20.0), A, B, 0.02).mode == "to_A"
    assert g.update_waypoint((0.0, 1200.0), (0.0, 20.0), A, B, 0.02).mode == "to_B"
    assert g.update_waypoint((0.0, 500.0), (0.0, 20.0), A, B, 0.02).mode == "track"


def test_turn_distance_like_arduplane():
    g = guidance()
    assert g.turn_distance(30.0, 90.0, 20.0) == pytest.approx(30.0)
    assert g.turn_distance(30.0, 45.0, 20.0) == pytest.approx(15.0)
    assert g.turn_distance(500.0, 120.0, 20.0) == pytest.approx(CFG.guidance.l1_distance(20.0))


def test_crosstrack_integrator_is_off_by_default_and_bounded_when_on():
    g0 = guidance()
    g0.update_waypoint((2.0, 0.0), (0.0, 20.0), A, B, 0.02)
    assert g0.xtrack_i == 0.0
    g = guidance(xtrack_i_gain=0.5)
    for _ in range(5000):
        g.update_waypoint((2.0, 0.0), (0.0, 20.0), A, B, 0.02)
    assert g.xtrack_i == pytest.approx(-0.1)
