import math

import numpy as np
import pytest

from uavguide.config import VehicleParams
from uavguide.vehicle import FixedWing, VehicleState, wrap_pi

P = VehicleParams()
DT = 0.02


def fly(s, phi_cmd, v_cmd, wind, seconds):
    m = FixedWing(P)
    out = [s]
    for _ in range(int(round(seconds / DT))):
        s = m.step(s, phi_cmd, v_cmd, wind, DT)
        out.append(s)
    return out


def test_full_bank_turn_radius_matches_coordinated_turn_formula():
    s0 = VehicleState(0.0, 0.0, 0.0, 20.0, phi=P.roll_max)
    states = fly(s0, P.roll_max, 20.0, (0.0, 0.0), 60.0)
    pts = np.array([(s.x, s.y) for s in states[len(states) // 2:]])
    # least-squares circle fit: x^2 + y^2 = 2 a x + 2 b y + c
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    a, b, c = np.linalg.lstsq(A, (pts ** 2).sum(axis=1), rcond=None)[0]
    radius = math.sqrt(c + a * a + b * b)
    assert radius == pytest.approx(P.min_turn_radius(20.0), rel=0.01)   # ~58.2 m


def test_wind_drifts_the_aircraft_with_the_air_mass():
    s0 = VehicleState(0.0, 0.0, 0.0, 20.0)
    s = fly(s0, 0.0, 20.0, (5.0, 0.0), 10.0)[-1]
    assert s.x == pytest.approx(50.0, abs=0.2)
    assert s.y == pytest.approx(200.0, abs=0.2)
    assert s.psi == pytest.approx(0.0, abs=1e-12)      # heading unchanged, only the track drifts


def test_roll_is_limited_in_angle_and_rate():
    states = fly(VehicleState(0.0, 0.0, 0.0, 20.0), math.radians(90), 20.0, (0.0, 0.0), 3.0)
    phi = np.array([s.phi for s in states])
    assert phi.max() <= P.roll_max + 1e-12
    assert np.max(np.abs(np.diff(phi))) / DT <= P.roll_rate_max + 1e-9
    assert phi[-1] == pytest.approx(P.roll_max, abs=math.radians(0.5))


def test_airspeed_is_acceleration_limited_and_kept_in_envelope():
    up = fly(VehicleState(0.0, 0.0, 0.0, 20.0), 0.0, 40.0, (0.0, 0.0), 10.0)
    v = np.array([s.v for s in up])
    assert v.max() == pytest.approx(P.v_max)
    assert np.max(np.diff(v)) / DT <= P.accel_max + 1e-9
    down = fly(VehicleState(0.0, 0.0, 0.0, 20.0), 0.0, 0.0, (0.0, 0.0), 10.0)
    assert down[-1].v == pytest.approx(P.v_stall)


@pytest.mark.parametrize("a, expected", [(0.0, 0.0), (math.pi, -math.pi), (3 * math.pi / 2, -math.pi / 2),
                                         (-3 * math.pi / 2, math.pi / 2)])
def test_wrap_pi(a, expected):
    assert wrap_pi(a) == pytest.approx(expected)
