import math

import pytest

from uavguide.geo import LocalFrame, state_from_telemetry

LAT0, LON0 = 41.0, 29.0          # roughly Istanbul


def test_origin_is_zero_and_round_trip_is_exact_enough():
    f = LocalFrame(LAT0, LON0)
    assert f.to_local(LAT0, LON0) == (0.0, 0.0)
    for east, north in ((600.0, 600.0), (-350.5, 120.25), (0.0, -4000.0)):
        lat, lon = f.to_geo(east, north)
        assert f.to_local(lat, lon) == pytest.approx((east, north), abs=1e-6)


def test_degree_lengths_match_the_standard_series():
    """Length of a degree at 41 N from the usual WGS-84 series: 111 053.9 m (lat), 84 135.2 m (lon)."""
    f = LocalFrame(LAT0, LON0)
    assert f.to_local(LAT0 + 0.001, LON0)[1] == pytest.approx(111.0539, rel=1e-4)
    assert f.to_local(LAT0, LON0 + 0.001)[0] == pytest.approx(84.1352, rel=1e-4)


def test_east_is_positive_longitude_and_north_is_positive_latitude():
    f = LocalFrame(LAT0, LON0)
    east, north = f.to_local(LAT0 + 0.002, LON0 - 0.003)
    assert east < 0 < north


def test_frame_shrinks_towards_the_pole():
    assert LocalFrame(60.0, 0.0).r_east < LocalFrame(10.0, 0.0).r_east


def test_state_from_mavlink_fields():
    f = LocalFrame(LAT0, LON0)
    lat, lon = f.to_geo(250.0, 400.0)
    st = state_from_telemetry(f, t=12.5, lat_e7=round(lat * 1e7), lon_e7=round(lon * 1e7),
                              vx_cms=1500.0, vy_cms=-800.0, roll=0.2, yaw=-math.pi / 2, airspeed=19.0)
    assert (st.x, st.y) == pytest.approx((250.0, 400.0), abs=0.02)      # degE7 resolves about 1 cm
    assert tuple(st.ground_vel) == pytest.approx((-8.0, 15.0))          # (east, north) from NED vx, vy
    assert st.psi == pytest.approx(1.5 * math.pi)                       # yaw -90 deg = heading west = 270 deg
    assert (st.t, st.phi, st.v) == (12.5, 0.2, 19.0)
