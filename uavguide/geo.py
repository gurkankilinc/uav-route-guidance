"""Between the autopilot's world (latitude / longitude, NED velocities) and the core's (east / north metres).

The core works in a flat local frame: x = east, y = north, in metres from an origin. Over a flight
area of a few kilometres a tangent-plane approximation with the WGS-84 radii of curvature at the
origin is accurate to centimetres, which is far below GNSS noise.

`state_from_telemetry` builds the core's `NavState` from the fields MAVLink already provides, so the
SITL bridge and a companion computer do not need their own conversions:

    GLOBAL_POSITION_INT   lat, lon [degE7], vx (north), vy (east) [cm/s]
    ATTITUDE              roll, yaw [rad]; yaw 0 = north, positive clockwise (same as the core)
    VFR_HUD               airspeed [m/s]
"""

from __future__ import annotations

import math

import numpy as np

from .navcore import NavState

WGS84_A = 6378137.0                 # semi-major axis, m
WGS84_F = 1.0 / 298.257223563       # flattening
_E2 = WGS84_F * (2.0 - WGS84_F)     # first eccentricity squared


class LocalFrame:
    """East / north metres around (lat0, lon0)."""

    def __init__(self, lat0_deg: float, lon0_deg: float):
        self.lat0, self.lon0 = float(lat0_deg), float(lon0_deg)
        phi = math.radians(self.lat0)
        w = math.sqrt(1.0 - _E2 * math.sin(phi) ** 2)
        self.r_north = WGS84_A * (1.0 - _E2) / w ** 3      # meridional radius of curvature
        self.r_east = WGS84_A / w * math.cos(phi)          # prime-vertical radius times cos(lat)

    def to_local(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        """(east, north) in metres."""
        return (math.radians(lon_deg - self.lon0) * self.r_east,
                math.radians(lat_deg - self.lat0) * self.r_north)

    def to_geo(self, east: float, north: float) -> tuple[float, float]:
        """(lat, lon) in degrees."""
        return (self.lat0 + math.degrees(north / self.r_north),
                self.lon0 + math.degrees(east / self.r_east))


def state_from_telemetry(frame: LocalFrame, t: float, lat_e7: int, lon_e7: int, vx_cms: float, vy_cms: float,
                         roll: float, yaw: float, airspeed: float) -> NavState:
    """MAVLink telemetry -> NavState. vx is the north and vy the east velocity (NED), in cm/s."""
    east, north = frame.to_local(lat_e7 * 1e-7, lon_e7 * 1e-7)
    return NavState(t=t, x=east, y=north, psi=yaw % (2.0 * math.pi), phi=roll, v=airspeed,
                    ground_vel=np.array([vy_cms * 0.01, vx_cms * 0.01]))
