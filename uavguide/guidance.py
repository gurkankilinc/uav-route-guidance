"""8.3 - L1 lateral guidance, ported from ArduPilot's AP_L1_Control.

The logic follows `AP_L1_Control::update_waypoint` and `turn_distance`
closely so that the simulation can later be compared with ArduPlane SITL:

    L1    = zeta * T * |V_g| / pi
    K     = 4 zeta^2
    a_lat = K * |V_g|^2 / L1 * sin(Nu),     Nu = Nu1 + Nu2
    Nu1   = asin(xtrack / L1)               (position error term)
    Nu2   = angle between ground velocity and the track (velocity term)
    roll  = atan(a_lat / g)

Internally the maths is done in a north/east frame exactly like ArduPilot
(`a % b = a.n * b.e - a.e * b.n`). The public API uses the package frame
(x = east, y = north) and reports cross-track error as *positive to the
right of the track*, which is how the task phrases it ("15 m to the right").

Not ported: `_prevent_indecision` (only matters when the target is almost
exactly behind the aircraft) and the loiter/heading-hold modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import G, GuidanceParams, VehicleParams

_SIN45 = 0.7071


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    """ArduPilot's Vector2f % operator on (north, east) vectors."""
    return a[0] * b[1] - a[1] * b[0]


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _ne(v) -> tuple[float, float]:
    """(east, north) -> (north, east)."""
    return (float(v[1]), float(v[0]))


@dataclass(frozen=True)
class L1Output:
    roll_cmd: float        # rad, positive = right wing down, already limited
    lat_acc: float         # m/s^2 demanded lateral acceleration (positive = right)
    xtrack_right: float    # m, cross-track error, positive = right of track A->B
    nu: float              # rad, total L1 angle after limiting
    l1_dist: float         # m
    mode: str              # "track" | "to_A" | "to_B"


class L1Guidance:
    def __init__(self, params: GuidanceParams, vehicle: VehicleParams):
        self.p = params
        self.roll_max = vehicle.roll_max
        self.xtrack_i = 0.0

    def reset(self) -> None:
        self.xtrack_i = 0.0

    def l1_distance(self, groundspeed: float) -> float:
        return self.p.l1_distance(max(groundspeed, 0.1))

    def update_waypoint(self, pos, ground_vel, wp_a, wp_b, dt: float) -> L1Output:
        p = self.p
        k_l1 = p.k_l1
        gv = _ne(ground_vel)
        gs = math.hypot(*gv)
        if gs < 0.1:
            gs = 0.1
        l1 = self.l1_distance(gs)

        pos_ne, a_ne, b_ne = _ne(pos), _ne(wp_a), _ne(wp_b)
        ab = (b_ne[0] - a_ne[0], b_ne[1] - a_ne[1])
        ab_len = math.hypot(*ab)
        if ab_len < 1e-6:                          # degenerate segment: aim at B directly
            ab = (b_ne[0] - pos_ne[0], b_ne[1] - pos_ne[1])
            ab_len = max(math.hypot(*ab), 1e-6)
        ab = (ab[0] / ab_len, ab[1] / ab_len)

        a_air = (pos_ne[0] - a_ne[0], pos_ne[1] - a_ne[1])
        xtrack = _cross(a_air, ab)                 # ArduPilot sign: negative = right of track
        wp_a_dist = math.hypot(*a_air)
        along = _dot(a_air, ab)

        if wp_a_dist > l1 and along / max(wp_a_dist, 1.0) < -_SIN45:
            # far behind A: fly towards A first
            u = (-a_air[0] / wp_a_dist, -a_air[1] / wp_a_dist)
            nu = math.atan2(_cross(gv, u), _dot(gv, u))
            mode = "to_A"
        elif along > ab_len + gs * 3.0:
            # well past B: fly back towards B
            b_air = (pos_ne[0] - b_ne[0], pos_ne[1] - b_ne[1])
            d = max(math.hypot(*b_air), 1e-6)
            u = (-b_air[0] / d, -b_air[1] / d)
            nu = math.atan2(_cross(gv, u), _dot(gv, u))
            mode = "to_B"
        else:
            nu2 = math.atan2(_cross(gv, ab), _dot(gv, ab))
            sine_nu1 = max(-_SIN45, min(_SIN45, xtrack / max(l1, 0.1)))
            nu1 = math.asin(sine_nu1)
            if p.xtrack_i_gain <= 0.0:
                self.xtrack_i = 0.0
            elif abs(nu1) < math.radians(5.0):
                self.xtrack_i += nu1 * p.xtrack_i_gain * dt
                self.xtrack_i = max(-0.1, min(0.1, self.xtrack_i))
            nu1 += self.xtrack_i
            nu = nu1 + nu2
            mode = "track"

        nu = max(-math.pi / 2, min(math.pi / 2, nu))
        lat_acc = k_l1 * gs * gs / l1 * math.sin(nu)
        roll = math.atan(lat_acc / G)
        roll = max(-self.roll_max, min(self.roll_max, roll))
        return L1Output(roll_cmd=roll, lat_acc=lat_acc, xtrack_right=-xtrack,
                        nu=nu, l1_dist=l1, mode=mode)

    def turn_distance(self, wp_radius: float, turn_angle_deg: float, groundspeed: float) -> float:
        """Distance before a waypoint at which ArduPlane starts the next leg."""
        d90 = min(wp_radius, self.l1_distance(groundspeed))
        ang = abs(turn_angle_deg)
        return d90 if ang >= 90.0 else d90 * ang / 90.0


def initial_roll_for_offset(offset_right: float, params: GuidanceParams, v: float = 20.0) -> float:
    """Roll command [rad] when flying parallel to the track, `offset_right` m to its right.

    Closed form of the L1 law: with L1 = zeta*T*V/pi and K = 4 zeta^2 the speed cancels,
    phi = -atan(4 pi^2 e / (g T^2)) (negative = left), valid while |e| < 0.7071 L1.
    """
    l1 = params.l1_distance(v)
    s = max(-_SIN45, min(_SIN45, -offset_right / l1))
    return math.atan(params.k_l1 * v * v / l1 * s / G)
