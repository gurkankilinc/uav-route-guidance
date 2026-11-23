"""ArduPlane's GUIDED mode as a companion computer sees it.

In GUIDED the aircraft is given one point (MAV_CMD_DO_REPOSITION). It does not
fly *through* the point: ArduPlane treats it as the centre of a loiter. As in
`Plane::update_loiter_update_nav`, while the point is more than three loiter
radii away the aircraft tracks the line from where it was when the command
arrived to the point (`AP_L1_Control::update_waypoint`); closer in it runs
`AP_L1_Control::update_loiter`, which eases onto the circle of radius
WP_LOITER_RAD around the point. `GuidedPlane` ports both on top of the
package's fixed-wing model, so a controller that only emits fly-to points can
be tested in closed loop without SITL.

It is a model, not ArduPlane: no TECS, no wind estimation, no sensor noise,
and `_prevent_indecision` is left out. Numbers from it are for comparing
controllers with each other, not for predicting a real flight.
"""

from __future__ import annotations

import math

import numpy as np

from .config import G, GuidanceParams, VehicleParams
from .guidance import L1Guidance
from .vehicle import FixedWing, VehicleState


class GuidedPlane:
    def __init__(self, vehicle: VehicleParams | None = None, guidance: GuidanceParams | None = None,
                 loiter_radius: float = 60.0, loiter_direction: int = 1):
        self.veh = vehicle or VehicleParams()
        self.gp = guidance or GuidanceParams()
        self.model = FixedWing(self.veh)
        self.radius = float(loiter_radius)
        self.direction = 1 if loiter_direction >= 0 else -1    # +1 clockwise
        self.l1 = L1Guidance(self.gp, self.veh)
        self.point: np.ndarray | None = None
        self.prev: np.ndarray | None = None       # where the aircraft was when the command arrived
        self.circling = False
        self._reached = False                     # ArduPlane's loiter.start_time_ms != 0
        self._pos: np.ndarray | None = None

    def command(self, point) -> None:
        """MAV_CMD_DO_REPOSITION: the new loiter centre (set_guided_WP: prev = current location)."""
        self.point = np.asarray(point, dtype=float).reshape(2).copy()
        self.prev = None if self._pos is None else self._pos.copy()
        self._reached = False

    def roll_command(self, s: VehicleState, wind, dt: float = 0.05) -> float:
        """Far from the point: line tracking (update_waypoint). Near it: AP_L1_Control::update_loiter,
        ported with the maths in north/east like ArduPilot."""
        self._pos = s.pos
        if self.point is None:
            return 0.0
        if self.prev is None:
            self.prev = s.pos.copy()
        if not self._reached and math.hypot(s.x - self.point[0], s.y - self.point[1]) > 3.0 * self.radius:
            self.circling = False
            return self.l1.update_waypoint(s.pos, s.ground_velocity(wind), self.prev, self.point, dt).roll_cmd
        period, damping = self.gp.l1_period, self.gp.l1_damping
        omega = 2.0 * math.pi / period
        kx, kv = omega * omega, 2.0 * damping * omega
        k_l1 = 4.0 * damping * damping
        gv_e, gv_n = s.ground_velocity(wind)
        gv = (float(gv_n), float(gv_e))
        gs = max(math.hypot(*gv), 1.0)
        l1 = damping * period * gs / math.pi

        a_air = (s.y - float(self.point[1]), s.x - float(self.point[0]))      # centre -> aircraft, (north, east)
        dist = math.hypot(*a_air)
        if dist > 0.1:
            u = (a_air[0] / dist, a_air[1] / dist)
        else:
            u = (gv[0] / gs, gv[1] / gs)

        xtrack_vel_cap = u[0] * gv[1] - u[1] * gv[0]           # velocity across the radial
        ltrack_vel_cap = -(gv[0] * u[0] + gv[1] * u[1])        # velocity along the radial, inbound
        nu = max(-math.pi / 2, min(math.pi / 2, math.atan2(xtrack_vel_cap, ltrack_vel_cap)))
        lat_cap = k_l1 * gs * gs / l1 * math.sin(nu)

        xtrack_vel_circ = -ltrack_vel_cap                       # radial outbound velocity
        xtrack_err_circ = dist - self.radius
        lat_pd = xtrack_err_circ * kx + xtrack_vel_circ * kv
        vel_tangent = xtrack_vel_cap * self.direction
        lat_ctr = vel_tangent * vel_tangent / max(0.5 * self.radius, self.radius + xtrack_err_circ)
        if ltrack_vel_cap < 0.0:
            if vel_tangent < 0.0:
                lat_pd = max(lat_pd, 0.0)
            elif xtrack_err_circ < 0.0:
                lat_pd = max(lat_pd, -lat_ctr)
        lat_circ = self.direction * (lat_pd + lat_ctr)

        if xtrack_err_circ > 0.0 and self.direction * lat_cap < self.direction * lat_circ:
            lat, self.circling = lat_cap, False
        else:
            lat, self.circling = lat_circ, True
            self._reached = True
        roll = math.atan(lat / G)
        return max(-self.veh.roll_max, min(self.veh.roll_max, roll))

    def step(self, s: VehicleState, v_cmd: float, wind, dt: float) -> VehicleState:
        return self.model.step(s, self.roll_command(s, wind, dt), v_cmd, wind, dt)
