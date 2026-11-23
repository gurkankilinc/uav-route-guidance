"""Artificial potential field (APF) guidance: no route, a steering direction straight from the field.

The aircraft is pulled towards the aim point (`d_follow` behind the target) and pushed away from
every known no-fly zone and from the border of the area. The resulting vector gives the desired
course; a heading loop turns that into a roll command for the coordinated-turn model. Speed
follows the same approach profile as the route planners, so only the steering differs.

Two variants are compared:

``APF``
    Classic Khatib-style field (Khatib 1986): unit attraction, repulsion
    ``k_rep * rho0 * (1/rho - 1/rho0)`` inside the influence distance ``rho0`` of a zone edge.

``APF + vortex``
    The two textbook remedies on top of it:

    * a tangential ("vortex", rotational) component around each zone, so a zone straight between
      the aircraft and the target is circled instead of balancing the pull (local minimum),
    * the Ge & Cui (2000) goal-distance factor, which fades the repulsion near the aim point, so a
      target flying next to a zone can still be reached (GNRON: goal non-reachable with
      obstacles nearby).

For a fixed-wing aircraft the influence distance has to be tied to the turn radius: the field can
ask for any direction instantly, the aircraft needs about one turn radius to change course by
90 degrees. By default ``rho0 = 2 R(V)`` with ``R(V) = V^2 / (g tan phi_max)`` at the current
airspeed, never less than at cruise (117 m at 20 m/s, 196 m at 26 m/s).
"""

from __future__ import annotations

import math
import numpy as np

from .config import FieldParams, G, VehicleParams


# gains picked on scenarios 100-119 (guard off), kept apart from the 0-19 evaluation set;
# see scripts/tune_potential.py
FIELD_METHODS: dict[str, FieldParams] = {
    "APF": FieldParams(k_rep=0.5, influence_turns=3.0),
    "APF + vortex": FieldParams(k_rep=0.5, vortex=1.0, goal_exp=2.0),
}


class PotentialField:
    def __init__(self, params: FieldParams, veh: VehicleParams, margin: float, bounds):
        self.p = params
        self.veh = veh
        self.margin = float(margin)
        self.bounds = bounds
        self.rho0 = params.influence_turns * veh.min_turn_radius()      # at cruise

    def influence(self, v: float | None) -> float:
        """rho0 for airspeed v: a faster aircraft turns wider, so zones must push it earlier."""
        if v is None:
            return self.rho0
        return self.p.influence_turns * self.veh.min_turn_radius(max(v, self.veh.v_cruise))

    def vector(self, pos, aim, course: float, C: np.ndarray, R: np.ndarray, v: float | None = None) -> np.ndarray:
        """Field at `pos` (east, north). `course` (rad, clockwise from north) picks the vortex side,
        `v` (airspeed) sets the influence distance."""
        p = self.p
        pos = np.asarray(pos, dtype=float)
        to_aim = np.asarray(aim, dtype=float) - pos
        d_aim = float(np.hypot(*to_aim))
        f = to_aim / d_aim if d_aim > 1e-6 else np.zeros(2)
        rho0 = self.influence(v)
        # Ge & Cui: repulsion scaled by (distance to goal / rho0)^n, capped at 1
        goal_scale = min(1.0, d_aim / rho0) ** p.goal_exp if p.goal_exp > 0 else 1.0
        heading = np.array([math.sin(course), math.cos(course)])

        if len(R):
            diff = pos[None, :] - C
            dist = np.hypot(diff[:, 0], diff[:, 1])
            rho = np.maximum(dist - (R + self.margin), 1.0)
            inside = rho < rho0
            for i in np.flatnonzero(inside):
                nrm = diff[i] / max(dist[i], 1e-9)               # points away from the zone
                mag = p.k_rep * rho0 * (1.0 / rho[i] - 1.0 / rho0) * goal_scale
                f = f + mag * nrm
                if p.vortex > 0.0:
                    tan = np.array([nrm[1], -nrm[0]])            # one of the two tangents
                    if tan @ heading < 0.0:                      # keep circling the way we already turn
                        tan = -tan
                    f = f + p.vortex * mag * tan

        if self.bounds is not None:
            x0, y0, x1, y1 = self.bounds
            m = p.wall_margin
            for rho_w, nrm in ((pos[0] - x0 - m, (1.0, 0.0)), (x1 - m - pos[0], (-1.0, 0.0)),
                               (pos[1] - y0 - m, (0.0, 1.0)), (y1 - m - pos[1], (0.0, -1.0))):
                if rho_w < rho0:
                    rho_w = max(rho_w, 1.0)
                    f = f + p.k_rep * rho0 * (1.0 / rho_w - 1.0 / rho0) * np.asarray(nrm)
        return f

    def roll_command(self, s, ground_vel, aim, C, R) -> float:
        """Course from the field, heading loop, coordinated-turn roll; limited to the bank limit."""
        gv = np.asarray(ground_vel, dtype=float)
        course = math.atan2(gv[0], gv[1]) if np.hypot(*gv) > 0.5 else s.psi
        f = self.vector(s.pos, aim, course, C, R, s.v)
        if np.hypot(*f) < 1e-9:
            return 0.0
        want = math.atan2(f[0], f[1])
        err = (want - course + math.pi) % (2.0 * math.pi) - math.pi
        rate = self.p.heading_gain * err
        phi = math.atan(s.v * rate / G)
        return float(np.clip(phi, -self.veh.roll_max, self.veh.roll_max))

    def streamline(self, pos, aim, course: float, C, R, length: float = 200.0, step: float = 5.0,
                   v: float | None = None) -> np.ndarray:
        """Where the field points from `pos` (for display only; ignores the aircraft's dynamics)."""
        pts = [np.asarray(pos, dtype=float)]
        c = course
        for _ in range(int(length / step)):
            f = self.vector(pts[-1], aim, c, C, R, v)
            n = float(np.hypot(*f))
            if n < 1e-9 or np.hypot(*(np.asarray(aim) - pts[-1])) < step:
                break
            c = math.atan2(f[0], f[1])
            pts.append(pts[-1] + step * f / n)
        return np.array(pts)
