"""Replanning navigator: one safe fly-to point per tick.

A companion computer that drives ArduPlane's GUIDED mode can only say "fly to
this point". The navigator turns the current picture (own position, course and
speed, the target, the no-fly zones, the area border) into that one point and
keeps four promises:

  1. the route it plans never crosses a zone or the border, and when no route
     exists it says so ("hold") instead of drawing a line through a zone,
  2. the aircraft is treated as something that has a course and a turn radius,
     not as a dot: the start of the route is chosen with the turn it needs in
     mind, and every commanded point is checked against a model of how the
     aircraft will really fly towards it ("guard"),
  3. the commanded point is far enough away that GUIDED turns hard towards it
     instead of easing onto a loiter circle around it,
  4. the answer comes within a predictable time: the search is over a fixed
     grid, so the worst case is bounded by the size of the area.

Layers, top to bottom (each one is a fallback for the one above it):

    plan    A* on the inflated grid + line-of-sight shortcut, heading-aware start
    stick   keep the previous route unless a new one is clearly better
    carrot  the farthest point of the route the aircraft can fly straight at
    guard   accept the point only if a hard half-turn out is still possible after flying
            towards it for a moment; otherwise command the nearest bearing that keeps one
    escape  inside a zone: the bearing that gets out soonest
    hold    no route: wait at a safe point the aircraft can reach
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import G
from .planner import GridMap, PlanningError, astar
from .shortcut import los_shortcut
from .world import Obstacle, World, points_in_polygon, points_polygon_distance


@dataclass(frozen=True)
class NavParams:
    grid_res: float = 5.0          # m per grid cell
    margin: float = 15.0           # m, planned distance to zones and to the border
    hard_margin: float = 5.0       # m, what the guard insists on for the predicted flight path
    roll_max_deg: float = 35.0     # bank limit used to predict turns
    roll_tau: float = 0.4          # s, roll response time constant
    roll_rate_max_deg: float = 60.0  # deg/s, roll rate limit
    l1_period: float = 17.0        # s, ArduPlane NAVL1_PERIOD
    l1_damping: float = 0.75       # ArduPlane NAVL1_DAMPING
    loiter_radius: float = 60.0    # m, ArduPlane WP_LOITER_RAD: GUIDED circles a point at this radius
    carrot: float = 150.0          # m, look-ahead along the route that sets the bearing to fly
    cmd_distance: float = 300.0    # m, the fly-to point is put this far out on that bearing: GUIDED turns
                                   # hard towards a far point but only eases onto the circle of a near one
    replan_period: float = 1.0     # s between route searches
    switch_gain: float = 0.15      # a new route must be this much cheaper to replace the old one
    turn_cost: float = 1.0         # weight of the first turn, in metres of arc flown
    guard_horizon: float = 1.5     # s flown towards a point before the 'is a way out still open' test
    slow_speed: float = 15.0       # m/s asked for while the guard or the escape is steering (tighter turns)
    escape_horizon: float = 15.0   # s rolled forward when looking for the way out of a zone
    d_follow: float = 40.0         # m, aim this far behind a moving target
    lead_tau: float = 0.5          # s, how far the target's position is extrapolated
    use_heading_start: bool = True
    use_stickiness: bool = True
    use_guard: bool = True
    use_carrot: bool = True        # False: command the next waypoint of the route (old behaviour)

    @property
    def roll_max(self) -> float:
        return math.radians(self.roll_max_deg)

    @property
    def roll_rate_max(self) -> float:
        return math.radians(self.roll_rate_max_deg)

    def turn_radius(self, v: float) -> float:
        return v * v / (G * math.tan(self.roll_max))


@dataclass
class NavCommand:
    point: np.ndarray              # (2,) where to fly, package frame (x east, y north)
    status: str                    # "route" | "guard" | "escape" | "hold"
    route: np.ndarray              # (N, 2) the route in force, starting at the aircraft
    goal: np.ndarray               # (2,) where the route ends
    plan_ms: float = 0.0           # wall-clock time of this update
    speed_limit: float | None = None   # m/s to slow down to (None = no request)
    replanned: bool = False
    reason: str = ""


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _bearing(a, b) -> float:
    return math.atan2(b[0] - a[0], b[1] - a[1])


def _unit(psi: float) -> np.ndarray:
    return np.array([math.sin(psi), math.cos(psi)])


def _length(pts) -> float:
    pts = np.asarray(pts, dtype=float)
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0


class Navigator:
    def __init__(self, polygon, params: NavParams | None = None):
        self.p = params or NavParams()
        self.polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
        self._zones = np.empty((0, 3))
        self._build()
        self._route: np.ndarray | None = None
        self._goal: np.ndarray | None = None
        self._last_plan_t = -math.inf
        self._guard_sign = 0.0
        self._guard_until = -math.inf
        self._blocked = False
        self._wind = (0.0, 0.0)
        self._phi = 0.0                           # bank estimated from how fast the course is changing
        self._last_course: tuple[float, float] | None = None

    # ------------------------------------------------------------------ map
    def set_zones(self, zones) -> bool:
        """Zones as rows (x, y, radius). Returns True if the set changed (the map is rebuilt)."""
        z = np.empty((0, 3)) if zones is None else np.asarray(zones, dtype=float).reshape(-1, 3)
        if z.shape == self._zones.shape and np.array_equal(z, self._zones):
            return False
        self._zones = z.copy()
        self._build()
        self._route = None                       # a new zone invalidates the route in force
        return True

    def _build(self) -> None:
        p = self.p
        self.world = World.from_polygon(self.polygon, [Obstacle(float(x), float(y), float(r)) for x, y, r in self._zones])
        self.grid = GridMap(self.world, p.grid_res, p.margin)
        g = self.grid
        xs = g.x0 + np.arange(g.nx) * g.res
        ys = g.y0 + np.arange(g.ny) * g.res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self._cells_xy = np.column_stack([X.ravel(), Y.ravel()])
        self._clear_field = self.clearance(self._cells_xy).reshape(g.nx, g.ny)   # physical, zones and border
        self._labels: np.ndarray | None = None

    def zone_clearance(self, pts) -> np.ndarray:
        """Distance of each point to the nearest zone edge (negative = inside); inf without zones."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(self._zones) == 0:
            return np.full(len(pts), np.inf)
        d = np.hypot(pts[:, None, 0] - self._zones[None, :, 0], pts[:, None, 1] - self._zones[None, :, 1])
        return (d - self._zones[None, :, 2]).min(axis=1)

    def border_clearance(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        d = points_polygon_distance(pts, self.polygon)
        return np.where(points_in_polygon(pts, self.polygon), d, -d)

    def clearance(self, pts) -> np.ndarray:
        return np.minimum(self.zone_clearance(pts), self.border_clearance(pts))

    def _components(self) -> np.ndarray:
        """Label the connected regions of free cells (lazy; only needed when a search fails)."""
        if self._labels is None:
            occ = self.grid.occ
            lab = np.full(occ.shape, -1, dtype=np.int32)
            nx, ny = occ.shape
            cur = 0
            for i0, j0 in np.argwhere(~occ):
                if lab[i0, j0] >= 0:
                    continue
                lab[i0, j0] = cur
                dq = deque([(int(i0), int(j0))])
                while dq:
                    i, j = dq.popleft()
                    for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                        if 0 <= ni < nx and 0 <= nj < ny and not occ[ni, nj] and lab[ni, nj] < 0:
                            lab[ni, nj] = cur
                            dq.append((ni, nj))
                cur += 1
            self._labels = lab
        return self._labels

    # ------------------------------------------------------------ geometry
    def _seg_ok(self, a, b, clearance: float) -> bool:
        return self.world.segment_clear(a, b, clearance) and self.world.segment_inside(a, b, max(clearance, 0.0))

    def free_point(self, p, max_candidates: int = 400):
        """p itself if its cell is free, else the nearest free cell reachable in a straight line.

        "Reachable" means the segment does not cross a zone or leave the area. A point that is
        already inside a zone or outside the area gets the nearest free cell (the shortest way out).
        Returns None if there is no free cell.
        """
        p = np.asarray(p, dtype=float)
        g = self.grid
        if g.is_free(*g.to_cell(*p)):
            return p
        free = ~g.occ.ravel()
        if not free.any():
            return None
        xy = self._cells_xy[free]
        order = np.argsort(np.hypot(xy[:, 0] - p[0], xy[:, 1] - p[1]))[:max_candidates]
        if float(self.clearance(p)[0]) < 0.0:
            return xy[order[0]]
        for k in order:
            if self._seg_ok(p, xy[k], 0.0):
                return xy[k]
        return None

    def _search(self, a, b) -> np.ndarray | None:
        """A* + line-of-sight shortcut between two free points; None if they are not connected."""
        m = self.p.margin
        try:
            raw = astar(self.grid, tuple(a), tuple(b)).path
        except PlanningError:
            return None
        if len(raw) < 2:
            return np.array([a, b], dtype=float)
        route, _ = los_shortcut(raw, lambda u, v: self._seg_ok(u, v, m))
        return route

    # ------------------------------------------------------------- planning
    def _turn(self, psi: float, a, b) -> float:
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        return 0.0 if d < 1e-6 else abs(_wrap(_bearing(a, b) - psi))

    def _plan(self, pos, psi: float, v: float, goal) -> tuple[np.ndarray | None, float]:
        """Cheapest route pos -> goal, where cost = length + the arc flown in the first turn."""
        p = self.p
        R = p.turn_radius(v)
        best, best_cost = None, math.inf

        begin = self.free_point(pos)
        if begin is not None:
            r = self._search(begin, goal)
            if r is not None:
                pts = np.vstack([pos, r]) if float(np.hypot(*(begin - pos))) > 1e-6 else r.copy()
                pts[0] = pos
                first = pts[1] if len(pts) > 1 else goal
                cost = _length(pts) + p.turn_cost * R * self._turn(psi, pos, first)
                best, best_cost = pts, cost

        if p.use_heading_start and (best is None or self._turn(psi, pos, best[1]) > math.radians(60.0)):
            # the direct route starts with a big turn: also try routes that start ahead of the aircraft
            lead = 1.5 * R
            for off in (0.0, math.radians(50.0), -math.radians(50.0)):
                q = pos + lead * _unit(psi + off)
                if not self.grid.is_free(*self.grid.to_cell(*q)) or not self._seg_ok(pos, q, p.hard_margin):
                    continue
                r = self._search(q, goal)
                if r is None:
                    continue
                nxt = r[1] if len(r) > 1 else goal
                cost = lead + _length(r) + p.turn_cost * R * (abs(off) + self._turn(psi + off, q, nxt))
                if cost < best_cost:
                    best, best_cost = np.vstack([pos, r]), cost
        return best, best_cost

    def _refresh(self, pos, goal) -> np.ndarray | None:
        """The route in force, seen from where the aircraft is now, ending at the new goal."""
        r = self._route
        if r is None or len(r) < 2:
            return None
        # leg the aircraft is on: the nearest one
        best_i, best_d = 0, math.inf
        for i in range(len(r) - 1):
            a, b = r[i], r[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 == 0.0 else float(np.clip((pos - a) @ ab / L2, 0.0, 1.0))
            d = float(np.hypot(*(pos - (a + t * ab))))
            if d < best_d - 1e-9:
                best_i, best_d = i, d
        rest = [w for w in r[best_i + 1:-1] if float(np.hypot(*(w - pos))) > self.p.grid_res]
        pts = np.array([pos, *rest, goal], dtype=float)
        m = self.p.margin
        for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            if not self._seg_ok(a, b, self.p.hard_margin if k == 0 else m - self.p.grid_res):
                return None
        return pts

    def _route_for(self, t: float, pos, psi: float, v: float, goal) -> tuple[np.ndarray | None, bool]:
        p = self.p
        old = self._refresh(pos, goal) if self._route is not None else None
        due = t - self._last_plan_t >= p.replan_period - 1e-9
        if old is not None and not due:
            return old, False
        new, new_cost = self._plan(pos, psi, v, goal)
        self._last_plan_t = t
        if new is None:
            return old, True                      # nothing better found: keep what still works
        if old is not None and p.use_stickiness:
            R = p.turn_radius(v)
            old_cost = _length(old) + p.turn_cost * R * self._turn(psi, pos, old[1])
            if old_cost <= new_cost * (1.0 + p.switch_gain):
                return old, True
        return new, True

    def _reachable_goal(self, pos, target) -> np.ndarray | None:
        """No route to the goal: the reachable free cell nearest to the target with room to circle."""
        begin = self.free_point(pos)
        if begin is None:
            return None
        lab = self._components()
        ci = self.grid.to_cell(*begin)
        mine = (lab == lab[ci]).ravel()
        if not mine.any():
            return None
        room = self._clear_field.ravel() >= self.p.loiter_radius + self.p.hard_margin
        cand = mine & room
        if not cand.any():                        # nowhere to circle safely: the roomiest cell
            idx = np.flatnonzero(mine)
            return self._cells_xy[idx[int(np.argmax(self._clear_field.ravel()[idx]))]]
        xy = self._cells_xy[cand]
        return xy[int(np.argmin(np.hypot(xy[:, 0] - target[0], xy[:, 1] - target[1])))]

    # ----------------------------------------------------------- fly-to point
    def _carrot(self, pos, route: np.ndarray, ext_dir: np.ndarray | None) -> np.ndarray:
        """Farthest point within `carrot` m along the route that the aircraft can fly straight at."""
        p = self.p
        thr = max(p.hard_margin, min(p.margin - p.grid_res, float(self.clearance(pos)[0]) - 1.0))
        pts = route
        if ext_dir is not None and _length(pts) < p.carrot:
            pts = np.vstack([pts, pts[-1] + ext_dir * (p.carrot - _length(pts))])   # fly on past the goal
        step = 10.0
        samples, used = [], 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.hypot(*(b - a)))
            if L < 1e-9:
                continue
            n = max(1, int(math.ceil(L / step)))
            for k in range(1, n + 1):
                s = min(L, k * L / n)
                if used + s > p.carrot + 1e-6:
                    break
                samples.append(a + (b - a) * (s / L))
            used += L
            if used >= p.carrot:
                break
        c = None
        for q in reversed(samples):
            if self._seg_ok(pos, q, thr):
                c = q
                break
        if c is None:
            c = route[1] if len(route) > 1 else route[0]
        d_min = 1.5 * p.loiter_radius
        d = float(np.hypot(*(c - pos)))
        if 1e-6 < d < d_min:
            # too close: GUIDED would start circling it. Push it out along the same bearing while that stays clear.
            u = (c - pos) / d
            e = d_min
            while e > d + 1e-6 and not self._seg_ok(pos, pos + u * e, p.hard_margin):
                e -= step
            c = pos + u * max(e, d)
        return np.asarray(c, dtype=float)

    # --------------------------------------------------------------- guard
    def _air(self, psi: float, v: float):
        """Air heading and airspeed from the ground track, ground speed and the wind estimate."""
        ax = v * math.sin(psi) - self._wind[0]
        ay = v * math.cos(psi) - self._wind[1]
        return math.atan2(ax, ay), max(math.hypot(ax, ay), 5.0)

    def _roll_step(self, phi, phi_cmd, dt: float):
        """Roll response of the aircraft: first-order lag with a rate limit (same shape as vehicle.FixedWing)."""
        p = self.p
        rate = np.clip((phi_cmd - phi) / p.roll_tau, -p.roll_rate_max, p.roll_rate_max)
        return phi + rate * dt

    def _rollout(self, pos, psi: float, v: float, bearings: np.ndarray, horizon: float, dt: float = 0.25):
        """Predicted flight for each bearing, the way GUIDED flies at a far point: the L1 law asks for a bank
        proportional to the sine of the track error, the roll follows with its lag and rate limit, the wind
        drifts everything. Starts from the bank the aircraft has now.
        Returns (paths (K, n, 2), air headings (K,), bank angles (K,))."""
        p = self.p
        ha, va = self._air(psi, v)
        wx, wy = self._wind
        k_l1 = 4.0 * p.l1_damping * math.pi / p.l1_period          # a_lat = k_l1 * V_ground * sin(error)
        K, n = len(bearings), max(2, int(round(horizon / dt)))
        hs = np.full(K, ha)
        phi = np.full(K, self._phi)
        x = np.full(K, float(pos[0]))
        y = np.full(K, float(pos[1]))
        out = np.empty((K, n, 2))
        for i in range(n):
            gx, gy = va * np.sin(hs) + wx, va * np.cos(hs) + wy
            err = np.clip(_wrap(bearings - np.arctan2(gx, gy)), -math.pi / 2, math.pi / 2)
            phi_cmd = np.clip(np.arctan(k_l1 * np.hypot(gx, gy) * np.sin(err) / G), -p.roll_max, p.roll_max)
            phi = self._roll_step(phi, phi_cmd, dt)
            hs = hs + G * np.tan(phi) / va * dt
            x = x + (va * np.sin(hs) + wx) * dt
            y = y + (va * np.cos(hs) + wy) * dt
            out[:, i, 0], out[:, i, 1] = x, y
        return out, hs, phi

    def _uturns(self, pts: np.ndarray, hs: np.ndarray, phis: np.ndarray, va: float, dt: float = 0.25) -> np.ndarray:
        """From each state: the hard half-turn to the left and to the right, (2, K, m, 2). The bank has to
        roll over from where it is first, so reversing a turn costs more room than tightening one."""
        p = self.p
        w = G * math.tan(p.roll_max) / va
        wx, wy = self._wind
        m = int(math.ceil((math.pi / w + 2.0 * p.roll_max / p.roll_rate_max + 2.0 * p.roll_tau) / dt))
        out = np.empty((2, len(pts), m, 2))
        for s_i, side in enumerate((-1.0, 1.0)):
            h, phi = hs.copy(), phis.copy()
            x, y = pts[:, 0].copy(), pts[:, 1].copy()
            for i in range(m):
                phi = self._roll_step(phi, side * p.roll_max, dt)
                h = h + G * np.tan(phi) / va * dt
                x = x + (va * np.sin(h) + wx) * dt
                y = y + (va * np.cos(h) + wy) * dt
                out[s_i, :, i, 0], out[s_i, :, i, 1] = x, y
        return out

    def _track_ok(self, pos, psi: float, v: float, route: np.ndarray, horizon: float = 12.0, dt: float = 0.25) -> bool:
        """Corridor case: can the route be followed from here with room to spare?

        True only if the predicted flight along the route keeps its distance for the whole `horizon`.
        A route that ends sooner is no corridor to fly through, so it cannot stand in for a way out.
        """
        p = self.p
        ha, va = self._air(psi, v)
        w = 0.8 * G * math.tan(p.roll_max) / va
        wx, wy = self._wind
        R = p.turn_radius(v)
        x, y, leg = float(pos[0]), float(pos[1]), 1
        lag = 2.0 * p.roll_tau + p.roll_max / p.roll_rate_max
        pts, ended = [], False
        for i in range(int(round(horizon / dt))):
            while leg < len(route) - 1 and math.hypot(route[leg][0] - x, route[leg][1] - y) < R:
                leg += 1
            gx, gy = va * math.sin(ha) + wx, va * math.cos(ha) + wy
            if i * dt >= lag:
                ha += float(np.clip(_wrap(math.atan2(route[leg][0] - x, route[leg][1] - y) - math.atan2(gx, gy)), -w * dt, w * dt))
            x += (va * math.sin(ha) + wx) * dt
            y += (va * math.cos(ha) + wy) * dt
            pts.append((x, y))
            if leg == len(route) - 1 and math.hypot(route[leg][0] - x, route[leg][1] - y) < R:
                ended = True
                break
        if ended:
            return False
        return float(self.clearance(np.array(pts)).min()) >= max(p.hard_margin, p.margin - p.grid_res)

    def _point_on_bearing(self, pos, bearing: float) -> np.ndarray:
        """The fly-to point for a bearing: `cmd_distance` out, pulled in while that would lie outside the area.

        The point only sets the direction (it is replaced several times a second), so it may lie beyond a
        zone; it is kept inside the area so that a lost link does not leave the aircraft heading out of it.
        """
        p = self.p
        d = p.cmd_distance
        u = _unit(bearing)
        while d > 2.5 * p.loiter_radius and float(self.border_clearance(pos + u * d)[0]) < p.hard_margin:
            d -= 10.0
        return pos + u * d

    _OFFSETS = np.radians([0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0])

    def _guard(self, t: float, pos, psi: float, v: float, point, route: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """Accept a fly-to point only if, after flying towards it for a moment, a way out is still open.

        "A way out" is a hard half-turn, to the left or to the right, that touches neither a zone nor
        the border (or, in a corridor too narrow to turn in, the planned route itself being flyable
        with room to spare). As long as every accepted command keeps one open, the aircraft is never
        committed to entering a zone. If the point fails the test, the nearest bearing that passes is
        commanded instead; if none passes, the half-turn that stays clearest.
        """
        p = self.p
        b0 = _bearing(pos, point)
        offs = self._OFFSETS
        if self._guard_sign != 0.0 and t < self._guard_until:     # keep turning the same way once started
            rest = offs[1:]
            same = np.sign(_wrap(b0 + rest - psi)) == self._guard_sign
            offs = np.concatenate([[0.0], rest[same], rest[~same]])
        _, va = self._air(psi, v)
        tr, hs, phis = self._rollout(pos, psi, v, b0 + offs, p.guard_horizon)
        K = len(offs)
        c = self.clearance(tr.reshape(-1, 2)).reshape(K, -1).min(axis=1)
        ut = self._uturns(tr[:, -1], hs, phis, va)
        cu = self.clearance(ut.reshape(-1, 2)).reshape(2, K, -1).min(axis=2)      # (side, K)
        ok = (c >= p.hard_margin) & (cu.max(axis=0) >= p.hard_margin)
        if ok[0]:
            return np.asarray(point, dtype=float), False
        if c[0] >= p.hard_margin and route is not None and self._track_ok(pos, psi, v, route):
            return np.asarray(point, dtype=float), False
        good = np.flatnonzero(ok)
        if len(good):
            bearing = b0 + offs[int(good[0])]
            self._guard_sign = float(np.sign(_wrap(bearing - psi))) or self._guard_sign
        else:
            # nothing keeps a way out open: fly the hard half-turn that stays clearest, starting now
            ha, _ = self._air(psi, v)
            now = self._uturns(pos[None, :], np.array([ha]), np.array([self._phi]), va)
            cn = self.clearance(now.reshape(-1, 2)).reshape(2, -1)
            score = np.where(cn.min(axis=1) >= 0.0, cn.min(axis=1), -(cn < 0.0).sum(axis=1).astype(float) * 1e3)
            side = -1.0 if score[0] > score[1] else 1.0
            if abs(score[0] - score[1]) < 1.0 and self._guard_sign != 0.0:
                side = self._guard_sign
            self._guard_sign = side
            bearing = psi + side * math.radians(120.0)
        self._guard_until = t + 3.0
        return self._point_on_bearing(pos, bearing), True

    def _escape(self, pos, psi: float, v: float) -> np.ndarray:
        """Inside a zone: the bearing whose predicted flight spends the least time inside (and stays in the area)."""
        p = self.p
        bearings = psi + np.radians(np.arange(0.0, 360.0, 30.0))
        tr, _, _ = self._rollout(pos, psi, v, bearings, p.escape_horizon)
        flat = tr.reshape(-1, 2)
        inside = (self.zone_clearance(flat) < 0.0).reshape(len(bearings), -1).sum(axis=1)
        out = (self.border_clearance(flat) < p.hard_margin).reshape(len(bearings), -1).sum(axis=1)
        score = inside + 10.0 * out + 1e-3 * np.abs(_wrap(bearings - psi))
        return self._point_on_bearing(pos, float(bearings[int(np.argmin(score))]))

    # ---------------------------------------------------------------- update
    def aim_point(self, target, target_vel=None) -> tuple[np.ndarray, np.ndarray | None]:
        """Where to go for a target: a little behind where it will be, so the aircraft joins from behind."""
        target = np.asarray(target, dtype=float)
        if target_vel is None:
            return target, None
        tv = np.asarray(target_vel, dtype=float)
        sp = float(np.hypot(*tv))
        if sp < 1e-6:
            return target, None
        u = tv / sp
        return target + tv * self.p.lead_tau - u * self.p.d_follow, u

    def update(self, t: float, pos, course_deg: float, speed: float, target, target_vel=None, wind=None) -> NavCommand:
        """One tick. `course_deg` is the ground track, clockwise from north; `speed` the ground speed;
        `wind` the (east, north) wind estimate in m/s if the autopilot reports one."""
        t0 = time.perf_counter()
        p = self.p
        pos = np.asarray(pos, dtype=float).reshape(2)
        psi = math.radians(course_deg)
        v = max(float(speed), 5.0)
        self._wind = (0.0, 0.0) if wind is None else (float(wind[0]), float(wind[1]))
        if self._last_course is not None and 1e-3 < t - self._last_course[0] < 1.0:
            rate = _wrap(psi - self._last_course[1]) / (t - self._last_course[0])
            self._phi = float(np.clip(math.atan(rate * v / G), -p.roll_max, p.roll_max))
        else:
            self._phi = 0.0
        self._last_course = (t, psi)

        def done(point, status, route, goal, replanned=False, reason=""):
            slow = p.slow_speed if status in ("guard", "escape") else None
            return NavCommand(np.asarray(point, dtype=float), status, np.asarray(route, dtype=float),
                              np.asarray(goal, dtype=float), (time.perf_counter() - t0) * 1e3, slow, replanned, reason)

        if float(self.zone_clearance(pos)[0]) < 0.0:
            self._route = None
            pt = self._escape(pos, psi, v)
            return done(pt, "escape", [pos, pt], pt, reason="inside a zone")

        aim, ext = self.aim_point(target, target_vel)
        goal = self.free_point(aim)
        blocked = False
        route, replanned = (None, False)
        if goal is not None:
            route, replanned = self._route_for(t, pos, psi, v, goal)
        if route is None:
            # no way to the goal from here: go to the reachable point nearest to the target and wait there
            blocked = True
            goal = self._reachable_goal(pos, np.asarray(target, dtype=float))
            if goal is not None:
                self._route = None
                route, _ = self._plan(pos, psi, v, goal)
                replanned = True
                self._last_plan_t = t
        if route is None:
            # not even that (the aircraft is boxed in): keep clear of everything with the guard alone
            pt = self._point_on_bearing(pos, psi)
            if p.use_guard:
                pt, _ = self._guard(t, pos, psi, v, pt)
            return done(pt, "hold", [pos, pt], pt, replanned=True, reason="no free route from here")
        self._route, self._goal = route, goal

        if not p.use_carrot:
            pt = route[1]
        elif blocked and _length(route) <= p.carrot:
            pt = goal                               # arrive and circle the waiting point (it has room for that)
        else:
            pt = self._carrot(pos, route, None if blocked else ext)
        status = "hold" if blocked else "route"
        circle_it = blocked and _length(route) <= p.carrot
        if p.use_guard:
            pt, hit = self._guard(t, pos, psi, v, pt, route)
            if hit:
                status = "guard"
                circle_it = False
        if p.use_carrot and not circle_it and not (status == "guard"):
            pt = self._point_on_bearing(pos, _bearing(pos, pt))   # same bearing, far point
        return done(pt, status, route, goal, replanned=replanned, reason="target not reachable" if blocked else "")
