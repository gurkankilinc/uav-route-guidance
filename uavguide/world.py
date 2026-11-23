"""Static environment: rectangular area with circular obstacles."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    r: float


@dataclass
class World:
    bounds: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    obstacles: list[Obstacle] = field(default_factory=list)
    polygon: np.ndarray | None = None          # (N, 2) area border; None = the `bounds` rectangle

    @classmethod
    def from_polygon(cls, vertices, obstacles=None) -> "World":
        """Area given by a simple polygon (e.g. the competition border); `bounds` is its bounding box."""
        poly = np.asarray(vertices, dtype=float).reshape(-1, 2)
        if len(poly) < 3:
            raise ValueError("a polygon needs at least 3 vertices")
        bounds = (float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max()))
        return cls(bounds, list(obstacles or []), poly)

    def contains(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.bounds
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
        return self.polygon is None or bool(points_in_polygon(np.array([[x, y]], dtype=float), self.polygon)[0])

    def border_clearance(self, p) -> float:
        """Distance from p to the area border (negative = outside)."""
        p = np.asarray(p, dtype=float)
        if self.polygon is None:
            x0, y0, x1, y1 = self.bounds
            return float(min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]))
        d = float(points_polygon_distance(p[None, :], self.polygon)[0])
        return d if self.contains(p[0], p[1]) else -d

    def nearest_border_point(self, p) -> np.ndarray:
        """Closest point to p on the polygon border (polygon worlds only)."""
        p = np.asarray(p, dtype=float)
        best, best_d = p, float("inf")
        for a, b in zip(self.polygon, np.roll(self.polygon, -1, axis=0)):
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 == 0.0 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
            q = a + t * ab
            d = float(np.hypot(*(p - q)))
            if d < best_d:
                best, best_d = q, d
        return best

    def segment_inside(self, a, b, clearance: float = 0.0) -> bool:
        """True if segment a-b stays inside the area, at least `clearance` m away from its border."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if self.polygon is None:        # a rectangle is convex: the endpoints decide
            return self.border_clearance(a) >= clearance and self.border_clearance(b) >= clearance
        if not (self.contains(*a) and self.contains(*b)):
            return False
        poly = self.polygon
        for p, q in zip(poly, np.roll(poly, -1, axis=0)):
            if _segments_distance(a, b, p, q) < max(clearance, 1e-9):
                return False
        return True

    def clearance(self, p) -> float:
        """Distance from point p to the nearest obstacle boundary (negative = inside)."""
        if not self.obstacles:
            return float("inf")
        c = np.array([(o.x, o.y) for o in self.obstacles])
        r = np.array([o.r for o in self.obstacles])
        return float(np.min(np.hypot(c[:, 0] - p[0], c[:, 1] - p[1]) - r))

    def segment_clear(self, a, b, clearance: float = 0.0) -> bool:
        """True if segment a-b stays at least `clearance` m away from every obstacle."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ab = b - a
        L2 = float(ab @ ab)
        for o in self.obstacles:
            c = np.array([o.x, o.y])
            t = 0.0 if L2 == 0.0 else float(np.clip((c - a) @ ab / L2, 0.0, 1.0))
            if np.hypot(*(a + t * ab - c)) < o.r + clearance:
                return False
        return True

    def path_clearance(self, path) -> float:
        """Minimum clearance along a polyline, checked segment by segment."""
        path = np.asarray(path, dtype=float)
        best = float("inf")
        for o in self.obstacles:
            c = np.array([o.x, o.y])
            for a, b in zip(path[:-1], path[1:]):
                ab = b - a
                L2 = float(ab @ ab)
                t = 0.0 if L2 == 0.0 else float(np.clip((c - a) @ ab / L2, 0.0, 1.0))
                best = min(best, float(np.hypot(*(a + t * ab - c))) - o.r)
        return best


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule for many points at once; pts (M, 2), poly (N, 2) -> bool (M,)."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if y1 == y2:
            continue
        cross = (y1 > y) != (y2 > y)
        xi = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cross & (x < xi)
    return inside


def points_polygon_distance(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Unsigned distance from each point to the polygon border."""
    best = np.full(len(pts), np.inf)
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        pq = q - p
        L2 = float(pq @ pq)
        t = np.zeros(len(pts)) if L2 == 0.0 else np.clip((pts - p) @ pq / L2, 0.0, 1.0)
        best = np.minimum(best, np.hypot(*(pts - (p + t[:, None] * pq)).T))
    return best


def _point_segment(p, a, b) -> float:
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 == 0.0 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.hypot(*(p - (a + t * ab))))


def _segments_distance(a, b, c, d) -> float:
    """Shortest distance between segments a-b and c-d (0 if they cross)."""
    def orient(p, q, r) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return 0.0
    return min(_point_segment(a, c, d), _point_segment(b, c, d), _point_segment(c, a, b), _point_segment(d, a, b))


def default_world(bounds=(0.0, 0.0, 600.0, 600.0)) -> World:
    """Hand-placed demo scenario: start (30, 30) -> goal (570, 570)."""
    obs = [
        (150, 120, 25), (120, 260, 20), (260, 200, 30), (330, 330, 25),
        (220, 400, 20), (420, 250, 25), (450, 450, 30), (350, 500, 15),
        (500, 350, 20), (300, 80, 15), (540, 180, 20), (150, 500, 25),
    ]
    return World(bounds, [Obstacle(float(x), float(y), float(r)) for x, y, r in obs])


def random_world(
    seed: int,
    n: int = 12,
    r_range: tuple[float, float] = (10.0, 30.0),
    keep_clear: tuple = ((30.0, 30.0), (570.0, 570.0)),
    clear_radius: float = 60.0,
    bounds=(0.0, 0.0, 600.0, 600.0),
) -> World:
    """Reproducible random obstacle field that never covers the start/goal."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bounds
    obstacles: list[Obstacle] = []
    tries = 0
    while len(obstacles) < n and tries < 10_000:
        tries += 1
        r = float(rng.uniform(*r_range))
        x = float(rng.uniform(x0 + r, x1 - r))
        y = float(rng.uniform(y0 + r, y1 - r))
        if any(np.hypot(x - px, y - py) < r + clear_radius for px, py in keep_clear):
            continue
        if any(np.hypot(x - o.x, y - o.y) < r + o.r + 10.0 for o in obstacles):
            continue
        obstacles.append(Obstacle(x, y, r))
    return World(bounds, obstacles)
