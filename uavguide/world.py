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

    def contains(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.bounds
        return x0 <= x <= x1 and y0 <= y <= y1

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
