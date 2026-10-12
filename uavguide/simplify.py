"""8.4 - Douglas-Peucker route simplification.

Plain DP only bounds the geometric error; a shortened segment can still
cut through an obstacle. `douglas_peucker` accepts an optional
`segment_ok(a, b)` predicate: a span is split whenever its error exceeds
epsilon OR its straight segment is not acceptable. Because consecutive
points of the input route are already collision-free, this always
terminates with a safe result.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

SegmentCheck = Callable[[np.ndarray, np.ndarray], bool]


def point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each of `points` (N, 2) to the segment a-b."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0.0:
        return np.hypot(*(points - a).T)
    t = np.clip((points - a) @ ab / L2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.hypot(*(points - proj).T)


def douglas_peucker(
    points,
    epsilon: float,
    segment_ok: SegmentCheck | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (simplified_points, keep_mask). Iterative, so no recursion limit."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return pts.copy(), keep
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        d = point_segment_distance(pts[s + 1:e], pts[s], pts[e])
        k = int(np.argmax(d))
        too_far = d[k] > epsilon
        unsafe = segment_ok is not None and not segment_ok(pts[s], pts[e])
        if too_far or unsafe:
            idx = s + 1 + k
            keep[idx] = True
            stack.append((s, idx))
            stack.append((idx, e))
    return pts[keep], keep
