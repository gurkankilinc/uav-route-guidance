"""Line-of-sight shortcut ("string pulling") for a grid route.

A* on an 8-connected grid can only move in 8 directions, so its route is a
few percent longer than necessary and has hundreds of points. Douglas-Peucker
removes points but keeps the route's shape. The shortcut instead jumps, from
each kept point, to the farthest later point it can reach in a straight line
that `segment_ok` accepts. With the full safety margin as the acceptance test
the result keeps that margin, which a DP tolerance does not.

Consecutive points of the input are assumed to be acceptable (they are for a
route on the inflated grid), so the result is always a valid route.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

SegmentCheck = Callable[[np.ndarray, np.ndarray], bool]


def los_shortcut(points, segment_ok: SegmentCheck) -> tuple[np.ndarray, np.ndarray]:
    """Return (shortened_points, keep_mask). Greedy: farthest reachable point first."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return pts.copy(), keep
    keep[0] = keep[-1] = True
    i = 0
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not segment_ok(pts[i], pts[j]):
            j -= 1
        keep[j] = True
        i = j
    return pts[keep], keep


def corner_cut(turn_radius: float, turn_deg: float) -> float:
    """How far a turn of radius R cuts inside a corner of `turn_deg`: R (sec(theta/2) - 1)."""
    half = np.radians(min(abs(turn_deg), 179.0)) / 2.0
    return float(turn_radius * (1.0 / np.cos(half) - 1.0))


def margin_for(turn_radius: float, max_turn_deg: float = 45.0, track_error: float = 5.0) -> float:
    """Safety margin from the aircraft, not from a guess: tracking error + corner cutting.

    A route that wraps around a zone has its corners on the zone side of the
    turn, so the aircraft cuts towards the zone by `corner_cut`. This is the
    geometry of an ideal circular turn; the value to fly with comes from the
    cross-track error measured in SITL.
    """
    return float(track_error + corner_cut(turn_radius, max_turn_deg))
