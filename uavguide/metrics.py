"""Numbers that say how a planner did in one pursuit, and a way to rank them.

Ranking is two-stage on purpose: a planner that flew into a no-fly zone, or
never caught the target, is *eliminated* rather than penalised, because no
amount of shortness makes up for a violation. Among the rest, the weighted
score is only as meaningful as the weights, so they are explicit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .config import Config
from .pursuit import PursuitLog, PursuitScenario


@dataclass(frozen=True)
class Metrics:
    planner: str
    captured: bool
    collided: bool                 # flown through a zone (physical radius, not the margin)
    left_area: bool                # flown outside the area
    success: bool                  # captured, no zone entered, never left the area
    time_to_capture: float         # s, start of the successful hold; nan if never caught
    flown_length: float            # m
    min_clearance: float           # m, flown, to the physical zone edge
    planned_clearance: float       # m, first planned route against the real zones
    n_replans: int
    plan_ms_mean: float
    plan_ms_max: float
    nodes_total: int
    mean_waypoints: float
    max_roll_deg: float
    mean_abs_roll_deg: float
    track_rms_m: float             # rms(distance - d_follow) after the capture; nan if never caught

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(log: PursuitLog, scenario: PursuitScenario, cfg: Config | None = None) -> Metrics:
    cfg = cfg or Config()
    ok = [e for e in log.replans if e.ok]
    ms = np.array([e.plan_ms for e in ok]) if ok else np.zeros(1)
    first = ok[0].route if ok else log.pos[:1]
    min_clr = float(np.min(log.clearance)) if len(log.clearance) else math.inf
    collided = min_clr < 0.0
    x0, y0, x1, y1 = scenario.world.bounds
    left_area = bool(np.any((log.pos[:, 0] < x0) | (log.pos[:, 0] > x1) | (log.pos[:, 1] < y0) | (log.pos[:, 1] > y1)))
    track = math.nan
    if log.captured:
        after = log.t >= log.capture_t
        track = float(np.sqrt(np.mean((log.dist[after] - cfg.approach.d_follow) ** 2)))
    return Metrics(
        planner=log.planner,
        captured=log.captured,
        collided=collided,
        left_area=left_area,
        success=log.captured and not collided and not left_area,
        time_to_capture=log.capture_t if log.captured else math.nan,
        flown_length=float(np.sum(np.hypot(*np.diff(log.pos, axis=0).T))),
        min_clearance=min_clr,
        planned_clearance=scenario.world.path_clearance(first) if len(first) > 1 else math.inf,
        n_replans=len(log.replans),
        plan_ms_mean=float(ms.mean()),
        plan_ms_max=float(ms.max()),
        nodes_total=int(sum(e.expanded for e in ok)),
        mean_waypoints=float(np.mean([len(e.route) for e in ok])) if ok else 0.0,
        max_roll_deg=float(np.degrees(np.max(np.abs(log.phi)))),
        mean_abs_roll_deg=float(np.degrees(np.mean(np.abs(log.phi)))),
        track_rms_m=track,
    )


# metric -> weight, lower is better except where negated below
DEFAULT_WEIGHTS = {
    "time_to_capture": 0.4,
    "flown_length": 0.2,
    "plan_ms_mean": 0.2,
    "min_clearance": 0.2,
}
_HIGHER_IS_BETTER = {"min_clearance"}


def rank(results: list[Metrics], weights: dict[str, float] | None = None) -> list[tuple[str, float | None]]:
    """Best first. Eliminated planners (not `success`) come last with score None."""
    weights = weights or DEFAULT_WEIGHTS
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    alive = [m for m in results if m.success]
    scores = {m.planner: 0.0 for m in alive}
    for key, w in weights.items():
        vals = np.array([getattr(m, key) for m in alive], dtype=float)
        if len(vals) == 0:
            continue
        if key in _HIGHER_IS_BETTER:
            vals = -vals
        span = vals.max() - vals.min()
        norm = (vals - vals.min()) / span if span > 1e-12 else np.zeros_like(vals)
        for m, x in zip(alive, norm):
            scores[m.planner] += w * float(x) / total
    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    out: list[tuple[str, float | None]] = [(name, sc) for name, sc in ordered]
    out += [(m.planner, None) for m in results if not m.success]
    return out
