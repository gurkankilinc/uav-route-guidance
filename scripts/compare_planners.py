"""Compare A* and Theta* on the default scenario and on random obstacle fields.

    python scripts/compare_planners.py
"""

from __future__ import annotations

import time

import numpy as np

from uavguide.config import Config
from uavguide.planner import PlanningError, plan_route
from uavguide.simplify import douglas_peucker
from uavguide.theta_star import plan_route_theta
from uavguide.world import default_world, random_world

START, GOAL = (30.0, 30.0), (570.0, 570.0)
PLANNERS = {"A*": plan_route, "Theta*": plan_route_theta}


def run(world, name, fn, cfg):
    t0 = time.perf_counter()
    res = fn(world, START, GOAL, cfg.grid_res, cfg.safety_margin)
    ms = (time.perf_counter() - t0) * 1e3
    safe = lambda a, b: world.segment_clear(a, b, cfg.dp_clearance)
    simple, _ = douglas_peucker(res.path, cfg.dp_epsilon, segment_ok=safe)
    return dict(length=res.length, raw=len(res.path), dp=len(simple), ms=ms,
                clearance=world.path_clearance(simple), expanded=res.expanded)


def main() -> None:
    cfg = Config().planner

    print("Default scenario")
    print(f"{'planner':<8} {'length m':>9} {'raw pts':>8} {'DP pts':>7} {'ms':>6} {'nodes':>7} {'min clr m':>10}")
    for name, fn in PLANNERS.items():
        r = run(default_world(), name, fn, cfg)
        print(f"{name:<8} {r['length']:>9.1f} {r['raw']:>8d} {r['dp']:>7d} {r['ms']:>6.0f} "
              f"{r['expanded']:>7d} {r['clearance']:>10.1f}")

    rows = {name: [] for name in PLANNERS}
    for seed in range(20):
        w = random_world(seed)
        try:
            out = {name: run(w, name, fn, cfg) for name, fn in PLANNERS.items()}
        except PlanningError:
            continue
        for name, r in out.items():
            rows[name].append(r)

    n = len(rows["A*"])
    print(f"\nRandom fields ({n} solvable of 20, seeds 0-19), mean values")
    print(f"{'planner':<8} {'length m':>9} {'raw pts':>8} {'DP pts':>7} {'ms':>6} {'min clr m':>10}")
    for name, rs in rows.items():
        m = lambda k: float(np.mean([r[k] for r in rs]))
        print(f"{name:<8} {m('length'):>9.1f} {m('raw'):>8.0f} {m('dp'):>7.1f} {m('ms'):>6.0f} "
              f"{min(r['clearance'] for r in rs):>10.1f}")


if __name__ == "__main__":
    main()
