"""Pursue a moving target with every planner on random scenarios and compare them.

    python scripts/compare_pursuit.py                    # 12 scenarios, replan every 2 s
    python scripts/compare_pursuit.py --seeds 30 --interval 1
    python scripts/compare_pursuit.py --interval 0       # plan once, never replan
    python scripts/compare_pursuit.py --no-guard         # without the look-ahead zone guard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uavguide.compare import monte_carlo, summarize
from uavguide.config import Config
from uavguide.pursuit import PursuitPolicy
from uavguide.scenarios import make_scenario


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--interval", type=float, default=2.0, help="replan interval in s; 0 = plan once")
    ap.add_argument("--zone-scale", type=float, default=1.0)
    ap.add_argument("--target", choices=("line", "circle", "fixed"), default="line")
    ap.add_argument("--detect", type=float, default=None, help="sensor radius in m; default: zones known from the start")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write the summary as JSON")
    a = ap.parse_args()

    cfg = Config()
    policy = PursuitPolicy(replan_interval=a.interval or None, guard=not a.no_guard)
    make = lambda s: make_scenario(s, cfg, zone_scale=a.zone_scale, target_mode=a.target, detect_radius=a.detect)
    rows = monte_carlo(range(a.seeds), make, policy=policy, cfg=cfg)
    summary = summarize(rows)

    print(f"{a.seeds} scenarios, target={a.target}, zone scale {a.zone_scale}, replan "
          f"{'once' if not a.interval else f'every {a.interval:g} s'}, guard {'off' if a.no_guard else 'on'}")
    print(f"{'planner':<12} {'success':>8} {'collided':>9} {'left area':>10} {'t_capture':>10} "
          f"{'flown m':>8} {'plan ms':>8} {'min clr m':>10}")
    for name, s in summary.items():
        left = sum(m.left_area for m in rows[name]) / len(rows[name])
        print(f"{name:<12} {s['success_rate']:>8.0%} {s['collision_rate']:>9.0%} {left:>10.0%} "
              f"{s['time_to_capture']:>10.1f} {s['flown_length']:>8.0f} {s['plan_ms_mean']:>8.1f} "
              f"{s['min_clearance']:>10.1f}")
    print("(time, length and plan time are medians over the successful runs; min clearance is the worst run)")
    if a.out:
        a.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
