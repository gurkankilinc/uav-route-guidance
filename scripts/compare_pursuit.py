"""Pursue a moving target with every method on random scenarios and compare them.

    python scripts/compare_pursuit.py                    # 100 held-out scenarios, replan every 2 s
    python scripts/compare_pursuit.py --seeds 20 --first-seed 100     # a quick look on tuning scenarios
    python scripts/compare_pursuit.py --interval 0       # plan once, never replan
    python scripts/compare_pursuit.py --no-guard         # without the look-ahead zone guard
    python scripts/compare_pursuit.py --procs 4 --out results.json
    python scripts/compare_pursuit.py --rate 10 --delay 0.2 --plan-latency 0.3 --source wind_est   # a realistic link

By default the scenarios are `scenarios.TEST_SEEDS` (200-299), which were never used for tuning.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from uavguide.compare import monte_carlo, paired_difference, summarize
from uavguide.config import Config
from uavguide.pursuit import PursuitPolicy
from uavguide.scenarios import TEST_SEEDS, make_scenario


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=len(TEST_SEEDS))
    ap.add_argument("--first-seed", type=int, default=TEST_SEEDS.start)
    ap.add_argument("--interval", type=float, default=2.0, help="replan interval in s; 0 = plan once")
    ap.add_argument("--zone-scale", type=float, default=1.0)
    ap.add_argument("--target", choices=("line", "circle", "fixed"), default="line")
    ap.add_argument("--detect", type=float, default=None, help="sensor radius in m; default: zones known from the start")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--rate", type=float, default=None, help="core update rate in Hz; default: every tick (50 Hz)")
    ap.add_argument("--delay", type=float, default=0.0, help="command delay in s")
    ap.add_argument("--plan-latency", type=float, default=0.0, help="time a new route takes to compute, in s")
    ap.add_argument("--source", choices=("true", "gnss", "wind_est", "air_only"), default="true",
                    help="where the core's ground velocity comes from")
    ap.add_argument("--procs", type=int, default=1, help="parallel processes")
    ap.add_argument("--out", type=Path, default=None, help="write summary, paired tests and per-run metrics as JSON")
    a = ap.parse_args()

    cfg = Config()
    policy = PursuitPolicy(replan_interval=a.interval or None, guard=not a.no_guard, control_rate_hz=a.rate,
                           command_delay=a.delay, plan_latency=a.plan_latency, velocity_source=a.source)
    make = partial(make_scenario, cfg=cfg, zone_scale=a.zone_scale, target_mode=a.target, detect_radius=a.detect)
    seeds = range(a.first_seed, a.first_seed + a.seeds)
    rows = monte_carlo(seeds, make, policy=policy, cfg=cfg, processes=a.procs)
    summary = summarize(rows)

    print(f"{a.seeds} scenarios (seeds {seeds.start}-{seeds.stop - 1}), target={a.target}, zone scale {a.zone_scale}, "
          f"replan {'once' if not a.interval else f'every {a.interval:g} s'}, guard {'off' if a.no_guard else 'on'}, "
          f"core at {'50' if a.rate is None else f'{a.rate:g}'} Hz, command delay {a.delay:g} s, "
          f"plan latency {a.plan_latency:g} s, velocity from {a.source}")
    print(f"{'method':<13} {'success':>8} {'95% interval':>14} {'collided':>9} {'left area':>10} "
          f"{'guard %':>8} {'t_capture':>10} {'flown m':>8} {'plan ms':>8}")
    for name, s in summary.items():
        print(f"{name:<13} {s['success_rate']:>8.0%} {f'''{s['success_lo']:.0%} - {s['success_hi']:.0%}''':>14} "
              f"{s['collision_rate']:>9.0%} {s['left_area_rate']:>10.0%} {s['guard_share']:>8.1%} "
              f"{s['time_to_capture']:>10.1f} {s['flown_length']:>8.0f} {s['plan_ms_mean']:>8.1f}")
    print("(time, length and plan time are medians over the successful runs; "
          "guard % = share of flight time the guard was steering)")

    best = max(summary, key=lambda n: summary[n]["success_rate"])
    pairs = [paired_difference(rows, best, other) for other in summary if other != best]
    print(f"\npaired comparison on the same scenarios, best = {best}")
    print(f"{'best minus':<13} {'difference':>11} {'95% interval':>16} {'only best':>10} {'only other':>11} {'p':>7}")
    for p in pairs:
        print(f"{p['b']:<13} {p['difference']:>+11.0%} {f'''{p['lo']:+.0%} .. {p['hi']:+.0%}''':>16} "
              f"{p['only_a']:>10} {p['only_b']:>11} {p['p_value']:>7.3f}")
    print("(p: exact McNemar test; below 0.05 the difference is unlikely to be chance)")

    if a.out:
        a.out.write_text(json.dumps(dict(
            args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
            summary=summary, paired=pairs,
            runs={name: [m.as_dict() for m in ms] for name, ms in rows.items()}), indent=1))


if __name__ == "__main__":
    main()
