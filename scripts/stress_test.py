"""Every method on the hand-built potential-field stress cases, guard on and off.

    python scripts/stress_test.py                     # table
    python scripts/stress_test.py --fig docs/img/stress_paths.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uavguide.compare import run_all
from uavguide.config import Config
from uavguide.metrics import outcome
from uavguide.pursuit import METHODS, PursuitPolicy
from uavguide.scenarios import stress_scenarios

SHORT = {"success": "ok", "collided": "ZONE", "left_area": "OUT", "not_caught": "miss"}


def run(guard: bool, cfg: Config):
    out = {}
    for name, sc in stress_scenarios(cfg).items():
        out[name] = run_all(sc, METHODS, PursuitPolicy(2.0, guard=guard), cfg)
    return out


def figure(results, cfg, path: Path, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scen = stress_scenarios(cfg)
    fig, axes = plt.subplots(len(scen), len(methods), figsize=(3.1 * len(methods), 3.1 * len(scen)))
    for r, (sname, sc) in enumerate(scen.items()):
        for c, m in enumerate(methods):
            ax = axes[r, c]
            log, met = results[sname][m]
            for o in sc.world.obstacles:
                ax.add_patch(plt.Circle((o.x, o.y), o.r, color="0.45"))
                ax.add_patch(plt.Circle((o.x, o.y), o.r + cfg.planner.safety_margin, fill=False, ls=":", color="0.5"))
            ax.plot(log.target[:, 0], log.target[:, 1], ":", color="#d62728", lw=1)
            ax.plot(log.pos[:, 0], log.pos[:, 1], color="#1f77b4", lw=1.3)
            ax.plot(*log.pos[0], "^", color="#2e7d32", ms=6)
            ax.plot(*log.target[-1], "*", color="#d62728", ms=10)
            res = outcome(met)
            ax.set_title(f"{m}\n{SHORT[res]}, clr {met.min_clearance:.0f} m", fontsize=8,
                         color="#2e7d32" if res == "success" else "#c62828")
            ax.set_xlim(0, 600); ax.set_ylim(80, 520); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(sname, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=Path, default=None, help="path figure (guard off)")
    ap.add_argument("--out", type=Path, default=None, help="results as JSON")
    a = ap.parse_args()
    cfg = Config()
    table = {}
    for guard in (False, True):
        res = run(guard, cfg)
        print(f"\nguard {'on' if guard else 'off'}   (ok = caught, no violation; ZONE = entered a zone; "
              f"miss = not caught in {cfg.sim.t_max:.0f} s; number = smallest zone clearance in m)")
        print(f"{'scenario':<12}" + "".join(f"{m:>14}" for m in METHODS))
        for sname, per in res.items():
            cells = []
            for m in METHODS:
                met = per[m][1]
                cells.append(f"{SHORT[outcome(met)]} {met.min_clearance:5.0f}")
                table.setdefault(f"guard_{'on' if guard else 'off'}", {}).setdefault(sname, {})[m] = dict(
                    outcome=outcome(met), min_clearance=met.min_clearance,
                    time_to_capture=None if np.isnan(met.time_to_capture) else met.time_to_capture)
            print(f"{sname:<12}" + "".join(f"{c:>14}" for c in cells))
        if a.fig and not guard:
            figure(res, cfg, a.fig, ["A* + DP", "Theta* + DP", "APF", "APF + vortex"])
    if a.out:
        a.out.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
