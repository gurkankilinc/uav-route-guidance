"""Figures for 8.1 (A*) and 8.4 (Douglas-Peucker).

    python scripts/demo_planning.py            # default scenario
    python scripts/demo_planning.py --seed 7   # random obstacle field
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uavguide.config import Config
from uavguide.planner import plan_route
from uavguide.simplify import douglas_peucker
from uavguide.world import default_world, random_world

START, GOAL = (30.0, 30.0), (570.0, 570.0)
OUT = Path(__file__).resolve().parents[1] / "docs" / "img"


def polyline_length(p: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(p, axis=0).T)))


def draw_world(ax, world, margin):
    for o in world.obstacles:
        ax.add_patch(plt.Circle((o.x, o.y), o.r, color="0.35", zorder=2))
        ax.add_patch(plt.Circle((o.x, o.y), o.r + margin, fill=False, ls="--", lw=0.8,
                                color="0.55", zorder=2))
    x0, y0, x1, y1 = world.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (east) [m]")
    ax.set_ylabel("y (north) [m]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = Config().planner
    world = default_world() if args.seed is None else random_world(args.seed)
    tag = "default" if args.seed is None else f"seed{args.seed}"

    t0 = time.perf_counter()
    plan = plan_route(world, START, GOAL, cfg.grid_res, cfg.safety_margin)
    t_plan = time.perf_counter() - t0

    def safe(a, b):
        return world.segment_clear(a, b, cfg.dp_clearance)

    simple, _ = douglas_peucker(plan.path, cfg.dp_epsilon, segment_ok=safe)
    plain, _ = douglas_peucker(plan.path, cfg.dp_epsilon)
    plain_unsafe = not all(safe(a, b) for a, b in zip(plain[:-1], plain[1:]))

    OUT.mkdir(parents=True, exist_ok=True)

    # --- figure 1: route + simplified route --------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_world(ax, world, cfg.safety_margin)
    ax.plot(*plan.path.T, color="tab:blue", lw=1.2, alpha=0.6,
            label=f"A* route: {len(plan.path)} pts, {plan.length:.0f} m")
    ax.plot(*simple.T, "-o", color="tab:red", lw=1.8, ms=5,
            label=f"Douglas-Peucker (eps={cfg.dp_epsilon:g} m): {len(simple)} pts, "
                  f"{polyline_length(simple):.0f} m")
    ax.plot(*START, "g^", ms=11, label="start", zorder=5)
    ax.plot(*GOAL, "k*", ms=14, label="goal", zorder=5)
    ax.set_title(f"A* on {cfg.grid_res:g} m grid, {cfg.safety_margin:g} m safety margin "
                 f"({t_plan*1e3:.0f} ms, {plan.expanded} nodes)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / f"planning_{tag}.png", dpi=140)
    plt.close(fig)

    # --- figure 2: epsilon sweep ---------------------------------------------
    eps = np.array([0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50])
    n_safe, n_plain, unsafe_flags = [], [], []
    for e in eps:
        s, _ = douglas_peucker(plan.path, e, segment_ok=safe)
        p, _ = douglas_peucker(plan.path, e)
        n_safe.append(len(s))
        n_plain.append(len(p))
        unsafe_flags.append(not all(safe(a, b) for a, b in zip(p[:-1], p[1:])))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eps, n_plain, "o--", color="0.5", label="plain DP")
    ax.plot(eps, n_safe, "o-", color="tab:red", label="collision-aware DP")
    bad = np.array(unsafe_flags)
    if bad.any():
        ax.plot(eps[bad], np.array(n_plain)[bad], "x", color="k", ms=10, mew=2,
                label="plain DP cuts an obstacle")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon [m]")
    ax.set_ylabel("waypoints")
    ax.set_title(f"Douglas-Peucker: waypoint count vs tolerance (A* route: {len(plan.path)} pts)",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"dp_sweep_{tag}.png", dpi=140)
    plt.close(fig)

    print(f"[{tag}] A*: {len(plan.path)} pts, {plan.length:.1f} m, {plan.expanded} nodes, "
          f"{t_plan*1e3:.0f} ms")
    print(f"[{tag}] DP eps={cfg.dp_epsilon:g}: {len(simple)} pts, {polyline_length(simple):.1f} m, "
          f"min clearance {world.path_clearance(simple):.1f} m")
    print(f"[{tag}] plain DP at eps={cfg.dp_epsilon:g} unsafe: {plain_unsafe}; "
          f"unsafe eps values: {eps[bad].tolist()}")


if __name__ == "__main__":
    main()
