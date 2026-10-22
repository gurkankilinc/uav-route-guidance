"""Run every planner on the same scenario(s) and put the results side by side."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .config import Config
from .metrics import Metrics, compute_metrics
from .planners import PLANNERS
from .pursuit import PursuitLog, PursuitPolicy, PursuitScenario, pursue


def run_all(
    scenario: PursuitScenario,
    planners: Iterable[str] | None = None,
    policy: PursuitPolicy | None = None,
    cfg: Config | None = None,
    seed: int = 0,
) -> dict[str, tuple[PursuitLog, Metrics]]:
    """Same world, same target, same wind, same noise seed for every planner."""
    cfg = cfg or Config()
    out = {}
    for name in (list(planners) if planners is not None else list(PLANNERS)):
        log = pursue(scenario, name, policy, cfg, seed=seed)
        out[name] = (log, compute_metrics(log, scenario, cfg))
    return out


def monte_carlo(
    seeds: Iterable[int],
    make: Callable[[int], PursuitScenario],
    planners: Iterable[str] | None = None,
    policy: PursuitPolicy | None = None,
    cfg: Config | None = None,
) -> dict[str, list[Metrics]]:
    """One scenario per seed, every planner on each; returns the raw per-run metrics."""
    names = list(planners) if planners is not None else list(PLANNERS)
    rows: dict[str, list[Metrics]] = {n: [] for n in names}
    for seed in seeds:
        scenario = make(seed)
        for name, (_, m) in run_all(scenario, names, policy, cfg, seed=seed).items():
            rows[name].append(m)
    return rows


def summarize(rows: dict[str, list[Metrics]]) -> dict[str, dict]:
    """Success / collision rates and medians over the runs that succeeded."""
    out = {}
    for name, ms in rows.items():
        ok = [m for m in ms if m.success]
        med = lambda key: float(np.median([getattr(m, key) for m in ok])) if ok else float("nan")
        out[name] = dict(
            runs=len(ms),
            success_rate=len(ok) / len(ms) if ms else 0.0,
            collision_rate=sum(m.collided for m in ms) / len(ms) if ms else 0.0,
            time_to_capture=med("time_to_capture"),
            flown_length=med("flown_length"),
            plan_ms_mean=med("plan_ms_mean"),
            min_clearance=float(np.min([m.min_clearance for m in ms])) if ms else float("nan"),
        )
    return out
