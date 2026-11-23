"""Run every method on the same scenario(s) and put the results side by side.

Statistics, because a success rate from a handful of scenarios is mostly noise:

* `wilson_interval`  95 % interval of a success rate (20 runs: about +-20 points, 100 runs: +-10),
* `paired_difference`  two methods on the *same* scenarios: difference of the rates with a
  bootstrap interval, and an exact McNemar test on the scenarios where exactly one of them
  succeeded. Pairing removes the scenario-to-scenario variation, so it sees smaller differences
  than comparing two intervals does.
"""

from __future__ import annotations

import math
from functools import partial
from multiprocessing import get_context
from typing import Callable, Iterable

import numpy as np

from .config import Config
from .metrics import Metrics, compute_metrics
from .pursuit import METHODS, PursuitLog, PursuitPolicy, PursuitScenario, pursue


def run_all(
    scenario: PursuitScenario,
    planners: Iterable[str] | None = None,
    policy: PursuitPolicy | None = None,
    cfg: Config | None = None,
    seed: int = 0,
) -> dict[str, tuple[PursuitLog, Metrics]]:
    """Same world, same target, same wind, same noise seed for every method."""
    cfg = cfg or Config()
    out = {}
    for name in (list(planners) if planners is not None else list(METHODS)):
        log = pursue(scenario, name, policy, cfg, seed=seed)
        out[name] = (log, compute_metrics(log, scenario, cfg))
    return out


def _one_seed(seed: int, make, names, policy, cfg) -> list[Metrics]:
    scenario = make(seed)
    return [m for _, m in run_all(scenario, names, policy, cfg, seed=seed).values()]


def monte_carlo(
    seeds: Iterable[int],
    make: Callable[[int], PursuitScenario],
    planners: Iterable[str] | None = None,
    policy: PursuitPolicy | None = None,
    cfg: Config | None = None,
    processes: int = 1,
) -> dict[str, list[Metrics]]:
    """One scenario per seed, every method on each; returns the raw per-run metrics, in seed order.

    With `processes` > 1 the seeds run in parallel; `make` must then be picklable
    (a module-level function or a `functools.partial`, not a lambda).
    """
    names = list(planners) if planners is not None else list(METHODS)
    seeds = list(seeds)
    job = partial(_one_seed, make=make, names=names, policy=policy, cfg=cfg)
    if processes > 1 and len(seeds) > 1:
        # "spawn" on every platform: the same behaviour on Linux and Windows, no fork-with-threads pitfalls
        with get_context("spawn").Pool(processes) as pool:
            per_seed = pool.map(job, seeds)
    else:
        per_seed = [job(s) for s in seeds]
    rows: dict[str, list[Metrics]] = {n: [] for n in names}
    for ms in per_seed:
        for name, m in zip(names, ms):
            rows[name].append(m)
    return rows


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n runs (95 % for z = 1.96)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(rows: dict[str, list[Metrics]]) -> dict[str, dict]:
    """Success / collision / left-area rates (success with its 95 % interval) and medians over
    the runs that succeeded."""
    out = {}
    for name, ms in rows.items():
        ok = [m for m in ms if m.success]

        def med(key, ok=ok):
            return float(np.median([getattr(m, key) for m in ok])) if ok else float("nan")

        lo, hi = wilson_interval(len(ok), len(ms))
        out[name] = dict(
            runs=len(ms),
            success_rate=len(ok) / len(ms) if ms else 0.0,
            success_lo=lo,
            success_hi=hi,
            collision_rate=sum(m.collided for m in ms) / len(ms) if ms else 0.0,
            left_area_rate=sum(m.left_area for m in ms) / len(ms) if ms else 0.0,
            guard_share=float(np.mean([m.guard_share for m in ms])) if ms else 0.0,
            time_to_capture=med("time_to_capture"),
            flown_length=med("flown_length"),
            plan_ms_mean=med("plan_ms_mean"),
            min_clearance=float(np.min([m.min_clearance for m in ms])) if ms else float("nan"),
        )
    return out


def mcnemar_p(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar test: could `only_a` vs `only_b` discordant scenarios be a coin flip?"""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def paired_difference(rows: dict[str, list[Metrics]], a: str, b: str, n_boot: int = 4000,
                      seed: int = 0) -> dict:
    """Success of method `a` minus method `b` on the same scenarios."""
    sa = np.array([m.success for m in rows[a]], dtype=float)
    sb = np.array([m.success for m in rows[b]], dtype=float)
    if len(sa) != len(sb):
        raise ValueError("both methods must have run the same scenarios")
    diff = sa - sb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    only_a, only_b = int(np.sum(diff > 0)), int(np.sum(diff < 0))
    return dict(a=a, b=b, runs=len(diff), difference=float(diff.mean()),
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                only_a=only_a, only_b=only_b, p_value=mcnemar_p(only_a, only_b))
