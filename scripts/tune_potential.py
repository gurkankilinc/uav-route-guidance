"""Pick the potential-field gains on scenarios that are *not* used for the comparison.

    python scripts/tune_potential.py              # seeds 100-119, guard off, 2 processes

The comparison (compare_pursuit.py) uses seeds 0-19, so the gains are not fitted to the
scenarios they are judged on. The guard is off here: the point is to see what the field alone does.
"""

from __future__ import annotations

import argparse
import itertools
from multiprocessing import Pool

import numpy as np

from uavguide import potential
from uavguide.metrics import compute_metrics
from uavguide.potential import FieldParams
from uavguide.pursuit import METHODS, PursuitPolicy, pursue
from uavguide.scenarios import make_scenario

NAME = "_candidate"


def evaluate(job):
    label, fp, seeds = job
    potential.FIELD_METHODS[NAME] = fp
    if NAME not in METHODS:
        METHODS.append(NAME)
    ok, times = 0, []
    for sd in seeds:
        sc = make_scenario(sd)
        m = compute_metrics(pursue(sc, NAME, PursuitPolicy(2.0, guard=False)), sc)
        if m.success:
            ok += 1
            times.append(m.time_to_capture)
    return label, ok, float(np.median(times)) if times else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-seed", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    seeds = list(range(a.first_seed, a.first_seed + a.seeds))
    grid = [(f"APF          k_rep={k:<4} influence={inf} R_min", FieldParams(k_rep=k, influence_turns=inf), seeds)
            for k, inf in itertools.product((0.5, 1.0, 2.0), (1.5, 2.0, 3.0))]
    grid += [(f"APF + vortex k_rep={k:<4} vortex={vx}", FieldParams(k_rep=k, vortex=vx, goal_exp=2.0), seeds)
             for k, vx in itertools.product((0.5, 1.0, 2.0), (0.5, 1.0, 2.0))]
    with Pool(a.procs) as pool:
        res = pool.map(evaluate, grid)
    print(f"{'candidate':<45} {'success':>8} {'median t':>9}")
    for label, ok, med in sorted(res, key=lambda r: (-r[1], r[2])):
        print(f"{label:<45} {ok:>5}/{len(seeds)} {med:>9.1f}")


if __name__ == "__main__":
    main()
