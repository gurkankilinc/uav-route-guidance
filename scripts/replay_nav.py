"""Play a recorded flight back through the navigation core.

    python scripts/replay_nav.py flight.jsonl                 # must reproduce the recorded commands
    python scripts/replay_nav.py flight.jsonl --method "APF + vortex"   # what would another method have done?
    python scripts/replay_nav.py --record flight.jsonl --seed 3 --method "Theta* + DP"   # make a recording
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from uavguide.pursuit import METHODS, PursuitPolicy, pursue
from uavguide.recorder import NavRecorder, load, replay
from uavguide.scenarios import make_scenario


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path)
    ap.add_argument("--method", choices=METHODS, default=None)
    ap.add_argument("--record", type=Path, default=None, help="fly a scenario in the simulation and record it")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.record:
        rec = NavRecorder()
        log = pursue(make_scenario(a.seed), a.method or "Theta* + DP", PursuitPolicy(2.0), recorder=rec)
        rec.save(a.record)
        print(f"recorded {len(rec.ticks)} ticks ({log.t[-1]:.1f} s, caught: {log.captured}) to {a.record}")
        return
    if not a.file:
        ap.error("give a recording to replay, or --record")

    header, zones, ticks = load(a.file)
    print(f"{a.file}: {header['method']}, {len(ticks)} ticks, {ticks[-1]['t']:.1f} s, "
          f"{len(zones)} zone set(s), replan every {header['replan_interval']} s, guard {header['guard']}")
    res = replay(a.file, method=a.method)
    if res.identical:
        print("replay reproduces every recorded command exactly: the core did what it always does "
              "with these inputs. If the flight looked wrong, look at the inputs.")
    else:
        print(f"commands differ from t = {res.first_mismatch_t:.2f} s: "
              f"max roll difference {math.degrees(res.max_roll_diff):.2f} deg, max speed difference "
              f"{res.max_speed_diff:.2f} m/s")
        if a.method is None:
            print("with the recorded method and configuration this should not happen: "
                  "the recording and the code are out of step (different version?)")
    print(f"guard steering on {res.guard_ticks} of {res.ticks} ticks")


if __name__ == "__main__":
    main()
