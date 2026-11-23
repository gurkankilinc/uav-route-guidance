"""Record what went into the navigation core and what came out, and play it back.

When the core is driven by something other than the built-in simulation (SITL, a real aircraft),
a strange flight raises one question first: did the core decide badly, or was it given bad data?
A recording answers it. `replay()` feeds the recorded inputs to a fresh core and compares the
commands; they must match exactly, because the core is deterministic. If they do, the core did
what it would always do with those inputs, and the inputs are the place to look.

The file is JSON lines: a header (method, settings, configuration), the zone sets as they became
known, and one line per tick. Plain text, so it can be read from C++ or a spreadsheet as well.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .navcore import NavCommand, NavState, NavTarget, NavZones, Navigator


def config_from_dict(d: dict) -> Config:
    """Inverse of dataclasses.asdict(Config()) after a trip through JSON (lists back to tuples)."""
    def build(cls, data):
        kw = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            v = data[f.name]
            sub = getattr(cls(), f.name) if is_dataclass(cls) else None
            if is_dataclass(sub):
                kw[f.name] = build(type(sub), v)
            else:
                kw[f.name] = tuple(v) if isinstance(v, list) else v
        return cls(**kw)
    return build(Config, d)


class NavRecorder:
    def __init__(self) -> None:
        self.header: dict | None = None
        self.zone_sets: list[dict] = []
        self.ticks: list[dict] = []
        self._zones_key = None

    def wrap(self, nav: Navigator) -> "_Recording":
        self.header = dict(type="header", method=nav.method, replan_interval=nav.replan_interval,
                           guard=nav.use_guard, plan_latency=nav.plan_latency, config=asdict(nav.cfg))
        return _Recording(nav, self)

    def _zone_id(self, zones: NavZones) -> int:
        key = (zones.bounds, zones.centres.tobytes(), zones.radii.tobytes())
        if key != self._zones_key:
            self._zones_key = key
            self.zone_sets.append(dict(type="zones", id=len(self.zone_sets), bounds=list(zones.bounds),
                                       centres=np.asarray(zones.centres, dtype=float).tolist(),
                                       radii=np.asarray(zones.radii, dtype=float).tolist()))
        return len(self.zone_sets) - 1

    def add(self, st: NavState, target: NavTarget, zones: NavZones, dt: float, wind, cmd: NavCommand) -> None:
        self.ticks.append(dict(
            type="tick", t=float(st.t), x=float(st.x), y=float(st.y), psi=float(st.psi), phi=float(st.phi),
            v=float(st.v), gv=[float(st.ground_vel[0]), float(st.ground_vel[1])],
            tp=[float(target.pos[0]), float(target.pos[1])], tv=[float(target.vel[0]), float(target.vel[1])],
            ts=None if target.speed is None else float(target.speed), dt=float(dt), z=self._zone_id(zones),
            w=None if wind is None else [float(wind[0]), float(wind[1])],
            roll=float(cmd.roll), speed=float(cmd.airspeed), guard=bool(cmd.guard_active)))

    def save(self, path) -> Path:
        path = Path(path)
        # zone sets are written before the first tick that uses them
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(self.header) + "\n")
            written = 0
            for tick in self.ticks:
                while written <= tick["z"]:
                    f.write(json.dumps(self.zone_sets[written]) + "\n")
                    written += 1
                f.write(json.dumps(tick) + "\n")
        return path


class _Recording:
    """A Navigator that writes every update to a recorder. Everything else passes through."""

    def __init__(self, nav: Navigator, rec: NavRecorder):
        self._nav, self._rec = nav, rec

    def update(self, st, target, zones, dt, wind=None):
        cmd = self._nav.update(st, target, zones, dt, wind=wind)
        self._rec.add(st, target, zones, dt, wind, cmd)
        return cmd

    def __getattr__(self, name):
        return getattr(self._nav, name)


@dataclass
class ReplayResult:
    ticks: int
    max_roll_diff: float            # rad
    max_speed_diff: float           # m/s
    first_mismatch_t: float | None  # time of the first tick whose command differs at all
    guard_ticks: int

    @property
    def identical(self) -> bool:
        return self.first_mismatch_t is None


def load(path) -> tuple[dict, dict[int, NavZones], list[dict]]:
    header, zones, ticks = None, {}, []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["type"] == "header":
                header = rec
            elif rec["type"] == "zones":
                zones[rec["id"]] = NavZones(np.array(rec["centres"], dtype=float).reshape(-1, 2),
                                            np.array(rec["radii"], dtype=float), tuple(rec["bounds"]))
            else:
                ticks.append(rec)
    if header is None:
        raise ValueError(f"{path}: no header line")
    return header, zones, ticks


def replay(path, cfg: Config | None = None, method: str | None = None) -> ReplayResult:
    """Run a fresh core on recorded inputs and compare its commands with the recorded ones.

    `cfg` / `method` override the recorded ones: "what would this other setting have done with the
    same inputs?" (the commands then differ, and from the first difference on the comparison is
    open-loop, because the recorded states came from the original commands).
    """
    header, zones, ticks = load(path)
    nav = Navigator(method or header["method"], cfg or config_from_dict(header["config"]),
                    replan_interval=header["replan_interval"], guard=header["guard"],
                    plan_latency=header.get("plan_latency", 0.0))
    max_roll = max_speed = 0.0
    first = None
    guard_ticks = 0
    for r in ticks:
        st = NavState(r["t"], r["x"], r["y"], r["psi"], r["phi"], r["v"], np.array(r["gv"]))
        cmd = nav.update(st, NavTarget(np.array(r["tp"]), np.array(r["tv"]), r["ts"]), zones[r["z"]], r["dt"],
                         wind=r["w"])
        dr, dv = abs(cmd.roll - r["roll"]), abs(cmd.airspeed - r["speed"])
        if first is None and (dr > 0.0 or dv > 0.0):
            first = r["t"]
        max_roll, max_speed = max(max_roll, dr), max(max_speed, dv)
        guard_ticks += cmd.guard_active
    return ReplayResult(len(ticks), max_roll, max_speed, first, guard_ticks)
