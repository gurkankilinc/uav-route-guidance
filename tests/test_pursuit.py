import math
from dataclasses import replace

import numpy as np
import pytest

from uavguide.config import Config, SimParams, TargetParams
from uavguide.planners import PLANNERS
from uavguide.pursuit import PursuitPolicy, PursuitScenario, pursue
from uavguide.target import MovingTarget
from uavguide.vehicle import VehicleState
from uavguide.world import Obstacle, World

BOUNDS = (0.0, 0.0, 600.0, 600.0)
EMPTY = World(BOUNDS, [])


def scenario(world=EMPTY, mode="line", start=(400.0, 300.0), heading=0.0, wind=(0.0, 0.0), detect=None,
             ac=(30.0, 30.0)):
    tgt = MovingTarget(TargetParams(mode=mode), BOUNDS, start=start, heading_deg=heading)
    t0, _ = tgt.state(0.0)
    psi = math.atan2(t0[0] - ac[0], t0[1] - ac[1])
    return PursuitScenario(world, tgt, VehicleState(ac[0], ac[1], psi, 20.0), wind=wind, detect_radius=detect)


def test_pursuer_catches_a_moving_target_in_open_air():
    log = pursue(scenario(), "Theta*")
    assert log.captured
    assert log.dist[log.t >= log.capture_t].max() <= Config().approach.capture_radius + 1.0
    assert log.clearance.min() == math.inf


def test_pursuer_catches_a_fixed_target():
    log = pursue(scenario(mode="fixed"), "A* + DP")
    assert log.captured and log.capture_t < 60.0


def test_capture_needs_the_hold_time():
    cfg = Config()
    log = pursue(scenario(mode="fixed"), "Theta*", cfg=cfg)
    assert log.t[-1] - log.capture_t == pytest.approx(cfg.approach.hold_time, abs=0.1)


def test_unreachable_speed_target_is_rejected():
    sc = scenario()
    fast = MovingTarget(TargetParams(speed=40.0, mode="line"), BOUNDS)
    with pytest.raises(ValueError):
        pursue(PursuitScenario(sc.world, fast, sc.start), "Theta*")


def test_plan_once_makes_one_plan_and_periodic_replans_repeatedly():
    sc = scenario()
    once = pursue(sc, "Theta*", PursuitPolicy(replan_interval=None))
    often = pursue(sc, "Theta*", PursuitPolicy(replan_interval=2.0))
    assert len(once.replans) == 1 and once.replans[0].reason == "initial"
    assert len(often.replans) >= int(often.t[-1] / 2.0)
    assert {e.reason for e in often.replans} <= {"initial", "periodic"}


def test_hidden_zones_are_learned_when_the_aircraft_gets_close():
    world = World(BOUNDS, [Obstacle(220.0, 160.0, 25.0), Obstacle(500.0, 500.0, 25.0)])
    sc = scenario(world, mode="fixed", start=(300.0, 250.0), detect=70.0)
    log = pursue(sc, "Theta*")
    assert log.known[0] == 0 and log.known.max() >= 1
    assert any(e.reason == "detected" for e in log.replans)
    assert log.known.max() <= 2


def test_all_zones_known_from_the_start_without_a_detection_radius():
    world = World(BOUNDS, [Obstacle(220.0, 160.0, 25.0)])
    log = pursue(scenario(world, mode="fixed", start=(300.0, 250.0)), "Theta*")
    assert log.known.min() == 1 and not any(e.reason == "detected" for e in log.replans)


@pytest.mark.parametrize("planner", list(PLANNERS))
def test_every_planner_flies_the_whole_loop(planner):
    world = World(BOUNDS, [Obstacle(200.0, 200.0, 30.0), Obstacle(350.0, 300.0, 25.0)])
    log = pursue(scenario(world, mode="fixed", start=(500.0, 480.0)), planner)
    assert log.captured
    assert log.clearance.min() > 0.0


def test_target_over_a_zone_does_not_break_the_planner():
    world = World(BOUNDS, [Obstacle(300.0, 300.0, 40.0)])
    cfg = replace(Config(), sim=SimParams(t_max=40.0))
    log = pursue(scenario(world, mode="fixed", start=(300.0, 300.0)), "Theta*", cfg=cfg)
    assert all(e.ok for e in log.replans)
    assert log.clearance.min() > 0.0                 # it stays outside, next to the target


def test_start_on_top_of_the_target_does_not_crash():
    cfg = replace(Config(), sim=SimParams(t_max=10.0))
    sc = scenario(mode="fixed", start=(30.0, 30.0))
    log = pursue(sc, "Theta*", cfg=cfg)
    assert len(log.t) > 10


def test_speed_command_respects_the_flight_envelope():
    cfg = Config()
    log = pursue(scenario(wind=(5.0, 0.0)), "Theta*")
    assert log.v_cmd.min() >= cfg.vehicle.v_stall - 1e-9 and log.v_cmd.max() <= cfg.vehicle.v_max + 1e-9
    assert np.abs(log.phi).max() <= cfg.vehicle.roll_max + 1e-9


def test_run_is_deterministic():
    sc = scenario(wind=(3.0, 2.0))
    a = pursue(sc, "A* + DP")
    b = pursue(sc, "A* + DP")
    assert np.array_equal(a.pos, b.pos) and a.capture_t == b.capture_t
