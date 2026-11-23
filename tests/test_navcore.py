"""The navigation core must be usable without the simulation: state in, command out."""

import math

import numpy as np
import pytest

from uavguide.config import Config, GuardParams, PursuitParams, TargetParams
from uavguide.navcore import METHODS, Navigator, NavState, NavTarget, NavZones
from uavguide.pursuit import PursuitPolicy, PursuitScenario, pursue
from uavguide.target import MovingTarget
from uavguide.vehicle import FixedWing, VehicleState
from uavguide.world import Obstacle, World

BOUNDS = (0.0, 0.0, 600.0, 600.0)
CFG = Config()
NO_ZONES = NavZones(np.empty((0, 2)), np.empty(0), BOUNDS)


def state(x=100.0, y=100.0, psi=0.0, v=20.0, t=0.0, wind=(0.0, 0.0)):
    gv = v * np.array([math.sin(psi), math.cos(psi)]) + np.asarray(wind)
    return NavState(t=t, x=x, y=y, psi=psi, phi=0.0, v=v, ground_vel=gv)


def fixed(x, y):
    return NavTarget(np.array([x, y]), np.zeros(2))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        Navigator("RRT*")


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_a_command_inside_the_flight_envelope(method):
    nav = Navigator(method, CFG)
    zones = NavZones(np.array([[300.0, 300.0]]), np.array([30.0]), BOUNDS)
    cmd = nav.update(state(), fixed(500, 500), zones, 0.02)
    assert abs(cmd.roll) <= CFG.vehicle.roll_max + 1e-12
    assert CFG.vehicle.v_stall <= cmd.airspeed <= CFG.vehicle.v_max
    assert cmd.event is not None                     # first tick: a plan or a field snapshot


def test_route_method_plans_once_then_follows():
    nav = Navigator("Theta*", CFG, replan_interval=None)
    first = nav.update(state(), fixed(500, 500), NO_ZONES, 0.02)
    second = nav.update(state(t=0.02), fixed(500, 500), NO_ZONES, 0.02)
    assert first.event.reason == "initial" and second.event is None
    assert nav.route[0] == pytest.approx([100, 100]) and nav.route[-1] == pytest.approx([500, 500])


def test_periodic_replanning_follows_the_interval():
    nav = Navigator("A* + DP", CFG, replan_interval=1.0)
    reasons = []
    for k in range(120):
        cmd = nav.update(state(t=k * 0.02), fixed(500, 500), NO_ZONES, 0.02)
        if cmd.event:
            reasons.append((round(cmd.event.t, 2), cmd.event.reason))
    assert reasons == [(0.0, "initial"), (1.0, "periodic"), (2.0, "periodic")]


def test_a_newly_known_zone_triggers_an_immediate_replan():
    nav = Navigator("Theta*", CFG, replan_interval=None)
    nav.update(state(), fixed(500, 500), NO_ZONES, 0.02)
    more = NavZones(np.array([[300.0, 300.0]]), np.array([40.0]), BOUNDS)
    cmd = nav.update(state(t=0.02), fixed(500, 500), more, 0.02)
    assert cmd.event is not None and cmd.event.reason == "detected"
    assert len(nav.route) > 2                        # the new route bends around the zone
    again = nav.update(state(t=0.04), fixed(500, 500), more, 0.02)
    assert again.event is None                       # same zones: nothing new


def test_last_waypoint_follows_a_moving_target_between_replans():
    nav = Navigator("Theta*", CFG, replan_interval=None)
    tgt = NavTarget(np.array([400.0, 400.0]), np.array([17.0, 0.0]))
    nav.update(state(), tgt, NO_ZONES, 0.02)
    moved = NavTarget(np.array([434.0, 400.0]), np.array([17.0, 0.0]))
    nav.update(state(t=2.0), moved, NO_ZONES, 0.02)
    aim = moved.pos - np.array([CFG.approach.d_follow, 0.0])     # d_follow behind the target
    assert nav.route[-1] == pytest.approx(aim)


def test_speed_command_uses_the_wind_the_aircraft_can_infer():
    """Tailwind towards the target: the same ground speed needs less airspeed."""
    tgt = NavTarget(np.array([100.0, 400.0]), np.array([0.0, 17.0]))
    calm = Navigator("Theta*", CFG).update(state(), tgt, NO_ZONES, 0.02)
    tail = Navigator("Theta*", CFG).update(state(wind=(0.0, 5.0)), tgt, NO_ZONES, 0.02)
    assert tail.airspeed == pytest.approx(calm.airspeed - 5.0)


def test_target_behind_and_close_slows_the_aircraft_for_a_tight_turn():
    behind = fixed(100, 40)                          # 60 m behind an aircraft heading north
    cmd = Navigator("Theta*", CFG).update(state(), behind, NO_ZONES, 0.02)
    assert cmd.airspeed == pytest.approx(CFG.pursuit.turn_back_factor * CFG.vehicle.v_stall)


def test_guard_overrides_only_when_enabled():
    zone = NavZones(np.array([[100.0, 160.0]]), np.array([25.0]), BOUNDS)     # dead ahead
    on = Navigator("A*", CFG, guard=True).update(state(), fixed(100, 500), zone, 0.02)
    off = Navigator("A*", CFG, guard=False).update(state(), fixed(100, 500), zone, 0.02)
    assert on.guard_active and abs(on.roll) == pytest.approx(CFG.vehicle.roll_max)
    assert not off.guard_active


def test_reset_forgets_the_route():
    nav = Navigator("Theta*", CFG)
    nav.update(state(), fixed(500, 500), NO_ZONES, 0.02)
    nav.reset()
    assert nav.route is None
    assert nav.update(state(), fixed(500, 500), NO_ZONES, 0.02).event.reason == "initial"


def test_core_driven_by_hand_matches_the_simulation():
    """pursue() adds nothing to the decisions: a bare loop around the core flies the same path."""
    world = World(BOUNDS, [Obstacle(250.0, 250.0, 30.0)])
    tgt = MovingTarget(TargetParams(mode="fixed"), BOUNDS, start=(480.0, 470.0))
    start = VehicleState(60.0, 60.0, math.radians(45), 20.0)
    log = pursue(PursuitScenario(world, tgt, start, wind=(3.0, -2.0)), "Theta* + DP", PursuitPolicy(1.0), CFG)

    nav = Navigator("Theta* + DP", CFG, replan_interval=1.0)
    model, s, wind = FixedWing(CFG.vehicle), start, np.array([3.0, -2.0])
    zones = NavZones(np.array([[250.0, 250.0]]), np.array([30.0]), BOUNDS)
    path = []
    for k in range(len(log.t)):
        path.append(s.pos)
        st = NavState(k * 0.02, s.x, s.y, s.psi, s.phi, s.v, s.ground_velocity(wind))
        cmd = nav.update(st, NavTarget(np.array([480.0, 470.0]), np.zeros(2)), zones, 0.02)
        s = model.step(s, cmd.roll, cmd.airspeed, wind, 0.02)
    assert np.array(path) == pytest.approx(log.pos, abs=1e-9)


def test_tunable_constants_live_in_config():
    assert PursuitParams().turn_back_range == 150.0 and GuardParams().horizon == 6.0
    slow = Config(pursuit=PursuitParams(turn_back_factor=1.1))
    cmd = Navigator("Theta*", slow).update(state(), fixed(100, 40), NO_ZONES, 0.02)
    assert cmd.airspeed == pytest.approx(1.1 * slow.vehicle.v_stall)
    tgt = MovingTarget(TargetParams(turn_radius=90.0), BOUNDS)
    assert tgt.turn_radius == 90.0


def test_plan_latency_keeps_the_old_route_until_the_new_one_is_ready():
    nav = Navigator("Theta*", CFG, replan_interval=1.0, plan_latency=0.4)
    events, switched = [], []
    route = None
    for k in range(120):
        t = k * 0.02
        cmd = nav.update(state(t=t), fixed(500 + 2 * k, 500), NO_ZONES, 0.02)
        if cmd.event:
            events.append(round(cmd.event.t, 2))
        if route is not None and nav.route is not route:       # a different route object: the new plan took over
            switched.append(round(t, 2))
        route = nav.route
    assert events == [0.0, 1.0, 2.0]                 # requested on schedule, the first one used at once
    assert switched == [1.4]                         # the plan asked for at 1.0 s is flown from 1.4 s
    # (the one asked for at 2.0 s would arrive at 2.4 s, after this loop ends)
