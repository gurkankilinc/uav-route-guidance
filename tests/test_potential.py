import math

import numpy as np
import pytest

from uavguide.config import Config, TargetParams, VehicleParams
from uavguide.metrics import compute_metrics
from uavguide.potential import FIELD_METHODS, FieldParams, PotentialField
from uavguide.pursuit import METHODS, PursuitPolicy, PursuitScenario, pursue
from uavguide.scenarios import stress_scenarios
from uavguide.target import MovingTarget
from uavguide.vehicle import VehicleState
from uavguide.world import World

VEH = VehicleParams()
BOUNDS = (0.0, 0.0, 600.0, 600.0)
NONE = (np.empty((0, 2)), np.empty(0))


def field(**kw):
    return PotentialField(FieldParams(**kw), VEH, margin=10.0, bounds=BOUNDS)


def test_influence_distance_follows_the_turn_radius():
    assert field().rho0 == pytest.approx(2.0 * VEH.min_turn_radius())
    assert field(influence_turns=3.0).rho0 == pytest.approx(3.0 * VEH.min_turn_radius())


def test_empty_sky_points_straight_at_the_aim():
    f = field().vector((300, 300), (400, 400), 0.0, *NONE)
    assert f / np.hypot(*f) == pytest.approx(np.array([1, 1]) / math.sqrt(2))


def test_zone_pushes_away_only_inside_its_influence():
    pf = field()
    C, R = np.array([[300.0, 340.0]]), np.array([20.0])
    near = pf.vector((300, 300), (300, 200), 0.0, C, R)          # aim south, zone 10 m north of the margin
    assert near[1] < -1.0                                          # pushed south harder than the pull alone
    far = pf.vector((150, 300), (100, 300), 0.0, C, R)            # 125 m from the inflated edge
    assert far == pytest.approx(np.array([-1.0, 0.0]))             # outside the influence: pull only


def test_collinear_zone_balances_classic_field_but_not_the_vortex():
    C, R = np.array([[300.0, 300.0]]), np.array([30.0])
    pos, aim = (300.0, 200.0), (300.0, 450.0)                      # zone exactly in between
    classic = field().vector(pos, aim, 0.0, C, R)
    assert abs(classic[0]) < 1e-9                                  # no sideways component: local-minimum line
    vortex = field(vortex=1.0).vector(pos, aim, math.radians(10), C, R)
    assert abs(vortex[0]) > 0.1                                    # the tangent breaks the symmetry


def test_vortex_turns_the_way_the_aircraft_already_goes():
    C, R = np.array([[300.0, 300.0]]), np.array([30.0])
    pos, aim = (300.0, 200.0), (300.0, 450.0)
    right = field(vortex=1.0).vector(pos, aim, math.radians(20), C, R)
    left = field(vortex=1.0).vector(pos, aim, math.radians(-20), C, R)
    assert right[0] > 0 > left[0]


def test_goal_factor_lets_the_aircraft_reach_an_aim_next_to_a_zone():
    C, R = np.array([[300.0, 300.0]]), np.array([30.0])
    aim = (300.0, 250.0)                                           # 10 m outside the inflated edge
    pos = (300.0, 220.0)                                           # 30 m south of the aim
    classic = field().vector(pos, aim, 0.0, C, R)
    gecui = field(goal_exp=2.0).vector(pos, aim, 0.0, C, R)
    assert classic[1] < 0 < gecui[1]                               # classic pushes away from the aim, Ge & Cui reaches it


def test_border_pushes_back_into_the_area():
    f = field().vector((595, 300), (700, 300), math.radians(90), *NONE)
    assert f[0] < 0


def test_roll_command_sign_and_limit():
    pf = field()
    s = VehicleState(300, 300, 0.0, 20.0)                          # heading north
    assert pf.roll_command(s, (0.0, 20.0), (400, 300), *NONE) > 0  # aim to the east: right turn
    assert pf.roll_command(s, (0.0, 20.0), (200, 300), *NONE) < 0
    assert abs(pf.roll_command(s, (0.0, 20.0), (300, 100), *NONE)) <= VEH.roll_max + 1e-12


def test_streamline_goes_to_the_aim_in_open_sky():
    line = field().streamline((100, 100), (100, 250), 0.0, *NONE, length=300)
    assert line[0] == pytest.approx([100, 100])
    assert np.hypot(*(line[-1] - [100, 250])) < 6.0


def test_field_methods_are_registered():
    assert set(FIELD_METHODS) <= set(METHODS)


@pytest.mark.parametrize("method", list(FIELD_METHODS))
def test_field_pursuit_catches_a_target_in_open_sky(method):
    tgt = MovingTarget(TargetParams(mode="line"), BOUNDS, start=(400.0, 300.0), heading_deg=0.0)
    sc = PursuitScenario(World(BOUNDS, []), tgt, VehicleState(30, 30, math.radians(45), 20.0))
    log = pursue(sc, method, PursuitPolicy(2.0, guard=False))
    assert log.captured
    assert all(e.reason == "field" for e in log.replans) and len(log.replans) >= 2
    m = compute_metrics(log, sc)
    assert m.n_replans == 0 and math.isnan(m.mean_waypoints) and m.plan_ms_mean > 0


def test_unknown_method_is_rejected():
    tgt = MovingTarget(TargetParams(mode="fixed"), BOUNDS)
    sc = PursuitScenario(World(BOUNDS, []), tgt, VehicleState(30, 30, 0.0, 20.0))
    with pytest.raises(ValueError):
        pursue(sc, "D* Lite")


def test_vortex_gets_around_a_zone_on_the_line_of_sight():
    sc = stress_scenarios()["behind_zone"]
    log = pursue(sc, "APF + vortex", PursuitPolicy(2.0, guard=False))
    assert log.captured and log.clearance.min() > 0.0
