import math

import numpy as np
import pytest

from uavguide.config import GuidanceParams, VehicleParams
from uavguide.guided import GuidedPlane
from uavguide.navigator import Navigator, NavParams
from uavguide.vehicle import VehicleState

POLY = [[405.0, 0.0], [0.0, 131.0], [235.0, 929.0], [654.0, 801.0]]      # a slanted quadrilateral, ~420 x 830 m
BOX = [[0.0, 0.0], [1150.0, 0.0], [1150.0, 800.0], [0.0, 800.0]]


def bearing_deg(a, b):
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0


def ang_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def route_clearance(nav, route):
    best = math.inf
    for a, b in zip(route[:-1], route[1:]):
        n = max(2, int(np.hypot(*(b - a)) / 2.0))
        pts = a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]
        best = min(best, float(nav.zone_clearance(pts).min()))
    return best


def test_no_zones_flies_straight_at_the_target():
    nav = Navigator(BOX)
    cmd = nav.update(0.0, (200.0, 400.0), 90.0, 20.0, (900.0, 400.0))
    assert cmd.status == "route"
    assert ang_diff(bearing_deg((200.0, 400.0), cmd.point), 90.0) < 1.0
    assert np.hypot(*(cmd.point - np.array([200.0, 400.0]))) >= 2.5 * nav.p.loiter_radius   # far: GUIDED will not circle it


def test_set_zones_reports_changes_and_drops_the_route():
    nav = Navigator(BOX)
    assert nav.set_zones([[500.0, 400.0, 80.0]]) is True
    assert nav.set_zones([[500.0, 400.0, 80.0]]) is False
    nav.update(0.0, (200.0, 400.0), 90.0, 20.0, (900.0, 400.0))
    assert nav._route is not None
    assert nav.set_zones([]) is True and nav._route is None


def test_route_goes_around_a_zone_with_the_margin():
    nav = Navigator(BOX)
    nav.set_zones([[500.0, 400.0, 100.0]])
    cmd = nav.update(0.0, (200.0, 400.0), 90.0, 20.0, (900.0, 400.0))
    assert cmd.status in ("route", "guard")
    assert len(cmd.route) >= 3
    assert route_clearance(nav, cmd.route) >= nav.p.margin - nav.p.grid_res


@pytest.mark.parametrize("seed", range(6))
def test_seeded_routes_never_cross_a_zone_or_the_border(seed):
    rng = np.random.default_rng(seed)
    nav = Navigator(POLY)
    lo, hi = nav.polygon.min(axis=0), nav.polygon.max(axis=0)

    def free(clear):
        while True:
            q = rng.uniform(lo, hi)
            if float(nav.clearance(q)[0]) >= clear:
                return q

    zones = []
    for r in (50.0, 50.0, 75.0):
        while True:
            c = rng.uniform(lo, hi)
            if float(nav.border_clearance(c)[0]) > 0.0:
                zones.append([c[0], c[1], r])
                break
    nav.set_zones(zones)
    for k in range(15):
        s, g = free(30.0), free(30.0)
        cmd = nav.update(float(k) * 10.0, s, float(rng.uniform(0.0, 360.0)), 20.0, g)
        assert cmd.plan_ms < 200.0
        assert float(nav.border_clearance(cmd.point)[0]) >= 0.0          # the commanded point is inside the area
        if cmd.status == "route":
            assert route_clearance(nav, cmd.route[1:]) >= nav.p.margin - 2.0 * nav.p.grid_res
            assert float(nav.border_clearance(cmd.route).min()) >= 0.0
        nav._route = None


def test_wall_without_a_gap_holds_at_a_reachable_point():
    nav = Navigator(POLY)
    nav.set_zones([[x, 465.0, 60.0] for x in range(0, 661, 60)])
    cmd = nav.update(0.0, (330.0, 200.0), 0.0, 20.0, (330.0, 750.0))
    assert cmd.status in ("hold", "guard") and cmd.reason == "target not reachable"
    assert cmd.goal[1] < 465.0                                            # waits on its own side of the wall
    assert float(nav.clearance(cmd.goal)[0]) >= nav.p.loiter_radius      # with room to circle
    assert route_clearance(nav, cmd.route) >= 0.0


def test_inside_a_zone_escapes_by_the_quickest_way_out():
    nav = Navigator(BOX)
    nav.set_zones([[500.0, 400.0, 100.0]])
    # 20 m inside the eastern edge, flying east: straight on is the way out
    cmd = nav.update(0.0, (580.0, 400.0), 90.0, 20.0, (200.0, 400.0))
    assert cmd.status == "escape"
    assert ang_diff(bearing_deg((580.0, 400.0), cmd.point), 90.0) <= 30.0
    assert cmd.speed_limit == nav.p.slow_speed


def test_target_inside_a_zone_is_not_followed_in():
    nav = Navigator(BOX)
    nav.set_zones([[500.0, 400.0, 100.0]])
    cmd = nav.update(0.0, (200.0, 400.0), 90.0, 20.0, (500.0, 400.0))
    assert float(nav.zone_clearance(cmd.goal)[0]) >= nav.p.margin - nav.p.grid_res


def test_guard_turns_away_when_a_zone_is_dead_ahead():
    nav = Navigator(BOX, NavParams(use_heading_start=False))
    nav.set_zones([[500.0, 400.0, 100.0]])
    pos = (360.0, 400.0)                                                  # 40 m from the edge, flying straight at it
    nav.update(0.0, (355.0, 400.0), 90.0, 20.0, (500.0, 400.0))
    cmd = nav.update(0.2, pos, 90.0, 20.0, (500.0, 400.0))
    assert cmd.status == "guard"
    assert ang_diff(bearing_deg(pos, cmd.point), 90.0) >= 60.0
    assert cmd.speed_limit == nav.p.slow_speed


def test_guard_can_be_switched_off():
    nav = Navigator(BOX, NavParams(use_guard=False))
    nav.set_zones([[500.0, 400.0, 100.0]])
    cmd = nav.update(0.0, (360.0, 400.0), 90.0, 20.0, (500.0, 400.0))
    assert cmd.status != "guard"


def test_stickiness_keeps_the_side_when_the_aircraft_drifts_across_the_centre_line():
    sides = {}
    for stick in (True, False):
        nav = Navigator(BOX, NavParams(use_stickiness=stick, use_guard=False))
        nav.set_zones([[575.0, 400.0, 100.0]])
        got = []
        for k, y in enumerate((400.6, 399.4, 400.6, 399.4)):
            cmd = nav.update(float(k) * 1.0, (200.0 + 2.0 * k, y), 90.0, 20.0, (950.0, 400.0))
            got.append(float(np.sign(cmd.route[1][1] - 400.0)))
        sides[stick] = got
    assert len(set(sides[True])) == 1                                     # one side, kept


def test_heading_aware_start_prefers_not_to_turn_around_for_a_small_gain():
    # target dead ahead but behind a zone; both ways round are equal, the aircraft is flying north-east
    nav = Navigator(BOX)
    nav.set_zones([[575.0, 400.0, 100.0]])
    cmd = nav.update(0.0, (300.0, 400.0), 45.0, 20.0, (950.0, 400.0))
    assert cmd.route[1][1] > 400.0                                        # goes round the north side


def test_wind_is_optional_and_accepted():
    nav = Navigator(BOX)
    a = nav.update(0.0, (200.0, 400.0), 90.0, 20.0, (900.0, 400.0))
    b = nav.update(0.2, (204.0, 400.0), 90.0, 20.0, (900.0, 400.0), wind=(3.0, -2.0))
    assert a.status == b.status == "route"


@pytest.mark.parametrize("seed", range(4))
def test_closed_loop_static_target_behind_zones_is_reached_without_entering(seed):
    """GUIDED model in the loop: fly to a fixed point on the far side of two zones."""
    rng = np.random.default_rng(100 + seed)
    nav = Navigator(BOX)
    zones = [[450.0 + rng.uniform(-40, 40), 300.0 + rng.uniform(-40, 40), 100.0],
             [650.0 + rng.uniform(-40, 40), 520.0 + rng.uniform(-40, 40), 75.0]]
    nav.set_zones(zones)
    veh = VehicleParams()
    plane = GuidedPlane(veh, GuidanceParams())
    wind = (float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4)))
    s = VehicleState(150.0, 200.0 + float(rng.uniform(0, 300)), math.radians(float(rng.uniform(30, 150))), 20.0)
    target = np.array([1000.0, 400.0 + float(rng.uniform(-150, 150))])
    dt, min_zone, min_border, best = 0.05, math.inf, math.inf, math.inf
    for k in range(int(120.0 / dt)):
        t = k * dt
        if k % 4 == 0:
            gv = s.ground_velocity(wind)
            cmd = nav.update(t, s.pos, math.degrees(math.atan2(gv[0], gv[1])) % 360.0, float(np.hypot(*gv)), target, wind=wind)
            plane.command(cmd.point)
        s = plane.step(s, cmd.speed_limit or 20.0, wind, dt)
        min_zone = min(min_zone, float(nav.zone_clearance(s.pos)[0]))
        min_border = min(min_border, float(nav.border_clearance(s.pos)[0]))
        best = min(best, float(np.hypot(*(s.pos - target))))
        if best < 80.0:
            break
    assert min_zone >= 0.0 and min_border >= 0.0
    assert best < 80.0
