import math

import numpy as np

from uavguide.config import GuidanceParams, VehicleParams
from uavguide.guided import GuidedPlane
from uavguide.vehicle import VehicleState


def fly(plane, s, seconds, wind=(0.0, 0.0), dt=0.05, v=20.0):
    out = []
    for _ in range(int(seconds / dt)):
        s = plane.step(s, v, wind, dt)
        out.append((s.x, s.y, s.psi, s.phi))
    return s, np.array(out)


def test_without_a_command_it_flies_straight():
    plane = GuidedPlane()
    s, log = fly(plane, VehicleState(0.0, 0.0, 0.0, 20.0), 5.0)
    assert abs(s.x) < 1e-6 and s.y > 95.0


def test_it_circles_the_point_at_the_loiter_radius_instead_of_flying_over_it():
    plane = GuidedPlane(loiter_radius=60.0)
    plane.command((300.0, 300.0))
    s, log = fly(plane, VehicleState(0.0, 0.0, 0.0, 20.0), 120.0)
    d = np.hypot(log[:, 0] - 300.0, log[:, 1] - 300.0)
    assert d.min() > 45.0                           # never over the point
    assert abs(d[-600:].mean() - 60.0) < 6.0        # settles on the circle
    assert plane.circling


def test_loiter_direction_can_be_reversed():
    turn = {}
    for direction in (1, -1):
        plane = GuidedPlane(loiter_direction=direction)
        plane.command((300.0, 300.0))
        s, log = fly(plane, VehicleState(0.0, 0.0, 0.0, 20.0), 120.0)
        turn[direction] = float(np.mean(log[-200:, 3]))
    assert turn[1] > 0.1 and turn[-1] < -0.1        # clockwise = right wing down


def test_a_far_point_is_flown_at_but_a_near_one_is_only_circled():
    """The reason the navigator commands far points. A point 35 degrees off the nose:
    400 m away the aircraft turns onto it; 150 m away (inside three loiter radii) it sets up
    the loiter circle instead and hardly turns towards the point at all."""
    off = {}
    for d in (150.0, 400.0):
        plane = GuidedPlane(VehicleParams(), GuidanceParams())
        pt = (d * math.sin(math.radians(35.0)), d * math.cos(math.radians(35.0)))
        plane.command(pt)
        s, log = fly(plane, VehicleState(0.0, 0.0, 0.0, 20.0), 4.0)
        bearing = math.degrees(math.atan2(pt[0] - s.x, pt[1] - s.y))
        off[d] = bearing - math.degrees(s.psi)
    assert abs(off[400.0]) < 10.0
    assert off[150.0] > 35.0


def test_a_new_command_replaces_the_old_one():
    plane = GuidedPlane()
    plane.command((0.0, 1000.0))
    plane.command((1000.0, 0.0))
    s, log = fly(plane, VehicleState(0.0, 0.0, 0.0, 20.0), 20.0)
    assert s.x > 150.0
