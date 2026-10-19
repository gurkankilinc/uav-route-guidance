"""Kinematic fixed-wing model: constant altitude, coordinated turns, wind drift.

State is kept in the package frame (x = east, y = north, heading clockwise
from north). The aircraft flies at airspeed `v` along its heading; the air
mass itself moves with the wind, so the ground velocity is

    ground = v * (sin psi, cos psi) + wind

The autopilot inner loops are modelled just enough to make the guidance
honest: roll follows the command with a first-order lag and a rate limit,
airspeed follows its command with an acceleration limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .config import G, VehicleParams


@dataclass(frozen=True)
class VehicleState:
    x: float          # east [m]
    y: float          # north [m]
    psi: float        # heading, clockwise from north [rad]
    v: float          # airspeed [m/s]
    phi: float = 0.0  # roll, positive = right wing down [rad]

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def air_velocity(self) -> np.ndarray:
        return self.v * np.array([math.sin(self.psi), math.cos(self.psi)])

    def ground_velocity(self, wind) -> np.ndarray:
        return self.air_velocity() + np.asarray(wind, dtype=float)


def wrap_pi(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class FixedWing:
    """One explicit-Euler step of the kinematic model."""

    def __init__(self, params: VehicleParams):
        self.p = params

    def step(self, s: VehicleState, phi_cmd: float, v_cmd: float, wind, dt: float) -> VehicleState:
        p = self.p
        # roll: saturate the command, first-order lag, then rate limit
        phi_cmd = float(np.clip(phi_cmd, -p.roll_max, p.roll_max))
        phi_rate = float(np.clip((phi_cmd - s.phi) / p.roll_tau, -p.roll_rate_max, p.roll_rate_max))
        phi = s.phi + phi_rate * dt

        # airspeed: acceleration limited, kept inside the flight envelope
        v_cmd = float(np.clip(v_cmd, p.v_stall, p.v_max))
        dv = float(np.clip(v_cmd - s.v, -p.accel_max * dt, p.accel_max * dt))
        v = s.v + dv

        # coordinated turn and position update (ground velocity includes wind)
        psi = wrap_pi(s.psi + G * math.tan(phi) / v * dt)
        wx, wy = wind
        x = s.x + (v * math.sin(psi) + wx) * dt
        y = s.y + (v * math.cos(psi) + wy) * dt
        return replace(s, x=x, y=y, psi=psi, v=v, phi=phi)
