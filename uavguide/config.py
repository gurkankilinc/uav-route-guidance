"""Modeling assumptions, kept in one place.

Every number the simulation depends on lives here, so the README's
assumptions table and the code can never drift apart.

Frame convention (aviation style, used everywhere in the package):
    x = east [m], y = north [m]
    heading psi = clockwise from north [rad]
    positive roll phi = right wing down = right turn
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

G = 9.81  # m/s^2


@dataclass(frozen=True)
class VehicleParams:
    """Small fixed-wing UAV, constant altitude, coordinated turns."""

    v_stall: float = 12.0          # m/s
    v_cruise: float = 20.0         # m/s
    v_max: float = 26.0            # m/s
    accel_max: float = 2.0         # m/s^2, |dV/dt| limit
    roll_max_deg: float = 35.0     # bank angle limit
    roll_tau: float = 0.4          # s, first-order roll response
    roll_rate_max_deg: float = 60.0  # deg/s

    @property
    def roll_max(self) -> float:
        return math.radians(self.roll_max_deg)

    @property
    def roll_rate_max(self) -> float:
        return math.radians(self.roll_rate_max_deg)

    def min_turn_radius(self, v: float | None = None) -> float:
        """R = V^2 / (g tan(phi_max)). ~58 m at cruise."""
        v = self.v_cruise if v is None else v
        return v * v / (G * math.tan(self.roll_max))


@dataclass(frozen=True)
class WindParams:
    """Constant, uniform wind. Unknown to the guidance law."""

    speed: float = 5.0             # m/s
    from_deg: float = 270.0        # meteorological: direction it blows FROM

    def vector(self) -> tuple[float, float]:
        """(east, north) velocity of the air mass."""
        a = math.radians(self.from_deg)
        return (-self.speed * math.sin(a), -self.speed * math.cos(a))


@dataclass(frozen=True)
class GuidanceParams:
    """L1 lateral guidance, tuned like ArduPlane's NAVL1_* parameters."""

    l1_period: float = 17.0        # s
    l1_damping: float = 0.75
    wp_accept_radius: float = 30.0  # m
    xtrack_i_gain: float = 0.0     # ArduPlane NAVL1_XTRACK_I; 0 = pure L1 (ArduPlane default 0.02)

    @property
    def k_l1(self) -> float:
        """Lateral gain K_L1 = 4 zeta^2 (2.25 for zeta = 0.75), as in AP_L1_Control."""
        return 4.0 * self.l1_damping ** 2

    def l1_distance(self, v: float) -> float:
        """L1 = zeta * T * V / pi. ~81 m at 20 m/s."""
        return self.l1_damping * self.l1_period * v / math.pi


@dataclass(frozen=True)
class PlannerParams:
    area: tuple[float, float, float, float] = (0.0, 0.0, 600.0, 600.0)  # xmin, ymin, xmax, ymax
    grid_res: float = 2.0          # m per A* cell
    safety_margin: float = 10.0    # m added to every obstacle radius
    dp_epsilon: float = 5.0        # m, Douglas-Peucker tolerance
    dp_clearance: float = 5.0      # m, min clearance a simplified segment must keep


@dataclass(frozen=True)
class TargetParams:
    speed: float = 17.0            # m/s, must satisfy v_stall < speed < v_max
    mode: str = "line"             # "line" | "circle"
    circle_radius: float = 120.0   # m


@dataclass(frozen=True)
class ApproachParams:
    """v(d) = v_t + (v_max - v_t) * tanh((d - d_follow) / d0)."""

    d_follow: float = 40.0         # m, desired trailing distance
    d0: float = 60.0               # m, how early to start slowing down
    capture_radius: float = 60.0   # m, "caught up" once closer than this ...
    hold_time: float = 5.0         # s, ... and staying that close for this long


@dataclass(frozen=True)
class SensorParams:
    """Measurement models used when the guidance does not get perfect state."""

    gps_rate_hz: float = 5.0       # GNSS velocity update rate
    gps_vel_sigma: float = 0.5     # m/s, per axis
    airspeed_sigma: float = 0.5    # m/s
    heading_sigma_deg: float = 1.0  # compass / AHRS heading noise


@dataclass(frozen=True)
class WindEstimatorParams:
    """2-state Kalman filter: wind (east, north) as a random walk."""

    process_sigma: float = 0.05    # m/s per sqrt(s), how fast the wind may drift
    initial_sigma: float = 5.0     # m/s, prior uncertainty (wind unknown at start)


@dataclass(frozen=True)
class SimParams:
    dt: float = 0.02               # s (50 Hz)
    t_max: float = 300.0           # s


@dataclass(frozen=True)
class Config:
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    wind: WindParams = field(default_factory=WindParams)
    guidance: GuidanceParams = field(default_factory=GuidanceParams)
    planner: PlannerParams = field(default_factory=PlannerParams)
    target: TargetParams = field(default_factory=TargetParams)
    approach: ApproachParams = field(default_factory=ApproachParams)
    sensors: SensorParams = field(default_factory=SensorParams)
    wind_estimator: WindEstimatorParams = field(default_factory=WindEstimatorParams)
    sim: SimParams = field(default_factory=SimParams)
