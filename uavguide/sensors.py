"""Where the guidance gets its ground velocity from.

The L1 law needs the velocity over ground. How that number is obtained is a
design choice, so it is modelled as an interchangeable *source*:

    TrueGroundVelocity   ideal reference (perfect GNSS/EKF)
    NoisyGnssVelocity    GNSS velocity, 5 Hz sample-and-hold with noise
    AirDataOnly          airspeed + heading only; the wind is ignored

`uavguide.wind.WindEstimateSource` adds a fourth one that combines air data
with an estimated wind.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .config import SensorParams
from .vehicle import VehicleState


class VelocitySource(Protocol):
    name: str

    def reset(self, rng: np.random.Generator) -> None: ...

    def ground_velocity(self, t: float, s: VehicleState, wind: np.ndarray) -> np.ndarray:
        """Ground velocity (east, north) the guidance will use at time t."""
        ...


class TrueGroundVelocity:
    name = "true ground velocity"

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def ground_velocity(self, t, s, wind):
        return s.ground_velocity(wind)


class NoisyGnssVelocity:
    """GNSS velocity: updated at `gps_rate_hz`, held in between, Gaussian noise per axis."""

    name = "noisy GNSS velocity"

    def __init__(self, params: SensorParams):
        self.p = params
        self._rng = np.random.default_rng(0)
        self._next_t = 0.0
        self._held = np.zeros(2)

    def reset(self, rng):
        self._rng = rng
        self._next_t = 0.0
        self._held = np.zeros(2)

    def sample(self, s: VehicleState, wind) -> np.ndarray:
        return s.ground_velocity(wind) + self._rng.normal(0.0, self.p.gps_vel_sigma, 2)

    def ground_velocity(self, t, s, wind):
        if t + 1e-9 >= self._next_t:
            self._held = self.sample(s, wind)
            self._next_t += 1.0 / self.p.gps_rate_hz
        return self._held


class AirDataOnly:
    """Velocity rebuilt from measured airspeed and heading; knows nothing about wind."""

    name = "air data only (wind ignored)"

    def __init__(self, params: SensorParams):
        self.p = params
        self._rng = np.random.default_rng(0)

    def reset(self, rng):
        self._rng = rng

    def air_velocity(self, s: VehicleState) -> np.ndarray:
        v = s.v + self._rng.normal(0.0, self.p.airspeed_sigma)
        psi = s.psi + math.radians(self._rng.normal(0.0, self.p.heading_sigma_deg))
        return v * np.array([math.sin(psi), math.cos(psi)])

    def ground_velocity(self, t, s, wind):
        return self.air_velocity(s)
