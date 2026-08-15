"""Weather effects engine: pure-Python logic for weather impact on driving.

No external dependencies. All formulas use piecewise-linear or clamped models
for simplicity and predictability.

ASSUMPTIONS:
  - Friction model is piecewise-linear; real tire-road friction depends on
    tire type, road surface, and water depth.
  - Wind model uses simple cosine/sine decomposition relative to vehicle
    heading; turbulence and gusting are not modelled.
  - Visibility uniformly affects all sensors; real sensor degradation varies
    by type (lidar vs camera vs radar).
  - Temperature effect on fuel is a step function; real cold-start and
    battery degradation is more complex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from convoy_commander.core.config import WeatherConfig


@dataclass
class WeatherState:
    """Current weather snapshot at a given simulation time."""

    temperature_c: float = 20.0
    precipitation_mm_h: float = 0.0
    wind_speed_ms: float = 0.0
    wind_direction_rad: float = 0.0
    visibility_m: float = 10000.0

    @property
    def is_icy(self) -> bool:
        """Ice conditions: near-freezing with precipitation."""
        return self.temperature_c < 2.0 and self.precipitation_mm_h > 0.0


@dataclass
class WeatherDataPoint:
    """A single weather observation from the API."""

    time_offset_s: float
    temperature_c: float
    precipitation_mm_h: float
    wind_speed_ms: float
    wind_direction_rad: float
    visibility_m: float


class WeatherEffects:
    """Computes weather impacts on vehicle dynamics.

    Initialized with a WeatherConfig. Can hold API data points for
    time-varying interpolation, or generate static/sinusoidal weather.
    """

    def __init__(self, config: WeatherConfig) -> None:
        self.config = config
        self.data_points: list[WeatherDataPoint] = []

    def set_data_points(self, points: list[WeatherDataPoint]) -> None:
        """Set API-sourced data points for time-varying interpolation."""
        self.data_points = sorted(points, key=lambda p: p.time_offset_s)

    def get_state(self, sim_time: float) -> WeatherState:
        """Get interpolated weather state at the given simulation time."""
        if self.data_points:
            return self._interpolate_api(sim_time)
        return self._static_state(sim_time)

    def _static_state(self, sim_time: float) -> WeatherState:
        """Generate weather from static config values with optional variation."""
        cfg = self.config
        var = cfg.weather_variability
        if var > 0 and cfg.weather_period_s > 0:
            phase = 2.0 * math.pi * sim_time / cfg.weather_period_s
            mod = math.sin(phase) * var
        else:
            mod = 0.0

        return WeatherState(
            temperature_c=cfg.static_temperature_c + mod * 5.0,
            precipitation_mm_h=max(0.0, cfg.static_precipitation_mm_h * (1.0 + mod * 0.5)),
            wind_speed_ms=max(0.0, cfg.static_wind_speed_ms * (1.0 + mod * 0.3)),
            wind_direction_rad=math.radians(cfg.static_wind_direction_deg) + mod * 0.2,
            visibility_m=max(50.0, cfg.static_visibility_m * (1.0 - mod * 0.3)),
        )

    def _interpolate_api(self, sim_time: float) -> WeatherState:
        """Linearly interpolate between API data points."""
        pts = self.data_points
        if not pts:
            return self._static_state(sim_time)
        if sim_time <= pts[0].time_offset_s:
            p = pts[0]
            return WeatherState(
                temperature_c=p.temperature_c,
                precipitation_mm_h=p.precipitation_mm_h,
                wind_speed_ms=p.wind_speed_ms,
                wind_direction_rad=p.wind_direction_rad,
                visibility_m=p.visibility_m,
            )
        if sim_time >= pts[-1].time_offset_s:
            p = pts[-1]
            return WeatherState(
                temperature_c=p.temperature_c,
                precipitation_mm_h=p.precipitation_mm_h,
                wind_speed_ms=p.wind_speed_ms,
                wind_direction_rad=p.wind_direction_rad,
                visibility_m=p.visibility_m,
            )
        # Find bounding points
        for i in range(len(pts) - 1):
            if pts[i].time_offset_s <= sim_time <= pts[i + 1].time_offset_s:
                a, b = pts[i], pts[i + 1]
                dt = b.time_offset_s - a.time_offset_s
                t = (sim_time - a.time_offset_s) / dt if dt > 0 else 0.0
                return WeatherState(
                    temperature_c=a.temperature_c + t * (b.temperature_c - a.temperature_c),
                    precipitation_mm_h=max(0.0, a.precipitation_mm_h + t * (b.precipitation_mm_h - a.precipitation_mm_h)),
                    wind_speed_ms=max(0.0, a.wind_speed_ms + t * (b.wind_speed_ms - a.wind_speed_ms)),
                    wind_direction_rad=a.wind_direction_rad + t * (b.wind_direction_rad - a.wind_direction_rad),
                    visibility_m=max(50.0, a.visibility_m + t * (b.visibility_m - a.visibility_m)),
                )
        return self._static_state(sim_time)

    # ------------------------------------------------------------------
    # Effect computations
    # ------------------------------------------------------------------

    def friction_factor(self, state: WeatherState) -> float:
        """Surface friction multiplier [0.3, 1.0]. Applied to max_speed, accel, decel."""
        factor = 1.0
        # Rain: up to 15% reduction at 10+ mm/h
        if state.precipitation_mm_h > 0:
            rain_penalty = 0.15 * min(state.precipitation_mm_h / 10.0, 1.0)
            factor -= rain_penalty
        # Ice: additional 30% reduction
        if state.is_icy:
            factor -= 0.30
        return max(0.3, min(1.0, factor))

    def stopping_distance_factor(self, state: WeatherState) -> float:
        """Multiplier >= 1.0 for min_separation increase (inverse of friction)."""
        ff = self.friction_factor(state)
        return 1.0 / ff

    def wind_effects(
        self, state: WeatherState, vehicle_heading: float, vehicle_speed: float,
    ) -> tuple[float, float]:
        """Compute wind impact on vehicle.

        Returns (heading_perturbation_rad, speed_delta_ms).
        """
        if state.wind_speed_ms < 0.5:
            return 0.0, 0.0
        relative_angle = state.wind_direction_rad - vehicle_heading
        # Crosswind → heading perturbation
        crosswind = state.wind_speed_ms * math.sin(relative_angle)
        heading_perturb = crosswind * self.config.crosswind_sensitivity
        # Head/tailwind → speed delta (positive = tailwind boost)
        along_wind = state.wind_speed_ms * math.cos(relative_angle)
        speed_delta = along_wind * self.config.wind_speed_effect
        return heading_perturb, speed_delta

    def visibility_effects(self, state: WeatherState) -> tuple[float, float, bool]:
        """Compute visibility impact.

        Returns (sensor_range_factor, comms_range_factor, trigger_safe_mode).
        """
        # Sensor range: full at 1km+, down to 20% at 200m
        sensor_factor = max(0.2, min(1.0, state.visibility_m / 1000.0))
        # Comms less affected than sensors
        comms_factor = max(0.5, min(1.0, state.visibility_m / 500.0))
        # Extremely dense fog triggers safe mode
        force_safe = state.visibility_m < 100.0
        return sensor_factor, comms_factor, force_safe

    def fuel_factor(self, state: WeatherState) -> float:
        """Fuel consumption multiplier >= 1.0 for adverse conditions."""
        factor = 1.0
        # Cold penalty: +20% below -10°C
        if state.temperature_c < -10.0:
            factor += 0.20
        # Wind resistance penalty
        if state.wind_speed_ms > 1.0:
            factor += 0.05 * state.wind_speed_ms / 10.0
        return factor
