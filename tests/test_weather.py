"""Tests for the weather effects module (Phase 10)."""

from __future__ import annotations

import math

import pytest

from convoy_commander.core.config import SimConfig, WeatherConfig
from convoy_commander.sim.scenarios import get_scenario
from convoy_commander.weather.effects import WeatherDataPoint, WeatherEffects, WeatherState


# ---------------------------------------------------------------------------
# WeatherState
# ---------------------------------------------------------------------------


class TestWeatherState:
    def test_icy_conditions(self):
        state = WeatherState(temperature_c=-3.0, precipitation_mm_h=2.0)
        assert state.is_icy is True

    def test_not_icy_warm(self):
        state = WeatherState(temperature_c=10.0, precipitation_mm_h=5.0)
        assert state.is_icy is False

    def test_not_icy_dry_cold(self):
        state = WeatherState(temperature_c=-10.0, precipitation_mm_h=0.0)
        assert state.is_icy is False


# ---------------------------------------------------------------------------
# WeatherEffects — friction
# ---------------------------------------------------------------------------


class TestFriction:
    def test_dry_conditions(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState()  # dry, warm, no wind
        assert effects.friction_factor(state) == pytest.approx(1.0)

    def test_light_rain(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState(precipitation_mm_h=5.0)
        ff = effects.friction_factor(state)
        assert 0.85 < ff < 1.0  # moderate reduction

    def test_heavy_rain(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState(precipitation_mm_h=10.0)
        ff = effects.friction_factor(state)
        assert ff == pytest.approx(0.85, abs=0.01)

    def test_ice(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState(temperature_c=-5.0, precipitation_mm_h=3.0)
        ff = effects.friction_factor(state)
        assert ff < 0.70  # significant reduction from ice + rain

    def test_friction_floor(self):
        """Friction must never go below 0.3."""
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState(temperature_c=-20.0, precipitation_mm_h=50.0)
        assert effects.friction_factor(state) >= 0.3

    def test_stopping_distance_inverse(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState(precipitation_mm_h=10.0)
        ff = effects.friction_factor(state)
        sd = effects.stopping_distance_factor(state)
        assert sd == pytest.approx(1.0 / ff)


# ---------------------------------------------------------------------------
# WeatherEffects — wind
# ---------------------------------------------------------------------------


class TestWind:
    def test_no_wind(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState(wind_speed_ms=0.0)
        h, s = effects.wind_effects(state, vehicle_heading=0.0, vehicle_speed=10.0)
        assert h == 0.0
        assert s == 0.0

    def test_crosswind_perturbation(self):
        cfg = WeatherConfig(enabled=True, crosswind_sensitivity=0.02)
        effects = WeatherEffects(cfg)
        # Wind from north (pi/2 rad), vehicle heading east (0 rad)
        state = WeatherState(wind_speed_ms=10.0, wind_direction_rad=math.pi / 2)
        h, _ = effects.wind_effects(state, vehicle_heading=0.0, vehicle_speed=10.0)
        assert abs(h) > 0.1  # measurable heading perturbation

    def test_headwind_reduces_speed(self):
        cfg = WeatherConfig(enabled=True, wind_speed_effect=0.1)
        effects = WeatherEffects(cfg)
        # Wind blowing opposite to vehicle heading
        state = WeatherState(wind_speed_ms=10.0, wind_direction_rad=math.pi)
        _, speed_delta = effects.wind_effects(state, vehicle_heading=0.0, vehicle_speed=10.0)
        assert speed_delta < 0  # headwind reduces speed

    def test_tailwind_increases_speed(self):
        cfg = WeatherConfig(enabled=True, wind_speed_effect=0.1)
        effects = WeatherEffects(cfg)
        # Wind blowing same direction as vehicle heading
        state = WeatherState(wind_speed_ms=10.0, wind_direction_rad=0.0)
        _, speed_delta = effects.wind_effects(state, vehicle_heading=0.0, vehicle_speed=10.0)
        assert speed_delta > 0  # tailwind boosts speed


# ---------------------------------------------------------------------------
# WeatherEffects — visibility
# ---------------------------------------------------------------------------


class TestVisibility:
    def test_clear_visibility(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState(visibility_m=10000.0)
        sensor, comms, safe = effects.visibility_effects(state)
        assert sensor == pytest.approx(1.0)
        assert comms == pytest.approx(1.0)
        assert safe is False

    def test_reduced_visibility(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState(visibility_m=200.0)
        sensor, comms, safe = effects.visibility_effects(state)
        assert sensor == pytest.approx(0.2, abs=0.01)
        assert comms < 1.0
        assert safe is False

    def test_extreme_fog_triggers_safe_mode(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState(visibility_m=50.0)
        _, _, safe = effects.visibility_effects(state)
        assert safe is True


# ---------------------------------------------------------------------------
# WeatherEffects — fuel
# ---------------------------------------------------------------------------


class TestFuel:
    def test_warm_no_wind(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState(temperature_c=20.0, wind_speed_ms=0.0)
        assert effects.fuel_factor(state) == pytest.approx(1.0)

    def test_cold_increases_fuel(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        state = WeatherState(temperature_c=-15.0)
        assert effects.fuel_factor(state) > 1.15


# ---------------------------------------------------------------------------
# WeatherEffects — interpolation
# ---------------------------------------------------------------------------


class TestInterpolation:
    def test_static_state(self):
        cfg = WeatherConfig(
            enabled=True,
            static_temperature_c=15.0,
            static_precipitation_mm_h=3.0,
        )
        effects = WeatherEffects(cfg)
        state = effects.get_state(50.0)
        assert state.temperature_c == pytest.approx(15.0, abs=0.1)
        assert state.precipitation_mm_h == pytest.approx(3.0, abs=0.5)

    def test_static_variability(self):
        cfg = WeatherConfig(
            enabled=True,
            static_temperature_c=20.0,
            weather_variability=1.0,
            weather_period_s=100.0,
        )
        effects = WeatherEffects(cfg)
        temps = [effects.get_state(t).temperature_c for t in range(0, 100, 10)]
        assert max(temps) > min(temps)  # temperature varies

    def test_api_interpolation(self):
        cfg = WeatherConfig(enabled=True)
        effects = WeatherEffects(cfg)
        points = [
            WeatherDataPoint(0.0, 10.0, 0.0, 5.0, 0.0, 5000.0),
            WeatherDataPoint(3600.0, 20.0, 10.0, 10.0, 1.0, 1000.0),
        ]
        effects.set_data_points(points)
        state = effects.get_state(1800.0)  # midpoint
        assert state.temperature_c == pytest.approx(15.0, abs=0.1)
        assert state.precipitation_mm_h == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# Config and scenarios
# ---------------------------------------------------------------------------


class TestWeatherConfig:
    def test_default_weather_disabled(self):
        config = SimConfig()
        assert config.weather.enabled is False

    def test_heavy_rain_scenario(self):
        config = get_scenario("heavy_rain", duration=10, vehicles=4, seed=42)
        assert config.weather.enabled is True
        assert config.weather.static_precipitation_mm_h == 8.0
        assert config.weather.static_visibility_m == 500.0

    def test_winter_storm_scenario(self):
        config = get_scenario("winter_storm", duration=10, vehicles=4, seed=42)
        assert config.weather.enabled is True
        assert config.weather.static_temperature_c == -5.0
        assert config.weather.static_wind_speed_ms == 12.0

    def test_weather_api_scenario(self):
        config = get_scenario("weather_api", duration=10, vehicles=4, seed=42)
        assert config.weather.enabled is True
        assert config.weather.weather_source == "api"
        assert config.weather.latitude == pytest.approx(52.52)


# ---------------------------------------------------------------------------
# Integration: short sim runs
# ---------------------------------------------------------------------------


class TestWeatherSim:
    def test_heavy_rain_runs(self):
        """Heavy rain scenario completes a short run without crashing."""
        from convoy_commander.sim.runner import SimRunner

        config = get_scenario("heavy_rain", duration=10, vehicles=4, seed=42)
        runner = SimRunner(config)
        result = runner.run()
        assert result is not None
        assert len(result.vehicles) == 4

    def test_winter_storm_runs(self):
        """Winter storm scenario completes a short run without crashing."""
        from convoy_commander.sim.runner import SimRunner

        config = get_scenario("winter_storm", duration=10, vehicles=4, seed=42)
        runner = SimRunner(config)
        result = runner.run()
        assert result is not None

    def test_weather_events_logged(self):
        """Weather events appear in the event log."""
        from convoy_commander.core.event_log import EventKind
        from convoy_commander.sim.runner import SimRunner

        config = get_scenario("heavy_rain", duration=15, vehicles=4, seed=42)
        runner = SimRunner(config)
        result = runner.run()
        weather_events = result.event_log.filter(kind=EventKind.WEATHER_UPDATED)
        assert len(weather_events) > 0

    def test_weather_metrics_recorded(self):
        """Weather metrics are computed in final metrics."""
        from convoy_commander.sim.runner import SimRunner

        config = get_scenario("heavy_rain", duration=15, vehicles=4, seed=42)
        runner = SimRunner(config)
        result = runner.run()
        metrics = result.collector.compute_final(
            result.vehicles,
            result.comms.total_sent,
            result.comms.total_delivered,
            result.comms.total_dropped,
            config.duration,
        )
        assert metrics.weather_source == "active"
        assert metrics.avg_friction_factor < 1.0  # rain reduces friction

    def test_weather_cli_override(self):
        """Weather can be enabled via CLI-style overrides."""
        config = get_scenario(
            "baseline", duration=10, vehicles=4, seed=42,
            precipitation=5.0, wind_speed=8.0,
        )
        assert config.weather.enabled is True
        assert config.weather.static_precipitation_mm_h == 5.0
        assert config.weather.static_wind_speed_ms == 8.0
