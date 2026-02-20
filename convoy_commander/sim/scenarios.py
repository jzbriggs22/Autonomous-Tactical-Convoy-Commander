"""Scenario definitions.

Each scenario returns a ``SimConfig`` with parameters chosen for that
operational context.  All defaults in ``SimConfig`` are conservative;
scenarios override only what is necessary for the test case.
"""

from __future__ import annotations

from typing import Callable

from convoy_commander.core.config import SimConfig


def get_scenario(name: str, **overrides: object) -> SimConfig:
    """Get a scenario configuration by name."""
    builders: dict[str, Callable[..., SimConfig]] = {
        "baseline": _baseline,
        "gps_denied": _gps_denied,
        "comms_degraded": _comms_degraded,
        "leader_failure": _leader_failure,
        "obstacle_pop": _obstacle_pop,
    }
    if name not in builders:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(builders.keys())}")
    config = builders[name](**overrides)
    config.scenario = name
    return config


def _apply_overrides(config: SimConfig, **overrides: object) -> SimConfig:
    """Apply CLI overrides to config."""
    if "seed" in overrides:
        config.seed = int(overrides["seed"])  # type: ignore[arg-type]
    if "vehicles" in overrides:
        config.num_vehicles = int(overrides["vehicles"])  # type: ignore[arg-type]
    if "loss" in overrides:
        config.comms.packet_loss = float(overrides["loss"])  # type: ignore[arg-type]
    if "latency" in overrides:
        config.comms.latency_mean_ms = float(overrides["latency"])  # type: ignore[arg-type]
    if "duration" in overrides:
        config.duration = float(overrides["duration"])  # type: ignore[arg-type]
    return config


def _baseline(**overrides: object) -> SimConfig:
    """Baseline: GPS available, low loss."""
    config = SimConfig()
    config.comms.packet_loss = 0.02
    config.comms.latency_mean_ms = 30.0
    return _apply_overrides(config, **overrides)


def _gps_denied(**overrides: object) -> SimConfig:
    """GPS denied: no GPS, IMU drift present."""
    config = SimConfig(
        gps_available=False,
        gps_intermittent_prob=0.0,
    )
    config.estimator.drift_rate = 0.08
    config.estimator.drift_bias_rate = 0.004
    return _apply_overrides(config, **overrides)


def _comms_degraded(**overrides: object) -> SimConfig:
    """Comms degraded: high loss + latency."""
    config = SimConfig()
    config.comms.packet_loss = 0.3
    config.comms.latency_mean_ms = 200.0
    config.comms.latency_std_ms = 80.0
    config.comms.max_range = 120.0
    return _apply_overrides(config, **overrides)


def _leader_failure(**overrides: object) -> SimConfig:
    """Leader fails at t=120s."""
    config = SimConfig(duration=300.0)
    return _apply_overrides(config, **overrides)


def _obstacle_pop(**overrides: object) -> SimConfig:
    """New obstacle appears at t=90s forcing reroute."""
    config = SimConfig(duration=300.0)
    return _apply_overrides(config, **overrides)
