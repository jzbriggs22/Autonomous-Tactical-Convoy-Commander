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
        # Phase 2 scenarios
        "gps_spoofed": _gps_spoofed,
        "silent_running": _silent_running,
        "comms_blackout": _comms_blackout,
        # Phase 3 scenarios
        "sensor_drift_spike": _sensor_drift_spike,
        # Phase 6 scenarios
        "platooning": _platooning,
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


# ---------------------------------------------------------------------------
# Phase 2 scenarios
# ---------------------------------------------------------------------------


def _gps_spoofed(**overrides: object) -> SimConfig:
    """GPS spoofing attack: spoof regions along the convoy route.

    The world generates 3 GPS spoofing zones with up to 60m offset.
    Innovation gating (5-sigma) is enabled to detect and reject spoofed fixes.
    """
    config = SimConfig(gps_available=True)
    config.world.spoof_region_count = 3
    config.world.spoof_offset_max = 60.0
    config.estimator.innovation_gate_sigma = 5.0
    return _apply_overrides(config, **overrides)


def _silent_running(**overrides: object) -> SimConfig:
    """Silent running: all vehicles suppress broadcasts to reduce RF signature.

    The broadcast interval is increased 5x compared to baseline.
    Comms loss timeout is extended proportionally so vehicles don't enter
    safe mode immediately due to the longer broadcast interval.
    """
    config = SimConfig()
    config.comms.packet_loss = 0.05
    config.comms.broadcast_interval = 5.0          # 5x longer broadcast interval
    config.coordination.comms_lost_timeout = 30.0  # Allow for longer silence
    return _apply_overrides(config, **overrides)


def _comms_blackout(**overrides: object) -> SimConfig:
    """Communications blackout zone: a large region in the convoy path where
    comms are degraded by 20x loss multiplier.

    The blackout region is added to the CommsNetwork in the SimRunner
    initialiser when ``scenario == "comms_blackout"``.
    """
    config = SimConfig(duration=300.0)
    config.comms.packet_loss = 0.05
    return _apply_overrides(config, **overrides)


# ---------------------------------------------------------------------------
# Phase 3 scenarios
# ---------------------------------------------------------------------------


def _sensor_drift_spike(**overrides: object) -> SimConfig:
    """Sensor drift spike: a sudden IMU bias injection at t=60s.

    Simulates a hardware shock event (e.g., road bump, vibration) that
    corrupts the IMU state and causes a sudden positional drift jump.

    GPS is unavailable so the spike's effect is not immediately corrected.
    Landmark fixes will gradually pull the estimate back, but safe mode
    should engage (uncertainty > threshold) immediately after the spike.

    Runner injects an 8m magnitude spike to all operational vehicles'
    estimators at t=60s via ``scenario == "sensor_drift_spike"``.
    """
    config = SimConfig(
        gps_available=False,
        duration=300.0,
    )
    config.estimator.drift_rate = 0.06       # slightly elevated baseline drift
    config.estimator.drift_bias_rate = 0.003
    return _apply_overrides(config, **overrides)


# ---------------------------------------------------------------------------
# Phase 6 scenarios
# ---------------------------------------------------------------------------


def _platooning(**overrides: object) -> SimConfig:
    """Platooning with realism upgrades: actuator lag, time headway,
    corridor adherence, and structured IMU noise.

    Leader speed perturbation at t=40s (brake to 6 m/s for 5s) is injected
    by the runner when ``scenario == "platooning"``.  String stability is
    computed on the disturbance window (t=40-60s).
    """
    config = SimConfig(
        gps_available=True,
        duration=300.0,
    )
    # Actuator lag
    config.vehicle.actuator_lag = 0.2
    # Time headway
    config.coordination.time_headway = 1.5
    # Corridor adherence
    config.road_corridor_width = 25.0
    # Elevated IMU noise
    config.estimator.bias_instability = 0.02
    config.estimator.angle_random_walk = 0.01
    return _apply_overrides(config, **overrides)
