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
        # Phase 8 scenarios
        "mesh_relay": _mesh_relay,
        # Phase 9 scenarios
        "terrain_real": _terrain_real,
        # Phase 10 weather scenarios
        "heavy_rain": _heavy_rain,
        "winter_storm": _winter_storm,
        "weather_api": _weather_api,
        # Phase 11 EW scenarios
        "jammed_corridor": _jammed_corridor,
        "mobile_jammer": _mobile_jammer,
        "multi_threat": _multi_threat,
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
    # Geospatial overrides (Phase 9)
    if "elevation" in overrides and overrides["elevation"]:
        config.world.elevation_tiff = str(overrides["elevation"])
    if "osm_source" in overrides and overrides["osm_source"]:
        config.world.osm_source = str(overrides["osm_source"])
    if "geo_bounds" in overrides and overrides["geo_bounds"]:
        config.world.geo_bounds = overrides["geo_bounds"]  # type: ignore[assignment]
    if "slope_weight" in overrides and overrides["slope_weight"] is not None:
        config.planning.w_slope = float(overrides["slope_weight"])  # type: ignore[arg-type]
    # Weather overrides (Phase 10)
    if overrides.get("weather_enabled"):
        config.weather.enabled = True
    if "weather_source" in overrides and overrides["weather_source"]:
        config.weather.weather_source = str(overrides["weather_source"])
    if "weather_lat" in overrides and overrides["weather_lat"] is not None:
        config.weather.latitude = float(overrides["weather_lat"])  # type: ignore[arg-type]
    if "weather_lon" in overrides and overrides["weather_lon"] is not None:
        config.weather.longitude = float(overrides["weather_lon"])  # type: ignore[arg-type]
    if "precipitation" in overrides and overrides["precipitation"] is not None:
        config.weather.enabled = True
        config.weather.static_precipitation_mm_h = float(overrides["precipitation"])  # type: ignore[arg-type]
    if "wind_speed" in overrides and overrides["wind_speed"] is not None:
        config.weather.enabled = True
        config.weather.static_wind_speed_ms = float(overrides["wind_speed"])  # type: ignore[arg-type]
    if "visibility" in overrides and overrides["visibility"] is not None:
        config.weather.enabled = True
        config.weather.static_visibility_m = float(overrides["visibility"])  # type: ignore[arg-type]
    if "temperature" in overrides and overrides["temperature"] is not None:
        config.weather.enabled = True
        config.weather.static_temperature_c = float(overrides["temperature"])  # type: ignore[arg-type]
    # EW overrides (Phase 11)
    if overrides.get("threats_enabled"):
        config.ew.enabled = True
    if "jammer_power" in overrides and overrides["jammer_power"] is not None:
        config.ew.enabled = True
    if "jammer_radius" in overrides and overrides["jammer_radius"] is not None:
        config.ew.enabled = True
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
    # Time headway — increased for safer following
    config.coordination.time_headway = 2.0
    # Increased standoff and formation spacing for platoon safety
    config.coordination.standoff_distance = 12.0
    config.coordination.formation_spacing = 30.0
    # Corridor adherence
    config.road_corridor_width = 25.0
    # Elevated IMU noise
    config.estimator.bias_instability = 0.02
    config.estimator.angle_random_walk = 0.01
    return _apply_overrides(config, **overrides)


# ---------------------------------------------------------------------------
# Phase 8 scenarios
# ---------------------------------------------------------------------------


def _mesh_relay(**overrides: object) -> SimConfig:
    """Mesh relay: reduced comms range that requires multi-hop forwarding.

    Comms range is halved to 100m so direct LOS connectivity is fragmented.
    Multi-hop relay with 2 hops is enabled so vehicles can maintain
    coordination through intermediate relay nodes.  Per-hop loss is 10%.
    """
    config = SimConfig(
        gps_available=True,
        duration=300.0,
    )
    config.comms.max_range = 100.0
    config.comms.packet_loss = 0.05
    config.comms.max_relay_hops = 2
    config.comms.relay_loss_per_hop = 0.1
    return _apply_overrides(config, **overrides)


# ---------------------------------------------------------------------------
# Phase 9 scenarios
# ---------------------------------------------------------------------------


def _terrain_real(**overrides: object) -> SimConfig:
    """Real-world terrain: OSM road network + DEM elevation.

    Requires ``--elevation`` and/or ``--osm-source`` CLI flags to supply
    geospatial data.  Without them, falls back to procedural generation
    but with slope-aware routing enabled.
    """
    config = SimConfig(
        gps_available=True,
        duration=600.0,  # longer for real-world scale
    )
    config.planning.w_slope = 0.4  # enable slope-aware routing
    return _apply_overrides(config, **overrides)


# ---------------------------------------------------------------------------
# Phase 10 weather scenarios
# ---------------------------------------------------------------------------


def _heavy_rain(**overrides: object) -> SimConfig:
    """Heavy rain: reduced friction, moderate visibility degradation.

    Tests convoy behaviour under wet-road conditions with reduced
    traction and increased stopping distances.
    """
    config = SimConfig(duration=300.0)
    config.weather.enabled = True
    config.weather.static_precipitation_mm_h = 8.0
    config.weather.static_visibility_m = 500.0
    config.weather.static_temperature_c = 10.0
    config.weather.weather_variability = 0.3
    return _apply_overrides(config, **overrides)


def _winter_storm(**overrides: object) -> SimConfig:
    """Winter storm: ice, strong crosswind, low visibility.

    Combined adverse conditions: sub-zero temperatures with precipitation
    create icy roads, high crosswind disturbs heading, and low visibility
    degrades sensors and may trigger safe mode.
    """
    config = SimConfig(duration=300.0)
    config.weather.enabled = True
    config.weather.static_precipitation_mm_h = 5.0
    config.weather.static_temperature_c = -5.0
    config.weather.static_wind_speed_ms = 12.0
    config.weather.static_wind_direction_deg = 90.0
    config.weather.static_visibility_m = 200.0
    config.weather.weather_variability = 0.5
    return _apply_overrides(config, **overrides)


def _weather_api(**overrides: object) -> SimConfig:
    """Live weather from Open-Meteo API (requires internet).

    Uses Berlin coordinates by default; override with --weather-lat and
    --weather-lon CLI flags.  Falls back to static weather if the API
    is unreachable.
    """
    config = SimConfig(duration=600.0)
    config.weather.enabled = True
    config.weather.weather_source = "api"
    config.weather.latitude = 52.52
    config.weather.longitude = 13.41
    return _apply_overrides(config, **overrides)


# ---------------------------------------------------------------------------
# Phase 11 EW scenarios
# ---------------------------------------------------------------------------


def _jammed_corridor(**overrides: object) -> SimConfig:
    """Jammed corridor: static RF jammers blocking the route.

    Two RF jammers and one GPS jammer placed along the convoy's path.
    Tests detection, triangulation, ECM activation, and route avoidance.
    """
    config = SimConfig(duration=300.0)
    config.ew.enabled = True
    config.planning.w_threat = 0.8
    return _apply_overrides(config, **overrides)


def _mobile_jammer(**overrides: object) -> SimConfig:
    """Mobile jammer: a moving jammer deployed at t=30s.

    Tests adaptive ECM response as jammer position changes over time.
    """
    config = SimConfig(duration=300.0)
    config.ew.enabled = True
    config.planning.w_threat = 0.8
    return _apply_overrides(config, **overrides)


def _multi_threat(**overrides: object) -> SimConfig:
    """Multi-threat: combined RF jammers, GPS jamming, and comms blackout.

    Stress test for convoy operations under multiple simultaneous threats.
    """
    config = SimConfig(duration=300.0)
    config.ew.enabled = True
    config.planning.w_threat = 0.8
    config.comms.max_range = 150.0
    return _apply_overrides(config, **overrides)
