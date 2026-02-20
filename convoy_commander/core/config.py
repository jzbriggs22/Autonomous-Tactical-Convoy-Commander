"""Simulation configuration using pydantic models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VehicleConfig(BaseModel):
    """Per-vehicle kinematic and energy parameters."""

    max_speed: float = Field(default=15.0, description="Max speed m/s")
    max_accel: float = Field(default=3.0, description="Max acceleration m/s^2")
    max_decel: float = Field(default=5.0, description="Max braking deceleration m/s^2")
    max_turn_rate: float = Field(default=0.8, description="Max yaw rate rad/s")
    length: float = Field(default=6.0, description="Vehicle length m")
    width: float = Field(default=2.5, description="Vehicle width m")
    fuel_capacity: float = Field(default=100.0, description="Fuel units")
    fuel_rate_idle: float = Field(default=0.01, description="Fuel/s at idle")
    fuel_rate_per_speed: float = Field(default=0.005, description="Fuel/s per m/s speed")


class EstimatorConfig(BaseModel):
    """Dead-reckoning / position estimator parameters."""

    drift_rate: float = Field(default=0.05, description="DR drift std m/s")
    drift_bias_rate: float = Field(default=0.002, description="Bias random walk m/s^2")
    landmark_fix_std: float = Field(default=1.0, description="Landmark fix noise std m")
    gps_fix_std: float = Field(default=0.5, description="GPS fix noise std m")
    uncertainty_safe_threshold: float = Field(
        default=20.0, description="Position uncertainty threshold to trigger safe mode (m)"
    )


class CommsConfig(BaseModel):
    """Communications model parameters."""

    max_range: float = Field(default=200.0, description="Max comms range m")
    packet_loss: float = Field(default=0.05, description="Baseline packet loss probability")
    latency_mean_ms: float = Field(default=50.0, description="Mean latency ms")
    latency_std_ms: float = Field(default=10.0, description="Latency std ms")
    broadcast_interval: float = Field(default=1.0, description="State broadcast interval s")


class CoordinationConfig(BaseModel):
    """Coordination and formation parameters."""

    formation_spacing: float = Field(default=20.0, description="Target inter-vehicle spacing m")
    min_separation: float = Field(default=8.0, description="Hard min separation m")
    collision_radius: float = Field(default=4.0, description="Collision detection radius m")
    comms_lost_timeout: float = Field(default=10.0, description="Seconds without comms before safe mode")
    safe_mode_speed_factor: float = Field(default=0.4, description="Speed factor in safe mode")
    safe_mode_spacing_factor: float = Field(default=2.0, description="Spacing multiplier in safe mode")
    leader_heartbeat_timeout: float = Field(default=5.0, description="Leader heartbeat timeout s")


class WorldConfig(BaseModel):
    """World / map configuration."""

    width: float = Field(default=1000.0, description="World width m")
    height: float = Field(default=1000.0, description="World height m")
    road_graph_density: int = Field(default=12, description="Grid nodes per axis for road graph")
    obstacle_count: int = Field(default=15, description="Number of random obstacles")
    obstacle_radius_range: tuple[float, float] = Field(
        default=(5.0, 25.0), description="Min/max obstacle radius m"
    )
    nogo_zone_count: int = Field(default=3, description="Number of no-go zones")
    nogo_zone_radius_range: tuple[float, float] = Field(
        default=(30.0, 60.0), description="Min/max no-go zone radius m"
    )
    landmark_count: int = Field(default=8, description="Number of landmarks for position fixes")


class SimConfig(BaseModel):
    """Top-level simulation configuration."""

    seed: int = Field(default=42, description="Random seed for reproducibility")
    dt: float = Field(default=0.1, description="Simulation timestep s")
    duration: float = Field(default=300.0, description="Total sim duration s")
    num_vehicles: int = Field(default=8, description="Number of vehicles")
    gps_available: bool = Field(default=True, description="Whether GPS is available")
    gps_intermittent_prob: float = Field(
        default=0.0, description="Probability of GPS fix each second (0 = never when gps_available=False)"
    )
    vehicle: VehicleConfig = Field(default_factory=VehicleConfig)
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    comms: CommsConfig = Field(default_factory=CommsConfig)
    coordination: CoordinationConfig = Field(default_factory=CoordinationConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
    scenario: str = Field(default="baseline", description="Scenario name")
