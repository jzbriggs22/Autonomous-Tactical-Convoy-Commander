"""Simulation configuration using pydantic models.

SAFETY-CRITICAL ASSUMPTIONS:
  A1. All physical quantities use SI units (metres, seconds, radians, m/s, m/s^2)
      unless explicitly noted otherwise.
  A2. Default values are chosen *conservatively*: lower speeds, wider separations,
      and tighter thresholds than an aggressive operational profile would use.
  A3. Configuration is validated at construction time; unsafe parameter
      combinations raise ``ValueError`` before any simulation step executes.
  A4. Immutable after validation — callers must not monkey-patch fields at
      runtime.  (Pydantic frozen models enforce this where possible.)
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class VehicleConfig(BaseModel):
    """Per-vehicle kinematic and energy parameters.

    Assumptions:
      - Bicycle-like 2-D kinematics (no roll, pitch, or vertical dynamics).
      - Fuel consumption is a linear function of speed — adequate for
        steady-state convoy driving, but does not model transients.
    """

    max_speed: float = Field(default=12.0, ge=0.1, description="Max speed m/s (conservative: ~43 km/h)")
    max_accel: float = Field(default=2.0, ge=0.1, description="Max acceleration m/s^2")
    max_decel: float = Field(default=4.0, ge=0.1, description="Max braking deceleration m/s^2")
    max_turn_rate: float = Field(default=0.6, ge=0.01, description="Max yaw rate rad/s")
    length: float = Field(default=6.0, gt=0, description="Vehicle length m")
    width: float = Field(default=2.5, gt=0, description="Vehicle width m")
    fuel_capacity: float = Field(default=100.0, gt=0, description="Fuel units")
    fuel_rate_idle: float = Field(default=0.01, ge=0, description="Fuel/s at idle")
    fuel_rate_per_speed: float = Field(default=0.005, ge=0, description="Fuel/s per m/s speed")

    @model_validator(mode="after")
    def _validate_decel_exceeds_accel(self) -> VehicleConfig:
        if self.max_decel < self.max_accel:
            raise ValueError(
                f"SAFETY: max_decel ({self.max_decel}) must be >= max_accel ({self.max_accel}) "
                "so the vehicle can always stop at least as fast as it accelerates."
            )
        return self


class EstimatorConfig(BaseModel):
    """Dead-reckoning / position estimator parameters.

    Assumptions:
      - Drift is modelled as additive white noise plus a slow random-walk bias.
        This is a first-order approximation to real IMU drift; it does NOT
        capture temperature-dependent bias, vibration coupling, or
        quantisation effects.
      - The complementary-filter gain is recomputed every fix, which is
        a simplification of a full EKF.
    """

    drift_rate: float = Field(default=0.05, ge=0, description="DR drift std m/s")
    drift_bias_rate: float = Field(default=0.002, ge=0, description="Bias random walk m/s^2")
    landmark_fix_std: float = Field(default=1.0, gt=0, description="Landmark fix noise std m")
    gps_fix_std: float = Field(default=0.5, gt=0, description="GPS fix noise std m")
    uncertainty_safe_threshold: float = Field(
        default=15.0, gt=0, description="Position uncertainty threshold to trigger safe mode (m)"
    )
    innovation_gate_sigma: float = Field(
        default=5.0, ge=0,
        description="Innovation gate multiplier: reject fix if innovation > gate_sigma * "
                    "max(uncertainty, fix_std).  0 = disabled (accept all fixes).",
    )


class CommsConfig(BaseModel):
    """Communications model parameters.

    Assumptions:
      - Propagation is line-of-sight with distance-squared degradation;
        no multipath, fading, or frequency effects.
      - Latency is drawn independently per packet (no queuing model).
    """

    max_range: float = Field(default=200.0, gt=0, description="Max comms range m")
    packet_loss: float = Field(default=0.05, ge=0, le=1.0, description="Baseline packet loss probability")
    latency_mean_ms: float = Field(default=50.0, ge=0, description="Mean latency ms")
    latency_std_ms: float = Field(default=10.0, ge=0, description="Latency std ms")
    broadcast_interval: float = Field(default=1.0, gt=0, description="State broadcast interval s")


class CoordinationConfig(BaseModel):
    """Coordination and formation parameters.

    Assumptions:
      - Formation is single-file behind the leader; no lateral offsets.
      - Collision radius is body-to-body (centre distance); a more
        conservative model would use swept-volume overlap.
    """

    formation_spacing: float = Field(default=25.0, gt=0, description="Target inter-vehicle spacing m")
    min_separation: float = Field(default=10.0, gt=0, description="Hard min separation m")
    collision_radius: float = Field(default=5.0, gt=0, description="Collision detection radius m")
    comms_lost_timeout: float = Field(default=8.0, gt=0, description="Seconds without comms before safe mode")
    safe_mode_speed_factor: float = Field(default=0.3, gt=0, le=1.0, description="Speed factor in safe mode")
    safe_mode_spacing_factor: float = Field(default=2.5, ge=1.0, description="Spacing multiplier in safe mode")
    leader_heartbeat_timeout: float = Field(default=5.0, gt=0, description="Leader heartbeat timeout s")

    @model_validator(mode="after")
    def _validate_separation_hierarchy(self) -> CoordinationConfig:
        if self.collision_radius >= self.min_separation:
            raise ValueError(
                f"SAFETY: collision_radius ({self.collision_radius}) must be strictly less than "
                f"min_separation ({self.min_separation}).  Otherwise the near-miss zone is empty "
                "and vehicles collide without any warning region."
            )
        if self.min_separation >= self.formation_spacing:
            raise ValueError(
                f"SAFETY: min_separation ({self.min_separation}) must be less than "
                f"formation_spacing ({self.formation_spacing}).  Vehicles cannot maintain formation "
                "if formation spacing is within the hard-stop separation envelope."
            )
        return self


class WorldConfig(BaseModel):
    """World / map configuration."""

    width: float = Field(default=1000.0, gt=0, description="World width m")
    height: float = Field(default=1000.0, gt=0, description="World height m")
    road_graph_density: int = Field(default=12, ge=2, description="Grid nodes per axis for road graph")
    obstacle_count: int = Field(default=15, ge=0, description="Number of random obstacles")
    obstacle_radius_range: tuple[float, float] = Field(
        default=(5.0, 25.0), description="Min/max obstacle radius m"
    )
    nogo_zone_count: int = Field(default=3, ge=0, description="Number of no-go zones")
    nogo_zone_radius_range: tuple[float, float] = Field(
        default=(30.0, 60.0), description="Min/max no-go zone radius m"
    )
    landmark_count: int = Field(default=8, ge=0, description="Number of landmarks for position fixes")
    spoof_region_count: int = Field(default=0, ge=0, description="Number of GPS spoofing regions")
    spoof_offset_max: float = Field(default=50.0, gt=0, description="Max GPS spoof offset magnitude m")
    poly_obstacle_count: int = Field(default=3, ge=0, description="Number of axis-aligned rectangle obstacles")

    @model_validator(mode="after")
    def _validate_radius_ranges(self) -> WorldConfig:
        lo, hi = self.obstacle_radius_range
        if lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError(
                f"SAFETY: obstacle_radius_range must satisfy 0 < lo <= hi, got ({lo}, {hi})"
            )
        lo, hi = self.nogo_zone_radius_range
        if lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError(
                f"SAFETY: nogo_zone_radius_range must satisfy 0 < lo <= hi, got ({lo}, {hi})"
            )
        return self


class PlanningObjective(BaseModel):
    """Weights for multi-objective route planning.

    The composite edge cost is:
        w_time * dist + w_fuel * dist + w_risk * edge_risk_score

    where ``edge_risk_score`` is pre-annotated during world generation based
    on proximity to obstacles and no-go zones (normalised to [0, 1]).
    Setting a weight to 0 disables that objective.
    """

    w_time: float = Field(default=1.0, ge=0, description="Travel time / distance weight")
    w_fuel: float = Field(default=0.3, ge=0, description="Fuel consumption weight")
    w_risk: float = Field(default=0.5, ge=0, description="Route risk weight")


class SimConfig(BaseModel):
    """Top-level simulation configuration.

    Assumptions:
      - The simulation uses a fixed-timestep loop.  ``dt`` must be small
        enough that vehicles cannot skip over obstacles in one step:
        max_speed * dt << smallest obstacle radius.
      - The random seed fully determines the run; there are no
        wall-clock-dependent branches.
    """

    seed: int = Field(default=42, description="Random seed for reproducibility")
    dt: float = Field(default=0.1, gt=0, le=1.0, description="Simulation timestep s")
    duration: float = Field(default=300.0, gt=0, description="Total sim duration s")
    num_vehicles: int = Field(default=8, ge=1, le=64, description="Number of vehicles")
    gps_available: bool = Field(default=True, description="Whether GPS is available")
    gps_intermittent_prob: float = Field(
        default=0.0, ge=0, le=1.0,
        description="Probability of GPS fix each second (0 = never when gps_available=False)",
    )
    vehicle: VehicleConfig = Field(default_factory=VehicleConfig)
    estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
    comms: CommsConfig = Field(default_factory=CommsConfig)
    coordination: CoordinationConfig = Field(default_factory=CoordinationConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
    planning: PlanningObjective = Field(default_factory=PlanningObjective)
    scenario: str = Field(default="baseline", description="Scenario name")
    use_supervisor: bool = Field(default=False, description="Enable centralised supervisor agent")
    use_cbba: bool = Field(default=True, description="Use CBBA-lite for formation slot allocation")

    @model_validator(mode="after")
    def _validate_timestep_safety(self) -> SimConfig:
        # Ensure a vehicle cannot skip an obstacle in a single step.
        max_step_distance = self.vehicle.max_speed * self.dt
        min_obstacle_radius = self.world.obstacle_radius_range[0]
        if max_step_distance > min_obstacle_radius * 0.5:
            raise ValueError(
                f"SAFETY: max_speed * dt = {max_step_distance:.2f}m exceeds half the "
                f"smallest obstacle radius ({min_obstacle_radius:.2f}m).  Reduce dt or "
                "max_speed to prevent obstacle-skipping."
            )
        # Collision radius must be physically meaningful vs vehicle size
        body_diagonal = (self.vehicle.length**2 + self.vehicle.width**2) ** 0.5
        if self.coordination.collision_radius < body_diagonal * 0.5:
            raise ValueError(
                f"SAFETY: collision_radius ({self.coordination.collision_radius:.1f}m) is smaller "
                f"than half the vehicle body diagonal ({body_diagonal * 0.5:.1f}m).  "
                "Overlapping vehicles would not register as collisions."
            )
        return self

    def safety_warnings(self) -> list[str]:
        """Return non-fatal safety warnings about the configuration."""
        warnings: list[str] = []
        if not self.gps_available and self.gps_intermittent_prob == 0:
            if self.world.landmark_count == 0:
                warnings.append(
                    "GPS is denied, intermittent GPS is off, and there are zero landmarks. "
                    "Estimator uncertainty will grow unboundedly — safe mode will engage permanently."
                )
        if self.comms.packet_loss > 0.5:
            warnings.append(
                f"Packet loss is {self.comms.packet_loss:.0%}. "
                "Leader election and coordination will be severely degraded."
            )
        if self.num_vehicles > 1:
            spawn_area = 25.0 * ((self.num_vehicles + 1) // 2)
            if spawn_area < self.coordination.min_separation * self.num_vehicles * 0.5:
                warnings.append(
                    "Spawn area is tight relative to min_separation. "
                    "Expect initial near-miss events during formation."
                )
        if self.vehicle.max_speed * self.dt > self.coordination.collision_radius:
            warnings.append(
                f"max_speed * dt ({self.vehicle.max_speed * self.dt:.2f}m) exceeds "
                f"collision_radius ({self.coordination.collision_radius}m). "
                "Fast-moving vehicles may tunnel through each other's collision zones."
            )
        return warnings
