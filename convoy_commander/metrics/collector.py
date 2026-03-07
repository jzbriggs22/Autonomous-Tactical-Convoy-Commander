"""Metrics collection and computation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from convoy_commander.vehicles.vehicle import Vehicle, VehicleStatus


@dataclass
class SimMetrics:
    """Aggregated simulation metrics."""

    mission_success: bool = False
    vehicles_arrived: int = 0
    vehicles_total: int = 0
    avg_time_to_destination: float = 0.0
    total_fuel_used: float = 0.0
    avg_fuel_used: float = 0.0
    convoy_cohesion_score: float = 0.0
    near_miss_count: int = 0
    collision_count: int = 0
    comms_total_sent: int = 0
    comms_total_delivered: int = 0
    comms_total_dropped: int = 0
    comms_delivery_ratio: float = 0.0
    avg_position_error: float = 0.0
    max_position_error: float = 0.0
    total_distance_traveled: float = 0.0
    num_leader_elections: int = 0
    num_safe_mode_activations: int = 0
    sim_duration: float = 0.0
    comms_by_type: dict = field(default_factory=dict)  # per-MessageType {sent, delivered, dropped}
    # Phase 6 realism metrics
    string_stability_max: float = 0.0
    string_stability_median: float = 0.0


@dataclass
class TimeSeriesEntry:
    """A single timestep of recorded data for one vehicle."""

    time: float
    vehicle_id: int
    true_x: float
    true_y: float
    est_x: float
    est_y: float
    speed: float
    heading: float
    fuel: float
    uncertainty: float
    status: str


class MetricsCollector:
    """Collects per-step data and computes final metrics."""

    def __init__(self) -> None:
        self.time_series: list[TimeSeriesEntry] = []
        self.cohesion_samples: list[float] = []
        self.arrival_times: dict[int, float] = {}
        self.leader_elections: int = 0
        self.safe_mode_activations: int = 0
        # Phase 6: string stability (set by runner after sim)
        self.string_stability_max: float = 0.0
        self.string_stability_median: float = 0.0

        # Per-step snapshots for plotting
        self.comms_adjacency_snapshots: list[dict[int, list[int]]] = []
        self.time_stamps: list[float] = []

        # Phase 7: per-step headway gaps for visualization
        self.headway_samples: list[dict] = []
        # Phase 7: per-step corridor distances for visualization
        self.corridor_samples: list[dict] = []
        # Phase 7: per-step spacing errors for string stability viz
        self.spacing_error_samples: list[dict] = []

    def record_step(
        self, time: float, vehicles: list[Vehicle]
    ) -> None:
        """Record data for one timestep."""
        for v in vehicles:
            self.time_series.append(
                TimeSeriesEntry(
                    time=time,
                    vehicle_id=v.id,
                    true_x=v.state.x,
                    true_y=v.state.y,
                    est_x=v.estimator.state.x,
                    est_y=v.estimator.state.y,
                    speed=v.state.speed,
                    heading=v.state.heading,
                    fuel=v.fuel.fuel,
                    uncertainty=v.estimator.state.uncertainty,
                    status=v.status.name,
                )
            )

        # Compute cohesion (avg pairwise distance of operational vehicles)
        operational = [v for v in vehicles if v.is_operational]
        if len(operational) >= 2:
            dists: list[float] = []
            for i in range(len(operational)):
                for j in range(i + 1, len(operational)):
                    d = operational[i].state.distance_to(operational[j].state)
                    dists.append(d)
            self.cohesion_samples.append(float(np.mean(dists)))

    def record_arrival(self, vehicle_id: int, time: float) -> None:
        if vehicle_id not in self.arrival_times:
            self.arrival_times[vehicle_id] = time

    def record_headway(self, time: float, vehicle_id: int, actual_gap: float, desired_gap: float) -> None:
        """Record a headway gap sample for visualization."""
        self.headway_samples.append({
            "time": time, "vehicle_id": vehicle_id,
            "actual_gap": actual_gap, "desired_gap": desired_gap,
        })

    def record_corridor_distance(self, time: float, vehicle_id: int, corridor_dist: float) -> None:
        """Record corridor distance sample for visualization."""
        self.corridor_samples.append({
            "time": time, "vehicle_id": vehicle_id, "corridor_dist": corridor_dist,
        })

    def record_spacing_error(self, time: float, vehicle_id: int, error: float) -> None:
        """Record spacing error sample for string stability visualization."""
        self.spacing_error_samples.append({
            "time": time, "vehicle_id": vehicle_id, "error": error,
        })

    def record_comms_adjacency(self, time: float, adj: dict[int, list[int]]) -> None:
        self.comms_adjacency_snapshots.append(adj)
        self.time_stamps.append(time)

    def compute_final(
        self,
        vehicles: list[Vehicle],
        comms_sent: int,
        comms_delivered: int,
        comms_dropped: int,
        duration: float,
        comms_by_type: dict | None = None,
    ) -> SimMetrics:
        """Compute final aggregate metrics."""
        m = SimMetrics()
        m.vehicles_total = len(vehicles)
        m.sim_duration = duration

        # Arrivals
        m.vehicles_arrived = sum(1 for v in vehicles if v.has_reached_destination())
        m.mission_success = m.vehicles_arrived >= max(1, m.vehicles_total // 2)

        # Time to destination
        if self.arrival_times:
            m.avg_time_to_destination = float(np.mean(list(self.arrival_times.values())))

        # Fuel
        fuel_used = [v.config.vehicle.fuel_capacity - v.fuel.fuel for v in vehicles]
        m.total_fuel_used = sum(fuel_used)
        m.avg_fuel_used = m.total_fuel_used / len(vehicles) if vehicles else 0

        # Cohesion
        m.convoy_cohesion_score = float(np.mean(self.cohesion_samples)) if self.cohesion_samples else 0.0

        # Collisions / near misses
        m.near_miss_count = sum(v.near_miss_count for v in vehicles)
        m.collision_count = sum(v.collision_count for v in vehicles)

        # Comms
        m.comms_total_sent = comms_sent
        m.comms_total_delivered = comms_delivered
        m.comms_total_dropped = comms_dropped
        m.comms_delivery_ratio = comms_delivered / comms_sent if comms_sent > 0 else 1.0

        # Position error
        errors: list[float] = []
        for v in vehicles:
            for (tx, ty), (ex, ey) in zip(v.true_history, v.est_history):
                errors.append(math.hypot(tx - ex, ty - ey))
        m.avg_position_error = float(np.mean(errors)) if errors else 0.0
        m.max_position_error = float(np.max(errors)) if errors else 0.0

        # Distance
        m.total_distance_traveled = sum(v.total_distance for v in vehicles)

        # Elections
        m.num_leader_elections = self.leader_elections
        m.num_safe_mode_activations = self.safe_mode_activations

        # Per-type comms stats
        m.comms_by_type = comms_by_type or {}

        # String stability (Phase 6, set by runner)
        m.string_stability_max = self.string_stability_max
        m.string_stability_median = self.string_stability_median

        return m

    def save_metrics(self, metrics: SimMetrics, path: Path) -> None:
        """Save metrics to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

    def save_time_series(self, path: Path) -> None:
        """Save time series to JSON lines."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in self.time_series:
                f.write(json.dumps(asdict(entry)) + "\n")
